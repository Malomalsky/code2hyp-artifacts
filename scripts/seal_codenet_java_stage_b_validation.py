from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from geometry_profile_research.codenet_eligibility import canonical_json_bytes, stable_sha256
from geometry_profile_research.codenet_stage_a_evaluation import summarize_problem_macro_retrieval
from geometry_profile_research.codenet_stage_a_runner import all_role_curvature_cell_id, curvature_cell_id
from geometry_profile_research.codenet_stage_b import build_stage_b_validation_selection
from scripts.run_codenet_java_stage_b_validation import RUNNER_TAG


def _validation_metadata(path: Path) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    rows = [json.loads(line) for line in path.read_bytes().splitlines() if line]
    query = [row for row in rows if row["role"] == "query"]
    gallery = [row for row in rows if row["role"] == "gallery"]
    return (
        tuple(str(row["source_relpath"]) for row in query),
        tuple(str(row["cluster_id"]) for row in query),
        tuple(str(row["source_relpath"]) for row in gallery),
        tuple(str(row["cluster_id"]) for row in gallery),
    )


def seal_stage_b_seed_result(
    *,
    result_path: Path,
    design: Mapping[str, Any],
    design_sha256: str,
    calibration_manifest_sha256: str,
    validation_programs_path: Path,
    model: str,
    output_path: Path,
) -> dict[str, Any]:
    """Recompute one Stage B validation seed from its distance matrices and seal it."""

    if model not in {"prefix", "label_only"}:
        raise ValueError("Stage B model must be prefix or label_only")
    result_bytes = result_path.read_bytes()
    result = json.loads(result_bytes)
    if result.get("status") != "complete":
        raise ValueError("Stage B validation seed is incomplete")
    if result.get("protocol_sha256") != design_sha256:
        raise ValueError("Stage B validation seed used a different design")
    if result.get("calibration_manifest_sha256") != calibration_manifest_sha256:
        raise ValueError("Stage B validation seed used different calibration pairs")
    if any(
        bool(result.get(flag))
        for flag in ("test_program_ids_materialized", "test_relevance_labels_opened", "test_retrieval_metrics_computed")
    ):
        raise ValueError("Stage B validation result indicates forbidden test access")
    implementation = result.get("implementation", {})
    if implementation.get("commit") != design["freeze"]["implementation_commit"]:
        raise ValueError("Stage B validation result used a different implementation commit")
    if implementation.get("tag") != RUNNER_TAG or implementation.get("container_digest") != design["freeze"]["container_digest"]:
        raise ValueError("Stage B validation result used a different runner tag or container")
    _verify_execution_config(result["execution_config"], design=design, model=model)

    checkpoint = result["checkpoint"]
    checkpoint_path = result_path.parent / str(checkpoint["path"])
    if stable_sha256(checkpoint_path.read_bytes()) != str(checkpoint["sha256"]):
        raise ValueError("Stage B encoder checkpoint hash mismatch")
    with torch.serialization.safe_globals([torch.torch_version.TorchVersion]):
        checkpoint_payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if int(checkpoint_payload.get("seed", -1)) != int(result["seed"]):
        raise ValueError("Stage B checkpoint seed mismatch")
    if checkpoint_payload.get("execution_config") != result["execution_config"]:
        raise ValueError("Stage B checkpoint execution configuration mismatch")
    if checkpoint_payload.get("training_metadata", {}).get("compute_device") != str(
        design["encoder_training"]["compute_device"]
    ):
        raise ValueError("Stage B checkpoint was trained on a different compute device")
    state = checkpoint_payload.get("model_state_dict")
    if not isinstance(state, Mapping) or not state or any(
        not isinstance(value, torch.Tensor) or not bool(torch.isfinite(value).all())
        for value in state.values()
    ):
        raise ValueError("Stage B checkpoint contains an invalid model state")

    active = tuple(float(value) for value in design["geometry"]["active_curvature_candidates"])
    expected_cells = {
        "EEE_true_LCA",
        "EEE_zero_anchor",
        "HEE_near_zero_true_LCA",
        *(curvature_cell_id(value) for value in active),
    }
    if model == "prefix":
        expected_cells.update(all_role_curvature_cell_id(value) for value in active)
    if set(result["cells"]) != expected_cells:
        raise ValueError("Stage B seed does not contain the frozen validation cells")

    query_ids, query_clusters, gallery_ids, gallery_clusters = _validation_metadata(validation_programs_path)
    expected_shape = [len(query_ids), len(gallery_ids)]
    expected_problem_count = int(design["eligibility"]["primary_role_upper_bound"]["validation_clusters"])
    artifacts = [
        {
            "role": "encoder_checkpoint",
            "path": checkpoint_path.name,
            "bytes": checkpoint_path.stat().st_size,
            "sha256": str(checkpoint["sha256"]),
        }
    ]
    for cell_id, cell in sorted(result["cells"].items()):
        distance = cell["distance_matrix"]
        if distance["shape"] != expected_shape or distance["dtype"] != "float64":
            raise ValueError(f"Stage B cell {cell_id} has an unexpected distance contract")
        distance_path = result_path.parent / str(distance["path"])
        distance_sha = stable_sha256(distance_path.read_bytes())
        if distance_sha != str(distance["sha256"]):
            raise ValueError(f"Stage B cell {cell_id} distance hash mismatch")
        values = torch.load(distance_path, map_location="cpu", weights_only=True)
        if not isinstance(values, torch.Tensor) or list(values.shape) != expected_shape or values.dtype != torch.float64:
            raise ValueError(f"Stage B cell {cell_id} distance tensor violates shape or dtype")
        if not bool(torch.isfinite(values).all()):
            raise ValueError(f"Stage B cell {cell_id} contains non-finite distances")
        if (
            float(values.min()) != float(distance["minimum"])
            or float(values.max()) != float(distance["maximum"])
            or int((values < 0.0).sum()) != int(distance["negative_count"])
        ):
            raise ValueError(f"Stage B cell {cell_id} distance metadata mismatch")
        recomputed = summarize_problem_macro_retrieval(
            values,
            query_ids=query_ids,
            query_cluster_ids=query_clusters,
            gallery_ids=gallery_ids,
            gallery_cluster_ids=gallery_clusters,
            r=8,
        )
        if canonical_json_bytes(asdict(recomputed)) != canonical_json_bytes(cell["metrics"]):
            raise ValueError(f"Stage B cell {cell_id} metrics do not match its distance matrix")
        if int(cell["metrics"]["problem_count"]) != expected_problem_count:
            raise ValueError(f"Stage B cell {cell_id} has an unexpected problem count")
        artifacts.append(
            {
                "role": f"distance_matrix:{cell_id}",
                "path": distance_path.name,
                "bytes": distance_path.stat().st_size,
                "sha256": distance_sha,
            }
        )
    seal = {
        "schema_version": "code2hyp-codenet-java-stage-b-validation-seed-seal-v1",
        "model": model,
        "seed": int(result["seed"]),
        "inputs": {
            "result": {"path": result_path.name, "bytes": len(result_bytes), "sha256": stable_sha256(result_bytes)},
            "design_sha256": design_sha256,
            "calibration_manifest_sha256": calibration_manifest_sha256,
            "validation_programs_sha256": stable_sha256(validation_programs_path.read_bytes()),
        },
        "artifacts": artifacts,
        "checks": {
            "validation_only": True,
            "checkpoint_content_validated": True,
            "checkpoint_compute_device_validated": True,
            "all_metrics_recomputed_from_distance_matrices": True,
            "cell_set_matches_design": True,
        },
        "test_program_ids_materialized": False,
        "test_relevance_labels_opened": False,
        "test_retrieval_metrics_computed": False,
    }
    content = canonical_json_bytes(seal)
    if output_path.exists() and output_path.read_bytes() != content:
        raise FileExistsError(f"refusing to overwrite a different Stage B seed seal: {output_path}")
    output_path.write_bytes(content)
    return seal


def seal_stage_b_validation(
    *,
    design_path: Path,
    calibration_manifest_path: Path,
    sampling_manifest_path: Path,
    selected_ast_manifest_path: Path,
    validation_programs_path: Path,
    validation_output_dir: Path,
) -> dict[str, Any]:
    design_bytes = design_path.read_bytes()
    design = json.loads(design_bytes)
    design_sha = stable_sha256(design_bytes)
    calibration_sha = stable_sha256(calibration_manifest_path.read_bytes())
    seeds = tuple(int(seed) for seed in design["encoder_training"]["model_seeds"])
    payloads: dict[str, list[dict[str, Any]]] = {"prefix": [], "label_only": []}
    seed_inputs = []
    for model in ("prefix", "label_only"):
        for seed in seeds:
            result_path = validation_output_dir / model / f"seed_{seed}_validation.json"
            seal_path = validation_output_dir / model / f"seed_{seed}_validation_seal.json"
            seal = seal_stage_b_seed_result(
                result_path=result_path,
                design=design,
                design_sha256=design_sha,
                calibration_manifest_sha256=calibration_sha,
                validation_programs_path=validation_programs_path,
                model=model,
                output_path=seal_path,
            )
            payloads[model].append(json.loads(result_path.read_text(encoding="utf-8")))
            seed_inputs.append(
                {
                    "model": model,
                    "seed": seed,
                    "result_path": f"{model}/{result_path.name}",
                    "result_sha256": seal["inputs"]["result"]["sha256"],
                    "seal_path": f"{model}/{seal_path.name}",
                    "seal_sha256": stable_sha256(seal_path.read_bytes()),
                }
            )
    selection = build_stage_b_validation_selection(
        payloads["prefix"],
        payloads["label_only"],
        expected_seeds=seeds,
        active_curvatures=tuple(float(value) for value in design["geometry"]["active_curvature_candidates"]),
    )
    selection["input"] = {
        "design_sha256": design_sha,
        "program_sampling_manifest_sha256": stable_sha256(sampling_manifest_path.read_bytes()),
        "calibration_manifest_sha256": calibration_sha,
        "selected_source_ast_manifest_sha256": stable_sha256(selected_ast_manifest_path.read_bytes()),
    }
    selection_path = validation_output_dir / "validation_selection_record.json"
    selection_bytes = canonical_json_bytes(selection)
    if selection_path.exists() and selection_path.read_bytes() != selection_bytes:
        raise ValueError("stored Stage B validation selection differs from the recomputed result")
    selection_path.write_bytes(selection_bytes)
    seal = {
        "schema_version": "code2hyp-codenet-java-stage-b-validation-selection-seal-v1",
        "status": "validation_selection_sealed_test_unopened",
        "inputs": {
            "selection": {"path": selection_path.name, "sha256": stable_sha256(selection_bytes)},
            "seeds": seed_inputs,
        },
        "selected_active_curvature": selection["selected_active_curvature"],
        "selected_prefix_HEE_cell_id": selection["selected_prefix_HEE_cell_id"],
        "selected_prefix_HHH_cell_id": selection["selected_prefix_HHH_cell_id"],
        "test_cell_plan": selection["test_cell_plan"],
        "checks": {
            "registered_seed_set_complete_for_both_models": True,
            "all_seed_results_recomputed_and_sealed": True,
            "selection_recomputed_from_frozen_rule": True,
            "validation_only": True,
        },
        "java_test_program_ids_materialized": False,
        "java_test_relevance_labels_opened": False,
        "java_test_retrieval_metrics_computed": False,
    }
    seal_path = validation_output_dir / "validation_selection_record_seal.json"
    seal_bytes = canonical_json_bytes(seal)
    if seal_path.exists() and seal_path.read_bytes() != seal_bytes:
        raise FileExistsError("refusing to overwrite a different Stage B validation-selection seal")
    seal_path.write_bytes(seal_bytes)
    return seal


def _verify_execution_config(actual: Mapping[str, Any], *, design: Mapping[str, Any], model: str) -> None:
    encoder = design["encoder_training"]
    calibration = design["train_only_calibration"]
    transport = design["transport"]
    sampling = design["sampling"]
    expected = {
        "dim": int(encoder["dimension_per_role"]),
        "epochs": int(encoder["epochs"]),
        "batch_size": int(encoder["batch_size_programs"]),
        "learning_rate": float(encoder["learning_rate"]),
        "gradient_clip_norm": float(encoder["gradient_clip_global_norm"]),
        "lambda_edge": float(encoder["loss"]["edge_length_weight"]),
        "lambda_gromov": float(encoder["loss"]["soft_gromov_LCA_distortion_weight"]),
        "lambda_branch": float(encoder["loss"]["branch_length_weight"]),
        "max_paths": int(sampling["paths_per_program"]),
        "max_ball_fraction": float(calibration["maximum_ball_radius_fraction"]),
        "active_curvatures": [float(value) for value in design["geometry"]["active_curvature_candidates"]],
        "near_zero_curvature": float(design["geometry"]["near_zero_curvature"]),
        "sinkhorn_kappa": float(calibration["sinkhorn_kappa"]),
        "sinkhorn_iterations": int(transport["sinkhorn_iterations"]),
        "projection_iterations": int(transport["projection_iterations_max"]),
        "marginal_tolerance": float(transport["maximum_marginal_residual"]),
        "query_batch_size": int(transport["query_batch_size"]),
        "gallery_batch_size": int(transport["gallery_batch_size"]),
        "torch_num_threads": int(encoder["torch_num_threads_per_process"]),
        "compute_device": str(encoder["compute_device"]),
        "fit_all_roles_to_active_ball": True,
    }
    if model == "prefix":
        expected["node_input_mode"] = "label_depth_prefix"
        expected["include_all_role_hyperbolic"] = True
    if dict(actual) != expected:
        raise ValueError(f"Stage B {model} execution configuration differs from the frozen design")


def main() -> None:
    parser = argparse.ArgumentParser(description="Recompute and seal all CodeNet Java Stage B validation artifacts.")
    parser.add_argument("--design", type=Path, default=PROJECT_ROOT / "configs/codenet_java_stage_b_draft_v1.json")
    parser.add_argument("--calibration-manifest", type=Path, default=PROJECT_ROOT / "data/codenet_java_stage_b_calibration_pairs_v1/calibration_pair_manifest.json")
    parser.add_argument("--sampling-manifest", type=Path, default=PROJECT_ROOT / "data/codenet_java_stage_b_program_sampling_v1/program_sampling_manifest.json")
    parser.add_argument("--selected-ast-manifest", type=Path, default=PROJECT_ROOT / "data/codenet_java_stage_b_selected_source_ast_v1/selected_source_ast_manifest.json")
    parser.add_argument("--validation-programs", type=Path, default=PROJECT_ROOT / "data/codenet_java_stage_b_program_sampling_v1/validation_programs.jsonl")
    parser.add_argument("--validation-output-dir", type=Path, default=PROJECT_ROOT / "outputs/codenet_java_stage_b_validation_v1")
    args = parser.parse_args()
    seal = seal_stage_b_validation(
        design_path=args.design,
        calibration_manifest_path=args.calibration_manifest,
        sampling_manifest_path=args.sampling_manifest,
        selected_ast_manifest_path=args.selected_ast_manifest,
        validation_programs_path=args.validation_programs,
        validation_output_dir=args.validation_output_dir,
    )
    print(json.dumps({"selected_active_curvature": seal["selected_active_curvature"], "sealed_seed_results": len(seal["inputs"]["seeds"])}, indent=2))


if __name__ == "__main__":
    main()
