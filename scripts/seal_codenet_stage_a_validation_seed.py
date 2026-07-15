from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from geometry_profile_research.codenet_eligibility import canonical_json_bytes, stable_sha256
from geometry_profile_research.codenet_stage_a_runner import curvature_cell_id


RUNNER_COMMIT = "469cbabc6692d1bc6cfde8cbb33c7ad79f8c9093"
RUNNER_TAG = "codenet-stage-a-validation-runner-v3"


def seal_seed_result(
    *,
    result_path: Path,
    protocol_path: Path,
    calibration_manifest_path: Path,
    gate0_path: Path,
    rounding_addendum_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Verify and seal one complete validation seed without reading source data."""

    result_bytes = result_path.read_bytes()
    protocol_bytes = protocol_path.read_bytes()
    calibration_bytes = calibration_manifest_path.read_bytes()
    gate0_bytes = gate0_path.read_bytes()
    rounding_addendum_bytes = rounding_addendum_path.read_bytes()
    result = json.loads(result_bytes)
    protocol = json.loads(protocol_bytes)
    calibration = json.loads(calibration_bytes)
    gate0 = json.loads(gate0_bytes)
    rounding_addendum = json.loads(rounding_addendum_bytes)
    if result.get("status") != "complete":
        raise ValueError("validation seed result is not complete")
    if result.get("protocol_sha256") != stable_sha256(protocol_bytes):
        raise ValueError("seed result does not match the frozen protocol")
    if result.get("calibration_manifest_sha256") != stable_sha256(calibration_bytes):
        raise ValueError("seed result does not match the frozen calibration manifest")
    if gate0.get("status") != "passed":
        raise ValueError("numerical Gate 0 did not pass")
    if gate0.get("schema_version") != "code2hyp-stage-a-gate0-numerical-v2":
        raise ValueError("numerical Gate 0 is not the rounding-aware v2 gate")
    if gate0.get("protocol", {}).get("sha256") != stable_sha256(protocol_bytes):
        raise ValueError("numerical Gate 0 used a different protocol")
    rounding_sha256 = stable_sha256(rounding_addendum_bytes)
    if gate0.get("transport_rounding_addendum", {}).get("sha256") != rounding_sha256:
        raise ValueError("numerical Gate 0 used a different transport rounding addendum")
    if rounding_addendum.get("inputs", {}).get("model_analysis_protocol", {}).get("sha256") != stable_sha256(protocol_bytes):
        raise ValueError("transport rounding addendum used a different protocol")
    expected_implementation = {
        "repository": "https://github.com/Malomalsky/code2hyp-artifacts",
        "commit": RUNNER_COMMIT,
        "tag": RUNNER_TAG,
        "tracked_worktree_clean": True,
    }
    if result.get("implementation") != expected_implementation:
        raise ValueError("seed result was not produced by the immutable validation runner")
    _verify_execution_config(result["execution_config"], protocol)

    expected_cells = {
        "EEE_true_LCA",
        "EEE_zero_anchor",
        "HEE_near_zero_true_LCA",
        *(
            curvature_cell_id(float(cell["factor_curvatures"][0]))
            for cell in protocol["geometry_cells"]["gate_C_active_candidates"]
        ),
    }
    if set(result["cells"]) != expected_cells:
        raise ValueError("seed result does not contain exactly the frozen validation cells")
    result_dir = result_path.parent
    checkpoint = result["checkpoint"]
    checkpoint_path = result_dir / str(checkpoint["path"])
    expected_checkpoint_name = f"seed_{int(result['seed'])}_encoder.pt"
    if checkpoint_path.name != expected_checkpoint_name:
        raise ValueError("encoder checkpoint filename does not identify the validation seed")
    if stable_sha256(checkpoint_path.read_bytes()) != str(checkpoint["sha256"]):
        raise ValueError("encoder checkpoint hash mismatch")
    _verify_checkpoint(
        checkpoint_path,
        result=result,
        protocol=protocol,
    )

    artifact_rows = [
        {
            "role": "encoder_checkpoint",
            "path": checkpoint_path.name,
            "bytes": checkpoint_path.stat().st_size,
            "sha256": str(checkpoint["sha256"]),
        }
    ]
    observed_distance_paths: set[str] = set()
    for cell_id, cell in sorted(result["cells"].items()):
        metrics = cell["metrics"]
        _verify_metrics(metrics, cell_id=cell_id)
        _verify_cell_geometry(cell, cell_id=cell_id, protocol=protocol)
        distance = cell["distance_matrix"]
        if distance["shape"] != [776, 776] or distance["dtype"] != "float64":
            raise ValueError(f"cell {cell_id!r} has an unexpected distance matrix contract")
        distance_path = result_dir / str(distance["path"])
        expected_distance_name = f"seed_{int(result['seed'])}_{cell_id}_distances.pt"
        if distance_path.name != expected_distance_name:
            raise ValueError(f"cell {cell_id!r} distance filename does not match the frozen contract")
        if distance_path.name in observed_distance_paths:
            raise ValueError("validation cells reuse one distance artifact")
        observed_distance_paths.add(distance_path.name)
        distance_sha = stable_sha256(distance_path.read_bytes())
        if distance_sha != str(distance["sha256"]):
            raise ValueError(f"cell {cell_id!r} distance matrix hash mismatch")
        _verify_distance_tensor(distance_path, metadata=distance, cell_id=cell_id)
        artifact_rows.append(
            {
                "role": f"distance_matrix:{cell_id}",
                "path": distance_path.name,
                "bytes": distance_path.stat().st_size,
                "sha256": distance_sha,
            }
        )
    if (
        bool(result.get("test_program_ids_materialized"))
        or bool(result.get("test_relevance_labels_opened"))
        or bool(result.get("test_retrieval_metrics_computed"))
    ):
        raise ValueError("seed result indicates forbidden test access")

    manifest = {
        "schema_version": "code2hyp-stage-a-validation-seed-seal-v1",
        "experiment_role": "verified_validation_only_seed_result",
        "seed": int(result["seed"]),
        "implementation": {
            "repository": "https://github.com/Malomalsky/code2hyp-artifacts",
            "commit": RUNNER_COMMIT,
            "tag": RUNNER_TAG,
        },
        "inputs": {
            "result": {
                "path": result_path.name,
                "bytes": len(result_bytes),
                "sha256": stable_sha256(result_bytes),
            },
            "protocol": {
                "path": str(protocol_path.relative_to(PROJECT_ROOT)),
                "sha256": stable_sha256(protocol_bytes),
            },
            "calibration_manifest": {
                "path": str(calibration_manifest_path.relative_to(PROJECT_ROOT)),
                "sha256": stable_sha256(calibration_bytes),
            },
            "gate0": {
                "path": str(gate0_path.relative_to(PROJECT_ROOT)),
                "sha256": stable_sha256(gate0_bytes),
            },
            "transport_rounding_addendum": {
                "path": str(rounding_addendum_path.relative_to(PROJECT_ROOT)),
                "sha256": rounding_sha256,
            },
        },
        "artifacts": artifact_rows,
        "checks": {
            "status_complete": True,
            "execution_config_matches_protocol": True,
            "immutable_runner_provenance_verified": True,
            "rounding_aware_Gate_0_passed": True,
            "cell_set_matches_protocol": True,
            "query_count_per_cell": 776,
            "problem_count_per_cell": 97,
            "distance_shape_per_cell": [776, 776],
            "distance_dtype": "float64",
            "all_hashes_match": True,
            "checkpoint_content_validated": True,
            "distance_tensor_content_validated": True,
            "MAP_at_8_aggregates_recomputed": True,
            "validation_only": True,
        },
        "test_program_ids_materialized": False,
        "test_relevance_labels_opened": False,
        "test_retrieval_metrics_computed": False,
    }
    content = canonical_json_bytes(manifest)
    if output_path.exists() and output_path.read_bytes() != content:
        raise ValueError(f"refusing to overwrite a different seed seal: {output_path}")
    output_path.write_bytes(content)
    return manifest


def _verify_checkpoint(
    checkpoint_path: Path,
    *,
    result: Mapping[str, Any],
    protocol: Mapping[str, Any],
) -> None:
    with torch.serialization.safe_globals([torch.torch_version.TorchVersion]):
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, Mapping):
        raise ValueError("encoder checkpoint is not a mapping")
    if int(checkpoint.get("seed", -1)) != int(result["seed"]):
        raise ValueError("encoder checkpoint seed mismatch")
    if checkpoint.get("protocol_sha256") != result.get("protocol_sha256"):
        raise ValueError("encoder checkpoint protocol hash mismatch")
    if checkpoint.get("calibration_manifest_sha256") != result.get("calibration_manifest_sha256"):
        raise ValueError("encoder checkpoint calibration hash mismatch")
    if checkpoint.get("execution_config") != result.get("execution_config"):
        raise ValueError("encoder checkpoint execution configuration mismatch")
    if checkpoint.get("implementation") != result.get("implementation"):
        raise ValueError("encoder checkpoint implementation provenance mismatch")
    if checkpoint.get("training_history") != result.get("training_history"):
        raise ValueError("encoder checkpoint and result contain different training histories")
    history = checkpoint.get("training_history")
    expected_epochs = int(protocol["encoder_training"]["epochs"])
    if not isinstance(history, list) or len(history) != expected_epochs:
        raise ValueError("encoder checkpoint contains an incomplete training history")
    for expected_epoch, row in enumerate(history, start=1):
        if int(float(row.get("epoch", -1))) != expected_epoch:
            raise ValueError("encoder checkpoint training epochs are not contiguous")
        if int(float(row.get("program_count", -1))) != 18_560:
            raise ValueError("encoder checkpoint epoch has an unexpected program count")
        if int(float(row.get("update_count", -1))) != 2_320:
            raise ValueError("encoder checkpoint epoch has an unexpected update count")
        if any(not math.isfinite(float(value)) for value in row.values()):
            raise ValueError("encoder checkpoint training history contains a non-finite value")
    state = checkpoint.get("model_state_dict")
    if not isinstance(state, Mapping) or not state:
        raise ValueError("encoder checkpoint has no model state")
    if any(
        not isinstance(value, torch.Tensor) or not bool(torch.isfinite(value).all())
        for value in state.values()
    ):
        raise ValueError("encoder checkpoint model state contains an invalid tensor")


def _verify_metrics(metrics: Mapping[str, Any], *, cell_id: str) -> None:
    if int(metrics["query_count"]) != 776 or int(metrics["problem_count"]) != 97:
        raise ValueError(f"cell {cell_id!r} has unexpected validation cardinalities")
    query_scores = metrics["query_scores"]
    task_scores = metrics["task_scores"]
    if len(query_scores) != 776 or len(task_scores) != 97:
        raise ValueError(f"cell {cell_id!r} has incomplete query/task scores")
    if any(not math.isfinite(float(value)) or not 0.0 <= float(value) <= 1.0 for value in query_scores.values()):
        raise ValueError(f"cell {cell_id!r} has an invalid query AP@8")
    if any(not math.isfinite(float(value)) or not 0.0 <= float(value) <= 1.0 for value in task_scores.values()):
        raise ValueError(f"cell {cell_id!r} has an invalid problem MAP@8")
    problem_macro = sum(float(value) for value in task_scores.values()) / len(task_scores)
    query_macro = sum(float(value) for value in query_scores.values()) / len(query_scores)
    if not math.isclose(problem_macro, float(metrics["problem_macro_map_at_r"]), rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(f"cell {cell_id!r} problem-macro MAP@8 does not match task scores")
    if not math.isclose(query_macro, float(metrics["query_macro_map_at_r"]), rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(f"cell {cell_id!r} query-macro MAP@8 does not match query scores")
    for name in ("mrr", "recall_at_1", "recall_at_5", "recall_at_10"):
        value = float(metrics[name])
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"cell {cell_id!r} has an invalid {name}")
    first_rank = float(metrics["mean_first_relevant_rank"])
    if not math.isfinite(first_rank) or not 1.0 <= first_rank <= 776.0:
        raise ValueError(f"cell {cell_id!r} has an invalid mean first relevant rank")


def _verify_cell_geometry(
    cell: Mapping[str, Any],
    *,
    cell_id: str,
    protocol: Mapping[str, Any],
) -> None:
    active = {
        curvature_cell_id(float(spec["factor_curvatures"][0])): list(spec["factor_curvatures"])
        for spec in protocol["geometry_cells"]["gate_C_active_candidates"]
    }
    near_zero = list(protocol["geometry_cells"]["gate_C_fixed_controls"][1]["factor_curvatures"])
    expected = {
        "EEE_true_LCA": [0.0, 0.0, 0.0],
        "EEE_zero_anchor": [0.0, 0.0, 0.0],
        "HEE_near_zero_true_LCA": near_zero,
        **active,
    }
    if [float(value) for value in cell["factor_curvatures"]] != [float(value) for value in expected[cell_id]]:
        raise ValueError(f"cell {cell_id!r} uses unexpected factor curvatures")
    weights = [float(value) for value in cell["factor_weights"]]
    if len(weights) != 3 or any(not math.isfinite(value) or value <= 0.0 for value in weights):
        raise ValueError(f"cell {cell_id!r} uses invalid factor weights")


def _verify_distance_tensor(
    distance_path: Path,
    *,
    metadata: Mapping[str, Any],
    cell_id: str,
) -> None:
    values = torch.load(distance_path, map_location="cpu", weights_only=True)
    if not isinstance(values, torch.Tensor):
        raise ValueError(f"cell {cell_id!r} distance artifact is not a tensor")
    if list(values.shape) != [776, 776] or values.dtype != torch.float64:
        raise ValueError(f"cell {cell_id!r} distance tensor violates shape or dtype")
    if not bool(torch.isfinite(values).all()):
        raise ValueError(f"cell {cell_id!r} distance tensor contains non-finite values")
    if float(values.min()) != float(metadata["minimum"]):
        raise ValueError(f"cell {cell_id!r} distance minimum metadata mismatch")
    if float(values.max()) != float(metadata["maximum"]):
        raise ValueError(f"cell {cell_id!r} distance maximum metadata mismatch")
    if int((values < 0.0).sum()) != int(metadata["negative_count"]):
        raise ValueError(f"cell {cell_id!r} negative distance count metadata mismatch")


def _verify_execution_config(actual: Mapping[str, Any], protocol: Mapping[str, Any]) -> None:
    expected = {
        "dim": int(protocol["encoder_training"]["dimension_per_role"]),
        "epochs": int(protocol["encoder_training"]["epochs"]),
        "batch_size": int(protocol["encoder_training"]["batch_size_programs"]),
        "learning_rate": float(protocol["encoder_training"]["learning_rate"]),
        "gradient_clip_norm": float(protocol["encoder_training"]["gradient_clip_global_norm"]),
        "lambda_edge": float(protocol["encoder_training"]["loss"]["edge_length_weight"]),
        "lambda_gromov": float(protocol["encoder_training"]["loss"]["soft_gromov_LCA_distortion_weight"]),
        "lambda_branch": float(protocol["encoder_training"]["loss"]["branch_length_weight"]),
        "max_paths": int(protocol["representation"]["path_count"]),
        "max_ball_fraction": float(
            protocol["train_only_calibration"]["coordinate_scaling"]["maximum_ball_radius_fraction"]
        ),
        "active_curvatures": [
            float(cell["factor_curvatures"][0])
            for cell in protocol["geometry_cells"]["gate_C_active_candidates"]
        ],
        "near_zero_curvature": float(
            protocol["geometry_cells"]["gate_C_fixed_controls"][1]["factor_curvatures"][0]
        ),
        "sinkhorn_kappa": float(protocol["train_only_calibration"]["sinkhorn_scale"]["kappa"]),
        "sinkhorn_iterations": int(protocol["transport"]["log_domain_iterations"]),
        "projection_iterations": int(protocol["transport"]["projection_iterations_max"]),
        "marginal_tolerance": float(protocol["transport"]["maximum_marginal_residual"]),
        "query_batch_size": int(protocol["transport"]["query_batch_size"]),
        "gallery_batch_size": int(protocol["transport"]["gallery_batch_size"]),
        "torch_num_threads": 1,
    }
    if dict(actual) != expected:
        raise ValueError("seed execution configuration differs from the frozen protocol")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Verify and seal one CodeNet Stage A validation seed.")
    parser.add_argument("result", type=Path)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=PROJECT_ROOT / "configs/codenet_python800_stage_a_model_analysis_protocol_v1.json",
    )
    parser.add_argument(
        "--calibration-manifest",
        type=Path,
        default=PROJECT_ROOT / "data/codenet_python800_stage_a_calibration_pairs/calibration_pair_manifest.json",
    )
    parser.add_argument(
        "--gate0",
        type=Path,
        default=PROJECT_ROOT / "reports/codenet_python800_stage_a_gate0_numerical_rounding_v2.json",
    )
    parser.add_argument(
        "--rounding-addendum",
        type=Path,
        default=PROJECT_ROOT / "configs/codenet_python800_stage_a_transport_rounding_addendum_v1.json",
    )
    parser.add_argument("--output", type=Path, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output = args.output or args.result.with_name(args.result.stem + "_seal.json")
    manifest = seal_seed_result(
        result_path=args.result,
        protocol_path=args.protocol,
        calibration_manifest_path=args.calibration_manifest,
        gate0_path=args.gate0,
        rounding_addendum_path=args.rounding_addendum,
        output_path=output,
    )
    print(json.dumps(manifest["checks"], indent=2, sort_keys=True))
    print(f"seal={output}")


if __name__ == "__main__":
    main()
