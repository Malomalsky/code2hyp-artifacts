from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNNER_TAG = "codenet-java-stage-b-validation-runner-v1"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from geometry_profile_research.codenet_eligibility import canonical_json_bytes, stable_sha256
from geometry_profile_research.codenet_stage_a import load_codenet_split
from geometry_profile_research.codenet_stage_a_runner import (
    configure_torch_runtime,
    iter_jsonl,
    run_stage_a_validation_seed,
    torch_device_metadata,
)
from geometry_profile_research.codenet_stage_b import (
    build_stage_b_validation_selection,
    validate_stage_b_registration,
)
from scripts.run_codenet_stage_a_validation import _verified_implementation_state


def _artifact_sha(manifest: dict[str, Any], filename: str) -> str:
    return next(str(item["sha256"]) for item in manifest["artifacts"] if item["path"] == filename)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the registered CodeNet Java Stage B validation protocol.")
    parser.add_argument("--design", type=Path, default=PROJECT_ROOT / "configs/codenet_java_stage_b_draft_v1.json")
    parser.add_argument("--registration", type=Path, default=PROJECT_ROOT / "registrations/codenet_java_stage_b_registration_v1.json")
    parser.add_argument("--sampling-manifest", type=Path, default=PROJECT_ROOT / "data/codenet_java_stage_b_program_sampling_v1/program_sampling_manifest.json")
    parser.add_argument("--train-programs", type=Path, default=PROJECT_ROOT / "data/codenet_java_stage_b_program_sampling_v1/train_programs.jsonl")
    parser.add_argument("--validation-programs", type=Path, default=PROJECT_ROOT / "data/codenet_java_stage_b_program_sampling_v1/validation_programs.jsonl")
    parser.add_argument("--calibration-manifest", type=Path, default=PROJECT_ROOT / "data/codenet_java_stage_b_calibration_pairs_v1/calibration_pair_manifest.json")
    parser.add_argument("--calibration-pairs", type=Path, default=PROJECT_ROOT / "data/codenet_java_stage_b_calibration_pairs_v1/calibration_pairs.jsonl")
    parser.add_argument("--selected-ast-manifest", type=Path, default=PROJECT_ROOT / "data/codenet_java_stage_b_selected_source_ast_v1/selected_source_ast_manifest.json")
    parser.add_argument("--ast-index", type=Path, default=PROJECT_ROOT / "data/codenet_java_stage_b_selected_source_ast_v1/selected_source_ast_index.jsonl")
    parser.add_argument("--source-root", type=Path, default=PROJECT_ROOT / "data/codenet_java_stage_b_sources_v1")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs/codenet_java_stage_b_validation_v1")
    parser.add_argument("--seeds", type=int, nargs="*", default=None)
    args = parser.parse_args()

    implementation = _verified_implementation_state(PROJECT_ROOT, runner_tag=RUNNER_TAG)
    design_bytes = args.design.read_bytes()
    design = json.loads(design_bytes)
    registration_bytes = args.registration.read_bytes()
    registration = json.loads(registration_bytes)
    validate_stage_b_registration(design=design, registration=registration, design_bytes=design_bytes)
    freeze = design["freeze"]
    if freeze.get("implementation_commit") != implementation["commit"]:
        raise ValueError("Stage B runner commit differs from the frozen design")
    if not str(freeze.get("container_digest") or "").startswith("sha256:"):
        raise ValueError("Stage B container digest is not frozen")
    encoder = design["encoder_training"]
    registered_seeds = tuple(int(seed) for seed in encoder["model_seeds"])
    device = configure_torch_runtime(
        compute_device=str(encoder["compute_device"]),
        torch_num_threads=int(encoder["torch_num_threads_per_process"]),
        seed=registered_seeds[0],
    )
    print(json.dumps({"phase": "runtime_preflight", **torch_device_metadata(device)}, sort_keys=True), flush=True)

    sampling_bytes = args.sampling_manifest.read_bytes()
    sampling_manifest = json.loads(sampling_bytes)
    if sampling_manifest.get("schema_version") != "codenet-java-stage-b-program-sampling-v1":
        raise ValueError("unsupported Stage B sampling manifest")
    train_sha = stable_sha256(args.train_programs.read_bytes())
    validation_sha = stable_sha256(args.validation_programs.read_bytes())
    if train_sha != _artifact_sha(sampling_manifest, "train_programs.jsonl"):
        raise ValueError("Stage B train rows differ from the sampling manifest")
    if validation_sha != _artifact_sha(sampling_manifest, "validation_programs.jsonl"):
        raise ValueError("Stage B validation rows differ from the sampling manifest")

    calibration_bytes = args.calibration_manifest.read_bytes()
    calibration_manifest = json.loads(calibration_bytes)
    if calibration_manifest.get("schema_version") != "codenet-java-stage-b-calibration-pairs-v1":
        raise ValueError("unsupported Stage B calibration manifest")
    if stable_sha256(args.calibration_pairs.read_bytes()) != _artifact_sha(
        calibration_manifest, "calibration_pairs.jsonl"
    ):
        raise ValueError("Stage B calibration pairs differ from their manifest")
    calibration_pairs = tuple(iter_jsonl(args.calibration_pairs))

    ast_bytes = args.selected_ast_manifest.read_bytes()
    ast_manifest = json.loads(ast_bytes)
    if ast_manifest.get("schema_version") != "codenet-java-stage-b-selected-source-ast-audit-v1":
        raise ValueError("unsupported Stage B selected-source AST manifest")
    if ast_manifest.get("valid_for_stage_b_modeling") is not True:
        raise ValueError("Stage B selected-source AST audit did not pass")
    if stable_sha256(args.ast_index.read_bytes()) != _artifact_sha(ast_manifest, "selected_source_ast_index.jsonl"):
        raise ValueError("Stage B selected-source AST index differs from its manifest")

    quotas = design["eligibility"]["primary_role_upper_bound"]
    sampling = design["sampling"]
    split = load_codenet_split(
        source_root=args.source_root,
        train_path=args.train_programs,
        validation_path=args.validation_programs,
        ast_index_path=args.ast_index,
        language="java",
        train_clusters=int(quotas["train_clusters"]),
        validation_clusters=int(quotas["validation_clusters"]),
        train_programs_per_cluster=int(sampling["train_programs_per_cluster"]),
        validation_queries_per_cluster=int(sampling["validation_queries_per_cluster"]),
        validation_gallery_per_cluster=int(sampling["validation_gallery_per_cluster"]),
    )
    seeds = registered_seeds if args.seeds is None else tuple(args.seeds)
    if not seeds or any(seed not in registered_seeds for seed in seeds):
        raise ValueError("requested Stage B seeds must belong to the frozen seed list")

    transport = design["transport"]
    calibration = design["train_only_calibration"]
    active_curvatures = tuple(float(value) for value in design["geometry"]["active_curvature_candidates"])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    start_time = time.perf_counter()

    def progress(event: dict[str, Any]) -> None:
        print(json.dumps({"elapsed_seconds": round(time.perf_counter() - start_time, 1), **event}, sort_keys=True), flush=True)

    common = {
        "split": split,
        "calibration_pairs": calibration_pairs,
        "protocol_sha256": stable_sha256(design_bytes),
        "calibration_manifest_sha256": stable_sha256(calibration_bytes),
        "dim": int(encoder["dimension_per_role"]),
        "epochs": int(encoder["epochs"]),
        "batch_size": int(encoder["batch_size_programs"]),
        "learning_rate": float(encoder["learning_rate"]),
        "gradient_clip_norm": float(encoder["gradient_clip_global_norm"]),
        "lambda_edge": float(encoder["loss"]["edge_length_weight"]),
        "lambda_gromov": float(encoder["loss"]["soft_gromov_LCA_distortion_weight"]),
        "lambda_branch": float(encoder["loss"]["branch_length_weight"]),
        "max_paths": int(sampling["paths_per_program"]),
        "fit_all_roles_to_active_ball": True,
        "max_ball_fraction": float(calibration["maximum_ball_radius_fraction"]),
        "active_curvatures": active_curvatures,
        "near_zero_curvature": float(design["geometry"]["near_zero_curvature"]),
        "sinkhorn_kappa": float(calibration["sinkhorn_kappa"]),
        "sinkhorn_iterations": int(transport["sinkhorn_iterations"]),
        "projection_iterations": int(transport["projection_iterations_max"]),
        "marginal_tolerance": float(transport["maximum_marginal_residual"]),
        "query_batch_size": int(transport["query_batch_size"]),
        "gallery_batch_size": int(transport["gallery_batch_size"]),
        "torch_num_threads": int(encoder["torch_num_threads_per_process"]),
        "compute_device": str(encoder["compute_device"]),
        "implementation": {
            **implementation,
            "container_digest": freeze["container_digest"],
            "design_sha256": stable_sha256(design_bytes),
        },
        "progress_callback": progress,
    }
    for seed in seeds:
        progress({"phase": "seed_start", "seed": seed, "model": "prefix"})
        run_stage_a_validation_seed(
            **common,
            seed=seed,
            output_dir=args.output_dir / "prefix",
            node_input_mode="label_depth_prefix",
            include_all_role_hyperbolic=True,
        )
        progress({"phase": "seed_start", "seed": seed, "model": "label_only"})
        run_stage_a_validation_seed(
            **common,
            seed=seed,
            output_dir=args.output_dir / "label_only",
            node_input_mode="label_only",
            include_all_role_hyperbolic=False,
        )

    prefix_payloads = _complete_seed_payloads(args.output_dir / "prefix", registered_seeds)
    label_payloads = _complete_seed_payloads(args.output_dir / "label_only", registered_seeds)
    if len(prefix_payloads) == len(label_payloads) == len(registered_seeds):
        selection = build_stage_b_validation_selection(
            prefix_payloads,
            label_payloads,
            expected_seeds=registered_seeds,
            active_curvatures=active_curvatures,
        )
        selection["input"] = {
            "design_sha256": stable_sha256(design_bytes),
            "registration_sha256": stable_sha256(registration_bytes),
            "program_sampling_manifest_sha256": stable_sha256(sampling_bytes),
            "calibration_manifest_sha256": stable_sha256(calibration_bytes),
            "selected_source_ast_manifest_sha256": stable_sha256(ast_bytes),
        }
        selection_path = args.output_dir / "validation_selection_record.json"
        selection_path.write_bytes(canonical_json_bytes(selection))
        progress({"phase": "validation_selection_complete", "selected_active_curvature": selection["selected_active_curvature"]})
    else:
        progress({"phase": "validation_selection_pending", "prefix_seeds": len(prefix_payloads), "label_only_seeds": len(label_payloads)})


def _complete_seed_payloads(directory: Path, seeds: tuple[int, ...]) -> list[dict[str, Any]]:
    payloads = []
    for seed in seeds:
        path = directory / f"seed_{seed}_validation.json"
        if path.exists():
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("status") == "complete":
                payloads.append(payload)
    return payloads


if __name__ == "__main__":
    main()
