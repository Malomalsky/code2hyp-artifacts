from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from geometry_profile_research.codenet_eligibility import canonical_json_bytes, stable_sha256
from geometry_profile_research.codenet_stage_a import load_stage_a_split
from geometry_profile_research.codenet_stage_a_runner import (
    iter_jsonl,
    run_stage_a_validation_seed,
    select_active_curvature,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the frozen CodeNet Python800 Stage A validation protocol without opening test IDs."
    )
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
        "--calibration-pairs",
        type=Path,
        default=PROJECT_ROOT / "data/codenet_python800_stage_a_calibration_pairs/calibration_pairs.jsonl",
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=PROJECT_ROOT / "data/external_raw/codenet_python800_extracted/Project_CodeNet_Python800",
    )
    parser.add_argument(
        "--train-programs",
        type=Path,
        default=PROJECT_ROOT / "data/codenet_python800_stage_a_program_sampling/train_programs.jsonl",
    )
    parser.add_argument(
        "--validation-programs",
        type=Path,
        default=PROJECT_ROOT / "data/codenet_python800_stage_a_program_sampling/validation_programs.jsonl",
    )
    parser.add_argument(
        "--ast-index",
        type=Path,
        default=PROJECT_ROOT / "data/codenet_python800_stage_a_selected_source_ast/selected_source_ast_index.jsonl",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs/codenet_python800_stage_a_validation_v1",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="*",
        default=None,
        help="Optional registered seed subset. The selection record is written only when all ten seeds exist.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    protocol_bytes = args.protocol.read_bytes()
    protocol = json.loads(protocol_bytes)
    if protocol.get("schema_version") != "code2hyp-stage-a-model-analysis-protocol-v1":
        raise ValueError("unexpected Stage A model-analysis protocol")
    if protocol.get("status") != "frozen_before_calibration_pair_materialization_or_validation_metrics":
        raise ValueError("Stage A protocol is not frozen in the expected state")
    calibration_manifest_bytes = args.calibration_manifest.read_bytes()
    calibration_manifest = json.loads(calibration_manifest_bytes)
    expected_protocol_hash = str(calibration_manifest["input"]["model_analysis_protocol"]["sha256"])
    protocol_sha256 = stable_sha256(protocol_bytes)
    if protocol_sha256 != expected_protocol_hash:
        raise ValueError("calibration manifest and model-analysis protocol disagree")
    expected_pairs_hash = next(
        item["sha256"]
        for item in calibration_manifest["artifacts"]
        if item["path"] == "calibration_pairs.jsonl"
    )
    if stable_sha256(args.calibration_pairs.read_bytes()) != expected_pairs_hash:
        raise ValueError("calibration pairs differ from their frozen manifest")
    calibration_pairs = tuple(iter_jsonl(args.calibration_pairs))

    registered_seeds = tuple(int(value) for value in protocol["encoder_training"]["model_seeds"])
    seeds = registered_seeds if args.seeds is None else tuple(args.seeds)
    if not seeds or any(seed not in registered_seeds for seed in seeds):
        raise ValueError("all requested seeds must belong to the frozen registered seed list")
    split = load_stage_a_split(
        source_root=args.source_root,
        train_path=args.train_programs,
        validation_path=args.validation_programs,
        ast_index_path=args.ast_index,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    start_time = time.perf_counter()

    def progress(event: dict[str, Any]) -> None:
        elapsed = time.perf_counter() - start_time
        print(json.dumps({"elapsed_seconds": round(elapsed, 1), **event}, sort_keys=True), flush=True)

    for seed in seeds:
        progress({"phase": "seed_start", "seed": seed})
        payload = run_stage_a_validation_seed(
            split=split,
            calibration_pairs=calibration_pairs,
            seed=seed,
            output_dir=args.output_dir,
            protocol_sha256=protocol_sha256,
            calibration_manifest_sha256=stable_sha256(calibration_manifest_bytes),
            dim=int(protocol["encoder_training"]["dimension_per_role"]),
            epochs=int(protocol["encoder_training"]["epochs"]),
            batch_size=int(protocol["encoder_training"]["batch_size_programs"]),
            learning_rate=float(protocol["encoder_training"]["learning_rate"]),
            gradient_clip_norm=float(protocol["encoder_training"]["gradient_clip_global_norm"]),
            lambda_edge=float(protocol["encoder_training"]["loss"]["edge_length_weight"]),
            lambda_gromov=float(protocol["encoder_training"]["loss"]["soft_gromov_LCA_distortion_weight"]),
            lambda_branch=float(protocol["encoder_training"]["loss"]["branch_length_weight"]),
            max_paths=int(protocol["representation"]["path_count"]),
            max_ball_fraction=float(
                protocol["train_only_calibration"]["coordinate_scaling"]["maximum_ball_radius_fraction"]
            ),
            active_curvatures=tuple(
                float(cell["factor_curvatures"][0])
                for cell in protocol["geometry_cells"]["gate_C_active_candidates"]
            ),
            near_zero_curvature=float(
                protocol["geometry_cells"]["gate_C_fixed_controls"][1]["factor_curvatures"][0]
            ),
            sinkhorn_kappa=float(protocol["train_only_calibration"]["sinkhorn_scale"]["kappa"]),
            sinkhorn_iterations=int(protocol["transport"]["log_domain_iterations"]),
            projection_iterations=int(protocol["transport"]["projection_iterations_max"]),
            marginal_tolerance=float(protocol["transport"]["maximum_marginal_residual"]),
            query_batch_size=int(protocol["transport"]["query_batch_size"]),
            gallery_batch_size=int(protocol["transport"]["gallery_batch_size"]),
            torch_num_threads=1,
            progress_callback=progress,
        )
        progress(
            {
                "phase": "seed_complete",
                "seed": seed,
                "cell_count": len(payload["cells"]),
            }
        )

    complete_payloads = []
    for seed in registered_seeds:
        path = args.output_dir / f"seed_{seed}_validation.json"
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("status") == "complete":
            complete_payloads.append(payload)
    if len(complete_payloads) == len(registered_seeds):
        selection = select_active_curvature(complete_payloads)
        selection.update(
            {
                "schema_version": "code2hyp-stage-a-validation-selection-v1",
                "protocol_sha256": protocol_sha256,
                "calibration_manifest_sha256": stable_sha256(calibration_manifest_bytes),
                "registered_seeds": list(registered_seeds),
            }
        )
        selection_path = args.output_dir / "validation_selection_record.json"
        selection_path.write_bytes(canonical_json_bytes(selection))
        progress(
            {
                "phase": "validation_selection_complete",
                "selected_active_curvature": selection["selected_active_curvature"],
                "selection_record": str(selection_path),
            }
        )
    else:
        progress(
            {
                "phase": "validation_selection_pending",
                "complete_seed_count": len(complete_payloads),
                "required_seed_count": len(registered_seeds),
            }
        )


if __name__ == "__main__":
    main()
