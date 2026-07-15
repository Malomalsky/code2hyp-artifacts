from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from geometry_profile_research.codenet_eligibility import canonical_json_bytes, stable_sha256
from geometry_profile_research.codenet_stage_a_runner import curvature_cell_id


RUNNER_COMMIT = "bf8cf512742330dc6b6d53bb7fa971e5ee131378"
RUNNER_TAG = "codenet-stage-a-validation-runner-v1"


def seal_seed_result(
    *,
    result_path: Path,
    protocol_path: Path,
    calibration_manifest_path: Path,
    gate0_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Verify and seal one complete validation seed without reading source data."""

    result_bytes = result_path.read_bytes()
    protocol_bytes = protocol_path.read_bytes()
    calibration_bytes = calibration_manifest_path.read_bytes()
    gate0_bytes = gate0_path.read_bytes()
    result = json.loads(result_bytes)
    protocol = json.loads(protocol_bytes)
    calibration = json.loads(calibration_bytes)
    gate0 = json.loads(gate0_bytes)
    if result.get("status") != "complete":
        raise ValueError("validation seed result is not complete")
    if result.get("protocol_sha256") != stable_sha256(protocol_bytes):
        raise ValueError("seed result does not match the frozen protocol")
    if result.get("calibration_manifest_sha256") != stable_sha256(calibration_bytes):
        raise ValueError("seed result does not match the frozen calibration manifest")
    if gate0.get("status") != "passed":
        raise ValueError("numerical Gate 0 did not pass")
    if gate0.get("protocol", {}).get("sha256") != stable_sha256(protocol_bytes):
        raise ValueError("numerical Gate 0 used a different protocol")
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
    if stable_sha256(checkpoint_path.read_bytes()) != str(checkpoint["sha256"]):
        raise ValueError("encoder checkpoint hash mismatch")

    artifact_rows = [
        {
            "role": "encoder_checkpoint",
            "path": checkpoint_path.name,
            "bytes": checkpoint_path.stat().st_size,
            "sha256": str(checkpoint["sha256"]),
        }
    ]
    for cell_id, cell in sorted(result["cells"].items()):
        metrics = cell["metrics"]
        if int(metrics["query_count"]) != 776 or int(metrics["problem_count"]) != 97:
            raise ValueError(f"cell {cell_id!r} has unexpected validation cardinalities")
        if len(metrics["query_scores"]) != 776 or len(metrics["task_scores"]) != 97:
            raise ValueError(f"cell {cell_id!r} has incomplete query/task scores")
        distance = cell["distance_matrix"]
        if distance["shape"] != [776, 776] or distance["dtype"] != "float64":
            raise ValueError(f"cell {cell_id!r} has an unexpected distance matrix contract")
        distance_path = result_dir / str(distance["path"])
        distance_sha = stable_sha256(distance_path.read_bytes())
        if distance_sha != str(distance["sha256"]):
            raise ValueError(f"cell {cell_id!r} distance matrix hash mismatch")
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
        },
        "artifacts": artifact_rows,
        "checks": {
            "status_complete": True,
            "execution_config_matches_protocol": True,
            "cell_set_matches_protocol": True,
            "query_count_per_cell": 776,
            "problem_count_per_cell": 97,
            "distance_shape_per_cell": [776, 776],
            "distance_dtype": "float64",
            "all_hashes_match": True,
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
        default=PROJECT_ROOT / "reports/codenet_python800_stage_a_gate0_numerical_v1.json",
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
        output_path=output,
    )
    print(json.dumps(manifest["checks"], indent=2, sort_keys=True))
    print(f"seal={output}")


if __name__ == "__main__":
    main()
