from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from geometry_profile_research.batched_transport import (
    batched_entropic_transport_objective,
    batched_marginal_residuals,
    batched_role_product_cost,
    batched_sinkhorn_plan,
    round_batched_plan_to_marginals,
)
from geometry_profile_research.codenet_eligibility import canonical_json_bytes, stable_sha256
from geometry_profile_research.codenet_stage_a import load_stage_a_split
from geometry_profile_research.codenet_stage_a_evaluation import (
    calibrate_euclidean_role_weights,
    calibration_cost_scale,
)
from geometry_profile_research.codenet_stage_a_runner import (
    _model_from_checkpoint,
    encode_stage_a_programs,
    iter_jsonl,
    scale_product_measure,
)
from geometry_profile_research.constant_curvature import RoleProductGeometry, scaled_sinkhorn_epsilon


def reproduce_incident(
    *,
    source_root: Path,
    checkpoint_path: Path,
    protocol_path: Path,
    calibration_pairs_path: Path,
    train_programs_path: Path,
    validation_programs_path: Path,
    ast_index_path: Path,
    output_path: Path,
) -> dict:
    protocol_bytes = protocol_path.read_bytes()
    protocol = json.loads(protocol_bytes)
    calibration_pairs = tuple(iter_jsonl(calibration_pairs_path))
    split = load_stage_a_split(
        source_root=source_root,
        train_path=train_programs_path,
        validation_path=validation_programs_path,
        ast_index_path=ast_index_path,
    )
    with torch.serialization.safe_globals([torch.torch_version.TorchVersion]):
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model = _model_from_checkpoint(checkpoint).eval()

    train_measures = encode_stage_a_programs(model, split.train, anchor_mode="true_lca")
    max_active_curvature = max(
        float(cell["factor_curvatures"][0])
        for cell in protocol["geometry_cells"]["gate_C_active_candidates"]
    )
    maximum_lca_norm = max(
        float(torch.linalg.vector_norm(measure.points[:, 0], dim=-1).max())
        for measure in train_measures.values()
    )
    max_ball_fraction = float(
        protocol["train_only_calibration"]["coordinate_scaling"]["maximum_ball_radius_fraction"]
    )
    allowed_norm = max_ball_fraction / max_active_curvature**0.5
    lca_scale = 1.0 if maximum_lca_norm <= 0.0 else min(1.0, allowed_norm / maximum_lca_norm)
    role_scales = (lca_scale, 1.0, 1.0)
    train_measures = {
        item_id: scale_product_measure(measure, role_scales=role_scales)
        for item_id, measure in train_measures.items()
    }
    role_calibration = calibrate_euclidean_role_weights(train_measures, calibration_pairs)
    geometry = RoleProductGeometry(
        factor_curvatures=(0.0, 0.0, 0.0),
        factor_weights=role_calibration.euclidean_weights,
        side_weight=0.0,
        unoriented=True,
    )
    scale = calibration_cost_scale(train_measures, calibration_pairs, geometry)
    epsilon = scaled_sinkhorn_epsilon(
        scale.cost_scale,
        kappa=float(protocol["train_only_calibration"]["sinkhorn_scale"]["kappa"]),
    )
    del train_measures

    query_measures_by_id = encode_stage_a_programs(model, split.query, anchor_mode="true_lca")
    queries = tuple(
        scale_product_measure(query_measures_by_id[item.item_id], role_scales=role_scales)
        for item in split.query
    )
    tolerance = float(protocol["transport"]["maximum_marginal_residual"])
    iterations = int(protocol["transport"]["log_domain_iterations"])
    projection_iterations = int(protocol["transport"]["projection_iterations_max"])
    query_batch_size = int(protocol["transport"]["query_batch_size"])
    incident = None
    for start in range(0, len(queries), query_batch_size):
        chunk = queries[start : start + query_batch_size]
        problem = batched_role_product_cost(geometry, chunk, chunk)
        unrounded = batched_sinkhorn_plan(
            problem.cost,
            problem.left_mass,
            problem.right_mass,
            epsilon=epsilon,
            iterations=iterations,
            projection_iterations=projection_iterations,
            marginal_tolerance=tolerance,
            round_marginals=False,
            enforce_marginal_tolerance=False,
        )
        old_residuals = batched_marginal_residuals(
            unrounded,
            problem.left_mass,
            problem.right_mass,
        )
        if float(old_residuals.max()) <= tolerance:
            continue
        rounded = round_batched_plan_to_marginals(
            unrounded,
            problem.left_mass,
            problem.right_mass,
        )
        new_residuals = batched_marginal_residuals(
            rounded,
            problem.left_mass,
            problem.right_mass,
        )
        old_objective = batched_entropic_transport_objective(
            unrounded,
            problem.cost,
            problem.left_mass,
            problem.right_mass,
            epsilon=epsilon,
        )
        new_objective = batched_entropic_transport_objective(
            rounded,
            problem.cost,
            problem.left_mass,
            problem.right_mass,
            epsilon=epsilon,
        )
        incident = {
            "query_batch_start": start,
            "query_batch_size": len(chunk),
            "query_source_relpaths": [item.item_id for item in split.query[start : start + len(chunk)]],
            "path_counts": list(problem.left_sizes),
            "unrounded_marginal_residuals": [float(value) for value in old_residuals],
            "rounded_marginal_residuals": [float(value) for value in new_residuals],
            "maximum_plan_absolute_correction": float(torch.max(torch.abs(rounded - unrounded))),
            "plan_L1_correction_by_problem": [
                float(value) for value in torch.sum(torch.abs(rounded - unrounded), dim=(1, 2))
            ],
            "unrounded_regularized_objective_by_problem": [
                float(value) for value in old_objective
            ],
            "rounded_regularized_objective_by_problem": [
                float(value) for value in new_objective
            ],
            "regularized_objective_shift_by_problem": [
                float(value) for value in new_objective - old_objective
            ],
            "relative_regularized_objective_shift_by_problem": [
                float(shift / torch.clamp(torch.abs(old), min=1e-15))
                for shift, old in zip(new_objective - old_objective, old_objective)
            ],
            "cost_minimum": float(problem.cost.min()),
            "cost_maximum": float(problem.cost.max()),
        }
        break
    if incident is None:
        raise ValueError("the runner-v1 marginal incident was not reproduced")
    if max(incident["rounded_marginal_residuals"]) > tolerance:
        raise ValueError("rounded incident batch still violates the frozen marginal tolerance")

    payload = {
        "schema_version": "code2hyp-stage-a-transport-incident-reproduction-v1",
        "experiment_role": "pre_metric_numerical_incident_reproduction",
        "inputs": {
            "checkpoint_sha256": stable_sha256(checkpoint_path.read_bytes()),
            "model_analysis_protocol_sha256": stable_sha256(protocol_bytes),
            "calibration_pairs_sha256": stable_sha256(calibration_pairs_path.read_bytes()),
        },
        "calibration": {
            "role_scales": list(role_scales),
            "role_weights": list(role_calibration.euclidean_weights),
            "role_weight_details": asdict(role_calibration),
            "cost_scale": asdict(scale),
            "epsilon": epsilon,
        },
        "incident": incident,
        "acceptance": {
            "original_residual_exceeds_1e-7": max(incident["unrounded_marginal_residuals"]) > tolerance,
            "rounded_residual_at_most_1e-7": max(incident["rounded_marginal_residuals"]) <= tolerance,
            "validation_retrieval_metrics_computed": False,
        },
        "test_program_ids_materialized": False,
        "test_relevance_labels_opened": False,
        "test_retrieval_metrics_computed": False,
    }
    content = canonical_json_bytes(payload)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and output_path.read_bytes() != content:
        raise ValueError(f"refusing to overwrite a different incident reproduction: {output_path}")
    output_path.write_bytes(content)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Reproduce the pre-metric Stage A Sinkhorn halt.")
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=PROJECT_ROOT / "outputs/codenet_python800_stage_a_validation_v1/seed_20260711_encoder.pt",
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        default=PROJECT_ROOT / "configs/codenet_python800_stage_a_model_analysis_protocol_v1.json",
    )
    parser.add_argument(
        "--calibration-pairs",
        type=Path,
        default=PROJECT_ROOT / "data/codenet_python800_stage_a_calibration_pairs/calibration_pairs.jsonl",
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
        "--output",
        type=Path,
        default=PROJECT_ROOT / "reports/codenet_python800_stage_a_transport_incident_reproduction_v1.json",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    payload = reproduce_incident(
        source_root=args.source_root,
        checkpoint_path=args.checkpoint,
        protocol_path=args.protocol,
        calibration_pairs_path=args.calibration_pairs,
        train_programs_path=args.train_programs,
        validation_programs_path=args.validation_programs,
        ast_index_path=args.ast_index,
        output_path=args.output,
    )
    print(json.dumps(payload["incident"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
