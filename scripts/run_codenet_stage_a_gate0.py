from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

import numpy as np
import ot
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from geometry_profile_research.batched_transport import (
    batched_marginal_residuals,
    batched_sinkhorn_plan,
    batched_sinkhorn_transport_objective,
)
from geometry_profile_research.codenet_eligibility import canonical_json_bytes, stable_sha256
from geometry_profile_research.constant_curvature import ProductMeasure, RoleProductGeometry
from geometry_profile_research.gromov_wasserstein import (
    entropic_transport_objective,
    sinkhorn_divergence,
    sinkhorn_plan,
    sinkhorn_transport_objective,
)


def run_gate0(*, protocol_path: Path) -> dict[str, object]:
    protocol_bytes = protocol_path.read_bytes()
    protocol = json.loads(protocol_bytes)
    tolerance = float(protocol["transport"]["maximum_marginal_residual"])
    epsilon = 0.23
    cost = torch.tensor(
        [[0.0, 0.7, 1.5], [0.4, 0.2, 0.9], [1.1, 0.6, 0.1]],
        dtype=torch.float64,
    )
    left = torch.tensor([0.2, 0.3, 0.5], dtype=torch.float64)
    right = torch.tensor([0.4, 0.35, 0.25], dtype=torch.float64)
    plan = sinkhorn_plan(
        cost,
        left,
        right,
        epsilon=epsilon,
        iterations=1000,
        projection_iterations=4096,
        marginal_tolerance=1e-10,
    )
    pot_plan = torch.from_numpy(
        np.asarray(
            ot.sinkhorn(
                left.numpy(),
                right.numpy(),
                cost.numpy(),
                reg=epsilon,
                method="sinkhorn_log",
                numItermax=20_000,
                stopThr=1e-13,
            )
        )
    )
    objective = sinkhorn_transport_objective(
        cost,
        left,
        right,
        epsilon=epsilon,
        iterations=1000,
        projection_iterations=4096,
    )
    pot_objective = entropic_transport_objective(
        pot_plan,
        cost,
        left,
        right,
        epsilon=epsilon,
    )

    batch_cost = torch.stack((cost, cost + 0.15))
    batch_left = torch.stack((left, left))
    batch_right = torch.stack((right, right))
    batch_plan = batched_sinkhorn_plan(
        batch_cost,
        batch_left,
        batch_right,
        epsilon=epsilon,
        iterations=1000,
        projection_iterations=4096,
        marginal_tolerance=1e-10,
    )
    batch_objective = batched_sinkhorn_transport_objective(
        batch_cost,
        batch_left,
        batch_right,
        epsilon=epsilon,
        iterations=1000,
        projection_iterations=4096,
        marginal_tolerance=1e-10,
    )
    scalar_objective = torch.stack(
        tuple(
            sinkhorn_transport_objective(
                matrix,
                left,
                right,
                epsilon=epsilon,
                iterations=1000,
                projection_iterations=4096,
            )
            for matrix in batch_cost
        )
    )

    generator = torch.Generator().manual_seed(20260715)
    x = torch.randn((16, 4), generator=generator, dtype=torch.float64) * 0.05
    y = torch.randn((16, 4), generator=generator, dtype=torch.float64) * 0.05
    euclidean = RoleProductGeometry(
        factor_curvatures=(0.0, 0.0, 0.0),
        factor_weights=(1.0, 1.0, 1.0),
        unoriented=True,
    ).point_distance(x, y, factor_index=0)
    near_zero = RoleProductGeometry(
        factor_curvatures=(1e-8, 0.0, 0.0),
        factor_weights=(1.0, 1.0, 1.0),
        unoriented=True,
    ).point_distance(x, y, factor_index=0)
    limit_relative_error = float(
        torch.linalg.vector_norm(near_zero - 2.0 * euclidean)
        / torch.clamp(torch.linalg.vector_norm(2.0 * euclidean), min=1e-15)
    )

    forward = ProductMeasure(
        points=torch.tensor([[[0.1], [0.2], [0.6]]], dtype=torch.float64),
        mass=torch.ones(1, dtype=torch.float64),
    )
    reversed_path = ProductMeasure(
        points=torch.tensor([[[0.1], [0.6], [0.2]]], dtype=torch.float64),
        mass=torch.ones(1, dtype=torch.float64),
    )
    reversal_cost = float(
        RoleProductGeometry(
            factor_curvatures=(1.0, 0.0, 0.0),
            factor_weights=(0.25, 1.0, 1.0),
            unoriented=True,
        ).path_cost_matrix(forward, reversed_path)[0, 0]
    )

    gradient_cost = torch.tensor(
        [[0.0, 0.8], [1.2, 0.0]],
        dtype=torch.float64,
        requires_grad=True,
    )
    gradient_mass = torch.tensor([0.4, 0.6], dtype=torch.float64)
    gradient_value = sinkhorn_transport_objective(
        gradient_cost,
        gradient_mass,
        gradient_mass,
        epsilon=0.3,
        iterations=1000,
        projection_iterations=4096,
    )
    gradient_value.backward()
    autograd = gradient_cost.grad.detach().clone()
    step = 1e-5
    finite_difference = torch.empty_like(gradient_cost)
    for row in range(gradient_cost.shape[0]):
        for column in range(gradient_cost.shape[1]):
            plus = gradient_cost.detach().clone()
            minus = gradient_cost.detach().clone()
            plus[row, column] += step
            minus[row, column] -= step
            plus_value = sinkhorn_transport_objective(
                plus,
                gradient_mass,
                gradient_mass,
                epsilon=0.3,
                iterations=1000,
                projection_iterations=4096,
            )
            minus_value = sinkhorn_transport_objective(
                minus,
                gradient_mass,
                gradient_mass,
                epsilon=0.3,
                iterations=1000,
                projection_iterations=4096,
            )
            finite_difference[row, column] = (plus_value - minus_value) / (2.0 * step)
    gradient_max_abs_error = float(torch.max(torch.abs(autograd - finite_difference)))

    identity_cost = torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=torch.float64)
    identity_mass = torch.tensor([0.25, 0.75], dtype=torch.float64)
    identity_divergence = float(
        sinkhorn_divergence(
            identity_cost,
            identity_mass,
            identity_mass,
            epsilon=0.2,
            iterations=1000,
            projection_iterations=4096,
        )
    )
    checks = {
        "pot_plan_max_absolute_error": {
            "value": float(torch.max(torch.abs(plan - pot_plan))),
            "threshold": 1e-8,
        },
        "pot_full_objective_absolute_error": {
            "value": float(torch.abs(objective - pot_objective)),
            "threshold": 1e-9,
        },
        "batch_scalar_objective_max_absolute_error": {
            "value": float(torch.max(torch.abs(batch_objective - scalar_objective))),
            "threshold": 1e-9,
        },
        "batched_max_marginal_residual": {
            "value": float(batched_marginal_residuals(batch_plan, batch_left, batch_right).max()),
            "threshold": tolerance,
        },
        "poincare_near_zero_relative_error": {
            "value": limit_relative_error,
            "threshold": 1e-5,
        },
        "endpoint_reversal_quotient_cost": {
            "value": abs(reversal_cost),
            "threshold": 1e-12,
        },
        "finite_difference_gradient_max_absolute_error": {
            "value": gradient_max_abs_error,
            "threshold": 1e-5,
        },
        "identity_sinkhorn_divergence_absolute_value": {
            "value": abs(identity_divergence),
            "threshold": 1e-10,
        },
    }
    for check in checks.values():
        check["passed"] = bool(check["value"] <= check["threshold"])
    passed = all(bool(check["passed"]) for check in checks.values())
    return {
        "schema_version": "code2hyp-stage-a-gate0-numerical-v1",
        "status": "passed" if passed else "failed",
        "protocol": {
            "path": str(protocol_path.relative_to(PROJECT_ROOT)),
            "sha256": stable_sha256(protocol_bytes),
        },
        "checks": checks,
        "software": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "POT": ot.__version__,
        },
        "validation_retrieval_metrics_computed": False,
        "test_program_ids_materialized": False,
        "test_relevance_labels_opened": False,
        "test_retrieval_metrics_computed": False,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run numerical Gate 0 before CodeNet Stage A validation.")
    parser.add_argument(
        "--protocol",
        type=Path,
        default=PROJECT_ROOT / "configs/codenet_python800_stage_a_model_analysis_protocol_v1.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "reports/codenet_python800_stage_a_gate0_numerical_v1.json",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    payload = run_gate0(protocol_path=args.protocol)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(canonical_json_bytes(payload))
    print(json.dumps(payload, indent=2, sort_keys=True))
    if payload["status"] != "passed":
        raise SystemExit("Stage A numerical Gate 0 failed")


if __name__ == "__main__":
    main()
