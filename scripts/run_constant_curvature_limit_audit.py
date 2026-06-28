from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Sequence

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from geometry_profile_research.constant_curvature import (
    ConstantCurvatureProduct,
    ProductMeasure,
    median_positive_cost_scale,
    scaled_sinkhorn_epsilon,
)


DEFAULT_CURVATURES = (0.0, 1e-8, 1e-6, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 4.0)


def run_audit(
    *,
    curvatures: Sequence[float] = DEFAULT_CURVATURES,
    seed: int = 20260625,
    method_count: int = 8,
    train_methods: int = 4,
    paths_per_method: int = 6,
    factors: int = 3,
    dim: int = 4,
    kappa: float = 0.05,
    sinkhorn_iterations: int = 96,
    unoriented: bool = False,
) -> dict[str, Any]:
    """Run a frozen-tensor Euclidean-limit audit for product path measures."""

    if method_count < 3:
        raise ValueError("method_count must be at least 3")
    if not 1 <= train_methods < method_count:
        raise ValueError("train_methods must be in [1, method_count)")
    if factors != 3:
        raise ValueError("the audit uses canonical (LCA, start, end) path objects")

    points, masses = _frozen_path_measure_tensors(
        seed=seed,
        method_count=method_count,
        paths_per_method=paths_per_method,
        factors=factors,
        dim=dim,
    )
    measures = tuple(ProductMeasure(points=points[index], mass=masses[index]) for index in range(method_count))

    baseline = _evaluate_curvature(
        0.0,
        measures,
        kappa=kappa,
        train_methods=train_methods,
        sinkhorn_iterations=sinkhorn_iterations,
        unoriented=unoriented,
    )
    rows = []
    for curvature in curvatures:
        current = baseline if curvature == 0.0 else _evaluate_curvature(
            float(curvature),
            measures,
            kappa=kappa,
            train_methods=train_methods,
            sinkhorn_iterations=sinkhorn_iterations,
            unoriented=unoriented,
        )
        rows.append(_compare_to_baseline(current, baseline))

    payload = {
        "audit": "constant_curvature_euclidean_limit",
        "interpretation": (
            "Mechanistic audit of the metric and Sinkhorn layer. This is not a downstream benchmark. "
            "The Poincare distance is normalized by 1/2 so that fixed-coordinate distances converge "
            "to the Euclidean metric as curvature tends to zero."
        ),
        "config": {
            "seed": seed,
            "method_count": method_count,
            "train_methods": train_methods,
            "paths_per_method": paths_per_method,
            "factors": factors,
            "dim": dim,
            "kappa": kappa,
            "sinkhorn_iterations": sinkhorn_iterations,
            "unoriented": unoriented,
            "curvatures": [float(value) for value in curvatures],
            "epsilon_policy": "epsilon(c) = kappa * median positive train ground cost at curvature c",
        },
        "rows": rows,
        "quality_gate": _quality_gate(rows),
    }
    return payload


def write_report(payload: dict[str, Any], path: Path) -> None:
    """Write a compact Markdown report for the audit."""

    lines = [
        "# Constant-curvature Euclidean-limit audit",
        "",
        "This is a mechanistic audit of the Code2Hyp product metric and entropic transport layer.",
        "It does not replace downstream retrieval experiments. Its purpose is to verify that the",
        "constant-curvature implementation has a well-defined Euclidean limit and that distances,",
        "Sinkhorn divergences, transport plans, rankings and embedding gradients change smoothly as",
        "curvature increases from zero.",
        "",
        "The Poincare geodesic is normalized by `1/2`, so fixed-coordinate distances satisfy",
        "`D_c(x, y) -> ||x-y||` as `c -> 0`. Sinkhorn regularization is scaled as",
        "`epsilon(c) = kappa * median_positive_train_cost(c)` and then held fixed for all pairs at",
        "the same curvature.",
        "",
        "## Results",
        "",
        "| curvature | epsilon | distance rel. Frobenius | Sinkhorn rel. Frobenius | plan L1 | grad cosine | top-1 agreement | top-3 Jaccard | rank Spearman |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["rows"]:
        lines.append(
            "| {curvature:g} | {epsilon:.6g} | {distance_matrix_relative_frobenius_error:.3e} | "
            "{sinkhorn_matrix_relative_frobenius_error:.3e} | {transport_plan_l1_error:.3e} | "
            "{gradient_cosine:.6f} | {ranking_top1_agreement:.3f} | {ranking_top3_jaccard:.3f} | "
            "{ranking_spearman:.3f} |".format(**row)
        )
    lines.extend(
        [
            "",
            "## Gate",
            "",
            f"- Near-zero distance convergence: `{payload['quality_gate']['near_zero_distance_converges']}`",
            f"- Near-zero Sinkhorn convergence: `{payload['quality_gate']['near_zero_sinkhorn_converges']}`",
            f"- Near-zero transport-plan convergence: `{payload['quality_gate']['near_zero_transport_converges']}`",
            f"- Near-zero gradient convergence: `{payload['quality_gate']['near_zero_gradients_converge']}`",
            f"- Near-zero ranking stability: `{payload['quality_gate']['near_zero_rankings_stable']}`",
            f"- Active-curvature deviation present: `{payload['quality_gate']['active_curvature_deviation_present']}`",
            "",
            "## Interpretation",
            "",
            payload["quality_gate"]["interpretation"],
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _evaluate_curvature(
    curvature: float,
    measures: Sequence[ProductMeasure],
    *,
    kappa: float,
    train_methods: int,
    sinkhorn_iterations: int,
    unoriented: bool,
) -> dict[str, Any]:
    geometry = ConstantCurvatureProduct(curvature=curvature, unoriented=unoriented)
    train_costs = []
    for left_index in range(train_methods):
        for right_index in range(train_methods):
            train_costs.append(geometry.path_cost_matrix(measures[left_index], measures[right_index]))
    cost_scale = median_positive_cost_scale(train_costs)
    epsilon = scaled_sinkhorn_epsilon(cost_scale, kappa=kappa)
    distance_matrix = _pairwise_product_distance_matrix(geometry, measures)
    sinkhorn_matrix = _pairwise_sinkhorn_matrix(
        geometry,
        measures,
        epsilon=epsilon,
        sinkhorn_iterations=sinkhorn_iterations,
    )
    transport_plan = geometry.transport_plan(
        measures[0],
        measures[1],
        epsilon=epsilon,
        iterations=sinkhorn_iterations,
    )
    gradient = _probe_gradient(
        geometry,
        measures,
        epsilon=epsilon,
        sinkhorn_iterations=sinkhorn_iterations,
    )
    return {
        "curvature": float(curvature),
        "cost_scale": float(cost_scale),
        "epsilon": float(epsilon),
        "distance_matrix": distance_matrix,
        "sinkhorn_matrix": sinkhorn_matrix,
        "transport_plan": transport_plan,
        "gradient": gradient,
    }


def _pairwise_product_distance_matrix(geometry: ConstantCurvatureProduct, measures: Sequence[ProductMeasure]) -> Tensor:
    n = len(measures)
    output = torch.zeros((n, n), dtype=torch.float32)
    for i in range(n):
        for j in range(i + 1, n):
            cross = geometry.path_cost_matrix(measures[i], measures[j])
            value = torch.sqrt(torch.clamp(cross.mean(), min=0.0))
            output[i, j] = output[j, i] = value.detach().to(dtype=output.dtype)
    return output


def _pairwise_sinkhorn_matrix(
    geometry: ConstantCurvatureProduct,
    measures: Sequence[ProductMeasure],
    *,
    epsilon: float,
    sinkhorn_iterations: int,
) -> Tensor:
    n = len(measures)
    output = torch.zeros((n, n), dtype=torch.float32)
    for i in range(n):
        for j in range(i + 1, n):
            value = geometry.sinkhorn_divergence(
                measures[i],
                measures[j],
                epsilon=epsilon,
                iterations=sinkhorn_iterations,
            )
            value = torch.sqrt(torch.clamp(value, min=0.0))
            output[i, j] = output[j, i] = value.detach().to(dtype=output.dtype)
    return output


def _probe_gradient(
    geometry: ConstantCurvatureProduct,
    measures: Sequence[ProductMeasure],
    *,
    epsilon: float,
    sinkhorn_iterations: int,
) -> Tensor:
    left_points = measures[0].points.detach().clone().requires_grad_(True)
    right_points = measures[1].points.detach().clone().requires_grad_(True)
    left = ProductMeasure(points=left_points, mass=measures[0].mass)
    right = ProductMeasure(points=right_points, mass=measures[1].mass)
    value = geometry.sinkhorn_divergence(left, right, epsilon=epsilon, iterations=sinkhorn_iterations)
    value.backward()
    return torch.cat((left_points.grad.reshape(-1), right_points.grad.reshape(-1))).detach()


def _compare_to_baseline(current: dict[str, Any], baseline: dict[str, Any]) -> dict[str, float]:
    distance_error = _matrix_error(current["distance_matrix"], baseline["distance_matrix"])
    sinkhorn_error = _matrix_error(current["sinkhorn_matrix"], baseline["sinkhorn_matrix"])
    ranking = _ranking_diagnostics(current["sinkhorn_matrix"], baseline["sinkhorn_matrix"])
    gradient = _gradient_diagnostics(current["gradient"], baseline["gradient"])
    plan_l1 = torch.mean(torch.abs(current["transport_plan"] - baseline["transport_plan"]))
    return {
        "curvature": float(current["curvature"]),
        "cost_scale": float(current["cost_scale"]),
        "epsilon": float(current["epsilon"]),
        "distance_matrix_max_abs_error": float(distance_error["max_abs"]),
        "distance_matrix_mean_abs_error": float(distance_error["mean_abs"]),
        "distance_matrix_relative_frobenius_error": float(distance_error["relative_frobenius"]),
        "sinkhorn_matrix_max_abs_error": float(sinkhorn_error["max_abs"]),
        "sinkhorn_matrix_mean_abs_error": float(sinkhorn_error["mean_abs"]),
        "sinkhorn_matrix_relative_frobenius_error": float(sinkhorn_error["relative_frobenius"]),
        "transport_plan_l1_error": float(plan_l1.detach()),
        "gradient_relative_error": float(gradient["relative_error"]),
        "gradient_cosine": float(gradient["cosine"]),
        "ranking_top1_agreement": float(ranking["top1_agreement"]),
        "ranking_top3_jaccard": float(ranking["top3_jaccard"]),
        "ranking_spearman": float(ranking["spearman"]),
    }


def _matrix_error(current: Tensor, baseline: Tensor) -> dict[str, float]:
    current = torch.as_tensor(current, dtype=torch.float64)
    baseline = torch.as_tensor(baseline, dtype=current.dtype, device=current.device)
    error = current - baseline
    denominator = torch.clamp(torch.linalg.vector_norm(baseline), min=1e-12)
    return {
        "max_abs": float(torch.max(torch.abs(error)).detach()),
        "mean_abs": float(torch.mean(torch.abs(error)).detach()),
        "relative_frobenius": float((torch.linalg.vector_norm(error) / denominator).detach()),
    }


def _gradient_diagnostics(current: Tensor, baseline: Tensor) -> dict[str, float]:
    current = torch.as_tensor(current, dtype=torch.float64)
    baseline = torch.as_tensor(baseline, dtype=current.dtype, device=current.device)
    denominator = torch.clamp(torch.linalg.vector_norm(baseline), min=1e-12)
    relative = torch.linalg.vector_norm(current - baseline) / denominator
    cosine = torch.dot(current, baseline) / torch.clamp(torch.linalg.vector_norm(current) * torch.linalg.vector_norm(baseline), min=1e-12)
    return {"relative_error": float(relative.detach()), "cosine": float(cosine.detach())}


def _ranking_diagnostics(current: Tensor, baseline: Tensor, *, top_k: int = 3) -> dict[str, float]:
    current_orders = _orders_without_self(current)
    baseline_orders = _orders_without_self(baseline)
    n = current_orders.shape[0]
    top1 = []
    topk_jaccard = []
    spearman = []
    for index in range(n):
        current_order = current_orders[index]
        baseline_order = baseline_orders[index]
        top1.append(float(current_order[0] == baseline_order[0]))
        k = min(top_k, current_order.numel())
        current_top = set(int(value) for value in current_order[:k])
        baseline_top = set(int(value) for value in baseline_order[:k])
        topk_jaccard.append(len(current_top & baseline_top) / max(len(current_top | baseline_top), 1))
        spearman.append(_spearman_order_correlation(current_order, baseline_order))
    return {
        "top1_agreement": sum(top1) / len(top1),
        "top3_jaccard": sum(topk_jaccard) / len(topk_jaccard),
        "spearman": sum(spearman) / len(spearman),
    }


def _orders_without_self(matrix: Tensor) -> Tensor:
    values = torch.as_tensor(matrix, dtype=torch.float64).clone()
    diagonal = torch.eye(values.shape[0], dtype=torch.bool, device=values.device)
    values = values.masked_fill(diagonal, float("inf"))
    return torch.argsort(values, dim=1)[:, : values.shape[0] - 1]


def _spearman_order_correlation(left_order: Tensor, right_order: Tensor) -> float:
    n = left_order.numel()
    left_rank = torch.empty((n + 1,), dtype=torch.float64)
    right_rank = torch.empty((n + 1,), dtype=torch.float64)
    for rank, item in enumerate(left_order):
        left_rank[int(item)] = float(rank)
    for rank, item in enumerate(right_order):
        right_rank[int(item)] = float(rank)
    valid_items = [int(item) for item in left_order]
    left = left_rank[valid_items]
    right = right_rank[valid_items]
    left = left - left.mean()
    right = right - right.mean()
    denominator = torch.clamp(torch.linalg.vector_norm(left) * torch.linalg.vector_norm(right), min=1e-12)
    return float((torch.dot(left, right) / denominator).detach())


def _quality_gate(rows: Sequence[dict[str, float]]) -> dict[str, Any]:
    by_curvature = {float(row["curvature"]): row for row in rows}
    near_zero = by_curvature.get(1e-8) or by_curvature.get(1e-6)
    active = by_curvature.get(1.0)
    if near_zero is None:
        near_zero = min((row for row in rows if row["curvature"] > 0.0), key=lambda row: row["curvature"])
    near_distance = near_zero["distance_matrix_relative_frobenius_error"] < 1e-5
    near_sinkhorn = near_zero["sinkhorn_matrix_relative_frobenius_error"] < 1e-4
    near_transport = near_zero["transport_plan_l1_error"] < 1e-4
    near_gradient = near_zero["gradient_relative_error"] < 1e-3 and near_zero["gradient_cosine"] > 0.999
    near_ranking = near_zero["ranking_top1_agreement"] == 1.0 and near_zero["ranking_top3_jaccard"] == 1.0
    active_deviation = bool(active and active["sinkhorn_matrix_relative_frobenius_error"] > near_zero["sinkhorn_matrix_relative_frobenius_error"] * 10.0)
    if near_distance and near_sinkhorn and near_transport and near_gradient and near_ranking and active_deviation:
        interpretation = (
            "The implementation has the required Euclidean limit under frozen coordinates and fixed relative "
            "Sinkhorn scale. Active curvature changes the transport geometry beyond numerical noise."
        )
    else:
        interpretation = (
            "The audit exposes a scale or implementation issue. Curvature-dependent downstream claims should "
            "remain conservative until the failed gate is resolved."
        )
    return {
        "near_zero_distance_converges": near_distance,
        "near_zero_sinkhorn_converges": near_sinkhorn,
        "near_zero_transport_converges": near_transport,
        "near_zero_gradients_converge": near_gradient,
        "near_zero_rankings_stable": near_ranking,
        "active_curvature_deviation_present": active_deviation,
        "interpretation": interpretation,
    }


def _frozen_path_measure_tensors(
    *,
    seed: int,
    method_count: int,
    paths_per_method: int,
    factors: int,
    dim: int,
) -> tuple[Tensor, Tensor]:
    generator = torch.Generator().manual_seed(seed)
    method_centers = torch.randn((method_count, 1, 1, dim), generator=generator) * 0.055
    factor_offsets = torch.randn((1, 1, factors, dim), generator=generator) * 0.025
    path_offsets = torch.randn((method_count, paths_per_method, factors, dim), generator=generator) * 0.018
    points = method_centers + factor_offsets + path_offsets
    raw_mass = torch.rand((method_count, paths_per_method), generator=generator) + 0.2
    masses = raw_mass / raw_mass.sum(dim=1, keepdim=True)
    return points.to(dtype=torch.float32), masses.to(dtype=torch.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit the Euclidean limit of the constant-curvature product metric.")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "outputs/constant_curvature_limit_audit.json")
    parser.add_argument("--report", type=Path, default=PROJECT_ROOT / "reports/constant_curvature_limit_audit.md")
    parser.add_argument("--seed", type=int, default=20260625)
    parser.add_argument("--kappa", type=float, default=0.05)
    parser.add_argument("--sinkhorn-iterations", type=int, default=96)
    parser.add_argument("--curvatures", type=float, nargs="*", default=list(DEFAULT_CURVATURES))
    parser.add_argument("--unoriented", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = run_audit(
        curvatures=tuple(args.curvatures),
        seed=args.seed,
        kappa=args.kappa,
        sinkhorn_iterations=args.sinkhorn_iterations,
        unoriented=args.unoriented,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    write_report(payload, args.report)
    print(json.dumps(payload["quality_gate"], ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
