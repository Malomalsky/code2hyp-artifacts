from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import torch
from torch import Tensor

from .raw_ast import RawAstPath, RawAstTree, raw_ast_path_relation_matrices


RawAstRelation = Literal[
    "oriented_endpoint",
    "unoriented_endpoint",
    "edge_symmetric_difference",
    "edge_jaccard",
    "lca_depth_difference",
    "lca_anchored_product",
    "path_length_difference",
]


@dataclass(frozen=True)
class MetricMeasureSpace:
    """Finite metric-measure space ``(P, D, mu)`` for method-level AST paths.

    ``distance[i, k]`` stores an internal relation between two paths of the same
    method. The mass vector is normalized during construction, so callers can
    pass counts, attention weights, or an already normalized distribution.
    """

    distance: Tensor
    mass: Tensor | None = None

    def __post_init__(self) -> None:
        distance = _as_floating_tensor(self.distance)
        if distance.ndim != 2 or distance.shape[0] != distance.shape[1]:
            raise ValueError("distance must be a square matrix")
        if not torch.isfinite(distance).all():
            raise ValueError("distance must contain only finite values")
        if bool((distance < 0).any()):
            raise ValueError("distance values must be non-negative")

        n_items = distance.shape[0]
        if self.mass is None:
            mass = torch.full((n_items,), 1.0 / max(n_items, 1), dtype=distance.dtype, device=distance.device)
        else:
            mass = torch.as_tensor(self.mass, dtype=distance.dtype, device=distance.device)
        mass = _normalize_mass(mass, expected_size=n_items)

        object.__setattr__(self, "distance", distance)
        object.__setattr__(self, "mass", mass)


@dataclass(frozen=True)
class GromovWassersteinResult:
    """Numerical result of an entropic GW/FGW fixed-point solve."""

    coupling: Tensor
    objective: Tensor
    objective_history: tuple[float, ...]
    feature_term: Tensor | None
    structure_term: Tensor
    max_marginal_residual: float
    plan_entropy: float


def metric_measure_space_from_raw_ast_paths(
    tree: RawAstTree,
    paths: Sequence[RawAstPath],
    *,
    relation: RawAstRelation = "unoriented_endpoint",
    mass: Tensor | Sequence[float] | None = None,
) -> MetricMeasureSpace:
    """Build ``(P, D, mu)`` from raw-AST path contexts of one method."""

    matrices = raw_ast_path_relation_matrices(tree, paths)
    if relation == "oriented_endpoint":
        distance = matrices.oriented_endpoint_distance
    elif relation == "unoriented_endpoint":
        distance = matrices.unoriented_endpoint_distance
    elif relation == "edge_symmetric_difference":
        distance = matrices.edge_symmetric_difference
    elif relation == "edge_jaccard":
        distance = matrices.edge_jaccard_distance
    elif relation == "lca_depth_difference":
        distance = matrices.lca_depth_difference
    elif relation == "lca_anchored_product":
        distance = matrices.lca_anchored_product_distance
    elif relation == "path_length_difference":
        distance = matrices.path_length_difference
    else:
        raise ValueError(f"unknown raw-AST relation: {relation!r}")
    return MetricMeasureSpace(distance=torch.as_tensor(distance, dtype=torch.float32), mass=mass)


def uniform_coupling(left_mass: Tensor, right_mass: Tensor) -> Tensor:
    """Independent coupling with the requested marginals."""

    left = _normalize_mass(left_mass)
    right = _normalize_mass(right_mass)
    return left.unsqueeze(1) * right.unsqueeze(0)


def sinkhorn_plan(
    cost_matrix: Tensor,
    left_mass: Tensor,
    right_mass: Tensor,
    *,
    epsilon: float = 0.05,
    iterations: int = 128,
    projection_iterations: int = 2048,
    marginal_tolerance: float = 1e-4,
) -> Tensor:
    """Solve an entropic OT subproblem in log-domain arithmetic."""

    if epsilon <= 0.0:
        raise ValueError("epsilon must be positive")
    if iterations <= 0:
        raise ValueError("iterations must be positive")
    if projection_iterations < 0:
        raise ValueError("projection_iterations must be non-negative")
    if marginal_tolerance <= 0.0:
        raise ValueError("marginal_tolerance must be positive")
    cost = torch.as_tensor(cost_matrix, dtype=torch.float64)
    if cost.ndim != 2:
        raise ValueError("cost_matrix must be two-dimensional")
    left = _normalize_mass(left_mass).to(dtype=cost.dtype, device=cost.device)
    right = _normalize_mass(right_mass).to(dtype=cost.dtype, device=cost.device)
    if cost.shape != (left.numel(), right.numel()):
        raise ValueError("cost_matrix shape must match transport marginals")

    log_left = torch.log(torch.clamp(left, min=1e-12))
    log_right = torch.log(torch.clamp(right, min=1e-12))
    log_kernel = -cost / epsilon
    log_u = torch.zeros_like(log_left)
    log_v = torch.zeros_like(log_right)
    for _ in range(iterations):
        log_u = log_left - torch.logsumexp(log_kernel + log_v.unsqueeze(0), dim=1)
        log_v = log_right - torch.logsumexp(log_kernel.transpose(0, 1) + log_u.unsqueeze(0), dim=1)
    plan = torch.clamp(torch.exp(log_u.unsqueeze(1) + log_kernel + log_v.unsqueeze(0)), min=1e-300)
    for _ in range(projection_iterations):
        plan = plan * (left / torch.clamp(plan.sum(dim=1), min=1e-12)).unsqueeze(1)
        plan = plan * (right / torch.clamp(plan.sum(dim=0), min=1e-12)).unsqueeze(0)
        if _max_marginal_residual(plan, left, right) <= marginal_tolerance:
            break
    if _max_marginal_residual(plan, left, right) > marginal_tolerance:
        for _ in range(max(4096, projection_iterations)):
            plan = plan * (left / torch.clamp(plan.sum(dim=1), min=1e-12)).unsqueeze(1)
            plan = plan * (right / torch.clamp(plan.sum(dim=0), min=1e-12)).unsqueeze(0)
            if _max_marginal_residual(plan, left, right) <= marginal_tolerance:
                break
    return _validate_coupling(plan, left, right, tolerance=marginal_tolerance)


def sinkhorn_plan_diagnostics(
    plan: Tensor,
    left_mass: Tensor,
    right_mass: Tensor,
    *,
    tolerance: float = 1e-4,
) -> dict[str, float]:
    """Return marginal residuals and entropy for an already computed plan."""

    coupling = _validate_coupling(plan, left_mass, right_mass, tolerance=tolerance)
    left = _normalize_mass(left_mass).to(dtype=coupling.dtype, device=coupling.device)
    right = _normalize_mass(right_mass).to(dtype=coupling.dtype, device=coupling.device)
    entropy = -torch.sum(coupling * torch.log(torch.clamp(coupling, min=1e-12)))
    return {
        "max_row_residual": float(torch.max(torch.abs(coupling.sum(dim=1) - left)).detach()),
        "max_column_residual": float(torch.max(torch.abs(coupling.sum(dim=0) - right)).detach()),
        "plan_entropy": float(entropy.detach()),
    }


def sinkhorn_transport_cost(
    cost_matrix: Tensor,
    left_mass: Tensor,
    right_mass: Tensor,
    *,
    epsilon: float = 0.05,
    iterations: int = 128,
    projection_iterations: int = 2048,
) -> Tensor:
    """Unregularized transport cost evaluated at the entropic Sinkhorn plan."""

    plan = sinkhorn_plan(
        cost_matrix,
        left_mass,
        right_mass,
        epsilon=epsilon,
        iterations=iterations,
        projection_iterations=projection_iterations,
    )
    cost = torch.as_tensor(cost_matrix, dtype=plan.dtype, device=plan.device)
    return torch.sum(plan * cost)


def entropic_plan_kl(coupling: Tensor, left_mass: Tensor, right_mass: Tensor) -> Tensor:
    """KL divergence ``KL(pi || left_mass tensor_product right_mass)``."""

    plan = _validate_coupling(coupling, left_mass, right_mass)
    left = _normalize_mass(left_mass).to(dtype=plan.dtype, device=plan.device)
    right = _normalize_mass(right_mass).to(dtype=plan.dtype, device=plan.device)
    product = torch.clamp(left.unsqueeze(1) * right.unsqueeze(0), min=1e-300)
    positive = plan > 0.0
    log_ratio = torch.log(torch.clamp(plan, min=1e-300)) - torch.log(product)
    return torch.sum(torch.where(positive, plan * log_ratio, torch.zeros_like(plan)))


def entropic_transport_objective(
    coupling: Tensor,
    cost_matrix: Tensor,
    left_mass: Tensor,
    right_mass: Tensor,
    *,
    epsilon: float,
) -> Tensor:
    """Full objective ``<pi,C> + epsilon KL(pi || left tensor_product right)``."""

    if epsilon <= 0.0:
        raise ValueError("epsilon must be positive")
    plan = _validate_coupling(coupling, left_mass, right_mass)
    cost = torch.as_tensor(cost_matrix, dtype=plan.dtype, device=plan.device)
    if cost.shape != plan.shape:
        raise ValueError("cost_matrix shape must match coupling shape")
    return torch.sum(plan * cost) + float(epsilon) * entropic_plan_kl(plan, left_mass, right_mass)


def sinkhorn_transport_objective(
    cost_matrix: Tensor,
    left_mass: Tensor,
    right_mass: Tensor,
    *,
    epsilon: float = 0.05,
    iterations: int = 128,
    projection_iterations: int = 2048,
) -> Tensor:
    """Evaluate the full entropic OT objective at its Sinkhorn minimizer."""

    plan = sinkhorn_plan(
        cost_matrix,
        left_mass,
        right_mass,
        epsilon=epsilon,
        iterations=iterations,
        projection_iterations=projection_iterations,
    )
    return entropic_transport_objective(plan, cost_matrix, left_mass, right_mass, epsilon=epsilon)


def sinkhorn_regularized_cost(
    cost_matrix: Tensor,
    left_mass: Tensor,
    right_mass: Tensor,
    *,
    epsilon: float = 0.05,
    iterations: int = 128,
    projection_iterations: int = 2048,
) -> Tensor:
    """Backward-compatible alias for ``sinkhorn_transport_objective``."""

    return sinkhorn_transport_objective(
        cost_matrix,
        left_mass,
        right_mass,
        epsilon=epsilon,
        iterations=iterations,
        projection_iterations=projection_iterations,
    )


def sinkhorn_divergence(
    cross_cost: Tensor,
    left_mass: Tensor,
    right_mass: Tensor,
    *,
    left_self_cost: Tensor | None = None,
    right_self_cost: Tensor | None = None,
    epsilon: float = 0.05,
    iterations: int = 128,
    projection_iterations: int = 2048,
    objective: Literal["regularized", "transport_cost"] = "regularized",
) -> Tensor:
    """Debias either the full entropic objective or its transport-only part."""

    if objective not in {"regularized", "transport_cost"}:
        raise ValueError("objective must be 'regularized' or 'transport_cost'")
    cross = _as_floating_tensor(cross_cost)
    left = _normalize_mass(left_mass).to(dtype=cross.dtype, device=cross.device)
    right = _normalize_mass(right_mass).to(dtype=cross.dtype, device=cross.device)
    if cross.shape != (left.numel(), right.numel()):
        raise ValueError("cross_cost shape must match left and right masses")
    if left_self_cost is None:
        if cross.shape[0] != cross.shape[1] or left.numel() != right.numel():
            raise ValueError("left_self_cost is required for non-square or unequal spaces")
        left_self = cross
    else:
        left_self = torch.as_tensor(left_self_cost, dtype=cross.dtype, device=cross.device)
    if right_self_cost is None:
        if cross.shape[0] != cross.shape[1] or left.numel() != right.numel():
            raise ValueError("right_self_cost is required for non-square or unequal spaces")
        right_self = cross
    else:
        right_self = torch.as_tensor(right_self_cost, dtype=cross.dtype, device=cross.device)
    if left_self.shape != (left.numel(), left.numel()):
        raise ValueError("left_self_cost shape must match the left mass")
    if right_self.shape != (right.numel(), right.numel()):
        raise ValueError("right_self_cost shape must match the right mass")
    transport = sinkhorn_transport_objective if objective == "regularized" else sinkhorn_transport_cost
    cross_value = transport(
        cross,
        left,
        right,
        epsilon=epsilon,
        iterations=iterations,
        projection_iterations=projection_iterations,
    )
    left_value = transport(
        left_self,
        left,
        left,
        epsilon=epsilon,
        iterations=iterations,
        projection_iterations=projection_iterations,
    )
    right_value = transport(
        right_self,
        right,
        right,
        epsilon=epsilon,
        iterations=iterations,
        projection_iterations=projection_iterations,
    )
    return cross_value - 0.5 * left_value - 0.5 * right_value


def permutation_coupling(left_mass: Tensor, right_mass: Tensor, left_to_right_index: Tensor) -> Tensor:
    """Deterministic coupling for a known path correspondence.

    ``left_to_right_index[i]`` is the right-space index corresponding to left
    item ``i``. The function validates that right marginals match this mapping.
    """

    left = _normalize_mass(left_mass)
    right = _normalize_mass(right_mass)
    mapping = torch.as_tensor(left_to_right_index, dtype=torch.long, device=left.device)
    if mapping.ndim != 1 or mapping.numel() != left.numel():
        raise ValueError("left_to_right_index must contain one target index per left item")
    if int(mapping.min().detach()) < 0 or int(mapping.max().detach()) >= right.numel():
        raise ValueError("left_to_right_index contains an out-of-range target index")
    if torch.unique(mapping).numel() != mapping.numel():
        raise ValueError("left_to_right_index must be one-to-one")

    coupling = torch.zeros((left.numel(), right.numel()), dtype=left.dtype, device=left.device)
    coupling[torch.arange(left.numel(), device=left.device), mapping] = left
    right_marginal = coupling.sum(dim=0)
    if not torch.allclose(right_marginal, right, atol=1e-6, rtol=1e-5):
        raise ValueError("permutation coupling does not match the right marginal")
    return coupling


def gromov_wasserstein_objective(
    left: MetricMeasureSpace,
    right: MetricMeasureSpace,
    coupling: Tensor,
) -> Tensor:
    """Evaluate the quadratic GW structural objective for a fixed coupling.

    This is the core raw-AST comparison recommended by the reviewer: two
    methods can live in different AST trees, so we compare their internal path
    relations instead of requiring a direct cross-tree path distance.
    """

    plan = _validate_coupling(coupling, left.mass, right.mass)
    left_distance = left.distance.to(dtype=plan.dtype, device=plan.device)
    right_distance = right.distance.to(dtype=plan.dtype, device=plan.device)
    relation_gap = left_distance[:, None, :, None] - right_distance[None, :, None, :]
    pair_weight = plan[:, :, None, None] * plan[None, None, :, :]
    return torch.sum(relation_gap.square() * pair_weight)


def fused_gromov_wasserstein_objective(
    left: MetricMeasureSpace,
    right: MetricMeasureSpace,
    feature_cost: Tensor,
    coupling: Tensor,
    *,
    alpha: float = 0.5,
) -> Tensor:
    """Evaluate fixed-coupling FGW objective with feature and structure terms."""

    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1]")
    plan = _validate_coupling(coupling, left.mass, right.mass)
    feature = torch.as_tensor(feature_cost, dtype=plan.dtype, device=plan.device)
    if feature.shape != plan.shape:
        raise ValueError("feature_cost shape must match coupling shape")
    feature_term = torch.sum(feature * plan)
    structure_term = gromov_wasserstein_objective(left, right, plan)
    return (1.0 - alpha) * feature_term + alpha * structure_term


def gromov_wasserstein_linearized_cost(
    left: MetricMeasureSpace,
    right: MetricMeasureSpace,
    coupling: Tensor,
) -> Tensor:
    """Linearized GW cost matrix for squared-loss entropic updates."""

    plan = _validate_coupling(coupling, left.mass, right.mass)
    left_distance = left.distance.to(dtype=plan.dtype, device=plan.device)
    right_distance = right.distance.to(dtype=plan.dtype, device=plan.device)
    left_mass = left.mass.to(dtype=plan.dtype, device=plan.device)
    right_mass = right.mass.to(dtype=plan.dtype, device=plan.device)
    left_const = (left_distance.square() @ left_mass).unsqueeze(1)
    right_const = (right_mass.unsqueeze(0) @ right_distance.square()).squeeze(0).unsqueeze(0)
    interaction = left_distance @ plan @ right_distance.transpose(0, 1)
    return left_const + right_const - 2.0 * interaction


def entropic_gromov_wasserstein(
    left: MetricMeasureSpace,
    right: MetricMeasureSpace,
    *,
    epsilon: float = 0.05,
    iterations: int = 24,
    sinkhorn_iterations: int = 128,
    initial_coupling: Tensor | None = None,
) -> GromovWassersteinResult:
    """Approximate GW by alternating linearized OT/Sinkhorn subproblems."""

    if iterations <= 0:
        raise ValueError("iterations must be positive")
    if initial_coupling is None:
        coupling = uniform_coupling(left.mass, right.mass)
    else:
        coupling = _validate_coupling(initial_coupling, left.mass, right.mass)
    history: list[float] = [float(gromov_wasserstein_objective(left, right, coupling).detach())]
    for _ in range(iterations):
        linear_cost = gromov_wasserstein_linearized_cost(left, right, coupling)
        coupling = sinkhorn_plan(linear_cost, left.mass, right.mass, epsilon=epsilon, iterations=sinkhorn_iterations)
        history.append(float(gromov_wasserstein_objective(left, right, coupling).detach()))
    objective = gromov_wasserstein_objective(left, right, coupling)
    return _build_result(left, right, coupling, objective, history, feature_term=None, structure_term=objective)


def entropic_fused_gromov_wasserstein(
    left: MetricMeasureSpace,
    right: MetricMeasureSpace,
    feature_cost: Tensor,
    *,
    alpha: float = 0.5,
    epsilon: float = 0.05,
    iterations: int = 24,
    sinkhorn_iterations: int = 128,
    initial_coupling: Tensor | None = None,
) -> GromovWassersteinResult:
    """Approximate FGW with entropic alternating linearization."""

    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1]")
    if iterations <= 0:
        raise ValueError("iterations must be positive")
    feature = torch.as_tensor(feature_cost, dtype=torch.float32)
    if feature.shape != (left.mass.numel(), right.mass.numel()):
        raise ValueError("feature_cost shape must match left and right spaces")
    if initial_coupling is None:
        coupling = sinkhorn_plan(feature, left.mass, right.mass, epsilon=max(epsilon, 1e-6), iterations=sinkhorn_iterations)
    else:
        coupling = _validate_coupling(initial_coupling, left.mass, right.mass)
    history: list[float] = [float(fused_gromov_wasserstein_objective(left, right, feature, coupling, alpha=alpha).detach())]
    for _ in range(iterations):
        linear_cost = alpha * gromov_wasserstein_linearized_cost(left, right, coupling)
        linear_cost = linear_cost + (1.0 - alpha) * feature.to(dtype=linear_cost.dtype, device=linear_cost.device)
        coupling = sinkhorn_plan(linear_cost, left.mass, right.mass, epsilon=epsilon, iterations=sinkhorn_iterations)
        history.append(float(fused_gromov_wasserstein_objective(left, right, feature, coupling, alpha=alpha).detach()))
    feature_term = torch.sum(coupling * feature.to(dtype=coupling.dtype, device=coupling.device))
    structure_term = gromov_wasserstein_objective(left, right, coupling)
    objective = (1.0 - alpha) * feature_term + alpha * structure_term
    return _build_result(left, right, coupling, objective, history, feature_term=feature_term, structure_term=structure_term)


def _normalize_mass(mass: Tensor, *, expected_size: int | None = None) -> Tensor:
    values = torch.as_tensor(mass)
    if not torch.is_floating_point(values):
        values = values.to(dtype=torch.float64)
    if values.ndim != 1:
        raise ValueError("mass must be a one-dimensional vector")
    if expected_size is not None and values.numel() != expected_size:
        raise ValueError("mass length must match the distance matrix size")
    if not torch.isfinite(values).all():
        raise ValueError("mass must contain only finite values")
    if bool((values < 0).any()):
        raise ValueError("mass values must be non-negative")
    total = values.sum()
    if float(total.detach()) <= 0.0:
        raise ValueError("mass must have a positive total")
    return values / total


def _build_result(
    left: MetricMeasureSpace,
    right: MetricMeasureSpace,
    coupling: Tensor,
    objective: Tensor,
    history: list[float],
    *,
    feature_term: Tensor | None,
    structure_term: Tensor,
) -> GromovWassersteinResult:
    diagnostics = sinkhorn_plan_diagnostics(coupling, left.mass, right.mass)
    return GromovWassersteinResult(
        coupling=coupling,
        objective=objective,
        objective_history=tuple(history),
        feature_term=feature_term,
        structure_term=structure_term,
        max_marginal_residual=max(diagnostics["max_row_residual"], diagnostics["max_column_residual"]),
        plan_entropy=diagnostics["plan_entropy"],
    )


def _validate_coupling(
    coupling: Tensor,
    left_mass: Tensor,
    right_mass: Tensor,
    *,
    tolerance: float | None = None,
) -> Tensor:
    plan = torch.as_tensor(coupling)
    if not torch.is_floating_point(plan):
        plan = plan.to(dtype=torch.float64)
    if plan.ndim != 2:
        raise ValueError("coupling must be a two-dimensional matrix")
    left = _normalize_mass(left_mass).to(dtype=plan.dtype, device=plan.device)
    right = _normalize_mass(right_mass).to(dtype=plan.dtype, device=plan.device)
    if plan.shape != (left.numel(), right.numel()):
        raise ValueError("coupling shape must match left and right masses")
    if not torch.isfinite(plan).all():
        raise ValueError("coupling must contain only finite values")
    if bool((plan < -1e-8).any()):
        raise ValueError("coupling values must be non-negative")
    tolerance_value = tolerance if tolerance is not None else 1e-4
    if not torch.allclose(plan.sum(dim=1), left, atol=tolerance_value, rtol=tolerance_value):
        raise ValueError("coupling row sums must equal left mass")
    if not torch.allclose(plan.sum(dim=0), right, atol=tolerance_value, rtol=tolerance_value):
        raise ValueError("coupling column sums must equal right mass")
    return torch.clamp(plan, min=0.0)


def _max_marginal_residual(plan: Tensor, left: Tensor, right: Tensor) -> float:
    row_residual = torch.max(torch.abs(plan.sum(dim=1) - left))
    column_residual = torch.max(torch.abs(plan.sum(dim=0) - right))
    return float(torch.maximum(row_residual, column_residual).detach())


def _as_floating_tensor(value: Tensor) -> Tensor:
    tensor = torch.as_tensor(value)
    return tensor if torch.is_floating_point(tensor) else tensor.to(dtype=torch.float64)
