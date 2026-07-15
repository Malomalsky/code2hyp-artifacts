from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import Tensor

from geometry_profile_research.constant_curvature import ProductMeasure, RoleProductGeometry


@dataclass(frozen=True)
class BatchedTransportProblem:
    """Padded pairwise transport problems with zero-mass padding."""

    cost: Tensor
    left_mass: Tensor
    right_mass: Tensor
    left_sizes: tuple[int, ...]
    right_sizes: tuple[int, ...]

    def __post_init__(self) -> None:
        cost = _floating_tensor(self.cost)
        left = torch.as_tensor(self.left_mass, dtype=cost.dtype, device=cost.device)
        right = torch.as_tensor(self.right_mass, dtype=cost.dtype, device=cost.device)
        if cost.ndim != 3:
            raise ValueError("cost must have shape (batch, n_left, n_right)")
        if left.shape != cost.shape[:2] or right.shape != (cost.shape[0], cost.shape[2]):
            raise ValueError("mass tensors must match the padded cost shape")
        batch = cost.shape[0]
        if len(self.left_sizes) != batch or len(self.right_sizes) != batch:
            raise ValueError("size metadata must contain one entry per batch item")
        if any(size <= 0 or size > cost.shape[1] for size in self.left_sizes):
            raise ValueError("left_sizes contains an invalid value")
        if any(size <= 0 or size > cost.shape[2] for size in self.right_sizes):
            raise ValueError("right_sizes contains an invalid value")
        _validate_batched_masses(left, name="left_mass")
        _validate_batched_masses(right, name="right_mass")
        if not torch.isfinite(cost).all() or bool((cost < 0.0).any()):
            raise ValueError("cost must contain finite non-negative values")
        object.__setattr__(self, "cost", cost)
        object.__setattr__(self, "left_mass", _normalize_batched_mass(left))
        object.__setattr__(self, "right_mass", _normalize_batched_mass(right))


def batched_role_product_cost(
    geometry: RoleProductGeometry,
    left: ProductMeasure | Sequence[ProductMeasure],
    right: ProductMeasure | Sequence[ProductMeasure],
) -> BatchedTransportProblem:
    """Build exact padded ground-cost matrices for paired product measures.

    A singleton side is broadcast across the other side. Padding never enters
    an OT objective because padded rows and columns receive exactly zero mass.
    """

    left_measures = _measure_sequence(left)
    right_measures = _measure_sequence(right)
    batch = max(len(left_measures), len(right_measures))
    if len(left_measures) not in {1, batch} or len(right_measures) not in {1, batch}:
        raise ValueError("measure batches must have equal size or one side must be a singleton")
    if len(left_measures) == 1:
        left_measures = left_measures * batch
    if len(right_measures) == 1:
        right_measures = right_measures * batch

    _validate_measure_batch(left_measures, right_measures, geometry=geometry)
    left_points, left_mass, left_sizes = _pad_measures(left_measures)
    right_points, right_mass, right_sizes = _pad_measures(right_measures)
    dtype = torch.promote_types(left_points.dtype, right_points.dtype)
    device = left_points.device
    right_points = right_points.to(dtype=dtype, device=device)
    right_mass = right_mass.to(dtype=dtype, device=device)
    left_points = left_points.to(dtype=dtype)
    left_mass = left_mass.to(dtype=dtype)

    terms: list[Tensor] = []
    for factor_index, weight in enumerate(geometry.factor_weights):
        left_factor = left_points[:, :, factor_index].unsqueeze(2)
        right_factor = right_points[:, :, factor_index].unsqueeze(1)
        distance = geometry.point_distance(left_factor, right_factor, factor_index=factor_index)
        terms.append(float(weight) * distance.square())
    direct = torch.stack(terms, dim=0).sum(dim=0)
    direct = direct + _batched_side_cost(geometry, left_measures, right_measures, reverse_right=False)

    if geometry.unoriented:
        reversed_right = torch.stack(
            (right_points[:, :, 0], right_points[:, :, 2], right_points[:, :, 1]),
            dim=2,
        )
        reversed_terms: list[Tensor] = []
        for factor_index, weight in enumerate(geometry.factor_weights):
            left_factor = left_points[:, :, factor_index].unsqueeze(2)
            right_factor = reversed_right[:, :, factor_index].unsqueeze(1)
            distance = geometry.point_distance(left_factor, right_factor, factor_index=factor_index)
            reversed_terms.append(float(weight) * distance.square())
        reversed_cost = torch.stack(reversed_terms, dim=0).sum(dim=0)
        reversed_cost = reversed_cost + _batched_side_cost(
            geometry,
            left_measures,
            right_measures,
            reverse_right=True,
        )
        direct = torch.minimum(direct, reversed_cost)

    return BatchedTransportProblem(
        cost=direct,
        left_mass=left_mass,
        right_mass=right_mass,
        left_sizes=left_sizes,
        right_sizes=right_sizes,
    )


def batched_sinkhorn_plan(
    cost: Tensor,
    left_mass: Tensor,
    right_mass: Tensor,
    *,
    epsilon: float,
    iterations: int = 128,
    projection_iterations: int = 2048,
    marginal_tolerance: float = 1e-7,
) -> Tensor:
    """Solve independent padded entropic OT problems in one tensor batch."""

    if epsilon <= 0.0:
        raise ValueError("epsilon must be positive")
    if iterations <= 0:
        raise ValueError("iterations must be positive")
    if projection_iterations < 0:
        raise ValueError("projection_iterations must be non-negative")
    if marginal_tolerance <= 0.0:
        raise ValueError("marginal_tolerance must be positive")
    values = _floating_tensor(cost).to(dtype=torch.float64)
    if values.ndim != 3:
        raise ValueError("cost must have shape (batch, n_left, n_right)")
    left = torch.as_tensor(left_mass, dtype=values.dtype, device=values.device)
    right = torch.as_tensor(right_mass, dtype=values.dtype, device=values.device)
    if left.shape != values.shape[:2] or right.shape != (values.shape[0], values.shape[2]):
        raise ValueError("mass tensors must match the cost shape")
    if not torch.isfinite(values).all() or bool((values < 0.0).any()):
        raise ValueError("cost must contain finite non-negative values")
    _validate_batched_masses(left, name="left_mass")
    _validate_batched_masses(right, name="right_mass")
    left = _normalize_batched_mass(left)
    right = _normalize_batched_mass(right)

    negative_infinity = values.new_tensor(float("-inf"))
    log_left = torch.where(left > 0.0, torch.log(torch.clamp(left, min=1e-300)), negative_infinity)
    log_right = torch.where(right > 0.0, torch.log(torch.clamp(right, min=1e-300)), negative_infinity)
    log_kernel = -values / float(epsilon)
    log_u = torch.zeros_like(left)
    log_v = torch.zeros_like(right)
    for _ in range(iterations):
        log_u = log_left - torch.logsumexp(log_kernel + log_v.unsqueeze(1), dim=2)
        log_v = log_right - torch.logsumexp(log_kernel.transpose(1, 2) + log_u.unsqueeze(1), dim=2)
    valid = (left > 0.0).unsqueeze(2) & (right > 0.0).unsqueeze(1)
    plan = torch.where(
        valid,
        torch.exp(log_u.unsqueeze(2) + log_kernel + log_v.unsqueeze(1)),
        torch.zeros_like(values),
    )

    for _ in range(projection_iterations):
        row_scale = torch.where(
            left > 0.0,
            left / torch.clamp(plan.sum(dim=2), min=1e-300),
            torch.zeros_like(left),
        )
        plan = plan * row_scale.unsqueeze(2)
        column_scale = torch.where(
            right > 0.0,
            right / torch.clamp(plan.sum(dim=1), min=1e-300),
            torch.zeros_like(right),
        )
        plan = plan * column_scale.unsqueeze(1)
        if bool((batched_marginal_residuals(plan, left, right) <= marginal_tolerance).all()):
            break
    residuals = batched_marginal_residuals(plan, left, right)
    if bool((residuals > marginal_tolerance).any()):
        worst = float(residuals.max().detach())
        raise ValueError(f"batched Sinkhorn marginal residual {worst:.3e} exceeds tolerance")
    return plan


def batched_marginal_residuals(plan: Tensor, left_mass: Tensor, right_mass: Tensor) -> Tensor:
    """Return the maximum row/column residual for every batch item."""

    coupling = _floating_tensor(plan)
    if coupling.ndim != 3:
        raise ValueError("plan must have shape (batch, n_left, n_right)")
    left = torch.as_tensor(left_mass, dtype=coupling.dtype, device=coupling.device)
    right = torch.as_tensor(right_mass, dtype=coupling.dtype, device=coupling.device)
    if left.shape != coupling.shape[:2] or right.shape != (coupling.shape[0], coupling.shape[2]):
        raise ValueError("mass tensors must match the plan shape")
    row = torch.max(torch.abs(coupling.sum(dim=2) - left), dim=1).values
    column = torch.max(torch.abs(coupling.sum(dim=1) - right), dim=1).values
    return torch.maximum(row, column)


def batched_entropic_transport_objective(
    plan: Tensor,
    cost: Tensor,
    left_mass: Tensor,
    right_mass: Tensor,
    *,
    epsilon: float,
) -> Tensor:
    """Evaluate ``<pi,C> + epsilon KL(pi || a tensor_product b)`` per item."""

    coupling = _floating_tensor(plan)
    values = torch.as_tensor(cost, dtype=coupling.dtype, device=coupling.device)
    left = torch.as_tensor(left_mass, dtype=coupling.dtype, device=coupling.device)
    right = torch.as_tensor(right_mass, dtype=coupling.dtype, device=coupling.device)
    if coupling.shape != values.shape or coupling.ndim != 3:
        raise ValueError("plan and cost must have the same three-dimensional shape")
    if left.shape != coupling.shape[:2] or right.shape != (coupling.shape[0], coupling.shape[2]):
        raise ValueError("mass tensors must match the plan shape")
    product = left.unsqueeze(2) * right.unsqueeze(1)
    positive = coupling > 0.0
    log_ratio = torch.log(torch.clamp(coupling, min=1e-300)) - torch.log(torch.clamp(product, min=1e-300))
    kl = torch.sum(torch.where(positive, coupling * log_ratio, torch.zeros_like(coupling)), dim=(1, 2))
    transport = torch.sum(coupling * values, dim=(1, 2))
    return transport + float(epsilon) * kl


def batched_sinkhorn_transport_objective(
    cost: Tensor,
    left_mass: Tensor,
    right_mass: Tensor,
    *,
    epsilon: float,
    iterations: int = 128,
    projection_iterations: int = 2048,
    marginal_tolerance: float = 1e-7,
) -> Tensor:
    """Evaluate the full entropic OT objective for each padded problem."""

    plan = batched_sinkhorn_plan(
        cost,
        left_mass,
        right_mass,
        epsilon=epsilon,
        iterations=iterations,
        projection_iterations=projection_iterations,
        marginal_tolerance=marginal_tolerance,
    )
    values = torch.as_tensor(cost, dtype=plan.dtype, device=plan.device)
    left = _normalize_batched_mass(torch.as_tensor(left_mass, dtype=plan.dtype, device=plan.device))
    right = _normalize_batched_mass(torch.as_tensor(right_mass, dtype=plan.dtype, device=plan.device))
    return batched_entropic_transport_objective(plan, values, left, right, epsilon=epsilon)


def debiased_objective_from_self_terms(
    cross_objective: Tensor,
    left_self_objective: Tensor,
    right_self_objective: Tensor,
) -> Tensor:
    """Debias precomputed entropic OT values without recomputing self terms."""

    cross = _floating_tensor(cross_objective)
    left = torch.as_tensor(left_self_objective, dtype=cross.dtype, device=cross.device)
    right = torch.as_tensor(right_self_objective, dtype=cross.dtype, device=cross.device)
    try:
        left, right = torch.broadcast_tensors(left, right)
        cross, left = torch.broadcast_tensors(cross, left)
        right = torch.broadcast_to(right, cross.shape)
    except RuntimeError as error:
        raise ValueError("self objectives are not broadcast-compatible with cross_objective") from error
    return cross - 0.5 * left - 0.5 * right


def _measure_sequence(value: ProductMeasure | Sequence[ProductMeasure]) -> tuple[ProductMeasure, ...]:
    if isinstance(value, ProductMeasure):
        return (value,)
    result = tuple(value)
    if not result or not all(isinstance(measure, ProductMeasure) for measure in result):
        raise ValueError("measure batch must contain ProductMeasure instances")
    return result


def _validate_measure_batch(
    left: Sequence[ProductMeasure],
    right: Sequence[ProductMeasure],
    *,
    geometry: RoleProductGeometry,
) -> None:
    reference = left[0]
    factors, dimension = reference.points.shape[1:]
    if factors != len(geometry.factor_curvatures):
        raise ValueError("measure factor count must match role geometry")
    if geometry.unoriented and factors != 3:
        raise ValueError("unoriented role geometry requires three path factors")
    device = reference.points.device
    for measure in tuple(left) + tuple(right):
        if measure.points.shape[1:] != (factors, dimension):
            raise ValueError("all measures must share factor count and point dimension")
        if measure.points.device != device:
            raise ValueError("all measures in a batch must use the same device")
    if geometry.side_weight > 0.0:
        widths = {
            int(measure.side_features.shape[1])
            for measure in tuple(left) + tuple(right)
            if measure.side_features is not None
        }
        if len(widths) != 1 or any(measure.side_features is None for measure in tuple(left) + tuple(right)):
            raise ValueError("positive side_weight requires equal-width side features for every measure")


def _pad_measures(measures: Sequence[ProductMeasure]) -> tuple[Tensor, Tensor, tuple[int, ...]]:
    sizes = tuple(int(measure.points.shape[0]) for measure in measures)
    max_size = max(sizes)
    reference = measures[0].points
    points = reference.new_zeros((len(measures), max_size, *reference.shape[1:]))
    mass = reference.new_zeros((len(measures), max_size))
    for index, measure in enumerate(measures):
        size = sizes[index]
        points[index, :size] = measure.points.to(dtype=reference.dtype, device=reference.device)
        mass[index, :size] = measure.mass.to(dtype=reference.dtype, device=reference.device)
    return points, mass, sizes


def _batched_side_cost(
    geometry: RoleProductGeometry,
    left: Sequence[ProductMeasure],
    right: Sequence[ProductMeasure],
    *,
    reverse_right: bool,
) -> Tensor:
    batch = len(left)
    max_left = max(measure.points.shape[0] for measure in left)
    max_right = max(measure.points.shape[0] for measure in right)
    reference = left[0].points
    if geometry.side_weight == 0.0:
        return reference.new_zeros((batch, max_left, max_right))
    side_dim = int(left[0].side_features.shape[1])
    left_features = reference.new_zeros((batch, max_left, side_dim))
    right_features = reference.new_zeros((batch, max_right, side_dim))
    for index, (left_measure, right_measure) in enumerate(zip(left, right)):
        left_features[index, : left_measure.points.shape[0]] = left_measure.side_features
        selected_right = (
            right_measure.reversed_side_features
            if reverse_right and right_measure.reversed_side_features is not None
            else right_measure.side_features
        )
        right_features[index, : right_measure.points.shape[0]] = selected_right
    difference = left_features.unsqueeze(2) - right_features.unsqueeze(1)
    return float(geometry.side_weight) * torch.sum(difference.square(), dim=-1)


def _floating_tensor(value: Tensor) -> Tensor:
    tensor = torch.as_tensor(value)
    return tensor if torch.is_floating_point(tensor) else tensor.to(dtype=torch.float64)


def _validate_batched_masses(mass: Tensor, *, name: str) -> None:
    if mass.ndim != 2 or mass.shape[0] == 0 or mass.shape[1] == 0:
        raise ValueError(f"{name} must have shape (batch, n_items)")
    if not torch.isfinite(mass).all() or bool((mass < 0.0).any()):
        raise ValueError(f"{name} must contain finite non-negative values")
    if bool((mass.sum(dim=1) <= 0.0).any()):
        raise ValueError(f"every row of {name} must have positive total mass")


def _normalize_batched_mass(mass: Tensor) -> Tensor:
    return mass / mass.sum(dim=1, keepdim=True)
