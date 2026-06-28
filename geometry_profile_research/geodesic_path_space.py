from __future__ import annotations

import torch
from torch import Tensor

from geometry_profile_research.code2hyp_torch import (
    torch_expmap,
    torch_logmap,
    torch_poincare_distance,
    torch_project_to_ball,
)


def poincare_geodesic_points(
    start: Tensor,
    end: Tensor,
    *,
    curvature: Tensor | float,
    num_points: int = 16,
    eps: float = 1e-5,
) -> Tensor:
    """Discretize the oriented geodesic segment from ``start`` to ``end``.

    The segment is represented by
    ``gamma(t) = exp_start(t log_start(end))`` for ``t in [0, 1]``.
    """

    if num_points < 2:
        raise ValueError("num_points must be at least 2")
    curvature_tensor = torch.as_tensor(curvature, dtype=start.dtype, device=start.device)
    start = torch_project_to_ball(start, curvature_tensor, eps=eps)
    end = torch_project_to_ball(end, curvature_tensor, eps=eps)
    tangent = torch_logmap(start, end, curvature=curvature_tensor, eps=eps)
    steps = torch.linspace(0.0, 1.0, num_points, dtype=start.dtype, device=start.device)
    view_shape = (1,) * (start.ndim - 1) + (num_points, 1)
    scaled_tangent = steps.view(view_shape) * tangent.unsqueeze(-2)
    expanded_start = start.unsqueeze(-2).expand_as(scaled_tangent)
    return torch_expmap(expanded_start, scaled_tangent, curvature=curvature_tensor, eps=eps)


def poincare_directed_endpoint_product_distance(
    left_start: Tensor,
    left_end: Tensor,
    right_start: Tensor,
    right_end: Tensor,
    *,
    curvature: Tensor | float,
    left_weight: float = 1.0,
    right_weight: float = 1.0,
) -> Tensor:
    """Directed product distance between oriented geodesic path endpoints."""

    if left_weight < 0 or right_weight < 0:
        raise ValueError("endpoint weights must be non-negative")
    start_distance = torch_poincare_distance(left_start, right_start, curvature=curvature)
    end_distance = torch_poincare_distance(left_end, right_end, curvature=curvature)
    squared = left_weight * start_distance.square() + right_weight * end_distance.square()
    return torch.sqrt(torch.clamp(squared, min=0.0))


def poincare_unoriented_endpoint_product_distance(
    left_start: Tensor,
    left_end: Tensor,
    right_start: Tensor,
    right_end: Tensor,
    *,
    curvature: Tensor | float,
    left_weight: float = 1.0,
    right_weight: float = 1.0,
) -> Tensor:
    """Endpoint product distance on the quotient that identifies path reversal."""

    directed = poincare_directed_endpoint_product_distance(
        left_start,
        left_end,
        right_start,
        right_end,
        curvature=curvature,
        left_weight=left_weight,
        right_weight=right_weight,
    )
    reversed_directed = poincare_directed_endpoint_product_distance(
        left_start,
        left_end,
        right_end,
        right_start,
        curvature=curvature,
        left_weight=left_weight,
        right_weight=right_weight,
    )
    return torch.minimum(directed, reversed_directed)


def poincare_geodesic_discrete_hausdorff_distance(
    left_start: Tensor,
    left_end: Tensor,
    right_start: Tensor,
    right_end: Tensor,
    *,
    curvature: Tensor | float,
    num_points: int = 16,
) -> Tensor:
    """Discrete Hausdorff distance between two geodesic segments."""

    left_points = poincare_geodesic_points(left_start, left_end, curvature=curvature, num_points=num_points)
    right_points = poincare_geodesic_points(right_start, right_end, curvature=curvature, num_points=num_points)
    pairwise = torch_poincare_distance(
        left_points.unsqueeze(-2),
        right_points.unsqueeze(-3),
        curvature=curvature,
    )
    left_to_right = pairwise.min(dim=-1).values.max(dim=-1).values
    right_to_left = pairwise.min(dim=-2).values.max(dim=-1).values
    return torch.maximum(left_to_right, right_to_left)


def poincare_gromov_product_at_origin(
    left: Tensor,
    right: Tensor,
    *,
    curvature: Tensor | float,
) -> Tensor:
    """Gromov product ``(left|right)_o`` with the Poincare origin as basepoint."""

    origin = torch.zeros_like(left)
    return 0.5 * (
        torch_poincare_distance(origin, left, curvature=curvature)
        + torch_poincare_distance(origin, right, curvature=curvature)
        - torch_poincare_distance(left, right, curvature=curvature)
    )
