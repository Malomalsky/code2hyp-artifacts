from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import Tensor

from geometry_profile_research.gromov_wasserstein import sinkhorn_divergence, sinkhorn_plan


@dataclass(frozen=True)
class ProductMeasure:
    """Finite measure over LCA-anchored ordered product path objects.

    ``points[i]`` stores one path object. In the canonical Code2Hyp setting the
    three product factors are ``(x_lca, x_start, x_end)``. The class is geometry
    agnostic; the associated metric is supplied by ``ConstantCurvatureProduct``.
    """

    points: Tensor
    mass: Tensor
    side_features: Tensor | None = None

    def __post_init__(self) -> None:
        points = torch.as_tensor(self.points)
        if not torch.is_floating_point(points):
            points = points.to(dtype=torch.float32)
        if points.ndim != 3:
            raise ValueError("points must have shape (n_paths, n_factors, dim)")
        mass = torch.as_tensor(self.mass, dtype=points.dtype, device=points.device)
        if mass.ndim != 1 or mass.numel() != points.shape[0]:
            raise ValueError("mass must contain one value per path object")
        if not torch.isfinite(points).all():
            raise ValueError("points must contain only finite values")
        if not torch.isfinite(mass).all():
            raise ValueError("mass must contain only finite values")
        if bool((mass < 0.0).any()):
            raise ValueError("mass values must be non-negative")
        total = mass.sum()
        if float(total.detach()) <= 0.0:
            raise ValueError("mass must have a positive total")
        if self.side_features is None:
            side_features = None
        else:
            side_features = torch.as_tensor(self.side_features, dtype=points.dtype, device=points.device)
            if side_features.ndim != 2 or side_features.shape[0] != points.shape[0]:
                raise ValueError("side_features must have shape (n_paths, side_dim)")
            if not torch.isfinite(side_features).all():
                raise ValueError("side_features must contain only finite values")
        object.__setattr__(self, "points", points)
        object.__setattr__(self, "mass", mass / total)
        object.__setattr__(self, "side_features", side_features)


@dataclass(frozen=True)
class ConstantCurvatureProduct:
    """Product metric with an explicit Euclidean curvature limit.

    Curvature ``c=0`` is the Euclidean control. Curvature ``c>0`` is the
    Poincare ball with sectional curvature ``-c``. The Poincare geodesic is
    multiplied by ``1/2`` so that, for fixed ball coordinates, ``D_c`` converges
    to the Euclidean distance as ``c -> 0``. This normalization is required for
    a fair curvature-continuation audit; otherwise the usual Poincare formula
    converges to ``2 * ||x-y||``.
    """

    curvature: float = 0.0
    factor_weights: Sequence[float] = (1.0, 1.0, 1.0)
    side_weight: float = 1.0
    unoriented: bool = False
    eps: float = 1e-5

    def __post_init__(self) -> None:
        if self.curvature < 0.0:
            raise ValueError("curvature must be non-negative")
        weights = tuple(float(value) for value in self.factor_weights)
        if not weights:
            raise ValueError("factor_weights must not be empty")
        if any(value < 0.0 for value in weights):
            raise ValueError("factor_weights must be non-negative")
        if self.side_weight < 0.0:
            raise ValueError("side_weight must be non-negative")
        object.__setattr__(self, "factor_weights", weights)

    @property
    def is_euclidean(self) -> bool:
        return self.curvature == 0.0

    def project(self, points: Tensor) -> Tensor:
        """Project points into the admissible ball for ``c>0``."""

        points = torch.as_tensor(points)
        if not torch.is_floating_point(points):
            points = points.to(dtype=torch.float32)
        if self.is_euclidean:
            return points
        curvature = points.new_tensor(self.curvature)
        radius = ((1.0 - self.eps) / torch.sqrt(curvature)) * (1.0 - 1e-12)
        norm = torch.linalg.vector_norm(points, dim=-1, keepdim=True)
        scale = torch.minimum(torch.ones_like(norm), radius / torch.clamp(norm, min=1e-15))
        return points * scale

    def point_distance(self, left: Tensor, right: Tensor) -> Tensor:
        """Distance between points under the selected constant-curvature metric."""

        left = torch.as_tensor(left)
        if not torch.is_floating_point(left):
            left = left.to(dtype=torch.float32)
        right = torch.as_tensor(right, dtype=left.dtype, device=left.device)
        if self.is_euclidean:
            return torch.linalg.vector_norm(left - right, dim=-1)
        left = self.project(left)
        right = self.project(right)
        curvature = left.new_tensor(self.curvature)
        sqrt_c = torch.sqrt(curvature)
        left_norm2 = torch.sum(left * left, dim=-1)
        right_norm2 = torch.sum(right * right, dim=-1)
        diff_norm = torch.linalg.vector_norm(left - right, dim=-1)
        denominator = torch.sqrt(
            torch.clamp(
                (1.0 - curvature * left_norm2) * (1.0 - curvature * right_norm2),
                min=1e-30,
            )
        )
        argument = sqrt_c * diff_norm / denominator
        poincare_distance = 2.0 * torch.asinh(argument) / sqrt_c
        return 0.5 * poincare_distance

    def path_cost_matrix(self, left: ProductMeasure, right: ProductMeasure) -> Tensor:
        """Squared product cost between two path-object measures."""

        if left.points.shape[1] != right.points.shape[1]:
            raise ValueError("left and right measures must have the same number of product factors")
        if left.points.shape[2] != right.points.shape[2]:
            raise ValueError("left and right measures must have the same point dimension")
        if len(self.factor_weights) != left.points.shape[1]:
            raise ValueError("factor_weights length must match the number of product factors")
        weights = left.points.new_tensor(self.factor_weights)
        left_points = self.project(left.points)
        right_points = self.project(right.points)
        direct = self._ordered_product_cost(left_points, right_points, weights)
        direct = direct + self._side_cost_matrix(left, right)
        if not self.unoriented or left_points.shape[1] != 3:
            return direct
        reversed_right = torch.stack((right_points[:, 0], right_points[:, 2], right_points[:, 1]), dim=1)
        reversed_cost = self._ordered_product_cost(left_points, reversed_right, weights)
        reversed_cost = reversed_cost + self._side_cost_matrix(left, right)
        return torch.minimum(direct, reversed_cost)

    def sinkhorn_divergence(
        self,
        left: ProductMeasure,
        right: ProductMeasure,
        *,
        epsilon: float,
        iterations: int = 128,
        projection_iterations: int = 2048,
    ) -> Tensor:
        """Debiased Sinkhorn divergence under the product ground cost."""

        cross = self.path_cost_matrix(left, right)
        left_self = self.path_cost_matrix(left, left)
        right_self = self.path_cost_matrix(right, right)
        return sinkhorn_divergence(
            cross,
            left.mass,
            right.mass,
            left_self_cost=left_self,
            right_self_cost=right_self,
            epsilon=epsilon,
            iterations=iterations,
            projection_iterations=projection_iterations,
        )

    def transport_plan(
        self,
        left: ProductMeasure,
        right: ProductMeasure,
        *,
        epsilon: float,
        iterations: int = 128,
        projection_iterations: int = 2048,
    ) -> Tensor:
        """Entropic transport plan for the cross cost."""

        return sinkhorn_plan(
            self.path_cost_matrix(left, right),
            left.mass,
            right.mass,
            epsilon=epsilon,
            iterations=iterations,
            projection_iterations=projection_iterations,
        )

    def _ordered_product_cost(self, left_points: Tensor, right_points: Tensor, weights: Tensor) -> Tensor:
        terms = []
        for factor_index in range(left_points.shape[1]):
            left_factor = left_points[:, factor_index].unsqueeze(1)
            right_factor = right_points[:, factor_index].unsqueeze(0)
            terms.append(weights[factor_index] * self.point_distance(left_factor, right_factor).square())
        return torch.stack(terms, dim=0).sum(dim=0)

    def _side_cost_matrix(self, left: ProductMeasure, right: ProductMeasure) -> Tensor:
        if self.side_weight == 0.0:
            return left.points.new_zeros((left.points.shape[0], right.points.shape[0]))
        if left.side_features is None and right.side_features is None:
            return left.points.new_zeros((left.points.shape[0], right.points.shape[0]))
        if left.side_features is None or right.side_features is None:
            raise ValueError("both measures must either provide side_features or omit them")
        if left.side_features.shape[1] != right.side_features.shape[1]:
            raise ValueError("left and right side_features must have the same dimension")
        diff = left.side_features.unsqueeze(1) - right.side_features.unsqueeze(0)
        return float(self.side_weight) * torch.sum(diff.square(), dim=-1)


def median_positive_cost_scale(cost_matrices: Sequence[Tensor], *, fallback: float = 1.0) -> float:
    """Median positive finite ground cost used to freeze Sinkhorn scale."""

    positives = []
    for matrix in cost_matrices:
        values = torch.as_tensor(matrix).detach().reshape(-1)
        values = values[torch.isfinite(values) & (values > 0.0)]
        if values.numel():
            positives.append(values)
    if not positives:
        return float(fallback)
    median = torch.quantile(torch.cat(positives), 0.5)
    value = float(median.detach())
    if value <= 0.0:
        return float(fallback)
    return value


def scaled_sinkhorn_epsilon(cost_scale: float, *, kappa: float = 0.05, min_epsilon: float = 1e-6) -> float:
    """Relative Sinkhorn regularization ``epsilon = kappa * cost_scale``."""

    if kappa <= 0.0:
        raise ValueError("kappa must be positive")
    if min_epsilon <= 0.0:
        raise ValueError("min_epsilon must be positive")
    return max(float(kappa) * float(cost_scale), float(min_epsilon))
