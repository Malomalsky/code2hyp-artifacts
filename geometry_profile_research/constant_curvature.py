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
    Floating-point input dtypes are preserved so numerical audits can run in
    ``float64`` without silent downcasts.
    """

    points: Tensor
    mass: Tensor
    side_features: Tensor | None = None
    reversed_side_features: Tensor | None = None

    def __post_init__(self) -> None:
        points = torch.as_tensor(self.points)
        if not torch.is_floating_point(points):
            points = points.to(dtype=torch.float64)
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
        side_features = self._validate_side_features(self.side_features, points=points, field_name="side_features")
        reversed_side_features = self._validate_side_features(
            self.reversed_side_features,
            points=points,
            field_name="reversed_side_features",
        )
        if reversed_side_features is not None and side_features is None:
            raise ValueError("reversed_side_features requires side_features")
        if (
            reversed_side_features is not None
            and side_features is not None
            and reversed_side_features.shape[1] != side_features.shape[1]
        ):
            raise ValueError("side_features and reversed_side_features must have the same width")
        object.__setattr__(self, "points", points)
        object.__setattr__(self, "mass", mass / total)
        object.__setattr__(self, "side_features", side_features)
        object.__setattr__(self, "reversed_side_features", reversed_side_features)

    @staticmethod
    def _validate_side_features(features: Tensor | None, *, points: Tensor, field_name: str) -> Tensor | None:
        if features is None:
            return None
        values = torch.as_tensor(features, dtype=points.dtype, device=points.device)
        if values.ndim != 2 or values.shape[0] != points.shape[0]:
            raise ValueError(f"{field_name} must have shape (n_paths, side_dim)")
        if not torch.isfinite(values).all():
            raise ValueError(f"{field_name} must contain only finite values")
        return values


@dataclass(frozen=True)
class ConstantCurvatureProduct:
    """Product metric with explicit Euclidean and Poincare branches.

    Curvature ``c=0`` is the Euclidean control. Curvature ``c>0`` is the
    standard Poincare ball with sectional curvature ``-c``. For fixed ball
    coordinates the geodesic distance converges to ``2 * ||x-y||`` as
    ``c -> 0``; matched Euclidean controls must therefore handle scale through
    prespecified factor weights rather than by changing the metric formula.
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

    @property
    def sectional_curvature(self) -> float:
        """Sectional curvature of the active point geometry."""

        return 0.0 if self.is_euclidean else -float(self.curvature)

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
        return 2.0 * torch.asinh(argument) / sqrt_c

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
        if not torch.isclose(weights[1], weights[2]):
            raise ValueError("unoriented endpoint quotient requires equal start/end factor weights")
        reversed_right = torch.stack((right_points[:, 0], right_points[:, 2], right_points[:, 1]), dim=1)
        reversed_cost = self._ordered_product_cost(left_points, reversed_right, weights)
        reversed_cost = reversed_cost + self._side_cost_matrix(left, right, reverse_right=True)
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

    def _side_cost_matrix(self, left: ProductMeasure, right: ProductMeasure, *, reverse_right: bool = False) -> Tensor:
        if self.side_weight == 0.0:
            return left.points.new_zeros((left.points.shape[0], right.points.shape[0]))
        if left.side_features is None and right.side_features is None:
            return left.points.new_zeros((left.points.shape[0], right.points.shape[0]))
        if left.side_features is None or right.side_features is None:
            raise ValueError("both measures must either provide side_features or omit them")
        right_side_features = (
            right.reversed_side_features
            if reverse_right and right.reversed_side_features is not None
            else right.side_features
        )
        if left.side_features.shape[1] != right_side_features.shape[1]:
            raise ValueError("left and right side_features must have the same dimension")
        diff = left.side_features.unsqueeze(1) - right_side_features.unsqueeze(0)
        return float(self.side_weight) * torch.sum(diff.square(), dim=-1)


@dataclass(frozen=True)
class RoleProductGeometry:
    """Product geometry with separately specified curvature for each path role.

    For the canonical factors ``(LCA, start, end)``, curvatures ``(c, 0, 0)``
    define ``H_{-c} x E x E``. Endpoint exchange is a valid quotient operation
    only when start and end use equal curvatures and weights.
    """

    factor_curvatures: Sequence[float] = (1.0, 0.0, 0.0)
    factor_weights: Sequence[float] = (1.0, 1.0, 1.0)
    side_weight: float = 0.0
    unoriented: bool = True
    eps: float = 1e-5

    def __post_init__(self) -> None:
        curvatures = tuple(float(value) for value in self.factor_curvatures)
        weights = tuple(float(value) for value in self.factor_weights)
        if not curvatures or len(curvatures) != len(weights):
            raise ValueError("factor_curvatures and factor_weights must have the same nonzero length")
        if any(value < 0.0 for value in curvatures):
            raise ValueError("factor curvatures must be non-negative")
        if any(value < 0.0 for value in weights):
            raise ValueError("factor weights must be non-negative")
        if self.side_weight < 0.0:
            raise ValueError("side_weight must be non-negative")
        if self.unoriented:
            if len(curvatures) != 3:
                raise ValueError("unoriented path quotient requires exactly three factors")
            if curvatures[1] != curvatures[2]:
                raise ValueError("unoriented endpoints require equal start/end curvatures")
            if weights[1] != weights[2]:
                raise ValueError("unoriented endpoints require equal start/end weights")
        object.__setattr__(self, "factor_curvatures", curvatures)
        object.__setattr__(self, "factor_weights", weights)

    @property
    def factor_sectional_curvatures(self) -> tuple[float, ...]:
        return tuple(
            0.0 if curvature == 0.0 or weight == 0.0 else -curvature / weight
            for curvature, weight in zip(self.factor_curvatures, self.factor_weights)
        )

    def point_distance(self, left: Tensor, right: Tensor, *, factor_index: int) -> Tensor:
        if factor_index < 0 or factor_index >= len(self.factor_curvatures):
            raise IndexError("factor_index is out of range")
        geometry = ConstantCurvatureProduct(
            curvature=self.factor_curvatures[factor_index],
            factor_weights=(1.0,),
            side_weight=0.0,
            eps=self.eps,
        )
        return geometry.point_distance(left, right)

    def path_cost_matrix(self, left: ProductMeasure, right: ProductMeasure) -> Tensor:
        if left.points.shape[1:] != right.points.shape[1:]:
            raise ValueError("left and right measures must have matching factor and point dimensions")
        if left.points.shape[1] != len(self.factor_curvatures):
            raise ValueError("measure factor count must match role geometry")
        direct = self._ordered_product_cost(left.points, right.points) + self._side_cost_matrix(left, right)
        if not self.unoriented:
            return direct
        reversed_right = torch.stack((right.points[:, 0], right.points[:, 2], right.points[:, 1]), dim=1)
        reversed_cost = self._ordered_product_cost(left.points, reversed_right)
        reversed_cost = reversed_cost + self._side_cost_matrix(left, right, reverse_right=True)
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
        return sinkhorn_divergence(
            self.path_cost_matrix(left, right),
            left.mass,
            right.mass,
            left_self_cost=self.path_cost_matrix(left, left),
            right_self_cost=self.path_cost_matrix(right, right),
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
        return sinkhorn_plan(
            self.path_cost_matrix(left, right),
            left.mass,
            right.mass,
            epsilon=epsilon,
            iterations=iterations,
            projection_iterations=projection_iterations,
        )

    def _ordered_product_cost(self, left_points: Tensor, right_points: Tensor) -> Tensor:
        terms = []
        for factor_index, weight in enumerate(self.factor_weights):
            left_factor = left_points[:, factor_index].unsqueeze(1)
            right_factor = right_points[:, factor_index].unsqueeze(0)
            distance = self.point_distance(left_factor, right_factor, factor_index=factor_index)
            terms.append(float(weight) * distance.square())
        return torch.stack(terms, dim=0).sum(dim=0)

    def _side_cost_matrix(self, left: ProductMeasure, right: ProductMeasure, *, reverse_right: bool = False) -> Tensor:
        if self.side_weight == 0.0:
            return left.points.new_zeros((left.points.shape[0], right.points.shape[0]))
        if left.side_features is None and right.side_features is None:
            return left.points.new_zeros((left.points.shape[0], right.points.shape[0]))
        if left.side_features is None or right.side_features is None:
            raise ValueError("both measures must either provide side_features or omit them")
        right_features = right.reversed_side_features if reverse_right and right.reversed_side_features is not None else right.side_features
        if left.side_features.shape[1] != right_features.shape[1]:
            raise ValueError("side feature dimensions must match")
        diff = left.side_features.unsqueeze(1) - right_features.unsqueeze(0)
        return float(self.side_weight) * torch.sum(diff.square(), dim=-1)


@dataclass(frozen=True)
class ConcatenatedEuclideanGeometry:
    """Independent 3d concatenation control for Euclidean role products.

    Scaling each role by the square root of its product weight makes squared
    Euclidean distance in the concatenated vector algebraically identical to
    the weighted Euclidean product cost. The separate implementation serves as
    a dimensional-capacity and numerical-consistency control.
    """

    factor_weights: Sequence[float] = (1.0, 1.0, 1.0)
    side_weight: float = 0.0
    unoriented: bool = True

    def __post_init__(self) -> None:
        weights = tuple(float(value) for value in self.factor_weights)
        if not weights or any(value < 0.0 for value in weights):
            raise ValueError("factor weights must be a non-empty non-negative sequence")
        if self.side_weight < 0.0:
            raise ValueError("side_weight must be non-negative")
        if self.unoriented and len(weights) != 3:
            raise ValueError("unoriented concatenation requires exactly three factors")
        if self.unoriented and weights[1] != weights[2]:
            raise ValueError("unoriented endpoints require equal start/end weights")
        object.__setattr__(self, "factor_weights", weights)

    def path_cost_matrix(self, left: ProductMeasure, right: ProductMeasure) -> Tensor:
        if left.points.shape[1:] != right.points.shape[1:]:
            raise ValueError("left and right measures must have matching factor and point dimensions")
        if left.points.shape[1] != len(self.factor_weights):
            raise ValueError("measure factor count must match concatenation weights")
        direct = self._ordered_cost(left.points, right.points) + self._side_cost_matrix(left, right)
        if not self.unoriented:
            return direct
        reversed_right = torch.stack((right.points[:, 0], right.points[:, 2], right.points[:, 1]), dim=1)
        reverse = self._ordered_cost(left.points, reversed_right)
        reverse = reverse + self._side_cost_matrix(left, right, reverse_right=True)
        return torch.minimum(direct, reverse)

    def sinkhorn_divergence(
        self,
        left: ProductMeasure,
        right: ProductMeasure,
        *,
        epsilon: float,
        iterations: int = 128,
        projection_iterations: int = 2048,
    ) -> Tensor:
        return sinkhorn_divergence(
            self.path_cost_matrix(left, right),
            left.mass,
            right.mass,
            left_self_cost=self.path_cost_matrix(left, left),
            right_self_cost=self.path_cost_matrix(right, right),
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
        return sinkhorn_plan(
            self.path_cost_matrix(left, right),
            left.mass,
            right.mass,
            epsilon=epsilon,
            iterations=iterations,
            projection_iterations=projection_iterations,
        )

    def _ordered_cost(self, left_points: Tensor, right_points: Tensor) -> Tensor:
        scale = left_points.new_tensor(self.factor_weights).sqrt().view(1, -1, 1)
        left_flat = (left_points * scale).reshape(left_points.shape[0], -1)
        right_flat = (right_points * scale).reshape(right_points.shape[0], -1)
        difference = left_flat.unsqueeze(1) - right_flat.unsqueeze(0)
        return torch.sum(difference.square(), dim=-1)

    def _side_cost_matrix(self, left: ProductMeasure, right: ProductMeasure, *, reverse_right: bool = False) -> Tensor:
        if self.side_weight == 0.0:
            return left.points.new_zeros((left.points.shape[0], right.points.shape[0]))
        if left.side_features is None and right.side_features is None:
            return left.points.new_zeros((left.points.shape[0], right.points.shape[0]))
        if left.side_features is None or right.side_features is None:
            raise ValueError("both measures must either provide side_features or omit them")
        right_features = right.reversed_side_features if reverse_right and right.reversed_side_features is not None else right.side_features
        if left.side_features.shape[1] != right_features.shape[1]:
            raise ValueError("side feature dimensions must match")
        difference = left.side_features.unsqueeze(1) - right_features.unsqueeze(0)
        return float(self.side_weight) * torch.sum(difference.square(), dim=-1)


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
