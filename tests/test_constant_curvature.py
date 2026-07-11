from __future__ import annotations

import torch
import pytest

from geometry_profile_research.constant_curvature import (
    ConcatenatedEuclideanGeometry,
    ConstantCurvatureProduct,
    ProductMeasure,
    RoleProductGeometry,
    median_positive_cost_scale,
    scaled_sinkhorn_epsilon,
)


def test_zero_curvature_distance_is_euclidean() -> None:
    left = torch.tensor([[0.0, 0.0], [1.0, 0.0]], dtype=torch.float32)
    right = torch.tensor([[3.0, 4.0], [1.0, 2.0]], dtype=torch.float32)
    geometry = ConstantCurvatureProduct(curvature=0.0)

    distance = geometry.point_distance(left, right)

    assert torch.allclose(distance, torch.linalg.vector_norm(left - right, dim=-1))


def test_small_positive_curvature_converges_to_standard_poincare_coordinate_limit() -> None:
    generator = torch.Generator().manual_seed(7)
    left = torch.randn((5, 3), generator=generator) * 0.1
    right = torch.randn((5, 3), generator=generator) * 0.1
    euclidean = ConstantCurvatureProduct(curvature=0.0).point_distance(left, right)
    near_zero = ConstantCurvatureProduct(curvature=1e-8).point_distance(left, right)

    assert torch.allclose(near_zero, 2.0 * euclidean, atol=1e-6, rtol=1e-5)


def test_path_cost_matrix_uses_single_curvature_family() -> None:
    points = torch.tensor(
        [
            [[0.00, 0.00], [0.03, 0.01], [0.04, -0.02]],
            [[0.01, 0.00], [0.02, 0.02], [0.05, -0.01]],
        ],
        dtype=torch.float32,
    )
    mass = torch.tensor([0.4, 0.6], dtype=torch.float32)
    measure = ProductMeasure(points=points, mass=mass)

    euclidean = ConstantCurvatureProduct(curvature=0.0).path_cost_matrix(measure, measure)
    near_zero = ConstantCurvatureProduct(curvature=1e-8).path_cost_matrix(measure, measure)

    assert torch.allclose(torch.diagonal(euclidean), torch.zeros(2), atol=1e-7)
    assert torch.allclose(near_zero, 4.0 * euclidean, atol=1e-7, rtol=1e-5)


def test_positive_curvature_reports_standard_sectional_curvature() -> None:
    assert ConstantCurvatureProduct(curvature=0.0).sectional_curvature == 0.0
    assert ConstantCurvatureProduct(curvature=2.5).sectional_curvature == -2.5


def test_path_cost_matrix_includes_euclidean_side_features() -> None:
    points = torch.zeros((2, 3, 2), dtype=torch.float32)
    mass = torch.tensor([0.5, 0.5], dtype=torch.float32)
    left = ProductMeasure(points=points, mass=mass, side_features=torch.tensor([[0.0, 0.0], [1.0, 0.0]]))
    right = ProductMeasure(points=points, mass=mass, side_features=torch.tensor([[0.0, 1.0], [1.0, 1.0]]))
    geometry = ConstantCurvatureProduct(curvature=0.0, side_weight=2.0)

    cost = geometry.path_cost_matrix(left, right)

    assert torch.allclose(cost, torch.tensor([[2.0, 4.0], [4.0, 2.0]]))


def test_unoriented_path_cost_reverses_side_features_when_available() -> None:
    left = ProductMeasure(
        points=torch.tensor([[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]], dtype=torch.float64),
        mass=torch.ones(1, dtype=torch.float64),
        side_features=torch.tensor([[0.0, 1.0, 2.0, 3.0]], dtype=torch.float64),
        reversed_side_features=torch.tensor([[0.0, 1.0, 3.0, 2.0]], dtype=torch.float64),
    )
    right_reversed = ProductMeasure(
        points=torch.tensor([[[0.0, 0.0], [0.0, 1.0], [1.0, 0.0]]], dtype=torch.float64),
        mass=torch.ones(1, dtype=torch.float64),
        side_features=torch.tensor([[0.0, 1.0, 3.0, 2.0]], dtype=torch.float64),
        reversed_side_features=torch.tensor([[0.0, 1.0, 2.0, 3.0]], dtype=torch.float64),
    )
    geometry = ConstantCurvatureProduct(curvature=0.0, side_weight=1.0, unoriented=True)

    cost = geometry.path_cost_matrix(left, right_reversed)

    assert torch.allclose(cost, torch.zeros((1, 1), dtype=torch.float64), atol=1e-12)


def test_unoriented_path_cost_requires_equal_endpoint_weights() -> None:
    measure = ProductMeasure(points=torch.zeros((1, 3, 2)), mass=torch.ones(1))
    geometry = ConstantCurvatureProduct(curvature=0.0, factor_weights=(1.0, 1.0, 0.5), unoriented=True)

    with pytest.raises(ValueError, match="equal start/end"):
        geometry.path_cost_matrix(measure, measure)


def test_role_product_geometry_supports_hyperbolic_anchor_and_euclidean_endpoints() -> None:
    geometry = RoleProductGeometry(
        factor_curvatures=(1.0, 0.0, 0.0),
        factor_weights=(2.0, 3.0, 3.0),
        unoriented=True,
    )
    left = ProductMeasure(points=torch.tensor([[[0.0], [1.0], [3.0]]], dtype=torch.float64), mass=torch.ones(1))
    right = ProductMeasure(points=torch.tensor([[[0.2], [2.0], [4.0]]], dtype=torch.float64), mass=torch.ones(1))

    anchor_distance = geometry.point_distance(left.points[:, 0], right.points[:, 0], factor_index=0)
    expected = 2.0 * anchor_distance.square().squeeze() + 3.0 * ((1.0 - 2.0) ** 2 + (3.0 - 4.0) ** 2)

    torch.testing.assert_close(geometry.path_cost_matrix(left, right).squeeze(), expected)
    assert geometry.factor_sectional_curvatures == (-0.5, 0.0, 0.0)


def test_role_product_geometry_is_invariant_to_endpoint_reversal_with_side_features() -> None:
    geometry = RoleProductGeometry(factor_curvatures=(1.0, 0.0, 0.0), unoriented=True, side_weight=1.0)
    forward = ProductMeasure(
        points=torch.tensor([[[0.1], [0.2], [0.4]]], dtype=torch.float64),
        mass=torch.ones(1),
        side_features=torch.tensor([[1.0, 2.0]], dtype=torch.float64),
        reversed_side_features=torch.tensor([[2.0, 1.0]], dtype=torch.float64),
    )
    reversed_path = ProductMeasure(
        points=torch.tensor([[[0.1], [0.4], [0.2]]], dtype=torch.float64),
        mass=torch.ones(1),
        side_features=torch.tensor([[2.0, 1.0]], dtype=torch.float64),
        reversed_side_features=torch.tensor([[1.0, 2.0]], dtype=torch.float64),
    )

    torch.testing.assert_close(geometry.path_cost_matrix(forward, reversed_path), torch.zeros((1, 1), dtype=torch.float64))


def test_role_product_geometry_rejects_mismatched_endpoint_spaces() -> None:
    with pytest.raises(ValueError, match="equal start/end curvatures"):
        RoleProductGeometry(factor_curvatures=(1.0, 0.0, 1.0), unoriented=True)


def test_weighted_euclidean_product_equals_weighted_3d_concatenation() -> None:
    generator = torch.Generator().manual_seed(17)
    left = ProductMeasure(
        points=torch.randn((4, 3, 5), generator=generator, dtype=torch.float64),
        mass=torch.full((4,), 0.25, dtype=torch.float64),
    )
    right = ProductMeasure(
        points=torch.randn((3, 3, 5), generator=generator, dtype=torch.float64),
        mass=torch.full((3,), 1.0 / 3.0, dtype=torch.float64),
    )
    weights = (2.0, 0.75, 0.75)
    product = RoleProductGeometry(
        factor_curvatures=(0.0, 0.0, 0.0),
        factor_weights=weights,
        unoriented=True,
    )
    concatenated = ConcatenatedEuclideanGeometry(factor_weights=weights, unoriented=True)

    torch.testing.assert_close(
        concatenated.path_cost_matrix(left, right),
        product.path_cost_matrix(left, right),
        atol=1e-12,
        rtol=1e-12,
    )
    torch.testing.assert_close(
        concatenated.sinkhorn_divergence(left, right, epsilon=0.2),
        product.sinkhorn_divergence(left, right, epsilon=0.2),
        atol=1e-10,
        rtol=1e-10,
    )


def test_relative_sinkhorn_scale_uses_positive_median() -> None:
    scale = median_positive_cost_scale(
        (
            torch.tensor([[0.0, 2.0], [4.0, 0.0]]),
            torch.tensor([[0.0, 10.0], [8.0, 0.0]]),
        )
    )

    assert scale == 6.0
    assert scaled_sinkhorn_epsilon(scale, kappa=0.25) == 1.5
