from __future__ import annotations

import torch

from geometry_profile_research.constant_curvature import (
    ConstantCurvatureProduct,
    ProductMeasure,
    median_positive_cost_scale,
    scaled_sinkhorn_epsilon,
)


def test_zero_curvature_distance_is_euclidean() -> None:
    left = torch.tensor([[0.0, 0.0], [1.0, 0.0]], dtype=torch.float32)
    right = torch.tensor([[3.0, 4.0], [1.0, 2.0]], dtype=torch.float32)
    geometry = ConstantCurvatureProduct(curvature=0.0)

    distance = geometry.point_distance(left, right)

    assert torch.allclose(distance, torch.linalg.vector_norm(left - right, dim=-1))


def test_small_positive_curvature_converges_to_euclidean_distance() -> None:
    generator = torch.Generator().manual_seed(7)
    left = torch.randn((5, 3), generator=generator) * 0.1
    right = torch.randn((5, 3), generator=generator) * 0.1
    euclidean = ConstantCurvatureProduct(curvature=0.0).point_distance(left, right)
    near_zero = ConstantCurvatureProduct(curvature=1e-8).point_distance(left, right)

    assert torch.allclose(near_zero, euclidean, atol=1e-6, rtol=1e-5)


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
    assert torch.allclose(near_zero, euclidean, atol=1e-7, rtol=1e-5)


def test_path_cost_matrix_includes_euclidean_side_features() -> None:
    points = torch.zeros((2, 3, 2), dtype=torch.float32)
    mass = torch.tensor([0.5, 0.5], dtype=torch.float32)
    left = ProductMeasure(points=points, mass=mass, side_features=torch.tensor([[0.0, 0.0], [1.0, 0.0]]))
    right = ProductMeasure(points=points, mass=mass, side_features=torch.tensor([[0.0, 1.0], [1.0, 1.0]]))
    geometry = ConstantCurvatureProduct(curvature=0.0, side_weight=2.0)

    cost = geometry.path_cost_matrix(left, right)

    assert torch.allclose(cost, torch.tensor([[2.0, 4.0], [4.0, 2.0]]))


def test_relative_sinkhorn_scale_uses_positive_median() -> None:
    scale = median_positive_cost_scale(
        (
            torch.tensor([[0.0, 2.0], [4.0, 0.0]]),
            torch.tensor([[0.0, 10.0], [8.0, 0.0]]),
        )
    )

    assert scale == 6.0
    assert scaled_sinkhorn_epsilon(scale, kappa=0.25) == 1.5
