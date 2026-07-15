from __future__ import annotations

import torch

from geometry_profile_research.batched_transport import (
    batched_marginal_residuals,
    batched_role_product_cost,
    batched_sinkhorn_plan,
    batched_sinkhorn_transport_objective,
    debiased_objective_from_self_terms,
)
from geometry_profile_research.constant_curvature import ProductMeasure, RoleProductGeometry
from geometry_profile_research.gromov_wasserstein import sinkhorn_transport_objective


def _measure(points: torch.Tensor, mass: torch.Tensor | None = None) -> ProductMeasure:
    count = points.shape[0]
    return ProductMeasure(
        points=points,
        mass=torch.full((count,), 1.0 / count, dtype=points.dtype) if mass is None else mass,
    )


def test_batched_role_cost_matches_scalar_with_variable_path_counts() -> None:
    generator = torch.Generator().manual_seed(123)
    query = _measure(torch.randn((3, 3, 4), generator=generator, dtype=torch.float64) * 0.08)
    gallery = (
        _measure(torch.randn((2, 3, 4), generator=generator, dtype=torch.float64) * 0.08),
        _measure(
            torch.randn((5, 3, 4), generator=generator, dtype=torch.float64) * 0.08,
            torch.tensor([0.1, 0.15, 0.2, 0.25, 0.3], dtype=torch.float64),
        ),
    )
    geometry = RoleProductGeometry(
        factor_curvatures=(0.7, 0.0, 0.0),
        factor_weights=(0.25, 1.5, 1.5),
        side_weight=0.0,
        unoriented=True,
    )

    batch = batched_role_product_cost(geometry, query, gallery)

    assert batch.left_sizes == (3, 3)
    assert batch.right_sizes == (2, 5)
    for index, measure in enumerate(gallery):
        expected = geometry.path_cost_matrix(query, measure)
        actual = batch.cost[index, : query.points.shape[0], : measure.points.shape[0]]
        torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
    assert torch.count_nonzero(batch.right_mass[0, 2:]) == 0


def test_batched_role_cost_matches_scalar_under_endpoint_reversal() -> None:
    left = ProductMeasure(
        points=torch.tensor([[[0.1], [0.2], [0.6]]], dtype=torch.float64),
        mass=torch.ones(1, dtype=torch.float64),
        side_features=torch.tensor([[1.0, 2.0]], dtype=torch.float64),
        reversed_side_features=torch.tensor([[2.0, 1.0]], dtype=torch.float64),
    )
    right = ProductMeasure(
        points=torch.tensor([[[0.1], [0.6], [0.2]]], dtype=torch.float64),
        mass=torch.ones(1, dtype=torch.float64),
        side_features=torch.tensor([[2.0, 1.0]], dtype=torch.float64),
        reversed_side_features=torch.tensor([[1.0, 2.0]], dtype=torch.float64),
    )
    geometry = RoleProductGeometry(
        factor_curvatures=(1.0, 0.0, 0.0),
        factor_weights=(0.25, 1.0, 1.0),
        side_weight=0.5,
        unoriented=True,
    )

    batch = batched_role_product_cost(geometry, left, right)

    torch.testing.assert_close(batch.cost[0, :1, :1], geometry.path_cost_matrix(left, right))
    torch.testing.assert_close(batch.cost[0, 0, 0], torch.zeros((), dtype=torch.float64), atol=1e-12, rtol=0.0)


def test_batched_sinkhorn_matches_scalar_regularized_objective() -> None:
    costs = torch.tensor(
        [
            [[0.0, 0.7, 0.0], [1.1, 0.2, 0.0], [0.0, 0.0, 0.0]],
            [[0.3, 1.2, 0.8], [0.4, 0.1, 1.3], [0.7, 0.9, 0.2]],
        ],
        dtype=torch.float64,
    )
    left_mass = torch.tensor([[0.4, 0.6, 0.0], [0.2, 0.3, 0.5]], dtype=torch.float64)
    right_mass = torch.tensor([[0.7, 0.3, 0.0], [0.1, 0.4, 0.5]], dtype=torch.float64)

    actual = batched_sinkhorn_transport_objective(
        costs,
        left_mass,
        right_mass,
        epsilon=0.25,
        iterations=300,
        projection_iterations=2048,
        marginal_tolerance=1e-9,
    )
    expected = torch.stack(
        (
            sinkhorn_transport_objective(
                costs[0, :2, :2],
                left_mass[0, :2],
                right_mass[0, :2],
                epsilon=0.25,
                iterations=300,
                projection_iterations=2048,
            ),
            sinkhorn_transport_objective(
                costs[1],
                left_mass[1],
                right_mass[1],
                epsilon=0.25,
                iterations=300,
                projection_iterations=2048,
            ),
        )
    )

    torch.testing.assert_close(actual, expected, atol=1e-9, rtol=1e-9)
    assert actual.dtype == torch.float64


def test_batched_sinkhorn_has_strict_individual_marginal_residuals() -> None:
    generator = torch.Generator().manual_seed(77)
    cost = torch.rand((3, 4, 5), generator=generator, dtype=torch.float64)
    left = torch.tensor(
        [[0.1, 0.2, 0.7, 0.0], [0.25, 0.25, 0.25, 0.25], [1.0, 0.0, 0.0, 0.0]],
        dtype=torch.float64,
    )
    right = torch.tensor(
        [[0.2, 0.3, 0.5, 0.0, 0.0], [0.1, 0.2, 0.3, 0.15, 0.25], [0.4, 0.6, 0.0, 0.0, 0.0]],
        dtype=torch.float64,
    )

    plan = batched_sinkhorn_plan(
        cost,
        left,
        right,
        epsilon=0.15,
        iterations=300,
        projection_iterations=2048,
        marginal_tolerance=1e-10,
    )

    residuals = batched_marginal_residuals(plan, left, right)
    assert bool((residuals <= 1e-10).all())
    assert torch.count_nonzero(plan[0, 3]) == 0
    assert torch.count_nonzero(plan[0, :, 3:]) == 0


def test_precomputed_self_debiasing_broadcasts_query_term() -> None:
    cross = torch.tensor([1.2, 1.5, 0.8], dtype=torch.float64)
    query_self = torch.tensor(0.2, dtype=torch.float64)
    gallery_self = torch.tensor([0.4, 0.8, 0.1], dtype=torch.float64)

    result = debiased_objective_from_self_terms(cross, query_self, gallery_self)

    torch.testing.assert_close(result, cross - 0.5 * query_self - 0.5 * gallery_self)


def test_batched_objective_preserves_finite_cost_gradient() -> None:
    cost = torch.tensor(
        [[[0.0, 0.8], [1.2, 0.0]], [[0.2, 1.0], [0.7, 0.3]]],
        dtype=torch.float64,
        requires_grad=True,
    )
    mass = torch.full((2, 2), 0.5, dtype=torch.float64)

    objective = batched_sinkhorn_transport_objective(
        cost,
        mass,
        mass,
        epsilon=0.3,
        iterations=200,
        projection_iterations=512,
    ).sum()
    objective.backward()

    assert cost.grad is not None
    assert torch.isfinite(cost.grad).all()
    assert float(torch.linalg.vector_norm(cost.grad)) > 0.0
