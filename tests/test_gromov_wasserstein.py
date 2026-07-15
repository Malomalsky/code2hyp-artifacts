from __future__ import annotations

import unittest

import numpy as np
import ot
import torch

from geometry_profile_research.gromov_wasserstein import (
    MetricMeasureSpace,
    entropic_plan_kl,
    entropic_transport_objective,
    entropic_fused_gromov_wasserstein,
    entropic_gromov_wasserstein,
    fused_gromov_wasserstein_objective,
    gromov_wasserstein_objective,
    metric_measure_space_from_raw_ast_paths,
    permutation_coupling,
    sinkhorn_plan,
    sinkhorn_plan_diagnostics,
    sinkhorn_divergence,
    sinkhorn_regularized_cost,
    sinkhorn_transport_cost,
    sinkhorn_transport_objective,
    uniform_coupling,
)
from geometry_profile_research.raw_ast import RawAstTree


def _toy_tree() -> RawAstTree:
    return RawAstTree.from_edges(
        root_id=0,
        edges=[
            (0, 1),
            (1, 2),
            (2, 3),
            (2, 4),
            (4, 5),
            (2, 6),
            (6, 7),
        ],
        labels={
            0: "CompilationUnit",
            1: "MethodDeclaration",
            2: "IfStatement",
            3: "Condition",
            4: "ThenBlock",
            5: "ReturnA",
            6: "ElseBlock",
            7: "ReturnB",
        },
    )


class GromovWassersteinTests(unittest.TestCase):
    def test_metric_measure_space_from_raw_ast_paths_uses_internal_relation_matrix(self) -> None:
        tree = _toy_tree()
        paths = (
            tree.path_between(5, 7),
            tree.path_between(5, 3),
            tree.path_between(3, 7),
        )

        space = metric_measure_space_from_raw_ast_paths(tree, paths, relation="unoriented_endpoint")

        self.assertIsInstance(space, MetricMeasureSpace)
        self.assertEqual(tuple(space.distance.shape), (3, 3))
        self.assertTrue(torch.allclose(space.mass, torch.full((3,), 1.0 / 3.0)))
        self.assertEqual(float(space.distance[0, 0]), 0.0)
        self.assertEqual(float(space.distance[0, 1]), 3.0)
        self.assertEqual(float(space.distance[1, 2]), 4.0)

    def test_lca_anchored_product_relation_uses_lca_position_and_unordered_endpoints(self) -> None:
        tree = RawAstTree.from_edges(
            root_id=0,
            edges=[
                (0, 1),
                (1, 2),
                (1, 3),
                (0, 4),
                (4, 5),
                (4, 6),
            ],
            labels={
                0: "Root",
                1: "LeftBranch",
                2: "LeftLeafA",
                3: "LeftLeafB",
                4: "RightBranch",
                5: "RightLeafA",
                6: "RightLeafB",
            },
        )
        left_path = tree.path_between(2, 3)
        right_path = tree.path_between(5, 6)
        reversed_right_path = right_path.reversed()

        space = metric_measure_space_from_raw_ast_paths(
            tree,
            (left_path, right_path, reversed_right_path),
            relation="lca_anchored_product",
        )

        self.assertAlmostEqual(float(space.distance[0, 1]), 6.0, places=6)
        self.assertAlmostEqual(float(space.distance[0, 2]), 6.0, places=6)
        self.assertAlmostEqual(float(space.distance[1, 2]), 0.0, places=6)

    def test_gromov_wasserstein_objective_is_zero_for_permuted_isomorphic_spaces(self) -> None:
        left = MetricMeasureSpace(
            distance=torch.tensor(
                [
                    [0.0, 1.0, 2.0],
                    [1.0, 0.0, 1.0],
                    [2.0, 1.0, 0.0],
                ]
            )
        )
        permutation = torch.tensor([2, 0, 1])
        right = MetricMeasureSpace(distance=left.distance[permutation][:, permutation])
        inverse_permutation = torch.argsort(permutation)
        coupling = permutation_coupling(left.mass, right.mass, inverse_permutation)

        loss = gromov_wasserstein_objective(left, right, coupling)

        self.assertAlmostEqual(float(loss), 0.0, places=7)

    def test_fused_gromov_wasserstein_objective_adds_feature_cost_to_structural_term(self) -> None:
        left = MetricMeasureSpace(distance=torch.tensor([[0.0, 2.0], [2.0, 0.0]]))
        right = MetricMeasureSpace(distance=torch.tensor([[0.0, 2.0], [2.0, 0.0]]))
        coupling = permutation_coupling(left.mass, right.mass, torch.arange(2))
        feature_cost = torch.tensor([[0.25, 1.25], [1.25, 0.25]])

        gw_loss = gromov_wasserstein_objective(left, right, coupling)
        fgw_loss = fused_gromov_wasserstein_objective(left, right, feature_cost, coupling, alpha=0.5)

        self.assertAlmostEqual(float(gw_loss), 0.0, places=7)
        self.assertGreater(float(fgw_loss), 0.0)
        self.assertAlmostEqual(float(fgw_loss), 0.5 * float((feature_cost * coupling).sum()), places=7)

    def test_sinkhorn_plan_matches_requested_marginals_and_reports_residuals(self) -> None:
        cost = torch.tensor([[0.0, 2.0], [1.0, 0.5], [2.0, 0.0]])
        left_mass = torch.tensor([0.2, 0.3, 0.5])
        right_mass = torch.tensor([0.4, 0.6])

        plan = sinkhorn_plan(cost, left_mass, right_mass, epsilon=0.1, iterations=200)
        diagnostics = sinkhorn_plan_diagnostics(plan, left_mass, right_mass)

        self.assertTrue(torch.allclose(plan.sum(dim=1), left_mass.to(dtype=plan.dtype), atol=1e-4))
        self.assertTrue(torch.allclose(plan.sum(dim=0), right_mass.to(dtype=plan.dtype), atol=1e-4))
        self.assertLess(diagnostics["max_row_residual"], 1e-4)
        self.assertLess(diagnostics["max_column_residual"], 1e-4)
        self.assertGreater(diagnostics["plan_entropy"], 0.0)

    def test_regularized_sinkhorn_cost_matches_closed_form_two_point_solution(self) -> None:
        epsilon = 0.7
        off_diagonal_cost = 1.0
        cost = torch.tensor(
            [[0.0, off_diagonal_cost], [off_diagonal_cost, 0.0]],
            dtype=torch.float64,
        )
        mass = torch.tensor([0.5, 0.5], dtype=torch.float64)
        diagonal_mass = 0.5 * torch.sigmoid(torch.tensor(off_diagonal_cost / epsilon, dtype=torch.float64))
        off_diagonal_mass = 0.5 - diagonal_mass
        analytic_plan = torch.stack(
            (
                torch.stack((diagonal_mass, off_diagonal_mass)),
                torch.stack((off_diagonal_mass, diagonal_mass)),
            )
        )
        reference = torch.full((2, 2), 0.25, dtype=torch.float64)
        expected = torch.sum(analytic_plan * cost) + epsilon * torch.sum(
            analytic_plan * torch.log(analytic_plan / reference)
        )

        actual = sinkhorn_regularized_cost(
            cost,
            mass,
            mass,
            epsilon=epsilon,
            iterations=300,
            projection_iterations=300,
        )
        transport_only = sinkhorn_transport_cost(
            cost,
            mass,
            mass,
            epsilon=epsilon,
            iterations=300,
            projection_iterations=300,
        )

        torch.testing.assert_close(actual, expected, atol=1e-10, rtol=1e-10)
        self.assertGreater(float(actual), float(transport_only))

    def test_named_entropic_objective_matches_plan_level_definition(self) -> None:
        cost = torch.tensor([[0.0, 2.0], [1.0, 0.0]], dtype=torch.float64)
        mass = torch.tensor([0.5, 0.5], dtype=torch.float64)
        plan = sinkhorn_plan(cost, mass, mass, epsilon=0.2, iterations=200)

        direct = entropic_transport_objective(plan, cost, mass, mass, epsilon=0.2)
        solved = sinkhorn_transport_objective(cost, mass, mass, epsilon=0.2, iterations=200)

        torch.testing.assert_close(solved, direct, atol=1e-10, rtol=1e-10)
        assert float(entropic_plan_kl(plan, mass, mass)) > 0.0

    def test_sinkhorn_plan_and_full_objective_match_independent_pot_solver(self) -> None:
        cost = torch.tensor(
            [[0.0, 0.7, 1.5], [0.4, 0.2, 0.9], [1.1, 0.6, 0.1]],
            dtype=torch.float64,
        )
        left = torch.tensor([0.2, 0.3, 0.5], dtype=torch.float64)
        right = torch.tensor([0.4, 0.35, 0.25], dtype=torch.float64)
        epsilon = 0.23

        actual_plan = sinkhorn_plan(
            cost,
            left,
            right,
            epsilon=epsilon,
            iterations=1000,
            projection_iterations=4096,
            marginal_tolerance=1e-10,
        )
        pot_plan = ot.sinkhorn(
            left.numpy(),
            right.numpy(),
            cost.numpy(),
            reg=epsilon,
            method="sinkhorn_log",
            numItermax=20_000,
            stopThr=1e-13,
        )
        reference_plan = torch.from_numpy(np.asarray(pot_plan))
        actual_objective = sinkhorn_transport_objective(
            cost,
            left,
            right,
            epsilon=epsilon,
            iterations=1000,
            projection_iterations=4096,
        )
        reference_objective = entropic_transport_objective(
            reference_plan,
            cost,
            left,
            right,
            epsilon=epsilon,
        )

        torch.testing.assert_close(actual_plan, reference_plan, atol=1e-8, rtol=1e-8)
        torch.testing.assert_close(actual_objective, reference_objective, atol=1e-9, rtol=1e-9)

    def test_sinkhorn_plan_supports_strict_marginal_tolerance(self) -> None:
        cost = torch.tensor([[0.0, 2.0], [1.0, 0.5], [2.0, 0.0]], dtype=torch.float64)
        left_mass = torch.tensor([0.2, 0.3, 0.5], dtype=torch.float64)
        right_mass = torch.tensor([0.4, 0.6], dtype=torch.float64)

        plan = sinkhorn_plan(
            cost,
            left_mass,
            right_mass,
            epsilon=0.1,
            iterations=300,
            projection_iterations=4096,
            marginal_tolerance=1e-7,
        )
        diagnostics = sinkhorn_plan_diagnostics(plan, left_mass, right_mass, tolerance=1e-7)

        self.assertLessEqual(diagnostics["max_row_residual"], 1e-7)
        self.assertLessEqual(diagnostics["max_column_residual"], 1e-7)

    def test_sinkhorn_divergence_preserves_float64_and_is_zero_on_identity(self) -> None:
        cost = torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=torch.float64)
        mass = torch.tensor([0.25, 0.75], dtype=torch.float64)

        divergence = sinkhorn_divergence(
            cost,
            mass,
            mass,
            left_self_cost=cost,
            right_self_cost=cost,
            epsilon=0.2,
        )

        assert divergence.dtype == torch.float64
        torch.testing.assert_close(divergence, torch.zeros((), dtype=torch.float64), atol=1e-12, rtol=0.0)

    def test_regularized_sinkhorn_cost_has_finite_ground_cost_gradient(self) -> None:
        cost = torch.tensor([[0.0, 0.8], [1.2, 0.0]], dtype=torch.float64, requires_grad=True)
        left = torch.tensor([0.4, 0.6], dtype=torch.float64)
        right = torch.tensor([0.7, 0.3], dtype=torch.float64)

        value = sinkhorn_regularized_cost(cost, left, right, epsilon=0.3, iterations=200)
        value.backward()

        assert cost.grad is not None
        assert torch.isfinite(cost.grad).all()
        assert float(torch.linalg.vector_norm(cost.grad)) > 0.0

    def test_entropic_gromov_wasserstein_keeps_zero_cost_for_known_isomorphism(self) -> None:
        left = MetricMeasureSpace(
            distance=torch.tensor(
                [
                    [0.0, 1.0, 3.0],
                    [1.0, 0.0, 2.0],
                    [3.0, 2.0, 0.0],
                ]
            )
        )
        permutation = torch.tensor([2, 0, 1])
        right = MetricMeasureSpace(distance=left.distance[permutation][:, permutation])
        initial = permutation_coupling(left.mass, right.mass, torch.argsort(permutation))

        result = entropic_gromov_wasserstein(left, right, epsilon=1e-3, iterations=4, sinkhorn_iterations=200, initial_coupling=initial)

        self.assertLess(float(result.objective), 1e-5)
        self.assertEqual(len(result.objective_history), 5)
        self.assertLess(result.max_marginal_residual, 1e-4)

    def test_entropic_fused_gromov_wasserstein_uses_feature_cost_and_self_divergence(self) -> None:
        left = MetricMeasureSpace(distance=torch.tensor([[0.0, 2.0], [2.0, 0.0]]))
        right = MetricMeasureSpace(distance=torch.tensor([[0.0, 2.0], [2.0, 0.0]]))
        feature_cost = torch.tensor([[0.0, 1.5], [1.5, 0.0]])

        result = entropic_fused_gromov_wasserstein(
            left,
            right,
            feature_cost,
            alpha=0.5,
            epsilon=0.05,
            iterations=3,
            sinkhorn_iterations=100,
        )
        divergence = sinkhorn_divergence(feature_cost, left.mass, right.mass, epsilon=0.05, iterations=100)

        self.assertLess(float(result.objective), 1e-3)
        self.assertLess(float(divergence), 1e-3)
        self.assertLess(result.max_marginal_residual, 1e-4)


if __name__ == "__main__":
    unittest.main()
