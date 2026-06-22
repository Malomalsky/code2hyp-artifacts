from __future__ import annotations

import unittest

import torch

from geometry_profile_research.code2hyp_torch import (
    Code2HypBatch,
    Code2HypTorchConfig,
    Code2HypTorchModel,
    ast_sequence_edit_distance,
    ast_sequence_jaccard_distance,
    ast_sequence_lca_depth,
    ast_sequence_tree_distance,
    ast_path_midpoint_branch_masks,
    batch_structural_distance_regularizer,
    batch_structural_neighbor_distribution_regularizer,
    batch_structural_neighbor_overlap_at_k,
    batch_structural_normalized_stress,
    batch_structural_rank_regularizer,
    batch_structural_spearman_correlation,
    Code2HypTorchOutput,
    batch_poincare_frechet_diagnostics,
    batch_poincare_radius_utilization,
    path_node_attention_monotonicity_loss,
    path_attention_soft_tree_distances,
    path_attention_tree_distance_loss,
    path_dual_attention_separation_loss,
    batch_structural_multi_metric_distance_regularizer,
    structural_distance_loss,
    structural_normalized_stress,
    structural_rank_loss,
    structural_spearman_correlation,
    tree_context_features,
    torch_expmap,
    torch_expmap0,
    torch_lorentz_distance,
    torch_lorentz_expmap0,
    torch_lorentz_logmap0,
    torch_lorentz_weighted_centroid,
    torch_logmap,
    torch_logmap0,
    torch_poincare_frechet_objective,
    torch_poincare_frechet_mean,
    torch_poincare_frechet_residual,
    torch_poincare_distance,
    torch_poincare_weighted_midpoint,
    batch_structural_neighbor_exact_overlap_at_k,
    batch_structural_distance_level_summary,
    _poincare_product_distance,
    _structural_embedding_distance,
)


def _toy_batch() -> Code2HypBatch:
    return Code2HypBatch(
        start_tokens=torch.tensor([[1, 3, 1], [2, 4, 0]], dtype=torch.long),
        end_tokens=torch.tensor([[2, 4, 5], [3, 1, 0]], dtype=torch.long),
        ast_paths=torch.tensor(
            [
                [[1, 2, 3, 0], [2, 4, 0, 0], [3, 0, 0, 0]],
                [[1, 5, 6, 0], [6, 7, 0, 0], [0, 0, 0, 0]],
            ],
            dtype=torch.long,
        ),
        ast_path_mask=torch.tensor(
            [
                [[True, True, True, False], [True, True, False, False], [True, False, False, False]],
                [[True, True, True, False], [True, True, False, False], [False, False, False, False]],
            ],
            dtype=torch.bool,
        ),
        context_mask=torch.tensor([[True, True, True], [True, True, False]], dtype=torch.bool),
    )


class Code2HypTorchGeometryTests(unittest.TestCase):
    def test_torch_exp_log_maps_are_inverse_near_origin(self) -> None:
        tangent = torch.tensor([[0.05, -0.02, 0.03], [0.01, 0.04, -0.03]], dtype=torch.float64)

        point = torch_expmap0(tangent, curvature=torch.tensor(1.4, dtype=torch.float64))
        recovered = torch_logmap0(point, curvature=torch.tensor(1.4, dtype=torch.float64))

        torch.testing.assert_close(recovered, tangent, atol=1e-8, rtol=1e-8)

    def test_torch_exp_log_maps_are_inverse_away_from_origin(self) -> None:
        curvature = torch.tensor(1.2, dtype=torch.float64)
        base = torch_expmap0(torch.tensor([[0.08, -0.04]], dtype=torch.float64), curvature=curvature)
        tangent = torch.tensor([[0.025, 0.015]], dtype=torch.float64)

        point = torch_expmap(base, tangent, curvature=curvature)
        recovered = torch_logmap(base, point, curvature=curvature)

        torch.testing.assert_close(recovered, tangent, atol=1e-7, rtol=1e-7)

    def test_torch_exp_log_maps_are_inverse_near_boundary(self) -> None:
        curvature = torch.tensor(1.0, dtype=torch.float64)
        base = torch.tensor([[0.82, 0.00]], dtype=torch.float64)
        tangent = torch.tensor([[0.002, 0.004]], dtype=torch.float64)

        point = torch_expmap(base, tangent, curvature=curvature)
        recovered = torch_logmap(base, point, curvature=curvature)

        self.assertLess(float(torch.linalg.vector_norm(point, dim=-1).max()), 1.0)
        torch.testing.assert_close(recovered, tangent, atol=1e-6, rtol=1e-6)

    def test_torch_poincare_distance_is_symmetric_and_differentiable(self) -> None:
        left = torch.tensor([[0.10, 0.20, 0.05]], dtype=torch.float64, requires_grad=True)
        right = torch.tensor([[0.20, -0.05, 0.03]], dtype=torch.float64, requires_grad=True)

        forward = torch_poincare_distance(left, right, curvature=torch.tensor(0.8, dtype=torch.float64))
        backward = torch_poincare_distance(right, left, curvature=torch.tensor(0.8, dtype=torch.float64))
        loss = forward.sum()
        loss.backward()

        torch.testing.assert_close(forward, backward, atol=1e-10, rtol=1e-10)
        self.assertTrue(torch.isfinite(left.grad).all())
        self.assertTrue(torch.isfinite(right.grad).all())

    def test_torch_poincare_distance_is_zero_on_identity_for_multiple_curvatures(self) -> None:
        points = torch.tensor(
            [[0.00, 0.00, 0.00], [0.10, -0.05, 0.03], [-0.20, 0.04, 0.02]],
            dtype=torch.float64,
        )

        for curvature_value in (1.0, 0.01, 0.0001):
            with self.subTest(curvature=curvature_value):
                curvature = torch.tensor(curvature_value, dtype=torch.float64)
                identity = torch_poincare_distance(points, points, curvature=curvature)

                torch.testing.assert_close(identity, torch.zeros_like(identity), atol=1e-12, rtol=0.0)

    def test_torch_poincare_distance_satisfies_sampled_triangle_inequality(self) -> None:
        curvature = torch.tensor(0.9, dtype=torch.float64)
        points = torch.tensor(
            [
                [0.00, 0.00],
                [0.12, -0.04],
                [-0.08, 0.18],
                [0.20, 0.06],
                [-0.15, -0.10],
            ],
            dtype=torch.float64,
        )

        for left_index in range(points.shape[0]):
            for mid_index in range(points.shape[0]):
                for right_index in range(points.shape[0]):
                    left = points[left_index : left_index + 1]
                    mid = points[mid_index : mid_index + 1]
                    right = points[right_index : right_index + 1]
                    direct = torch_poincare_distance(left, right, curvature=curvature)
                    via_mid = torch_poincare_distance(left, mid, curvature=curvature) + torch_poincare_distance(
                        mid,
                        right,
                        curvature=curvature,
                    )

                    self.assertLessEqual(float(direct), float(via_mid) + 1e-10)

    def test_poincare_and_lorentz_origin_charts_have_matching_distances(self) -> None:
        curvature = torch.tensor(1.3, dtype=torch.float64)
        left_tangent = torch.tensor([[0.08, -0.03, 0.02], [0.04, 0.05, -0.01]], dtype=torch.float64)
        right_tangent = torch.tensor([[-0.02, 0.06, 0.04], [0.03, -0.02, 0.07]], dtype=torch.float64)

        left_poincare = torch_expmap0(left_tangent, curvature=curvature)
        right_poincare = torch_expmap0(right_tangent, curvature=curvature)
        # The Poincare implementation uses Euclidean tangent coordinates where
        # the Riemannian norm at the origin is 2 * ||v||. The Lorentz chart uses
        # tangent vectors directly in Riemannian norm, so the factor 2 aligns
        # the two coordinate conventions.
        left_lorentz = torch_lorentz_expmap0(2.0 * left_tangent, curvature=curvature)
        right_lorentz = torch_lorentz_expmap0(2.0 * right_tangent, curvature=curvature)

        poincare_distance = torch_poincare_distance(left_poincare, right_poincare, curvature=curvature)
        lorentz_distance = torch_lorentz_distance(left_lorentz, right_lorentz, curvature=curvature)

        torch.testing.assert_close(poincare_distance, lorentz_distance, atol=1e-8, rtol=1e-8)

    def test_poincare_distance_gradient_matches_finite_difference(self) -> None:
        curvature = torch.tensor(0.8, dtype=torch.float64)
        left = torch.tensor([[0.10, 0.04]], dtype=torch.float64, requires_grad=True)
        right = torch.tensor([[0.21, -0.03]], dtype=torch.float64)
        distance = torch_poincare_distance(left, right, curvature=curvature).sum()
        distance.backward()

        step = 1e-6
        left_plus = left.detach().clone()
        left_minus = left.detach().clone()
        left_plus[0, 0] += step
        left_minus[0, 0] -= step
        finite_difference = (
            torch_poincare_distance(left_plus, right, curvature=curvature)
            - torch_poincare_distance(left_minus, right, curvature=curvature)
        ) / (2.0 * step)

        torch.testing.assert_close(left.grad[0, 0], finite_difference.squeeze(), atol=1e-5, rtol=1e-5)

    def test_poincare_weighted_midpoint_is_not_tangent_space_mean(self) -> None:
        tangents = torch.tensor(
            [[[0.30, 0.05], [0.05, 0.25], [-0.10, 0.15]]],
            dtype=torch.float64,
        )
        weights = torch.tensor([[0.55, 0.30, 0.15]], dtype=torch.float64)
        curvature = torch.tensor(1.7, dtype=torch.float64)
        points = torch_expmap0(tangents, curvature=curvature)

        midpoint = torch_poincare_weighted_midpoint(points, weights, curvature=curvature)
        midpoint_log = torch_logmap0(midpoint, curvature=curvature)
        tangent_mean = torch.sum(tangents * weights.unsqueeze(-1), dim=1)

        self.assertEqual(midpoint.shape, (1, 2))
        self.assertLess(float(torch.linalg.vector_norm(midpoint, dim=-1).max()), 1.0 / float(torch.sqrt(curvature)))
        self.assertGreater(float(torch.linalg.vector_norm(midpoint_log - tangent_mean)), 1e-4)

    def test_poincare_frechet_mean_refines_weighted_midpoint_objective(self) -> None:
        tangents = torch.tensor(
            [[[0.45, 0.05], [0.05, 0.35], [-0.25, 0.20], [0.12, -0.18]]],
            dtype=torch.float64,
        )
        weights = torch.tensor([[0.40, 0.25, 0.20, 0.15]], dtype=torch.float64)
        curvature = torch.tensor(1.1, dtype=torch.float64)
        points = torch_expmap0(tangents, curvature=curvature)
        midpoint = torch_poincare_weighted_midpoint(points, weights, curvature=curvature)

        frechet = torch_poincare_frechet_mean(
            points,
            weights,
            curvature=curvature,
            steps=6,
            step_size=0.5,
        )

        midpoint_objective = torch.sum(
            weights * torch_poincare_distance(midpoint.unsqueeze(1), points, curvature=curvature).square(),
            dim=1,
        )
        frechet_objective = torch.sum(
            weights * torch_poincare_distance(frechet.unsqueeze(1), points, curvature=curvature).square(),
            dim=1,
        )

        self.assertEqual(frechet.shape, (1, 2))
        self.assertLess(float(torch.linalg.vector_norm(frechet, dim=-1).max()), 1.0 / float(torch.sqrt(curvature)))
        self.assertLessEqual(float(frechet_objective), float(midpoint_objective) + 1e-8)

    def test_poincare_frechet_mean_reduces_karcher_residual(self) -> None:
        tangents = torch.tensor(
            [[[0.40, 0.06], [0.10, 0.32], [-0.20, 0.18], [0.06, -0.16]]],
            dtype=torch.float64,
        )
        weights = torch.tensor([[0.35, 0.30, 0.20, 0.15]], dtype=torch.float64)
        curvature = torch.tensor(0.9, dtype=torch.float64)
        points = torch_expmap0(tangents, curvature=curvature)
        midpoint = torch_poincare_weighted_midpoint(points, weights, curvature=curvature)
        frechet = torch_poincare_frechet_mean(points, weights, curvature=curvature, steps=8, step_size=0.5)

        midpoint_residual = torch_poincare_frechet_residual(points, weights, midpoint, curvature=curvature)
        frechet_residual = torch_poincare_frechet_residual(points, weights, frechet, curvature=curvature)
        frechet_objective = torch_poincare_frechet_objective(points, weights, frechet, curvature=curvature)

        self.assertLessEqual(float(frechet_residual), float(midpoint_residual) + 1e-8)
        self.assertGreaterEqual(float(frechet_objective), 0.0)

    def test_batch_poincare_diagnostics_report_frechet_and_radius_utilization(self) -> None:
        batch = Code2HypBatch(
            start_tokens=torch.tensor([[1, 1, 1]], dtype=torch.long),
            end_tokens=torch.tensor([[2, 2, 2]], dtype=torch.long),
            ast_paths=torch.tensor([[[1, 2], [1, 3], [1, 4]]], dtype=torch.long),
            ast_path_mask=torch.ones(1, 3, 2, dtype=torch.bool),
            context_mask=torch.ones(1, 3, dtype=torch.bool),
        )
        curvature = torch.tensor(1.0, dtype=torch.float64)
        points = torch_expmap0(
            torch.tensor([[[0.10, 0.00], [0.00, 0.12], [-0.08, 0.04]]], dtype=torch.float64),
            curvature=curvature,
        )
        attention = torch.tensor([[0.50, 0.30, 0.20]], dtype=torch.float64)
        mean = torch_poincare_frechet_mean(points, attention, curvature=curvature, steps=4, step_size=0.5)
        output = Code2HypTorchOutput(
            logits=torch.zeros(1, 1, dtype=torch.float64),
            representation=torch.zeros(1, 2, dtype=torch.float64),
            attention=attention,
            curvature=curvature,
            structural_points=mean,
            context_structural_points=points,
            structural_geometry="poincare",
        )

        frechet = batch_poincare_frechet_diagnostics(output)
        radius = batch_poincare_radius_utilization(output, batch)

        self.assertIsNotNone(frechet)
        self.assertIsNotNone(radius)
        assert frechet is not None
        assert radius is not None
        self.assertLess(float(frechet["residual_mean"]), 0.02)
        self.assertGreaterEqual(float(frechet["objective_mean"]), 0.0)
        self.assertGreater(float(radius["context_radius_ratio_mean"]), 0.0)
        self.assertLess(float(radius["context_radius_ratio_max"]), 1.0)
        self.assertEqual(float(radius["context_near_boundary_rate"]), 0.0)

    def test_lorentz_exp_log_maps_are_inverse_near_origin(self) -> None:
        tangent = torch.tensor([[0.05, -0.02, 0.03], [0.01, 0.04, -0.03]], dtype=torch.float64)
        curvature = torch.tensor(1.4, dtype=torch.float64)

        point = torch_lorentz_expmap0(tangent, curvature=curvature)
        recovered = torch_lorentz_logmap0(point, curvature=curvature)

        torch.testing.assert_close(recovered, tangent, atol=1e-8, rtol=1e-8)

    def test_lorentz_distance_is_symmetric_and_keeps_hyperboloid_norm(self) -> None:
        curvature = torch.tensor(1.7, dtype=torch.float64)
        left = torch_lorentz_expmap0(torch.tensor([[0.08, -0.03, 0.02]], dtype=torch.float64), curvature=curvature)
        right = torch_lorentz_expmap0(torch.tensor([[-0.02, 0.06, 0.04]], dtype=torch.float64), curvature=curvature)

        left_norm = -left[..., 0] * left[..., 0] + torch.sum(left[..., 1:] * left[..., 1:], dim=-1)
        right_norm = -right[..., 0] * right[..., 0] + torch.sum(right[..., 1:] * right[..., 1:], dim=-1)
        forward = torch_lorentz_distance(left, right, curvature=curvature)
        backward = torch_lorentz_distance(right, left, curvature=curvature)

        torch.testing.assert_close(left_norm, torch.full_like(left_norm, -1.0 / curvature), atol=1e-8, rtol=1e-8)
        torch.testing.assert_close(right_norm, torch.full_like(right_norm, -1.0 / curvature), atol=1e-8, rtol=1e-8)
        torch.testing.assert_close(forward, backward, atol=1e-10, rtol=1e-10)
        self.assertGreater(float(forward), 0.0)

    def test_lorentz_weighted_centroid_returns_valid_hyperboloid_point(self) -> None:
        tangents = torch.tensor(
            [[[0.20, 0.03], [0.05, 0.16], [-0.08, 0.09]]],
            dtype=torch.float64,
        )
        weights = torch.tensor([[0.50, 0.30, 0.20]], dtype=torch.float64)
        curvature = torch.tensor(1.3, dtype=torch.float64)
        points = torch_lorentz_expmap0(tangents, curvature=curvature)

        centroid = torch_lorentz_weighted_centroid(points, weights, curvature=curvature)
        centroid_norm = -centroid[..., 0] * centroid[..., 0] + torch.sum(
            centroid[..., 1:] * centroid[..., 1:],
            dim=-1,
        )

        self.assertEqual(centroid.shape, (1, 3))
        torch.testing.assert_close(
            centroid_norm,
            torch.full_like(centroid_norm, -1.0 / curvature),
            atol=1e-8,
            rtol=1e-8,
        )

    def test_structural_neighbor_overlap_rewards_ast_neighborhood_preservation(self) -> None:
        batch = Code2HypBatch(
            start_tokens=torch.tensor([[1, 1, 1]], dtype=torch.long),
            end_tokens=torch.tensor([[2, 2, 2]], dtype=torch.long),
            ast_paths=torch.tensor([[[1, 2, 3], [1, 2, 4], [8, 9, 10]]], dtype=torch.long),
            ast_path_mask=torch.ones(1, 3, 3, dtype=torch.bool),
            context_mask=torch.ones(1, 3, dtype=torch.bool),
        )
        good_output = Code2HypTorchOutput(
            logits=torch.zeros(1, 1),
            representation=torch.zeros(1, 2),
            attention=torch.full((1, 3), 1 / 3),
            curvature=torch.tensor(1.0),
            context_structural_embeddings=torch.tensor([[[0.0], [0.1], [5.0]]]),
        )
        bad_output = Code2HypTorchOutput(
            logits=torch.zeros(1, 1),
            representation=torch.zeros(1, 2),
            attention=torch.full((1, 3), 1 / 3),
            curvature=torch.tensor(1.0),
            context_structural_embeddings=torch.tensor([[[0.0], [5.0], [0.1]]]),
        )

        good_overlap = batch_structural_neighbor_overlap_at_k(good_output, batch, k=1)
        bad_overlap = batch_structural_neighbor_overlap_at_k(bad_output, batch, k=1)

        self.assertEqual(float(good_overlap), 1.0)
        self.assertLess(float(bad_overlap), float(good_overlap))

    def test_independent_ast_path_sequence_diagnostics_are_well_defined(self) -> None:
        left = torch.tensor([[1, 2, 3], [1, 2, 0]], dtype=torch.long)
        right = torch.tensor([[1, 2, 4], [8, 9, 0]], dtype=torch.long)
        left_mask = torch.tensor([[True, True, True], [True, True, False]])
        right_mask = torch.tensor([[True, True, True], [True, True, False]])

        edit = ast_sequence_edit_distance(left, left_mask, right, right_mask)
        jaccard = ast_sequence_jaccard_distance(left, left_mask, right, right_mask)

        torch.testing.assert_close(edit, torch.tensor([1.0, 2.0]))
        torch.testing.assert_close(jaccard, torch.tensor([2.0 / 3.0, 1.0]))

    def test_structural_neighbor_overlap_is_tie_tolerant(self) -> None:
        batch = Code2HypBatch(
            start_tokens=torch.tensor([[1, 1, 1]], dtype=torch.long),
            end_tokens=torch.tensor([[2, 2, 2]], dtype=torch.long),
            ast_paths=torch.tensor([[[1, 2], [1, 3], [1, 4]]], dtype=torch.long),
            ast_path_mask=torch.ones(1, 3, 2, dtype=torch.bool),
            context_mask=torch.ones(1, 3, dtype=torch.bool),
        )
        output = Code2HypTorchOutput(
            logits=torch.zeros(1, 1),
            representation=torch.zeros(1, 2),
            attention=torch.full((1, 3), 1 / 3),
            curvature=torch.tensor(1.0),
            context_structural_embeddings=torch.tensor([[[0.0], [10.0], [0.1]]]),
        )

        overlap = batch_structural_neighbor_overlap_at_k(output, batch, k=1)

        self.assertEqual(float(overlap), 1.0)

    def test_structural_neighbor_exact_overlap_does_not_expand_ties(self) -> None:
        batch = Code2HypBatch(
            start_tokens=torch.tensor([[1, 1, 1]], dtype=torch.long),
            end_tokens=torch.tensor([[2, 2, 2]], dtype=torch.long),
            ast_paths=torch.tensor([[[1, 2], [1, 3], [1, 4]]], dtype=torch.long),
            ast_path_mask=torch.ones(1, 3, 2, dtype=torch.bool),
            context_mask=torch.ones(1, 3, dtype=torch.bool),
        )
        output = Code2HypTorchOutput(
            logits=torch.zeros(1, 1),
            representation=torch.zeros(1, 2),
            attention=torch.full((1, 3), 1 / 3),
            curvature=torch.tensor(1.0),
            context_structural_embeddings=torch.tensor([[[0.0], [10.0], [0.1]]]),
        )

        tie_tolerant = batch_structural_neighbor_overlap_at_k(output, batch, k=1)
        exact = batch_structural_neighbor_exact_overlap_at_k(output, batch, k=1)

        self.assertEqual(float(tie_tolerant), 1.0)
        self.assertLess(float(exact), float(tie_tolerant))

    def test_structural_distance_level_summary_groups_by_target_distance(self) -> None:
        batch = Code2HypBatch(
            start_tokens=torch.tensor([[1, 1, 1]], dtype=torch.long),
            end_tokens=torch.tensor([[2, 2, 2]], dtype=torch.long),
            ast_paths=torch.tensor([[[1, 2, 3], [1, 2, 4], [8, 9, 10]]], dtype=torch.long),
            ast_path_mask=torch.ones(1, 3, 3, dtype=torch.bool),
            context_mask=torch.ones(1, 3, dtype=torch.bool),
        )
        output = Code2HypTorchOutput(
            logits=torch.zeros(1, 1),
            representation=torch.zeros(1, 2),
            attention=torch.full((1, 3), 1 / 3),
            curvature=torch.tensor(1.0),
            context_structural_embeddings=torch.tensor([[[0.0], [0.5], [2.0]]]),
        )

        summary = batch_structural_distance_level_summary(output, batch)

        self.assertEqual([item["target_distance"] for item in summary], [2.0, 6.0])
        self.assertEqual([item["pair_count"] for item in summary], [1, 2])
        self.assertTrue(all(item["model_distance_mean"] > 0.0 for item in summary))

    def test_structural_normalized_stress_is_zero_for_scaled_distances(self) -> None:
        ast_distances = torch.tensor([1.0, 2.0, 4.0, 6.0])
        embedding_distances = 0.5 * ast_distances

        stress = structural_normalized_stress(embedding_distances, ast_distances)

        self.assertAlmostEqual(float(stress), 0.0, places=6)

    def test_structural_normalized_stress_penalizes_metric_distortion(self) -> None:
        ast_distances = torch.tensor([1.0, 2.0, 4.0, 6.0])
        good_embedding_distances = torch.tensor([0.5, 1.0, 2.0, 3.0])
        bad_embedding_distances = torch.tensor([3.0, 0.5, 2.0, 1.0])

        good_stress = structural_normalized_stress(good_embedding_distances, ast_distances)
        bad_stress = structural_normalized_stress(bad_embedding_distances, ast_distances)

        self.assertLess(float(good_stress), float(bad_stress))
        self.assertGreater(float(bad_stress), 0.0)

    def test_batch_structural_normalized_stress_uses_model_geometry(self) -> None:
        batch = Code2HypBatch(
            start_tokens=torch.tensor([[1, 1, 1]], dtype=torch.long),
            end_tokens=torch.tensor([[2, 2, 2]], dtype=torch.long),
            ast_paths=torch.tensor([[[1, 2, 3], [1, 2, 4], [8, 9, 10]]], dtype=torch.long),
            ast_path_mask=torch.ones(1, 3, 3, dtype=torch.bool),
            context_mask=torch.ones(1, 3, dtype=torch.bool),
        )
        good_output = Code2HypTorchOutput(
            logits=torch.zeros(1, 1),
            representation=torch.zeros(1, 2),
            attention=torch.full((1, 3), 1 / 3),
            curvature=torch.tensor(1.0),
            context_structural_embeddings=torch.tensor([[[0.0], [0.5], [1.5]]]),
        )
        bad_output = Code2HypTorchOutput(
            logits=torch.zeros(1, 1),
            representation=torch.zeros(1, 2),
            attention=torch.full((1, 3), 1 / 3),
            curvature=torch.tensor(1.0),
            context_structural_embeddings=torch.tensor([[[0.0], [1.5], [0.5]]]),
        )

        good_stress = batch_structural_normalized_stress(good_output, batch)
        bad_stress = batch_structural_normalized_stress(bad_output, batch)

        self.assertLess(float(good_stress), float(bad_stress))

    def test_structural_neighbor_overlap_uses_poincare_distances(self) -> None:
        batch = Code2HypBatch(
            start_tokens=torch.tensor([[1, 1, 1]], dtype=torch.long),
            end_tokens=torch.tensor([[2, 2, 2]], dtype=torch.long),
            ast_paths=torch.tensor([[[1, 2, 3], [1, 2, 4], [8, 9, 10]]], dtype=torch.long),
            ast_path_mask=torch.ones(1, 3, 3, dtype=torch.bool),
            context_mask=torch.ones(1, 3, dtype=torch.bool),
        )
        output = Code2HypTorchOutput(
            logits=torch.zeros(1, 1),
            representation=torch.zeros(1, 2),
            attention=torch.full((1, 3), 1 / 3),
            curvature=torch.tensor(1.0),
            context_structural_embeddings=None,
            context_structural_points=torch.tensor([[[0.00, 0.00], [0.05, 0.00], [0.70, 0.00]]]),
            structural_geometry="poincare",
        )

        overlap = batch_structural_neighbor_overlap_at_k(output, batch, k=1)

        self.assertEqual(float(overlap), 1.0)

    def test_structural_neighbor_distribution_loss_rewards_local_ast_preservation(self) -> None:
        batch = Code2HypBatch(
            start_tokens=torch.tensor([[1, 1, 1, 1]], dtype=torch.long),
            end_tokens=torch.tensor([[2, 2, 2, 2]], dtype=torch.long),
            ast_paths=torch.tensor([[[1, 2, 3], [1, 2, 4], [1, 8, 9], [7, 8, 9]]], dtype=torch.long),
            ast_path_mask=torch.ones(1, 4, 3, dtype=torch.bool),
            context_mask=torch.ones(1, 4, dtype=torch.bool),
        )
        good_output = Code2HypTorchOutput(
            logits=torch.zeros(1, 1),
            representation=torch.zeros(1, 2),
            attention=torch.full((1, 4), 0.25),
            curvature=torch.tensor(1.0),
            context_structural_embeddings=torch.tensor([[[0.0], [0.1], [3.0], [6.0]]]),
        )
        bad_output = Code2HypTorchOutput(
            logits=torch.zeros(1, 1),
            representation=torch.zeros(1, 2),
            attention=torch.full((1, 4), 0.25),
            curvature=torch.tensor(1.0),
            context_structural_embeddings=torch.tensor([[[0.0], [6.0], [0.1], [3.0]]]),
        )

        good_loss = batch_structural_neighbor_distribution_regularizer(good_output, batch)
        bad_loss = batch_structural_neighbor_distribution_regularizer(bad_output, batch)

        self.assertTrue(torch.isfinite(good_loss))
        self.assertTrue(torch.isfinite(bad_loss))
        self.assertGreaterEqual(float(good_loss), 0.0)
        self.assertLess(float(good_loss), float(bad_loss))

    def test_structural_neighbor_distribution_uses_poincare_distances(self) -> None:
        batch = Code2HypBatch(
            start_tokens=torch.tensor([[1, 1, 1]], dtype=torch.long),
            end_tokens=torch.tensor([[2, 2, 2]], dtype=torch.long),
            ast_paths=torch.tensor([[[1, 2, 3], [1, 2, 4], [8, 9, 10]]], dtype=torch.long),
            ast_path_mask=torch.ones(1, 3, 3, dtype=torch.bool),
            context_mask=torch.ones(1, 3, dtype=torch.bool),
        )
        good_output = Code2HypTorchOutput(
            logits=torch.zeros(1, 1),
            representation=torch.zeros(1, 2),
            attention=torch.full((1, 3), 1 / 3),
            curvature=torch.tensor(1.0),
            context_structural_embeddings=None,
            context_structural_points=torch.tensor([[[0.00, 0.00], [0.05, 0.00], [0.70, 0.00]]]),
            structural_geometry="poincare",
        )
        bad_output = Code2HypTorchOutput(
            logits=torch.zeros(1, 1),
            representation=torch.zeros(1, 2),
            attention=torch.full((1, 3), 1 / 3),
            curvature=torch.tensor(1.0),
            context_structural_embeddings=None,
            context_structural_points=torch.tensor([[[0.00, 0.00], [0.70, 0.00], [0.05, 0.00]]]),
            structural_geometry="poincare",
        )

        good_loss = batch_structural_neighbor_distribution_regularizer(good_output, batch)
        bad_loss = batch_structural_neighbor_distribution_regularizer(bad_output, batch)

        self.assertLess(float(good_loss), float(bad_loss))


class Code2HypTorchModelTests(unittest.TestCase):
    def test_euclidean_and_product_variants_return_matching_logits_shape(self) -> None:
        config = Code2HypTorchConfig(
            token_vocab_size=16,
            ast_node_vocab_size=12,
            label_vocab_size=7,
            token_dim=4,
            structural_dim=5,
        )
        batch = _toy_batch()

        euclidean = Code2HypTorchModel(config, variant="euclidean")
        product = Code2HypTorchModel(config, variant="product")

        euclidean_output = euclidean(batch)
        product_output = product(batch)

        self.assertEqual(euclidean_output.logits.shape, (2, 7))
        self.assertEqual(product_output.logits.shape, (2, 7))
        self.assertEqual(euclidean_output.representation.shape, product_output.representation.shape)
        self.assertEqual(euclidean_output.context_structural_embeddings.shape, (2, 3, 5))
        self.assertEqual(product_output.context_structural_embeddings.shape, (2, 3, 5))
        torch.testing.assert_close(euclidean_output.attention.sum(dim=1), torch.ones(2), atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(product_output.attention.sum(dim=1), torch.ones(2), atol=1e-6, rtol=1e-6)

    def test_trainable_curvature_is_positive_and_receives_gradient(self) -> None:
        config = Code2HypTorchConfig(
            token_vocab_size=16,
            ast_node_vocab_size=12,
            label_vocab_size=7,
            token_dim=4,
            structural_dim=5,
            trainable_curvature=True,
        )
        model = Code2HypTorchModel(config, variant="product")
        labels = torch.tensor([1, 3], dtype=torch.long)

        output = model(_toy_batch())
        loss = torch.nn.functional.cross_entropy(output.logits, labels)
        loss.backward()

        self.assertGreater(float(output.curvature.detach()), 0.0)
        self.assertIsNotNone(model.raw_curvature.grad)
        self.assertTrue(torch.isfinite(model.raw_curvature.grad).all())

    def test_parameter_count_reports_geometry_overhead(self) -> None:
        config = Code2HypTorchConfig(
            token_vocab_size=64,
            ast_node_vocab_size=48,
            label_vocab_size=20,
            token_dim=16,
            structural_dim=16,
            trainable_curvature=True,
        )

        euclidean = Code2HypTorchModel(config, variant="euclidean")
        product = Code2HypTorchModel(config, variant="product")

        self.assertEqual(product.parameter_count() - euclidean.parameter_count(), 1)
        self.assertLess(product.relative_parameter_overhead(euclidean), 0.03)

    def test_gru_path_encoder_preserves_matched_capacity_geometry_overhead(self) -> None:
        config = Code2HypTorchConfig(
            token_vocab_size=64,
            ast_node_vocab_size=48,
            label_vocab_size=20,
            token_dim=16,
            structural_dim=16,
            trainable_curvature=True,
            path_encoder="gru",
        )
        batch = _toy_batch()

        euclidean = Code2HypTorchModel(config, variant="euclidean")
        product = Code2HypTorchModel(config, variant="product")
        euclidean_output = euclidean(batch)
        product_output = product(batch)

        self.assertEqual(euclidean_output.logits.shape, (2, 20))
        self.assertEqual(product_output.logits.shape, (2, 20))
        self.assertEqual(product.parameter_count() - euclidean.parameter_count(), 1)
        self.assertGreater(euclidean.parameter_count(), 64 * 16)

    def test_tanh_representation_transform_preserves_matched_capacity_geometry_overhead(self) -> None:
        config = Code2HypTorchConfig(
            token_vocab_size=64,
            ast_node_vocab_size=48,
            label_vocab_size=20,
            token_dim=16,
            structural_dim=16,
            trainable_curvature=True,
            path_encoder="gru",
            representation_transform="tanh",
        )
        batch = _toy_batch()

        euclidean = Code2HypTorchModel(config, variant="euclidean")
        product = Code2HypTorchModel(config, variant="product")

        self.assertEqual(euclidean(batch).logits.shape, (2, 20))
        self.assertEqual(product(batch).logits.shape, (2, 20))
        self.assertEqual(product.parameter_count() - euclidean.parameter_count(), 1)

    def test_hyperbolic_code2vec_variant_maps_full_contexts_to_poincare_ball(self) -> None:
        config = Code2HypTorchConfig(
            token_vocab_size=16,
            ast_node_vocab_size=12,
            label_vocab_size=7,
            token_dim=4,
            structural_dim=5,
            curvature=1.3,
        )
        model = Code2HypTorchModel(config, variant="hyperbolic")

        output = model(_toy_batch())

        self.assertEqual(output.logits.shape, (2, 7))
        self.assertEqual(output.representation.shape, (2, config.representation_dim))
        self.assertIsNotNone(output.structural_points)
        self.assertIsNotNone(output.context_structural_points)
        self.assertIsNotNone(output.context_structural_embeddings)
        self.assertEqual(output.context_structural_points.shape, (2, 3, config.representation_dim))
        radius = (1.0 / torch.sqrt(output.curvature)).detach()
        max_norm = torch.linalg.vector_norm(output.context_structural_points, dim=-1).max().detach()
        self.assertLess(float(max_norm), float(radius))
        torch.testing.assert_close(output.attention.sum(dim=1), torch.ones(2), atol=1e-6, rtol=1e-6)

    def test_hyperbolic_attention_variant_removes_poincare_midpoint_aggregation(self) -> None:
        config = Code2HypTorchConfig(
            token_vocab_size=16,
            ast_node_vocab_size=12,
            label_vocab_size=7,
            token_dim=4,
            structural_dim=5,
            curvature=1.3,
        )
        b4_model = Code2HypTorchModel(config, variant="hyperbolic")
        b7_model = Code2HypTorchModel(config, variant="hyperbolic_attention")

        output = b7_model(_toy_batch())

        self.assertEqual(output.logits.shape, (2, 7))
        self.assertEqual(output.representation.shape, (2, config.representation_dim))
        self.assertIsNone(output.structural_points)
        self.assertIsNotNone(output.context_structural_points)
        self.assertIsNotNone(output.context_structural_embeddings)
        self.assertEqual(output.context_structural_points.shape, (2, 3, config.representation_dim))
        self.assertEqual(b7_model.parameter_count(), b4_model.parameter_count())
        torch.testing.assert_close(output.attention.sum(dim=1), torch.ones(2), atol=1e-6, rtol=1e-6)

    def test_hyperbolic_frechet_variant_uses_intrinsic_iterative_aggregation(self) -> None:
        config = Code2HypTorchConfig(
            token_vocab_size=16,
            ast_node_vocab_size=12,
            label_vocab_size=7,
            token_dim=4,
            structural_dim=5,
            curvature=1.3,
            frechet_steps=3,
        )
        b4_model = Code2HypTorchModel(config, variant="hyperbolic")
        b8_model = Code2HypTorchModel(config, variant="hyperbolic_frechet")

        output = b8_model(_toy_batch())

        self.assertEqual(output.logits.shape, (2, 7))
        self.assertEqual(output.representation.shape, (2, config.representation_dim))
        self.assertIsNotNone(output.structural_points)
        self.assertIsNotNone(output.context_structural_points)
        self.assertIsNotNone(output.context_structural_embeddings)
        self.assertEqual(output.context_structural_points.shape, (2, 3, config.representation_dim))
        self.assertEqual(b8_model.parameter_count(), b4_model.parameter_count())
        radius = (1.0 / torch.sqrt(output.curvature)).detach()
        max_norm = torch.linalg.vector_norm(output.structural_points, dim=-1).max().detach()
        self.assertLess(float(max_norm), float(radius))
        torch.testing.assert_close(output.attention.sum(dim=1), torch.ones(2), atol=1e-6, rtol=1e-6)

    def test_lorentz_code2vec_variant_uses_hyperboloid_context_space(self) -> None:
        config = Code2HypTorchConfig(
            token_vocab_size=16,
            ast_node_vocab_size=12,
            label_vocab_size=7,
            token_dim=4,
            structural_dim=5,
            curvature=1.3,
        )
        b4_model = Code2HypTorchModel(config, variant="hyperbolic")
        b9_model = Code2HypTorchModel(config, variant="lorentz")

        output = b9_model(_toy_batch())
        structural_loss = batch_structural_distance_regularizer(output, _toy_batch())

        self.assertEqual(output.logits.shape, (2, 7))
        self.assertEqual(output.representation.shape, (2, config.representation_dim))
        self.assertIsNotNone(output.structural_points)
        self.assertIsNotNone(output.context_structural_points)
        self.assertIsNotNone(output.context_structural_embeddings)
        self.assertEqual(output.structural_points.shape, (2, config.representation_dim + 1))
        self.assertEqual(output.context_structural_points.shape, (2, 3, config.representation_dim + 1))
        self.assertEqual(output.structural_geometry, "lorentz")
        self.assertEqual(b9_model.parameter_count(), b4_model.parameter_count())
        self.assertTrue(torch.isfinite(structural_loss))
        torch.testing.assert_close(output.attention.sum(dim=1), torch.ones(2), atol=1e-6, rtol=1e-6)

    def test_factorized_product_variant_uses_mixed_curvature_context_metric(self) -> None:
        config = Code2HypTorchConfig(
            token_vocab_size=16,
            ast_node_vocab_size=12,
            label_vocab_size=7,
            token_dim=4,
            structural_dim=5,
            curvature=1.3,
        )
        b4_model = Code2HypTorchModel(config, variant="hyperbolic")
        b10_model = Code2HypTorchModel(config, variant="factorized_product")

        output = b10_model(_toy_batch())
        structural_loss = batch_structural_distance_regularizer(output, _toy_batch())

        self.assertEqual(output.logits.shape, (2, 7))
        self.assertEqual(output.representation.shape, (2, config.representation_dim))
        self.assertIsNotNone(output.structural_points)
        self.assertIsNotNone(output.context_structural_points)
        self.assertIsNotNone(output.context_structural_embeddings)
        self.assertEqual(output.structural_points.shape, (2, config.structural_dim))
        self.assertEqual(output.context_structural_points.shape, (2, 3, config.structural_dim))
        self.assertEqual(output.context_structural_embeddings.shape, (2, 3, config.structural_dim))
        self.assertEqual(output.structural_geometry, "poincare")
        self.assertEqual(b10_model.parameter_count(), b4_model.parameter_count())
        radius = (1.0 / torch.sqrt(output.curvature)).detach()
        max_norm = torch.linalg.vector_norm(output.context_structural_points, dim=-1).max().detach()
        self.assertLess(float(max_norm), float(radius))
        self.assertTrue(torch.isfinite(structural_loss))
        torch.testing.assert_close(output.attention.sum(dim=1), torch.ones(2), atol=1e-6, rtol=1e-6)

    def test_lorentz_product_variant_uses_hyperboloid_ast_path_channel(self) -> None:
        config = Code2HypTorchConfig(
            token_vocab_size=16,
            ast_node_vocab_size=12,
            label_vocab_size=7,
            token_dim=4,
            structural_dim=5,
            curvature=1.3,
        )
        b4_model = Code2HypTorchModel(config, variant="hyperbolic")
        b15_model = Code2HypTorchModel(config, variant="lorentz_product")

        output = b15_model(_toy_batch())
        structural_loss = batch_structural_distance_regularizer(output, _toy_batch())

        self.assertEqual(output.logits.shape, (2, 7))
        self.assertEqual(output.representation.shape, (2, config.representation_dim))
        self.assertIsNotNone(output.structural_points)
        self.assertIsNotNone(output.context_structural_points)
        self.assertIsNotNone(output.context_structural_embeddings)
        self.assertEqual(output.structural_points.shape, (2, config.structural_dim + 1))
        self.assertEqual(output.context_structural_points.shape, (2, 3, config.structural_dim + 1))
        self.assertEqual(output.context_structural_embeddings.shape, (2, 3, config.structural_dim))
        self.assertEqual(output.structural_geometry, "lorentz")
        self.assertEqual(b15_model.parameter_count(), b4_model.parameter_count())
        self.assertTrue(torch.isfinite(structural_loss))
        torch.testing.assert_close(output.attention.sum(dim=1), torch.ones(2), atol=1e-6, rtol=1e-6)

    def test_factorized_product_learned_metric_variant_trains_product_distance_weights(self) -> None:
        config = Code2HypTorchConfig(
            token_vocab_size=16,
            ast_node_vocab_size=12,
            label_vocab_size=7,
            token_dim=4,
            structural_dim=5,
            curvature=1.3,
        )
        b4_model = Code2HypTorchModel(config, variant="hyperbolic")
        b12_model = Code2HypTorchModel(config, variant="factorized_product_learned_metric")

        output = b12_model(_toy_batch())
        metric_weights = b12_model.factorized_metric_weights()
        loss = output.logits.square().sum()
        loss.backward()

        self.assertEqual(output.logits.shape, (2, 7))
        self.assertEqual(output.representation.shape, (2, config.representation_dim))
        self.assertEqual(output.structural_geometry, "poincare")
        self.assertEqual(output.context_structural_points.shape, (2, 3, config.structural_dim))
        self.assertEqual(b12_model.parameter_count(), b4_model.parameter_count() + 2)
        self.assertEqual(metric_weights.shape, (2,))
        self.assertTrue(torch.all(metric_weights > 0.0))
        self.assertIsNotNone(b12_model.raw_factorized_metric_weights.grad)
        self.assertTrue(torch.isfinite(b12_model.raw_factorized_metric_weights.grad).all())
        torch.testing.assert_close(output.attention.sum(dim=1), torch.ones(2), atol=1e-6, rtol=1e-6)

    def test_factorized_product_three_metric_variant_learns_start_path_end_weights(self) -> None:
        config = Code2HypTorchConfig(
            token_vocab_size=16,
            ast_node_vocab_size=12,
            label_vocab_size=7,
            token_dim=4,
            structural_dim=5,
            curvature=1.3,
        )
        b4_model = Code2HypTorchModel(config, variant="hyperbolic")
        b16_model = Code2HypTorchModel(config, variant="factorized_product_three_metric")

        output = b16_model(_toy_batch())
        metric_weights = b16_model.factorized_metric_weights()
        loss = output.logits.square().sum()
        loss.backward()

        self.assertEqual(output.logits.shape, (2, 7))
        self.assertEqual(output.representation.shape, (2, config.representation_dim))
        self.assertEqual(output.structural_geometry, "poincare")
        self.assertEqual(output.context_structural_points.shape, (2, 3, config.structural_dim))
        self.assertEqual(output.context_structural_embeddings.shape, (2, 3, config.structural_dim))
        self.assertEqual(b16_model.parameter_count(), b4_model.parameter_count() + 3)
        self.assertEqual(metric_weights.shape, (3,))
        self.assertTrue(torch.all(metric_weights > 0.0))
        self.assertIsNotNone(b16_model.raw_factorized_metric_weights.grad)
        self.assertTrue(torch.isfinite(b16_model.raw_factorized_metric_weights.grad).all())
        torch.testing.assert_close(output.attention.sum(dim=1), torch.ones(2), atol=1e-6, rtol=1e-6)

    def test_code2hyp_product_frechet_variant_keeps_code2vec_contexts_but_uses_intrinsic_path_mean(self) -> None:
        config = Code2HypTorchConfig(
            token_vocab_size=16,
            ast_node_vocab_size=12,
            label_vocab_size=7,
            token_dim=4,
            structural_dim=5,
            curvature=1.3,
            trainable_curvature=True,
            frechet_steps=3,
        )
        b16_model = Code2HypTorchModel(config, variant="factorized_product_three_metric")
        b35_model = Code2HypTorchModel(config, variant="code2hyp_product_frechet")

        output = b35_model(_toy_batch())
        metric_weights = b35_model.factorized_metric_weights()
        loss = output.logits.square().sum()
        loss.backward()

        self.assertEqual(output.logits.shape, (2, 7))
        self.assertEqual(output.representation.shape, (2, config.representation_dim))
        self.assertEqual(output.structural_geometry, "poincare")
        self.assertEqual(output.structural_points.shape, (2, config.structural_dim))
        self.assertEqual(output.context_structural_points.shape, (2, 3, config.structural_dim))
        self.assertEqual(output.context_structural_embeddings.shape, (2, 3, config.structural_dim))
        self.assertEqual(b35_model.parameter_count(), b16_model.parameter_count())
        self.assertEqual(metric_weights.shape, (3,))
        self.assertTrue(torch.all(metric_weights > 0.0))
        self.assertIsNotNone(b35_model.raw_curvature.grad)
        self.assertIsNotNone(b35_model.raw_factorized_metric_weights.grad)
        self.assertTrue(torch.isfinite(b35_model.raw_curvature.grad).all())
        self.assertTrue(torch.isfinite(b35_model.raw_factorized_metric_weights.grad).all())
        radius = (1.0 / torch.sqrt(output.curvature)).detach()
        max_norm = torch.linalg.vector_norm(output.structural_points, dim=-1).max().detach()
        self.assertLess(float(max_norm), float(radius))
        torch.testing.assert_close(output.attention.sum(dim=1), torch.ones(2), atol=1e-6, rtol=1e-6)

    def test_code2hyp_code2vec_attention_frechet_keeps_original_attention_form(self) -> None:
        config = Code2HypTorchConfig(
            token_vocab_size=16,
            ast_node_vocab_size=12,
            label_vocab_size=7,
            token_dim=4,
            structural_dim=5,
            curvature=1.3,
            trainable_curvature=True,
            frechet_steps=3,
        )
        batch = _toy_batch()
        euclidean_model = Code2HypTorchModel(config, variant="euclidean")
        b37_model = Code2HypTorchModel(config, variant="code2hyp_code2vec_attention_frechet")

        output = b37_model(batch)
        start_vectors = b37_model.token_embeddings(batch.start_tokens)
        end_vectors = b37_model.token_embeddings(batch.end_tokens)
        path_tangents = b37_model._path_tangents(batch.ast_paths, batch.ast_path_mask)
        path_points = torch_expmap0(path_tangents, curvature=output.curvature, eps=config.eps)
        path_logs = torch_logmap0(path_points, curvature=output.curvature)
        expected_contexts = torch.cat([start_vectors, path_logs, end_vectors], dim=-1)
        expected_attention = b37_model._attention(expected_contexts, batch.context_mask)
        loss = output.logits.square().sum()
        loss.backward()

        self.assertEqual(output.logits.shape, (2, 7))
        self.assertEqual(output.representation.shape, (2, config.representation_dim))
        self.assertEqual(output.structural_geometry, "poincare")
        self.assertEqual(output.structural_points.shape, (2, config.structural_dim))
        self.assertEqual(output.context_structural_points.shape, (2, 3, config.structural_dim))
        self.assertEqual(output.context_structural_embeddings.shape, (2, 3, config.structural_dim))
        self.assertEqual(b37_model.parameter_count(), euclidean_model.parameter_count() + 1)
        torch.testing.assert_close(output.attention, expected_attention)
        self.assertIsNotNone(b37_model.raw_curvature.grad)
        self.assertTrue(torch.isfinite(b37_model.raw_curvature.grad).all())

    def test_code2vec_context_transform_variant_matches_original_context_attention_flow(self) -> None:
        config = Code2HypTorchConfig(
            token_vocab_size=16,
            ast_node_vocab_size=12,
            label_vocab_size=7,
            token_dim=4,
            structural_dim=5,
        )
        batch = _toy_batch()
        b1_model = Code2HypTorchModel(config, variant="euclidean")
        b39_model = Code2HypTorchModel(config, variant="code2vec_context_transform")

        output = b39_model(batch)
        start_vectors = b39_model.token_embeddings(batch.start_tokens)
        end_vectors = b39_model.token_embeddings(batch.end_tokens)
        path_tangents = b39_model._path_tangents(batch.ast_paths, batch.ast_path_mask)
        raw_contexts = torch.cat([start_vectors, path_tangents, end_vectors], dim=-1)
        expected_contexts = b39_model._code2vec_context_transform(raw_contexts)
        expected_attention = b39_model._attention(expected_contexts, batch.context_mask)
        expected_representation = torch.sum(expected_contexts * expected_attention.unsqueeze(-1), dim=1)
        loss = output.logits.square().sum()
        loss.backward()

        self.assertEqual(output.logits.shape, (2, 7))
        self.assertEqual(output.structural_geometry, None)
        self.assertEqual(output.context_structural_embeddings.shape, (2, 3, config.structural_dim))
        self.assertGreater(b39_model.parameter_count(), b1_model.parameter_count())
        torch.testing.assert_close(output.attention, expected_attention)
        torch.testing.assert_close(output.representation, expected_representation)
        self.assertIsNotNone(b39_model.context_transform_layer.weight.grad)
        self.assertTrue(torch.isfinite(b39_model.context_transform_layer.weight.grad).all())

    def test_code2vec_context_transform_l1_variant_uses_manhattan_structural_metric(self) -> None:
        config = Code2HypTorchConfig(
            token_vocab_size=16,
            ast_node_vocab_size=12,
            label_vocab_size=7,
            token_dim=4,
            structural_dim=5,
        )
        batch = _toy_batch()
        model = Code2HypTorchModel(config, variant="code2vec_context_transform_l1")

        output = model(batch)
        left = torch.tensor([[0.0, 0.0], [1.0, -1.0]])
        right = torch.tensor([[3.0, 4.0], [-1.0, 2.0]])

        l1_distance = _structural_embedding_distance(left, right, metric="l1")
        l2_distance = _structural_embedding_distance(left, right, metric="l2")

        self.assertEqual(output.structural_geometry, None)
        self.assertEqual(output.structural_embedding_metric, "l1")
        self.assertEqual(output.context_structural_embeddings.shape, (2, 3, config.structural_dim))
        torch.testing.assert_close(l1_distance, torch.tensor([7.0, 5.0]))
        torch.testing.assert_close(l2_distance, torch.tensor([5.0, 3.6055512]))

    def test_code2hyp_context_transform_frechet_uses_code2vec_context_layer_and_hyperbolic_path_mean(self) -> None:
        config = Code2HypTorchConfig(
            token_vocab_size=16,
            ast_node_vocab_size=12,
            label_vocab_size=7,
            token_dim=4,
            structural_dim=5,
            curvature=1.3,
            trainable_curvature=True,
            frechet_steps=3,
        )
        batch = _toy_batch()
        b39_model = Code2HypTorchModel(config, variant="code2vec_context_transform")
        b40_model = Code2HypTorchModel(config, variant="code2hyp_context_transform_frechet")

        output = b40_model(batch)
        start_vectors = b40_model.token_embeddings(batch.start_tokens)
        end_vectors = b40_model.token_embeddings(batch.end_tokens)
        path_tangents = b40_model._path_tangents(batch.ast_paths, batch.ast_path_mask)
        path_points = torch_expmap0(path_tangents, curvature=output.curvature, eps=config.eps)
        path_logs = torch_logmap0(path_points, curvature=output.curvature)
        raw_contexts = torch.cat([start_vectors, path_logs, end_vectors], dim=-1)
        expected_contexts = b40_model._code2vec_context_transform(raw_contexts)
        expected_attention = b40_model._attention(expected_contexts, batch.context_mask)
        loss = output.logits.square().sum()
        loss.backward()

        self.assertEqual(output.logits.shape, (2, 7))
        self.assertEqual(output.representation.shape, (2, config.representation_dim))
        self.assertEqual(output.structural_geometry, "poincare")
        self.assertEqual(output.structural_points.shape, (2, config.structural_dim))
        self.assertEqual(output.context_structural_points.shape, (2, 3, config.structural_dim))
        self.assertEqual(output.context_structural_embeddings.shape, (2, 3, config.structural_dim))
        self.assertEqual(b40_model.parameter_count(), b39_model.parameter_count() + 1)
        torch.testing.assert_close(output.attention, expected_attention)
        self.assertIsNotNone(b40_model.raw_curvature.grad)
        self.assertIsNotNone(b40_model.context_transform_layer.weight.grad)
        self.assertTrue(torch.isfinite(b40_model.raw_curvature.grad).all())
        self.assertTrue(torch.isfinite(b40_model.context_transform_layer.weight.grad).all())

    def test_code2hyp_product_context_transform_uses_product_attention_and_code2vec_context_vector(self) -> None:
        config = Code2HypTorchConfig(
            token_vocab_size=16,
            ast_node_vocab_size=12,
            label_vocab_size=7,
            token_dim=4,
            structural_dim=5,
            curvature=1.3,
            trainable_curvature=True,
            frechet_steps=3,
        )
        batch = _toy_batch()
        b39_model = Code2HypTorchModel(config, variant="code2vec_context_transform")
        b42_model = Code2HypTorchModel(config, variant="code2hyp_product_context_transform_frechet")

        output = b42_model(batch)
        start_vectors = b42_model.token_embeddings(batch.start_tokens)
        end_vectors = b42_model.token_embeddings(batch.end_tokens)
        path_tangents = b42_model._path_tangents(batch.ast_paths, batch.ast_path_mask)
        path_points = torch_expmap0(path_tangents, curvature=output.curvature, eps=config.eps)
        path_logs = torch_logmap0(path_points, curvature=output.curvature)
        raw_contexts = torch.cat([start_vectors, path_logs, end_vectors], dim=-1)
        expected_contexts = b42_model._code2vec_context_transform(raw_contexts)
        expected_attention = b42_model._factorized_product_attention(
            start_vectors,
            path_points,
            end_vectors,
            batch.context_mask,
            output.curvature,
        )
        expected_representation = torch.sum(expected_contexts * expected_attention.unsqueeze(-1), dim=1)
        loss = output.logits.square().sum()
        loss.backward()

        self.assertEqual(output.logits.shape, (2, 7))
        self.assertEqual(output.representation.shape, (2, config.representation_dim))
        self.assertEqual(output.structural_geometry, "poincare")
        self.assertEqual(output.structural_points.shape, (2, config.structural_dim))
        self.assertEqual(output.context_structural_points.shape, (2, 3, config.structural_dim))
        self.assertEqual(output.context_structural_embeddings.shape, (2, 3, config.structural_dim))
        self.assertEqual(b42_model.parameter_count(), b39_model.parameter_count() + 4)
        torch.testing.assert_close(output.attention, expected_attention)
        torch.testing.assert_close(output.representation, expected_representation)
        self.assertIsNotNone(b42_model.raw_curvature.grad)
        self.assertIsNotNone(b42_model.raw_factorized_metric_weights.grad)
        self.assertIsNotNone(b42_model.context_transform_layer.weight.grad)
        self.assertTrue(torch.isfinite(b42_model.raw_curvature.grad).all())
        self.assertTrue(torch.isfinite(b42_model.raw_factorized_metric_weights.grad).all())
        self.assertTrue(torch.isfinite(b42_model.context_transform_layer.weight.grad).all())

    def test_code2hyp_context_transform_product_bias_adds_trainable_structural_attention_bias(self) -> None:
        config = Code2HypTorchConfig(
            token_vocab_size=16,
            ast_node_vocab_size=12,
            label_vocab_size=7,
            token_dim=4,
            structural_dim=5,
            curvature=1.3,
            trainable_curvature=True,
            frechet_steps=3,
        )
        batch = _toy_batch()
        b39_model = Code2HypTorchModel(config, variant="code2vec_context_transform")
        b44_model = Code2HypTorchModel(config, variant="code2hyp_context_transform_product_bias_frechet")

        output = b44_model(batch)
        start_vectors = b44_model.token_embeddings(batch.start_tokens)
        end_vectors = b44_model.token_embeddings(batch.end_tokens)
        path_tangents = b44_model._path_tangents(batch.ast_paths, batch.ast_path_mask)
        path_points = torch_expmap0(path_tangents, curvature=output.curvature, eps=config.eps)
        path_logs = torch_logmap0(path_points, curvature=output.curvature)
        raw_contexts = torch.cat([start_vectors, path_logs, end_vectors], dim=-1)
        transformed_contexts = b44_model._code2vec_context_transform(raw_contexts)
        expected_attention = b44_model._code2hyp_context_product_bias_attention(
            transformed_contexts,
            start_vectors,
            path_points,
            end_vectors,
            batch.context_mask,
            output.curvature,
        )
        expected_representation = torch.sum(transformed_contexts * expected_attention.unsqueeze(-1), dim=1)
        loss = output.logits.square().sum()
        loss.backward()

        self.assertEqual(output.logits.shape, (2, 7))
        self.assertEqual(output.representation.shape, (2, config.representation_dim))
        self.assertEqual(output.structural_geometry, "poincare")
        self.assertEqual(output.structural_points.shape, (2, config.structural_dim))
        self.assertEqual(output.context_structural_points.shape, (2, 3, config.structural_dim))
        self.assertEqual(output.context_structural_embeddings.shape, (2, 3, config.structural_dim))
        self.assertEqual(b44_model.parameter_count(), b39_model.parameter_count() + 5)
        torch.testing.assert_close(output.attention, expected_attention)
        torch.testing.assert_close(output.representation, expected_representation)
        self.assertGreater(float(b44_model.product_attention_bias_weight().detach()), 0.0)
        self.assertIsNotNone(b44_model.raw_curvature.grad)
        self.assertIsNotNone(b44_model.raw_factorized_metric_weights.grad)
        self.assertIsNotNone(b44_model.raw_product_attention_bias_weight.grad)
        self.assertIsNotNone(b44_model.context_transform_layer.weight.grad)
        self.assertTrue(torch.isfinite(b44_model.raw_curvature.grad).all())
        self.assertTrue(torch.isfinite(b44_model.raw_factorized_metric_weights.grad).all())
        self.assertTrue(torch.isfinite(b44_model.raw_product_attention_bias_weight.grad).all())
        self.assertTrue(torch.isfinite(b44_model.context_transform_layer.weight.grad).all())

    def test_code2hyp_branch_product_context_transform_splits_ast_path_into_two_factors(self) -> None:
        config = Code2HypTorchConfig(
            token_vocab_size=16,
            ast_node_vocab_size=12,
            label_vocab_size=7,
            token_dim=4,
            structural_dim=6,
            curvature=1.3,
            trainable_curvature=True,
            frechet_steps=3,
        )
        batch = _toy_batch()
        baseline = Code2HypTorchModel(config, variant="code2vec_context_transform")
        model = Code2HypTorchModel(config, variant="code2hyp_branch_product_context_transform_frechet")

        output = model(batch)
        start_vectors = model.token_embeddings(batch.start_tokens)
        end_vectors = model.token_embeddings(batch.end_tokens)
        left_tangents, right_tangents = model._branch_path_tangents(batch.ast_paths, batch.ast_path_mask)
        left_points = torch_expmap0(left_tangents, curvature=output.curvature, eps=config.eps)
        right_points = torch_expmap0(right_tangents, curvature=output.curvature, eps=config.eps)
        path_logs = torch.cat(
            [
                torch_logmap0(left_points, curvature=output.curvature),
                torch_logmap0(right_points, curvature=output.curvature),
            ],
            dim=-1,
        )
        transformed_contexts = model._code2vec_context_transform(torch.cat([start_vectors, path_logs, end_vectors], dim=-1))
        expected_attention = model._code2hyp_branch_product_bias_attention(
            transformed_contexts,
            start_vectors,
            left_points,
            right_points,
            end_vectors,
            batch.context_mask,
            output.curvature,
        )
        expected_representation = torch.sum(transformed_contexts * expected_attention.unsqueeze(-1), dim=1)
        loss = output.logits.square().sum()
        loss.backward()

        self.assertEqual(output.logits.shape, (2, 7))
        self.assertEqual(output.representation.shape, (2, config.representation_dim))
        self.assertEqual(output.structural_geometry, "poincare_product")
        self.assertIsNone(output.structural_points)
        self.assertIsNone(output.context_structural_points)
        self.assertIsNotNone(output.structural_product_points)
        self.assertIsNotNone(output.context_structural_product_points)
        self.assertEqual(output.structural_product_points[0].shape, (2, 3))
        self.assertEqual(output.structural_product_points[1].shape, (2, 3))
        self.assertEqual(output.context_structural_product_points[0].shape, (2, 3, 3))
        self.assertEqual(output.context_structural_product_points[1].shape, (2, 3, 3))
        self.assertEqual(output.context_structural_embeddings.shape, (2, 3, config.structural_dim))
        self.assertEqual(model.factorized_metric_weights().numel(), 4)
        self.assertGreater(model.parameter_count(), baseline.parameter_count())
        torch.testing.assert_close(output.attention, expected_attention)
        torch.testing.assert_close(output.representation, expected_representation)
        self.assertGreater(float(model.product_attention_bias_weight().detach()), 0.0)
        self.assertIsNotNone(model.raw_curvature.grad)
        self.assertIsNotNone(model.raw_factorized_metric_weights.grad)
        self.assertIsNotNone(model.raw_product_attention_bias_weight.grad)
        self.assertIsNotNone(model.branch_left_projection.weight.grad)
        self.assertIsNotNone(model.branch_right_projection.weight.grad)
        self.assertTrue(torch.isfinite(model.raw_curvature.grad).all())
        self.assertTrue(torch.isfinite(model.raw_factorized_metric_weights.grad).all())
        self.assertTrue(torch.isfinite(model.raw_product_attention_bias_weight.grad).all())
        self.assertTrue(torch.isfinite(model.branch_left_projection.weight.grad).all())
        self.assertTrue(torch.isfinite(model.branch_right_projection.weight.grad).all())

    def test_code2hyp_context_transform_branch_product_bias_keeps_whole_path_decoder_channel(self) -> None:
        config = Code2HypTorchConfig(
            token_vocab_size=16,
            ast_node_vocab_size=12,
            label_vocab_size=7,
            token_dim=4,
            structural_dim=6,
            curvature=1.3,
            trainable_curvature=True,
            frechet_steps=3,
        )
        batch = _toy_batch()
        model = Code2HypTorchModel(config, variant="code2hyp_context_transform_branch_product_bias_frechet")

        output = model(batch)
        start_vectors = model.token_embeddings(batch.start_tokens)
        end_vectors = model.token_embeddings(batch.end_tokens)
        path_tangents = model._path_tangents(batch.ast_paths, batch.ast_path_mask)
        path_points = torch_expmap0(path_tangents, curvature=output.curvature, eps=config.eps)
        path_logs = torch_logmap0(path_points, curvature=output.curvature)
        left_tangents, right_tangents = model._branch_path_tangents(batch.ast_paths, batch.ast_path_mask)
        left_points = torch_expmap0(left_tangents, curvature=output.curvature, eps=config.eps)
        right_points = torch_expmap0(right_tangents, curvature=output.curvature, eps=config.eps)
        transformed_contexts = model._code2vec_context_transform(torch.cat([start_vectors, path_logs, end_vectors], dim=-1))
        expected_attention = model._code2hyp_branch_product_bias_attention(
            transformed_contexts,
            start_vectors,
            left_points,
            right_points,
            end_vectors,
            batch.context_mask,
            output.curvature,
        )
        expected_representation = torch.sum(transformed_contexts * expected_attention.unsqueeze(-1), dim=1)
        loss = output.logits.square().sum()
        loss.backward()

        self.assertEqual(output.structural_geometry, "poincare_product")
        self.assertIsNone(output.context_structural_points)
        self.assertIsNotNone(output.context_structural_product_points)
        self.assertEqual(output.context_structural_embeddings.shape, (2, 3, config.structural_dim))
        torch.testing.assert_close(output.attention, expected_attention)
        torch.testing.assert_close(output.representation, expected_representation)
        self.assertIsNotNone(model.branch_left_projection.weight.grad)
        self.assertIsNotNone(model.branch_right_projection.weight.grad)
        self.assertTrue(torch.isfinite(model.branch_left_projection.weight.grad).all())
        self.assertTrue(torch.isfinite(model.branch_right_projection.weight.grad).all())

    def test_code2hyp_latent_lca_branch_product_bias_learns_branch_split(self) -> None:
        config = Code2HypTorchConfig(
            token_vocab_size=16,
            ast_node_vocab_size=12,
            label_vocab_size=7,
            token_dim=4,
            structural_dim=6,
            curvature=1.3,
            trainable_curvature=True,
            frechet_steps=3,
        )
        batch = _toy_batch()
        model = Code2HypTorchModel(
            config,
            variant="code2hyp_context_transform_latent_lca_branch_product_bias_frechet",
        )

        output = model(batch)
        start_vectors = model.token_embeddings(batch.start_tokens)
        end_vectors = model.token_embeddings(batch.end_tokens)
        path_tangents = model._path_tangents(batch.ast_paths, batch.ast_path_mask)
        path_points = torch_expmap0(path_tangents, curvature=output.curvature, eps=config.eps)
        path_logs = torch_logmap0(path_points, curvature=output.curvature)
        left_tangents, right_tangents, pivot_attention = model._latent_lca_branch_path_tangents(
            batch.ast_paths,
            batch.ast_path_mask,
        )
        left_points = torch_expmap0(left_tangents, curvature=output.curvature, eps=config.eps)
        right_points = torch_expmap0(right_tangents, curvature=output.curvature, eps=config.eps)
        transformed_contexts = model._code2vec_context_transform(torch.cat([start_vectors, path_logs, end_vectors], dim=-1))
        expected_attention = model._code2hyp_branch_product_bias_attention(
            transformed_contexts,
            start_vectors,
            left_points,
            right_points,
            end_vectors,
            batch.context_mask,
            output.curvature,
        )
        expected_representation = torch.sum(transformed_contexts * expected_attention.unsqueeze(-1), dim=1)
        valid_paths = batch.ast_path_mask.any(dim=-1)
        loss = output.logits.square().sum()
        loss.backward()

        self.assertEqual(output.structural_geometry, "poincare_product")
        self.assertIsNone(output.context_structural_points)
        self.assertIsNotNone(output.context_structural_product_points)
        self.assertIsNotNone(output.path_node_attention)
        self.assertEqual(output.context_structural_embeddings.shape, (2, 3, config.structural_dim))
        torch.testing.assert_close(output.attention, expected_attention)
        torch.testing.assert_close(output.representation, expected_representation)
        torch.testing.assert_close(output.path_node_attention, pivot_attention)
        torch.testing.assert_close(
            output.path_node_attention.sum(dim=-1)[valid_paths],
            torch.ones_like(output.path_node_attention.sum(dim=-1)[valid_paths]),
        )
        torch.testing.assert_close(
            output.path_node_attention.sum(dim=-1)[~valid_paths],
            torch.zeros_like(output.path_node_attention.sum(dim=-1)[~valid_paths]),
        )
        self.assertIsNotNone(model.branch_pivot_query.grad)
        self.assertIsNotNone(model.branch_left_projection.weight.grad)
        self.assertIsNotNone(model.branch_right_projection.weight.grad)
        self.assertTrue(torch.isfinite(model.branch_pivot_query.grad).all())
        self.assertTrue(torch.isfinite(model.branch_left_projection.weight.grad).all())
        self.assertTrue(torch.isfinite(model.branch_right_projection.weight.grad).all())

    def test_code2hyp_latent_lca_prior_branch_product_bias_has_trainable_center_prior(self) -> None:
        config = Code2HypTorchConfig(
            token_vocab_size=16,
            ast_node_vocab_size=12,
            label_vocab_size=7,
            token_dim=4,
            structural_dim=6,
            curvature=1.3,
            trainable_curvature=True,
            frechet_steps=3,
        )
        batch = _toy_batch()
        model = Code2HypTorchModel(
            config,
            variant="code2hyp_context_transform_latent_lca_prior_branch_product_bias_frechet",
        )

        output = model(batch)
        loss = output.logits.square().sum()
        loss.backward()

        self.assertEqual(output.structural_geometry, "poincare_product")
        self.assertIsNotNone(output.path_node_attention)
        self.assertGreater(float(model.branch_pivot_center_prior_weight().detach()), 0.0)
        self.assertIsNotNone(model.branch_pivot_query.grad)
        self.assertIsNotNone(model.raw_branch_pivot_center_prior_weight.grad)
        self.assertTrue(torch.isfinite(model.branch_pivot_query.grad).all())
        self.assertTrue(torch.isfinite(model.raw_branch_pivot_center_prior_weight.grad).all())

    def test_code2hyp_branch_sequence_product_bias_encodes_ordered_branches(self) -> None:
        config = Code2HypTorchConfig(
            token_vocab_size=16,
            ast_node_vocab_size=12,
            label_vocab_size=7,
            token_dim=4,
            structural_dim=6,
            curvature=1.3,
            trainable_curvature=True,
            frechet_steps=3,
        )
        batch = _toy_batch()
        model = Code2HypTorchModel(
            config,
            variant="code2hyp_context_transform_branch_sequence_product_bias_frechet",
        )

        output = model(batch)
        start_vectors = model.token_embeddings(batch.start_tokens)
        end_vectors = model.token_embeddings(batch.end_tokens)
        path_tangents = model._path_tangents(batch.ast_paths, batch.ast_path_mask)
        path_points = torch_expmap0(path_tangents, curvature=output.curvature, eps=config.eps)
        path_logs = torch_logmap0(path_points, curvature=output.curvature)
        left_tangents, right_tangents = model._branch_sequence_path_tangents(batch.ast_paths, batch.ast_path_mask)
        left_points = torch_expmap0(left_tangents, curvature=output.curvature, eps=config.eps)
        right_points = torch_expmap0(right_tangents, curvature=output.curvature, eps=config.eps)
        transformed_contexts = model._code2vec_context_transform(torch.cat([start_vectors, path_logs, end_vectors], dim=-1))
        expected_attention = model._code2hyp_branch_product_bias_attention(
            transformed_contexts,
            start_vectors,
            left_points,
            right_points,
            end_vectors,
            batch.context_mask,
            output.curvature,
        )
        expected_representation = torch.sum(transformed_contexts * expected_attention.unsqueeze(-1), dim=1)
        loss = output.logits.square().sum()
        loss.backward()

        self.assertEqual(output.structural_geometry, "poincare_product")
        self.assertIsNone(output.context_structural_points)
        self.assertIsNotNone(output.context_structural_product_points)
        self.assertEqual(output.context_structural_embeddings.shape, (2, 3, config.structural_dim))
        torch.testing.assert_close(output.attention, expected_attention)
        torch.testing.assert_close(output.representation, expected_representation)
        self.assertIsNotNone(model.branch_left_sequence_encoder.weight_ih_l0.grad)
        self.assertIsNotNone(model.branch_right_sequence_encoder.weight_ih_l0.grad)
        self.assertTrue(torch.isfinite(model.branch_left_sequence_encoder.weight_ih_l0.grad).all())
        self.assertTrue(torch.isfinite(model.branch_right_sequence_encoder.weight_ih_l0.grad).all())

    def test_hyperbolic_path_message_passing_variant_updates_ast_path_nodes(self) -> None:
        config = Code2HypTorchConfig(
            token_vocab_size=16,
            ast_node_vocab_size=12,
            label_vocab_size=7,
            token_dim=4,
            structural_dim=5,
            curvature=1.3,
            path_message_passing_steps=2,
        )
        b4_model = Code2HypTorchModel(config, variant="hyperbolic")
        b17_model = Code2HypTorchModel(config, variant="hyperbolic_path_message_passing")

        output = b17_model(_toy_batch())
        loss = output.logits.square().sum()
        loss.backward()

        self.assertEqual(output.logits.shape, (2, 7))
        self.assertEqual(output.representation.shape, (2, config.representation_dim))
        self.assertEqual(output.structural_geometry, "poincare")
        self.assertIsNotNone(output.structural_points)
        self.assertIsNotNone(output.context_structural_points)
        self.assertEqual(output.context_structural_points.shape, (2, 3, config.representation_dim))
        self.assertGreater(b17_model.parameter_count(), b4_model.parameter_count())
        self.assertIsNotNone(b17_model.path_message_linear.weight.grad)
        self.assertIsNotNone(b17_model.path_update_linear.weight.grad)
        self.assertTrue(torch.isfinite(b17_model.path_message_linear.weight.grad).all())
        self.assertTrue(torch.isfinite(b17_model.path_update_linear.weight.grad).all())
        radius = 1.0 / torch.sqrt(output.curvature)
        max_norm = torch.linalg.vector_norm(output.context_structural_points, dim=-1).max().detach()
        self.assertLess(float(max_norm), float(radius))
        torch.testing.assert_close(output.attention.sum(dim=1), torch.ones(2), atol=1e-6, rtol=1e-6)

    def test_hyperbolic_path_attention_message_passing_pools_ast_nodes_with_learned_attention(self) -> None:
        config = Code2HypTorchConfig(
            token_vocab_size=16,
            ast_node_vocab_size=12,
            label_vocab_size=7,
            token_dim=4,
            structural_dim=5,
            curvature=1.3,
            path_message_passing_steps=2,
        )
        b17_model = Code2HypTorchModel(config, variant="hyperbolic_path_message_passing")
        b23_model = Code2HypTorchModel(config, variant="hyperbolic_path_attention_message_passing")

        output = b23_model(_toy_batch())
        loss = output.logits.square().sum()
        loss.backward()

        self.assertEqual(output.logits.shape, (2, 7))
        self.assertEqual(output.representation.shape, (2, config.representation_dim))
        self.assertEqual(output.structural_geometry, "poincare")
        self.assertIsNotNone(output.structural_points)
        self.assertIsNotNone(output.context_structural_points)
        self.assertEqual(output.context_structural_points.shape, (2, 3, config.representation_dim))
        self.assertEqual(b23_model.parameter_count(), b17_model.parameter_count() + config.structural_dim)
        self.assertIsNotNone(b23_model.path_node_attention_query)
        self.assertIsNotNone(b23_model.path_node_attention_query.grad)
        self.assertTrue(torch.isfinite(b23_model.path_node_attention_query.grad).all())
        radius = 1.0 / torch.sqrt(output.curvature)
        max_norm = torch.linalg.vector_norm(output.context_structural_points, dim=-1).max().detach()
        self.assertLess(float(max_norm), float(radius))
        torch.testing.assert_close(output.attention.sum(dim=1), torch.ones(2), atol=1e-6, rtol=1e-6)
        self.assertIsNotNone(output.path_node_attention)
        self.assertEqual(output.path_node_attention.shape, _toy_batch().ast_path_mask.shape)
        self.assertIsNotNone(output.path_node_attention_monotonicity_loss)
        self.assertGreaterEqual(float(output.path_node_attention_monotonicity_loss.detach()), 0.0)
        valid_path_sums = output.path_node_attention.sum(dim=-1)[_toy_batch().ast_path_mask.any(dim=-1)]
        torch.testing.assert_close(valid_path_sums, torch.ones_like(valid_path_sums), atol=1e-6, rtol=1e-6)

    def test_path_node_attention_monotonicity_loss_accepts_either_root_or_leaf_direction(self) -> None:
        monotone_weights = torch.tensor(
            [
                [[0.10, 0.20, 0.70, 0.00], [0.60, 0.30, 0.10, 0.00]],
            ],
            dtype=torch.float32,
        )
        monotone_mask = torch.tensor(
            [
                [[True, True, True, False], [True, True, True, False]],
            ],
            dtype=torch.bool,
        )

        loss = path_node_attention_monotonicity_loss(monotone_weights, monotone_mask)

        torch.testing.assert_close(loss, torch.tensor(0.0), atol=1e-7, rtol=1e-7)

    def test_path_node_attention_monotonicity_loss_penalizes_nonmonotone_profiles(self) -> None:
        nonmonotone_weights = torch.tensor(
            [
                [[0.20, 0.60, 0.20, 0.00], [0.25, 0.25, 0.25, 0.25]],
            ],
            dtype=torch.float32,
        )
        nonmonotone_mask = torch.tensor(
            [
                [[True, True, True, False], [True, True, True, True]],
            ],
            dtype=torch.bool,
        )

        loss = path_node_attention_monotonicity_loss(nonmonotone_weights, nonmonotone_mask)

        self.assertGreater(float(loss), 0.0)

    def test_path_attention_soft_tree_distances_match_leaf_tree_distance_for_leaf_attention(self) -> None:
        batch = Code2HypBatch(
            start_tokens=torch.tensor([[1, 1, 1]], dtype=torch.long),
            end_tokens=torch.tensor([[2, 2, 2]], dtype=torch.long),
            ast_paths=torch.tensor([[[1, 2, 3], [1, 2, 4], [5, 6, 7]]], dtype=torch.long),
            ast_path_mask=torch.ones(1, 3, 3, dtype=torch.bool),
            context_mask=torch.ones(1, 3, dtype=torch.bool),
        )
        leaf_attention = torch.tensor([[[0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0]]])

        soft_distances, leaf_distances = path_attention_soft_tree_distances(
            leaf_attention,
            batch.ast_paths,
            batch.ast_path_mask,
            batch.context_mask,
        )

        torch.testing.assert_close(soft_distances, leaf_distances, atol=1e-7, rtol=1e-7)
        torch.testing.assert_close(leaf_distances, torch.tensor([2.0, 6.0, 6.0]))

    def test_path_attention_tree_distance_loss_penalizes_root_collapsed_attention(self) -> None:
        batch = Code2HypBatch(
            start_tokens=torch.tensor([[1, 1, 1]], dtype=torch.long),
            end_tokens=torch.tensor([[2, 2, 2]], dtype=torch.long),
            ast_paths=torch.tensor([[[1, 2, 3], [1, 2, 4], [5, 6, 7]]], dtype=torch.long),
            ast_path_mask=torch.ones(1, 3, 3, dtype=torch.bool),
            context_mask=torch.ones(1, 3, dtype=torch.bool),
        )
        leaf_attention = torch.tensor([[[0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0]]])
        root_attention = torch.tensor([[[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]])

        leaf_loss = path_attention_tree_distance_loss(
            leaf_attention,
            batch.ast_paths,
            batch.ast_path_mask,
            batch.context_mask,
        )
        root_loss = path_attention_tree_distance_loss(
            root_attention,
            batch.ast_paths,
            batch.ast_path_mask,
            batch.context_mask,
        )

        torch.testing.assert_close(leaf_loss, torch.tensor(0.0), atol=1e-7, rtol=1e-7)
        self.assertGreater(float(root_loss), 0.0)

    def test_path_dual_attention_separation_loss_rewards_root_leaf_split(self) -> None:
        split_attention = torch.tensor(
            [
                [
                    [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
                    [[0.8, 0.2, 0.0], [0.0, 0.2, 0.8]],
                ]
            ],
            dtype=torch.float32,
        )
        collapsed_attention = torch.tensor(
            [
                [
                    [[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
                    [[0.2, 0.6, 0.2], [0.2, 0.6, 0.2]],
                ]
            ],
            dtype=torch.float32,
        )
        mask = torch.ones(1, 2, 3, dtype=torch.bool)

        split_loss = path_dual_attention_separation_loss(split_attention, mask, margin=0.25)
        collapsed_loss = path_dual_attention_separation_loss(collapsed_attention, mask, margin=0.25)

        torch.testing.assert_close(split_loss, torch.tensor(0.0), atol=1e-7, rtol=1e-7)
        self.assertGreater(float(collapsed_loss), 0.0)

    def test_hyperbolic_path_depth_attention_message_passing_adds_monotone_depth_bias(self) -> None:
        config = Code2HypTorchConfig(
            token_vocab_size=16,
            ast_node_vocab_size=12,
            label_vocab_size=7,
            token_dim=4,
            structural_dim=5,
            curvature=1.3,
            path_message_passing_steps=2,
        )
        b23_model = Code2HypTorchModel(config, variant="hyperbolic_path_attention_message_passing")
        b25_model = Code2HypTorchModel(config, variant="hyperbolic_path_depth_attention_message_passing")

        output = b25_model(_toy_batch())
        loss = output.logits.square().sum()
        loss.backward()

        self.assertEqual(output.logits.shape, (2, 7))
        self.assertEqual(output.representation.shape, (2, config.representation_dim))
        self.assertEqual(output.structural_geometry, "poincare")
        self.assertEqual(output.context_structural_points.shape, (2, 3, config.representation_dim))
        self.assertEqual(b25_model.parameter_count(), b23_model.parameter_count() + 1)
        self.assertIsNotNone(b25_model.path_node_attention_query)
        self.assertIsNotNone(b25_model.raw_path_depth_attention_bias)
        self.assertIsNotNone(b25_model.path_node_attention_query.grad)
        self.assertIsNotNone(b25_model.raw_path_depth_attention_bias.grad)
        self.assertTrue(torch.isfinite(b25_model.path_node_attention_query.grad).all())
        self.assertTrue(torch.isfinite(b25_model.raw_path_depth_attention_bias.grad).all())
        radius = 1.0 / torch.sqrt(output.curvature)
        max_norm = torch.linalg.vector_norm(output.context_structural_points, dim=-1).max().detach()
        self.assertLess(float(max_norm), float(radius))
        torch.testing.assert_close(output.attention.sum(dim=1), torch.ones(2), atol=1e-6, rtol=1e-6)

    def test_hyperbolic_path_dual_attention_message_passing_splits_root_and_leaf_channels(self) -> None:
        config = Code2HypTorchConfig(
            token_vocab_size=16,
            ast_node_vocab_size=12,
            label_vocab_size=7,
            token_dim=4,
            structural_dim=5,
            curvature=1.3,
            path_message_passing_steps=2,
        )
        b23_model = Code2HypTorchModel(config, variant="hyperbolic_path_attention_message_passing")
        b29_model = Code2HypTorchModel(config, variant="hyperbolic_path_dual_attention_message_passing")

        output = b29_model(_toy_batch())
        loss = output.logits.square().sum()
        loss.backward()

        self.assertEqual(output.logits.shape, (2, 7))
        self.assertEqual(output.representation.shape, (2, config.representation_dim))
        self.assertEqual(output.structural_geometry, "poincare")
        self.assertEqual(output.context_structural_points.shape, (2, 3, config.representation_dim))
        self.assertGreater(b29_model.parameter_count(), b23_model.parameter_count())
        self.assertIsNotNone(b29_model.path_root_attention_query)
        self.assertIsNotNone(b29_model.path_leaf_attention_query)
        self.assertIsNotNone(b29_model.path_dual_attention_projection)
        self.assertIsNotNone(b29_model.path_root_attention_query.grad)
        self.assertIsNotNone(b29_model.path_leaf_attention_query.grad)
        self.assertIsNotNone(b29_model.path_dual_attention_projection.weight.grad)
        self.assertTrue(torch.isfinite(b29_model.path_root_attention_query.grad).all())
        self.assertTrue(torch.isfinite(b29_model.path_leaf_attention_query.grad).all())
        self.assertTrue(torch.isfinite(b29_model.path_dual_attention_projection.weight.grad).all())
        self.assertIsNotNone(output.path_node_attention_pair)
        self.assertEqual(output.path_node_attention_pair.shape, (2, 3, 2, 4))
        valid_path_sums = output.path_node_attention_pair.sum(dim=-1)[_toy_batch().ast_path_mask.any(dim=-1)]
        torch.testing.assert_close(valid_path_sums, torch.ones_like(valid_path_sums), atol=1e-6, rtol=1e-6)
        radius = 1.0 / torch.sqrt(output.curvature)
        max_norm = torch.linalg.vector_norm(output.context_structural_points, dim=-1).max().detach()
        self.assertLess(float(max_norm), float(radius))
        torch.testing.assert_close(output.attention.sum(dim=1), torch.ones(2), atol=1e-6, rtol=1e-6)

    def test_lorentz_path_dual_attention_message_passing_uses_hyperboloid_context_space(self) -> None:
        config = Code2HypTorchConfig(
            token_vocab_size=16,
            ast_node_vocab_size=12,
            label_vocab_size=7,
            token_dim=4,
            structural_dim=5,
            curvature=1.3,
            path_message_passing_steps=2,
        )
        b29_model = Code2HypTorchModel(config, variant="hyperbolic_path_dual_attention_message_passing")
        b32_model = Code2HypTorchModel(config, variant="lorentz_path_dual_attention_message_passing")

        output = b32_model(_toy_batch())
        loss = output.logits.square().sum()
        loss.backward()

        self.assertEqual(output.logits.shape, (2, 7))
        self.assertEqual(output.representation.shape, (2, config.representation_dim))
        self.assertEqual(output.structural_geometry, "lorentz")
        self.assertEqual(output.context_structural_points.shape, (2, 3, config.representation_dim + 1))
        self.assertEqual(b32_model.parameter_count(), b29_model.parameter_count())
        self.assertIsNotNone(b32_model.path_root_attention_query)
        self.assertIsNotNone(b32_model.path_leaf_attention_query)
        self.assertIsNotNone(b32_model.path_dual_attention_projection)
        self.assertIsNotNone(b32_model.path_root_attention_query.grad)
        self.assertIsNotNone(b32_model.path_leaf_attention_query.grad)
        self.assertIsNotNone(b32_model.path_dual_attention_projection.weight.grad)
        self.assertTrue(torch.isfinite(b32_model.path_root_attention_query.grad).all())
        self.assertTrue(torch.isfinite(b32_model.path_leaf_attention_query.grad).all())
        self.assertTrue(torch.isfinite(b32_model.path_dual_attention_projection.weight.grad).all())
        self.assertIsNotNone(output.path_node_attention_pair)
        self.assertEqual(output.path_node_attention_pair.shape, (2, 3, 2, 4))
        valid_path_sums = output.path_node_attention_pair.sum(dim=-1)[_toy_batch().ast_path_mask.any(dim=-1)]
        torch.testing.assert_close(valid_path_sums, torch.ones_like(valid_path_sums), atol=1e-6, rtol=1e-6)
        hyperboloid_norm = -output.context_structural_points[..., 0].square() + torch.sum(
            output.context_structural_points[..., 1:].square(),
            dim=-1,
        )
        torch.testing.assert_close(
            hyperboloid_norm,
            torch.full_like(hyperboloid_norm, -1.0 / output.curvature),
            atol=1e-5,
            rtol=1e-5,
        )
        torch.testing.assert_close(output.attention.sum(dim=1), torch.ones(2), atol=1e-6, rtol=1e-6)

    def test_factorized_product_channel_mixer_variant_adds_nonlinear_channel_interaction(self) -> None:
        config = Code2HypTorchConfig(
            token_vocab_size=16,
            ast_node_vocab_size=12,
            label_vocab_size=7,
            token_dim=4,
            structural_dim=5,
            curvature=1.3,
            factorized_mixer_rank=3,
        )
        b4_model = Code2HypTorchModel(config, variant="hyperbolic")
        b13_model = Code2HypTorchModel(config, variant="factorized_product_channel_mixer")

        output = b13_model(_toy_batch())
        loss = output.logits.square().sum()
        loss.backward()

        expected_extra_parameters = 2 * config.representation_dim * config.factorized_mixer_rank
        self.assertEqual(output.logits.shape, (2, 7))
        self.assertEqual(output.representation.shape, (2, config.representation_dim))
        self.assertEqual(output.structural_geometry, "poincare")
        self.assertEqual(output.context_structural_points.shape, (2, 3, config.structural_dim))
        self.assertEqual(b13_model.parameter_count(), b4_model.parameter_count() + expected_extra_parameters)
        self.assertIsNotNone(b13_model.factorized_channel_down.weight.grad)
        self.assertIsNotNone(b13_model.factorized_channel_up.weight.grad)
        self.assertTrue(torch.isfinite(b13_model.factorized_channel_down.weight.grad).all())
        self.assertTrue(torch.isfinite(b13_model.factorized_channel_up.weight.grad).all())
        torch.testing.assert_close(output.attention.sum(dim=1), torch.ones(2), atol=1e-6, rtol=1e-6)

    def test_euclidean_metric_code2vec_variant_matches_b4_without_hyperbolic_geometry(self) -> None:
        config = Code2HypTorchConfig(
            token_vocab_size=16,
            ast_node_vocab_size=12,
            label_vocab_size=7,
            token_dim=4,
            structural_dim=5,
        )
        model = Code2HypTorchModel(config, variant="euclidean_metric")

        output = model(_toy_batch())

        self.assertEqual(output.logits.shape, (2, 7))
        self.assertEqual(output.representation.shape, (2, config.representation_dim))
        self.assertIsNone(output.structural_points)
        self.assertIsNone(output.context_structural_points)
        self.assertIsNotNone(output.context_structural_embeddings)
        self.assertEqual(output.context_structural_embeddings.shape, (2, 3, config.representation_dim))
        torch.testing.assert_close(output.attention.sum(dim=1), torch.ones(2), atol=1e-6, rtol=1e-6)

    def test_bounded_euclidean_metric_variant_controls_for_ball_constraint_without_hyperbolic_geometry(self) -> None:
        config = Code2HypTorchConfig(
            token_vocab_size=16,
            ast_node_vocab_size=12,
            label_vocab_size=7,
            token_dim=4,
            structural_dim=5,
            curvature=1.3,
        )
        b4_model = Code2HypTorchModel(config, variant="hyperbolic")
        b6_model = Code2HypTorchModel(config, variant="euclidean_metric")
        b14_model = Code2HypTorchModel(config, variant="bounded_euclidean_metric")

        output = b14_model(_toy_batch())

        self.assertEqual(output.logits.shape, (2, 7))
        self.assertEqual(output.representation.shape, (2, config.representation_dim))
        self.assertIsNone(output.structural_points)
        self.assertIsNone(output.context_structural_points)
        self.assertIsNone(output.structural_geometry)
        self.assertIsNotNone(output.context_structural_embeddings)
        self.assertEqual(output.context_structural_embeddings.shape, (2, 3, config.representation_dim))
        radius = 1.0 / torch.sqrt(output.curvature)
        context_norms = torch.linalg.vector_norm(output.context_structural_embeddings, dim=-1)
        representation_norms = torch.linalg.vector_norm(output.representation, dim=-1)
        self.assertLess(float(context_norms.max().detach()), float(radius))
        self.assertLess(float(representation_norms.max().detach()), float(radius))
        self.assertEqual(b14_model.parameter_count(), b4_model.parameter_count())
        self.assertEqual(b14_model.parameter_count(), b6_model.parameter_count())
        torch.testing.assert_close(output.attention.sum(dim=1), torch.ones(2), atol=1e-6, rtol=1e-6)

    def test_tree_context_features_encode_length_distance_and_lca_controls(self) -> None:
        batch = Code2HypBatch(
            start_tokens=torch.tensor([[1, 1, 1]], dtype=torch.long),
            end_tokens=torch.tensor([[2, 2, 2]], dtype=torch.long),
            ast_paths=torch.tensor([[[1, 2, 3, 0], [1, 2, 4, 0], [1, 5, 0, 0]]], dtype=torch.long),
            ast_path_mask=torch.tensor(
                [[[True, True, True, False], [True, True, True, False], [True, True, False, False]]],
                dtype=torch.bool,
            ),
            context_mask=torch.tensor([[True, True, True]], dtype=torch.bool),
        )

        lca_depth = ast_sequence_lca_depth(
            batch.ast_paths[:, 0],
            batch.ast_path_mask[:, 0],
            batch.ast_paths[:, 1],
            batch.ast_path_mask[:, 1],
        )
        features = tree_context_features(batch)

        torch.testing.assert_close(lca_depth, torch.tensor([2.0]))
        torch.testing.assert_close(
            features[0],
            torch.tensor(
                [
                    [0.75, 0.3125, 0.3750, 0.3750],
                    [0.75, 0.3125, 0.3750, 0.3750],
                    [0.50, 0.3750, 0.3750, 0.2500],
                ]
            ),
        )

    def test_tree_context_features_prefers_precomputed_cache(self) -> None:
        cached_features = torch.tensor(
            [[[0.10, 0.20, 0.30, 0.40], [0.50, 0.60, 0.70, 0.80], [0.0, 0.0, 0.0, 0.0]]],
            dtype=torch.float32,
        )
        batch = Code2HypBatch(
            start_tokens=torch.tensor([[1, 1, 0]], dtype=torch.long),
            end_tokens=torch.tensor([[2, 2, 0]], dtype=torch.long),
            ast_paths=torch.tensor([[[1, 9, 9, 0], [8, 7, 6, 0], [0, 0, 0, 0]]], dtype=torch.long),
            ast_path_mask=torch.tensor(
                [[[True, True, True, False], [True, True, True, False], [False, False, False, False]]],
                dtype=torch.bool,
            ),
            context_mask=torch.tensor([[True, True, False]], dtype=torch.bool),
            context_tree_features=cached_features,
        )

        features = tree_context_features(batch)

        self.assertIs(features, cached_features)

    def test_euclidean_tree_variant_adds_explicit_lca_bias_without_hyperbolic_geometry(self) -> None:
        config = Code2HypTorchConfig(
            token_vocab_size=16,
            ast_node_vocab_size=12,
            label_vocab_size=7,
            token_dim=4,
            structural_dim=5,
        )
        metric_model = Code2HypTorchModel(config, variant="euclidean_metric")
        tree_model = Code2HypTorchModel(config, variant="euclidean_tree")

        output = tree_model(_toy_batch())

        self.assertEqual(output.logits.shape, (2, 7))
        self.assertEqual(output.representation.shape, (2, config.representation_dim))
        self.assertIsNone(output.structural_points)
        self.assertIsNone(output.context_structural_points)
        self.assertIsNotNone(output.context_structural_embeddings)
        self.assertIsNotNone(output.context_tree_features)
        self.assertEqual(output.context_structural_embeddings.shape, (2, 3, config.representation_dim))
        self.assertEqual(output.context_tree_features.shape, (2, 3, 4))
        self.assertEqual(tree_model.parameter_count() - metric_model.parameter_count(), 4)
        torch.testing.assert_close(output.attention.sum(dim=1), torch.ones(2), atol=1e-6, rtol=1e-6)

    def test_structural_distance_loss_is_zero_for_perfectly_scaled_distances(self) -> None:
        ast_distances = torch.tensor([1.0, 2.0, 3.0])
        embedding_distances = torch.tensor([2.0, 4.0, 6.0])

        loss = structural_distance_loss(embedding_distances, ast_distances)

        self.assertLess(float(loss), 1e-8)

    def test_structural_rank_loss_penalizes_inverted_distance_order(self) -> None:
        ast_distances = torch.tensor([1.0, 2.0, 3.0])
        monotone_embedding_distances = torch.tensor([2.0, 4.0, 6.0])
        inverted_embedding_distances = torch.tensor([6.0, 4.0, 2.0])

        monotone_loss = structural_rank_loss(monotone_embedding_distances, ast_distances, margin=0.1)
        inverted_loss = structural_rank_loss(inverted_embedding_distances, ast_distances, margin=0.1)

        self.assertLess(float(monotone_loss), 1e-8)
        self.assertGreater(float(inverted_loss), 1.0)

    def test_structural_rank_loss_uses_adjacent_sorted_comparisons(self) -> None:
        ast_distances = torch.tensor([1.0, 2.0, 3.0])
        embedding_distances = torch.tensor([0.0, 10.0, 10.0])

        loss = structural_rank_loss(embedding_distances, ast_distances, margin=1.0)

        torch.testing.assert_close(loss, torch.tensor(0.5))

    def test_structural_spearman_correlation_reports_monotonic_alignment(self) -> None:
        ast_distances = torch.tensor([1.0, 2.0, 3.0, 4.0])

        perfect = structural_spearman_correlation(torch.tensor([2.0, 4.0, 6.0, 8.0]), ast_distances)
        inverted = structural_spearman_correlation(torch.tensor([8.0, 6.0, 4.0, 2.0]), ast_distances)
        degenerate = structural_spearman_correlation(torch.tensor([1.0, 1.0, 1.0, 1.0]), ast_distances)

        torch.testing.assert_close(perfect, torch.tensor(1.0))
        torch.testing.assert_close(inverted, torch.tensor(-1.0))
        torch.testing.assert_close(degenerate, torch.tensor(0.0))

    def test_batch_structural_distance_regularizer_is_finite_for_b1_and_b3(self) -> None:
        config = Code2HypTorchConfig(
            token_vocab_size=16,
            ast_node_vocab_size=12,
            label_vocab_size=7,
            token_dim=4,
            structural_dim=5,
            trainable_curvature=True,
        )
        batch = _toy_batch()
        euclidean = Code2HypTorchModel(config, variant="euclidean")
        product = Code2HypTorchModel(config, variant="product")

        b1_loss = batch_structural_distance_regularizer(euclidean(batch), batch)
        b3_loss = batch_structural_distance_regularizer(product(batch), batch)

        self.assertTrue(torch.isfinite(b1_loss))
        self.assertTrue(torch.isfinite(b3_loss))
        self.assertGreaterEqual(float(b1_loss.detach()), 0.0)
        self.assertGreaterEqual(float(b3_loss.detach()), 0.0)

    def test_batch_structural_multi_metric_regularizer_is_finite_for_euclidean_and_product(self) -> None:
        config = Code2HypTorchConfig(
            token_vocab_size=16,
            ast_node_vocab_size=12,
            label_vocab_size=7,
            token_dim=4,
            structural_dim=5,
            trainable_curvature=True,
        )
        batch = _toy_batch()
        euclidean = Code2HypTorchModel(config, variant="euclidean")
        product = Code2HypTorchModel(config, variant="product")

        b1_loss = batch_structural_multi_metric_distance_regularizer(euclidean(batch), batch)
        b3_loss = batch_structural_multi_metric_distance_regularizer(product(batch), batch)

        self.assertTrue(torch.isfinite(b1_loss))
        self.assertTrue(torch.isfinite(b3_loss))
        self.assertGreaterEqual(float(b1_loss.detach()), 0.0)
        self.assertGreaterEqual(float(b3_loss.detach()), 0.0)

    def test_batch_structural_multi_metric_regularizer_rejects_empty_target_set(self) -> None:
        config = Code2HypTorchConfig(
            token_vocab_size=16,
            ast_node_vocab_size=12,
            label_vocab_size=7,
            token_dim=4,
            structural_dim=5,
        )
        batch = _toy_batch()
        output = Code2HypTorchModel(config, variant="euclidean")(batch)

        with self.assertRaisesRegex(ValueError, "target_distances"):
            batch_structural_multi_metric_distance_regularizer(output, batch, target_distances=())

    def test_batch_structural_rank_regularizer_is_finite_for_b1_and_b3(self) -> None:
        config = Code2HypTorchConfig(
            token_vocab_size=16,
            ast_node_vocab_size=12,
            label_vocab_size=7,
            token_dim=4,
            structural_dim=5,
            trainable_curvature=True,
        )
        batch = _toy_batch()
        euclidean = Code2HypTorchModel(config, variant="euclidean")
        product = Code2HypTorchModel(config, variant="product")

        b1_loss = batch_structural_rank_regularizer(euclidean(batch), batch)
        b3_loss = batch_structural_rank_regularizer(product(batch), batch)

        self.assertTrue(torch.isfinite(b1_loss))
        self.assertTrue(torch.isfinite(b3_loss))
        self.assertGreaterEqual(float(b1_loss.detach()), 0.0)
        self.assertGreaterEqual(float(b3_loss.detach()), 0.0)

    def test_batch_structural_spearman_correlation_is_bounded_for_b1_and_b3(self) -> None:
        config = Code2HypTorchConfig(
            token_vocab_size=16,
            ast_node_vocab_size=12,
            label_vocab_size=7,
            token_dim=4,
            structural_dim=5,
            trainable_curvature=True,
        )
        batch = _toy_batch()
        euclidean = Code2HypTorchModel(config, variant="euclidean")
        product = Code2HypTorchModel(config, variant="product")

        b1_correlation = batch_structural_spearman_correlation(euclidean(batch), batch)
        b3_correlation = batch_structural_spearman_correlation(product(batch), batch)

        self.assertTrue(torch.isfinite(b1_correlation))
        self.assertTrue(torch.isfinite(b3_correlation))
        self.assertGreaterEqual(float(b1_correlation.detach()), -1.0)
        self.assertLessEqual(float(b1_correlation.detach()), 1.0)
        self.assertGreaterEqual(float(b3_correlation.detach()), -1.0)
        self.assertLessEqual(float(b3_correlation.detach()), 1.0)

    def test_ast_sequence_tree_distance_uses_lca_prefix(self) -> None:
        left = torch.tensor([[1, 2, 3, 0], [1, 2, 4, 0]], dtype=torch.long)
        right = torch.tensor([[1, 2, 5, 0], [1, 6, 0, 0]], dtype=torch.long)
        left_mask = torch.tensor([[True, True, True, False], [True, True, True, False]])
        right_mask = torch.tensor([[True, True, True, False], [True, True, False, False]])

        distances = ast_sequence_tree_distance(left, left_mask, right, right_mask)

        torch.testing.assert_close(distances, torch.tensor([2.0, 3.0]))

    def test_ast_path_midpoint_branch_masks_share_the_pivot_as_lca_proxy(self) -> None:
        mask = torch.tensor(
            [
                [True, True, True, True, True, False],
                [True, True, True, True, False, False],
                [True, False, False, False, False, False],
                [False, False, False, False, False, False],
            ],
            dtype=torch.bool,
        )

        left, right = ast_path_midpoint_branch_masks(mask)

        expected_left = torch.tensor(
            [
                [True, True, True, False, False, False],
                [True, True, False, False, False, False],
                [True, False, False, False, False, False],
                [False, False, False, False, False, False],
            ],
            dtype=torch.bool,
        )
        expected_right = torch.tensor(
            [
                [False, False, True, True, True, False],
                [False, True, True, True, False, False],
                [True, False, False, False, False, False],
                [False, False, False, False, False, False],
            ],
            dtype=torch.bool,
        )
        torch.testing.assert_close(left, expected_left)
        torch.testing.assert_close(right, expected_right)

    def test_poincare_product_distance_combines_branch_factors(self) -> None:
        curvature = torch.tensor(1.0)
        left_a = torch_expmap0(torch.tensor([[0.10, 0.00]]), curvature=curvature)
        left_b = torch_expmap0(torch.tensor([[0.20, 0.00]]), curvature=curvature)
        right_a = torch_expmap0(torch.tensor([[0.00, 0.10]]), curvature=curvature)
        right_b = torch_expmap0(torch.tensor([[0.00, 0.30]]), curvature=curvature)

        product_distance = _poincare_product_distance((left_a, right_a), (left_b, right_b), curvature)
        left_distance = torch_poincare_distance(left_a, left_b, curvature)
        right_distance = torch_poincare_distance(right_a, right_b, curvature)

        torch.testing.assert_close(
            product_distance,
            torch.sqrt(left_distance.square() + right_distance.square()),
            atol=1e-6,
            rtol=1e-6,
        )

    def test_poincare_product_distance_has_finite_gradient_on_coincident_factors(self) -> None:
        curvature = torch.tensor(1.0)
        left = torch.zeros(2, 3, requires_grad=True)
        right = torch.zeros(2, 3, requires_grad=True)

        distance = _poincare_product_distance((left, right), (left, right), curvature)
        distance.sum().backward()

        self.assertTrue(torch.isfinite(left.grad).all())
        self.assertTrue(torch.isfinite(right.grad).all())


if __name__ == "__main__":
    unittest.main()
