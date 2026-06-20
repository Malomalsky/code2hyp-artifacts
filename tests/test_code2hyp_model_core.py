from __future__ import annotations

import unittest

import numpy as np

from geometry_profile_research.code2hyp_model import (
    Code2HypConfig,
    PathContext,
    ProductCode2HypModel,
    project_to_ball,
    expmap0,
    logmap0,
    poincare_distance,
)


class Code2HypGeometryTests(unittest.TestCase):
    def test_expmap_and_logmap_are_inverse_near_origin(self) -> None:
        tangent = np.array([[0.05, -0.02, 0.03], [0.01, 0.04, -0.03]], dtype=float)

        point = expmap0(tangent, curvature=1.3)
        recovered = logmap0(point, curvature=1.3)

        np.testing.assert_allclose(recovered, tangent, atol=1e-8)

    def test_projection_keeps_points_inside_poincare_ball(self) -> None:
        points = np.array([[10.0, 0.0], [0.3, 0.4]], dtype=float)

        projected = project_to_ball(points, curvature=4.0, eps=1e-5)

        radius = (1.0 - 1e-5) / 2.0
        self.assertLess(float(np.max(np.linalg.norm(projected, axis=-1))), radius)

    def test_poincare_distance_is_symmetric_and_zero_on_identity(self) -> None:
        left = np.array([[0.10, 0.20, 0.05]], dtype=float)
        right = np.array([[0.20, -0.05, 0.03]], dtype=float)

        forward = poincare_distance(left, right, curvature=0.8)
        backward = poincare_distance(right, left, curvature=0.8)
        identity = poincare_distance(left, left, curvature=0.8)

        np.testing.assert_allclose(forward, backward, atol=1e-10)
        np.testing.assert_allclose(identity, np.zeros_like(identity), atol=1e-10)
        self.assertGreater(float(forward[0]), 0.0)


class Code2HypForwardTests(unittest.TestCase):
    def _toy_contexts(self) -> list[list[PathContext]]:
        return [
            [
                PathContext(start_token=1, ast_path=(1, 2, 3), end_token=2),
                PathContext(start_token=3, ast_path=(2, 4), end_token=4),
                PathContext(start_token=1, ast_path=(3,), end_token=5),
            ],
            [
                PathContext(start_token=2, ast_path=(1, 5, 6), end_token=3),
                PathContext(start_token=4, ast_path=(6, 7), end_token=1),
            ],
        ]

    def test_euclidean_forward_returns_logits_attention_and_representations(self) -> None:
        config = Code2HypConfig(
            token_vocab_size=16,
            ast_node_vocab_size=12,
            label_vocab_size=7,
            token_dim=4,
            structural_dim=5,
        )
        model = ProductCode2HypModel.random(config, seed=7)

        output = model.forward_euclidean(self._toy_contexts())

        self.assertEqual(output.logits.shape, (2, 7))
        self.assertEqual(output.representation.shape, (2, 13))
        self.assertEqual(output.attention.shape, (2, 3))
        np.testing.assert_allclose(output.attention.sum(axis=1), np.ones(2), atol=1e-10)

    def test_product_forward_keeps_structural_centroid_inside_ball(self) -> None:
        config = Code2HypConfig(
            token_vocab_size=16,
            ast_node_vocab_size=12,
            label_vocab_size=7,
            token_dim=4,
            structural_dim=5,
            curvature=1.7,
        )
        model = ProductCode2HypModel.random(config, seed=11)

        output = model.forward_product(self._toy_contexts())

        self.assertEqual(output.logits.shape, (2, 7))
        self.assertEqual(output.representation.shape, (2, 13))
        self.assertEqual(output.structural_points.shape, (2, 5))
        np.testing.assert_allclose(output.attention.sum(axis=1), np.ones(2), atol=1e-10)
        max_norm = float(np.max(np.linalg.norm(output.structural_points, axis=1)))
        self.assertLess(max_norm, 1.0 / np.sqrt(config.curvature))

    def test_product_and_euclidean_use_same_decoder_representation_size(self) -> None:
        config = Code2HypConfig(
            token_vocab_size=20,
            ast_node_vocab_size=15,
            label_vocab_size=9,
            token_dim=6,
            structural_dim=4,
        )
        model = ProductCode2HypModel.random(config, seed=13)

        euclidean = model.forward_euclidean(self._toy_contexts())
        product = model.forward_product(self._toy_contexts())

        self.assertEqual(euclidean.representation.shape, product.representation.shape)
        self.assertEqual(euclidean.logits.shape, product.logits.shape)


if __name__ == "__main__":
    unittest.main()
