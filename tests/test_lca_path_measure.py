from __future__ import annotations

import unittest

import torch

from geometry_profile_research.lca_path_measure import (
    EuclideanPathMeasure,
    euclidean_lca_anchored_product_cost_matrix,
    lca_anchored_product_cost_matrix,
    poincare_gromov_product_at_origin,
    poincare_node_embeddings_from_tree,
    poincare_path_measure_from_paths,
    sinkhorn_euclidean_path_measure_distance,
    sinkhorn_path_measure_distance,
)
from geometry_profile_research.raw_ast import RawAstPath, RawAstTree


class LcaPathMeasureTests(unittest.TestCase):
    def _tree(self) -> RawAstTree:
        return RawAstTree.from_edges(
            root_id=0,
            edges=((0, 1), (0, 2), (1, 3), (1, 4), (2, 5), (2, 6)),
            labels={node: f"N{node}" for node in range(7)},
        )

    def test_poincare_node_embeddings_encode_depth_inside_ball(self) -> None:
        tree = self._tree()

        embeddings = poincare_node_embeddings_from_tree(tree, curvature=1.0, radial_scale=0.35)

        self.assertEqual(embeddings.shape, (7, 2))
        self.assertTrue(torch.all(torch.linalg.vector_norm(embeddings, dim=1) < 1.0))
        self.assertAlmostEqual(float(torch.linalg.vector_norm(embeddings[0]).item()), 0.0, places=8)
        self.assertGreater(
            float(torch.linalg.vector_norm(embeddings[3]).item()),
            float(torch.linalg.vector_norm(embeddings[1]).item()),
        )

    def test_label_hash_node_embeddings_are_aligned_across_trees(self) -> None:
        left = RawAstTree.from_edges(
            root_id=0,
            edges=((0, 1), (0, 2)),
            labels={0: "Root", 1: "Identifier", 2: "Literal"},
        )
        right = RawAstTree.from_edges(
            root_id=10,
            edges=((10, 11), (10, 12)),
            labels={10: "Root", 11: "Identifier", 12: "Literal"},
        )

        left_embeddings = poincare_node_embeddings_from_tree(left, angle_mode="label_hash")
        right_embeddings = poincare_node_embeddings_from_tree(right, angle_mode="label_hash")

        self.assertTrue(torch.allclose(left_embeddings[1], right_embeddings[11]))
        self.assertTrue(torch.allclose(left_embeddings[2], right_embeddings[12]))
        self.assertFalse(torch.allclose(left_embeddings[1], left_embeddings[2]))

    def test_path_signature_hash_distinguishes_same_label_in_different_branches(self) -> None:
        tree = RawAstTree.from_edges(
            root_id=0,
            edges=((0, 1), (0, 2), (1, 3), (2, 4)),
            labels={0: "Root", 1: "If", 2: "For", 3: "Identifier", 4: "Identifier"},
        )

        embeddings = poincare_node_embeddings_from_tree(tree, angle_mode="path_signature_hash")

        self.assertFalse(torch.allclose(embeddings[3], embeddings[4]))

    def test_depth_only_keeps_same_depth_nodes_on_same_ray(self) -> None:
        tree = self._tree()

        embeddings = poincare_node_embeddings_from_tree(tree, angle_mode="depth_only")

        self.assertTrue(torch.allclose(embeddings[3], embeddings[4]))
        self.assertGreater(float(torch.linalg.vector_norm(embeddings[3])), float(torch.linalg.vector_norm(embeddings[1])))

    def test_path_measure_represents_each_path_by_lca_start_and_end(self) -> None:
        tree = self._tree()
        paths = (tree.path_between(3, 4), tree.path_between(3, 5))
        embeddings = poincare_node_embeddings_from_tree(tree, curvature=1.0)

        measure = poincare_path_measure_from_paths(tree, paths, node_embeddings=embeddings, curvature=1.0)

        self.assertEqual(measure.points.shape, (2, 3, 2))
        self.assertTrue(torch.allclose(measure.mass, torch.full((2,), 0.5)))
        self.assertTrue(torch.allclose(measure.points[0, 0], embeddings[1]))
        self.assertTrue(torch.allclose(measure.points[0, 1], embeddings[3]))
        self.assertTrue(torch.allclose(measure.points[0, 2], embeddings[4]))
        self.assertTrue(torch.allclose(measure.points[1, 0], embeddings[0]))

    def test_lca_anchored_cost_is_zero_for_reversed_same_unoriented_path(self) -> None:
        tree = self._tree()
        embeddings = poincare_node_embeddings_from_tree(tree, curvature=1.0)
        path = tree.path_between(3, 5)
        reversed_path = RawAstPath(start=path.end, end=path.start, nodes=tuple(reversed(path.nodes)))
        left = poincare_path_measure_from_paths(tree, (path,), node_embeddings=embeddings, curvature=1.0)
        right = poincare_path_measure_from_paths(tree, (reversed_path,), node_embeddings=embeddings, curvature=1.0)

        unoriented = lca_anchored_product_cost_matrix(left, right, unoriented=True)
        oriented = lca_anchored_product_cost_matrix(left, right, unoriented=False)

        self.assertAlmostEqual(float(unoriented.item()), 0.0, places=8)
        self.assertGreater(float(oriented.item()), 0.0)

    def test_sinkhorn_path_measure_distance_is_zero_for_identical_measure(self) -> None:
        tree = self._tree()
        embeddings = poincare_node_embeddings_from_tree(tree, curvature=1.0)
        paths = (tree.path_between(3, 4), tree.path_between(3, 5), tree.path_between(4, 6))
        measure = poincare_path_measure_from_paths(tree, paths, node_embeddings=embeddings, curvature=1.0)

        distance = sinkhorn_path_measure_distance(measure, measure, epsilon=0.05, iterations=80)

        self.assertLess(abs(float(distance.item())), 1e-5)

    def test_sinkhorn_path_measure_distance_normalizes_large_ground_costs(self) -> None:
        tree = self._tree()
        embeddings = poincare_node_embeddings_from_tree(
            tree,
            curvature=1.0,
            radial_scale=1.4,
            angle_mode="label_hash",
        )
        left = poincare_path_measure_from_paths(
            tree,
            (tree.path_between(3, 4), tree.path_between(3, 5)),
            node_embeddings=embeddings,
            curvature=1.0,
        )
        right = poincare_path_measure_from_paths(
            tree,
            (tree.path_between(4, 6), tree.path_between(5, 6)),
            node_embeddings=embeddings,
            curvature=1.0,
        )

        distance = sinkhorn_path_measure_distance(left, right, epsilon=0.05, iterations=80)

        self.assertTrue(torch.isfinite(distance))

    def test_gromov_product_at_origin_matches_basic_identities(self) -> None:
        point = torch.tensor([[0.2, 0.0]], dtype=torch.float32)
        origin = torch.zeros_like(point)

        same = poincare_gromov_product_at_origin(point, point, curvature=1.0)
        with_origin = poincare_gromov_product_at_origin(point, origin, curvature=1.0)

        self.assertGreater(float(same), 0.0)
        self.assertAlmostEqual(float(with_origin), 0.0, places=5)

    def test_euclidean_path_measure_matches_product_protocol(self) -> None:
        left = EuclideanPathMeasure(
            points=torch.tensor([[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]]),
            mass=torch.tensor([2.0]),
        )
        right_reversed = EuclideanPathMeasure(
            points=torch.tensor([[[0.0, 0.0], [0.0, 1.0], [1.0, 0.0]]]),
            mass=torch.tensor([3.0]),
        )

        unoriented = euclidean_lca_anchored_product_cost_matrix(left, right_reversed, unoriented=True)
        oriented = euclidean_lca_anchored_product_cost_matrix(left, right_reversed, unoriented=False)

        self.assertAlmostEqual(float(unoriented.item()), 0.0, places=8)
        self.assertGreater(float(oriented.item()), 0.0)
        self.assertAlmostEqual(float(left.mass.sum()), 1.0, places=8)

    def test_sinkhorn_euclidean_path_measure_distance_is_zero_for_identical_measure(self) -> None:
        measure = EuclideanPathMeasure(
            points=torch.tensor(
                [
                    [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
                    [[0.0, 0.0], [2.0, 0.0], [0.0, 2.0]],
                ],
                dtype=torch.float32,
            ),
            mass=torch.ones(2),
        )

        distance = sinkhorn_euclidean_path_measure_distance(measure, measure, epsilon=0.05, iterations=80)

        self.assertLess(abs(float(distance.item())), 1e-5)


if __name__ == "__main__":
    unittest.main()
