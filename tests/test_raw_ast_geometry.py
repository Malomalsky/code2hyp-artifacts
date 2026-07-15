from __future__ import annotations

import unittest

from geometry_profile_research.raw_ast import (
    RawAstPath,
    RawAstTree,
    edge_jaccard_distance,
    edge_symmetric_difference_distance,
    gromov_lca_depth,
    oriented_endpoint_distance,
    raw_ast_order_relation_records,
    raw_ast_path_relation_matrices,
    raw_ast_path_relation_records,
    terminal_to_terminal_paths,
    unoriented_endpoint_distance,
)


def _toy_java_method_tree() -> RawAstTree:
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
            (1, 8),
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
            8: "CallExpression",
        },
    )


class RawAstGeometryTests(unittest.TestCase):
    def test_lca_depth_is_exact_gromov_product_in_tree_metric(self) -> None:
        tree = _toy_java_method_tree()

        self.assertEqual(tree.lca(5, 7), 2)
        self.assertEqual(tree.depth(2), 2)
        self.assertEqual(tree.tree_distance(5, 7), 4)
        self.assertEqual(gromov_lca_depth(tree, 5, 7), 2.0)

    def test_path_between_endpoints_is_additive_through_lca(self) -> None:
        tree = _toy_java_method_tree()
        path = tree.path_between(5, 7)

        self.assertEqual(path, RawAstPath(start=5, end=7, nodes=(5, 4, 2, 6, 7)))
        self.assertEqual(path.lca(tree), 2)
        self.assertEqual(path.length, 4)
        self.assertEqual(
            tree.tree_distance(path.start, path.end),
            tree.tree_distance(path.start, path.lca(tree)) + tree.tree_distance(path.lca(tree), path.end),
        )
        self.assertEqual(path.reversed(), RawAstPath(start=7, end=5, nodes=(7, 6, 2, 4, 5)))

    def test_edge_overlap_distinguishes_paths_beyond_endpoint_distance(self) -> None:
        tree = _toy_java_method_tree()
        left = tree.path_between(5, 7)
        right = tree.path_between(5, 3)

        self.assertEqual(left.undirected_edges, frozenset({(4, 5), (2, 4), (2, 6), (6, 7)}))
        self.assertEqual(right.undirected_edges, frozenset({(4, 5), (2, 4), (2, 3)}))
        self.assertEqual(edge_symmetric_difference_distance(left, right), 3)
        self.assertAlmostEqual(edge_jaccard_distance(left, right), 0.6)

    def test_endpoint_product_distance_keeps_orientation_explicit(self) -> None:
        tree = _toy_java_method_tree()
        path = tree.path_between(5, 7)
        same_left_different_right = tree.path_between(5, 3)
        reversed_candidate = same_left_different_right.reversed()

        self.assertEqual(oriented_endpoint_distance(tree, path, same_left_different_right), 3)
        self.assertEqual(oriented_endpoint_distance(tree, path, reversed_candidate), 7)
        self.assertEqual(unoriented_endpoint_distance(tree, path, reversed_candidate), 3)

    def test_terminal_to_terminal_paths_are_deterministic_leaf_pairs(self) -> None:
        tree = _toy_java_method_tree()

        paths = terminal_to_terminal_paths(tree)
        limited_paths = terminal_to_terminal_paths(tree, max_paths=3)

        self.assertEqual([path.nodes for path in limited_paths], [(3, 2, 4, 5), (3, 2, 6, 7), (3, 2, 1, 8)])
        self.assertEqual(len(paths), 6)
        self.assertTrue(all(path.length == tree.tree_distance(path.start, path.end) for path in paths))

    def test_terminal_to_terminal_paths_support_hash_sorted_sensitivity_policy(self) -> None:
        tree = _toy_java_method_tree()

        first = terminal_to_terminal_paths(tree, max_paths=3, selection_policy="hash_sorted")
        second = terminal_to_terminal_paths(tree, max_paths=3, selection_policy="hash_sorted")
        preorder = terminal_to_terminal_paths(tree, max_paths=3)

        self.assertEqual(first, second)
        self.assertEqual(len(first), 3)
        self.assertNotEqual([path.nodes for path in first], [path.nodes for path in preorder])

    def test_terminal_to_terminal_paths_support_lca_depth_stratified_policy(self) -> None:
        tree = _toy_java_method_tree()

        paths = terminal_to_terminal_paths(tree, max_paths=4, selection_policy="lca_depth_stratified")
        lca_depths = [tree.depth(path.lca(tree)) for path in paths]

        self.assertEqual(len(paths), 4)
        self.assertEqual(lca_depths.count(1), 2)
        self.assertEqual(lca_depths.count(2), 2)

    def test_affine_lca_sampler_is_deterministic_unique_and_depth_stratified(self) -> None:
        tree = _toy_java_method_tree()

        first = terminal_to_terminal_paths(tree, max_paths=4, selection_policy="lca_depth_affine_sampled")
        second = terminal_to_terminal_paths(tree, max_paths=4, selection_policy="lca_depth_affine_sampled")
        endpoint_pairs = [(path.start, path.end) for path in first]
        lca_depths = [tree.depth(path.lca(tree)) for path in first]

        self.assertEqual(first, second)
        self.assertEqual(len(first), 4)
        self.assertEqual(len(set(endpoint_pairs)), 4)
        self.assertEqual(lca_depths.count(1), 2)
        self.assertEqual(lca_depths.count(2), 2)

    def test_affine_lca_sampler_returns_all_pairs_when_capacity_is_smaller_than_k(self) -> None:
        tree = _toy_java_method_tree()

        paths = terminal_to_terminal_paths(tree, max_paths=64, selection_policy="lca_depth_affine_sampled")
        endpoint_pairs = {frozenset((path.start, path.end)) for path in paths}

        self.assertEqual(len(paths), 6)
        self.assertEqual(len(endpoint_pairs), 6)

    def test_affine_lca_sampler_ignores_terminal_values(self) -> None:
        edges = [(0, 1), (0, 2), (0, 3), (0, 4), (0, 5)]
        labels = {0: "Module", 1: "TerminalToken", 2: "TerminalToken", 3: "TerminalToken", 4: "TerminalToken", 5: "TerminalToken"}
        first_tree = RawAstTree.from_edges(
            root_id=0,
            edges=edges,
            labels=labels,
            attributes={node: {"terminal_token": f"name_{node}"} for node in range(1, 6)},
        )
        renamed_tree = RawAstTree.from_edges(
            root_id=0,
            edges=edges,
            labels=labels,
            attributes={node: {"terminal_token": f"renamed_{node}"} for node in range(1, 6)},
        )

        first = terminal_to_terminal_paths(first_tree, max_paths=6, selection_policy="lca_depth_affine_sampled")
        renamed = terminal_to_terminal_paths(renamed_tree, max_paths=6, selection_policy="lca_depth_affine_sampled")

        self.assertEqual([(path.start, path.end) for path in first], [(path.start, path.end) for path in renamed])

    def test_affine_lca_sampler_scales_without_materializing_all_leaf_pairs(self) -> None:
        leaf_count = 5_000
        tree = RawAstTree.from_edges(
            root_id=0,
            edges=[(0, node) for node in range(1, leaf_count + 1)],
            labels={0: "Module", **{node: "TerminalToken" for node in range(1, leaf_count + 1)}},
        )

        paths = terminal_to_terminal_paths(tree, max_paths=64, selection_policy="lca_depth_affine_sampled")

        self.assertEqual(len(paths), 64)
        self.assertEqual(len({(path.start, path.end) for path in paths}), 64)
        self.assertTrue(all(path.nodes[1] == 0 for path in paths))

    def test_terminal_to_terminal_paths_can_be_restricted_to_a_subtree(self) -> None:
        tree = _toy_java_method_tree()

        subtree_leaves = terminal_to_terminal_paths(tree, root_id=2)

        self.assertEqual([path.nodes for path in subtree_leaves], [(3, 2, 4, 5), (3, 2, 6, 7), (5, 4, 2, 6, 7)])

    def test_path_relation_matrices_preserve_orientation_and_true_lca_depths(self) -> None:
        tree = _toy_java_method_tree()
        left = tree.path_between(5, 7)
        reversed_left = left.reversed()
        sibling_branch = tree.path_between(5, 3)

        matrices = raw_ast_path_relation_matrices(tree, (left, reversed_left, sibling_branch))

        self.assertEqual(matrices.oriented_endpoint_distance[0][0], 0)
        self.assertEqual(matrices.oriented_endpoint_distance[0][1], 8)
        self.assertEqual(matrices.unoriented_endpoint_distance[0][1], 0)
        self.assertEqual(matrices.edge_symmetric_difference[0][2], 3)
        self.assertAlmostEqual(matrices.edge_jaccard_distance[0][2], 0.6)
        self.assertEqual(matrices.lca_depth[0], 2)
        self.assertEqual(matrices.lca_depth[2], 2)
        self.assertEqual(matrices.path_length[0], 4)
        self.assertEqual(matrices.path_length[2], 3)
        self.assertEqual(matrices.path_length_difference[0][2], 1)

    def test_path_relation_records_are_json_friendly_upper_triangle(self) -> None:
        tree = _toy_java_method_tree()
        paths = (tree.path_between(5, 7), tree.path_between(5, 3), tree.path_between(3, 8))

        records = raw_ast_path_relation_records(tree, paths)

        self.assertEqual([(record.left_index, record.right_index) for record in records], [(0, 1), (0, 2), (1, 2)])
        self.assertEqual(records[0].left_lca_depth, 2)
        self.assertEqual(records[0].right_lca_depth, 2)
        self.assertEqual(records[0].path_length_difference, 1)
        self.assertEqual(records[0].as_dict()["edge_symmetric_difference"], 3)
        self.assertIsInstance(records[0].as_dict()["edge_jaccard_distance"], float)

    def test_order_relation_records_encode_ancestor_descendant_pairs_in_subtree(self) -> None:
        tree = _toy_java_method_tree()

        records = raw_ast_order_relation_records(tree, root_id=2)

        pairs = [(record.ancestor, record.descendant) for record in records]
        self.assertEqual(pairs[:7], [(2, 3), (2, 4), (2, 5), (4, 5), (2, 6), (2, 7), (6, 7)])
        self.assertTrue(all(record.label == 1 for record in records))
        self.assertEqual(records[2].ancestor_depth, 2)
        self.assertEqual(records[2].descendant_depth, 4)
        self.assertEqual(records[2].tree_distance, 2)
        self.assertFalse(records[2].is_direct_edge)
        self.assertTrue(records[3].is_direct_edge)
        self.assertEqual(records[0].as_dict()["ancestor_label"], "IfStatement")

    def test_order_relation_records_add_deterministic_incomparable_controls(self) -> None:
        tree = _toy_java_method_tree()

        records = raw_ast_order_relation_records(tree, root_id=2, include_incomparable=True, max_records=9)

        labels = [record.label for record in records]
        pairs = [(record.ancestor, record.descendant) for record in records]
        self.assertEqual(labels.count(1), 7)
        self.assertEqual(labels.count(0), 2)
        self.assertEqual(pairs[-2:], [(3, 4), (3, 5)])
        self.assertEqual(records[-1].ancestor_depth, 3)
        self.assertEqual(records[-1].descendant_depth, 4)
        self.assertEqual(records[-1].tree_distance, 3)
        self.assertFalse(records[-1].is_direct_edge)


if __name__ == "__main__":
    unittest.main()
