from __future__ import annotations

import unittest

import torch

from geometry_profile_research.raw_ast import RawAstTree
from geometry_profile_research.raw_ast_code2hyp import (
    RawASTCode2Hyp,
    RawASTMethodMeasure,
    _root_to_node_tokens,
    build_raw_ast_token_vocab,
)
from geometry_profile_research.java_raw_ast import parse_java_ast_tree


class RawASTCode2HypTests(unittest.TestCase):
    def _tree(self) -> RawAstTree:
        return RawAstTree.from_edges(
            root_id=0,
            edges=((0, 1), (0, 2), (1, 3), (1, 4), (2, 5), (2, 6)),
            labels={
                0: "Method",
                1: "IfStatement",
                2: "ReturnStatement",
                3: "Identifier",
                4: "Literal",
                5: "Identifier",
                6: "Literal",
            },
        )

    def test_build_vocab_covers_node_and_child_edge_tokens(self) -> None:
        tree = self._tree()

        vocab = build_raw_ast_token_vocab((tree,))

        self.assertIn("node:Method", vocab)
        self.assertIn("node:Identifier", vocab)
        self.assertIn("edge:child_0", vocab)
        self.assertIn("edge:child_1", vocab)

    def test_build_vocab_uses_explicit_edge_metadata_when_available(self) -> None:
        tree = RawAstTree.from_edges(
            root_id=0,
            edges=((0, 1),),
            labels={0: "Method", 1: "Parameter"},
            attributes={1: {"edge_type": "parameters", "child_index": "0"}},
        )

        vocab = build_raw_ast_token_vocab((tree,))

        self.assertIn("edge:parameters:0", vocab)

    def test_vocab_can_include_terminal_token_values_for_downstream_tasks(self) -> None:
        tree = parse_java_ast_tree("class Demo { int value(int x) { return x + 1; } }")

        structural_vocab = build_raw_ast_token_vocab((tree,), terminal_policy="type")
        lexical_vocab = build_raw_ast_token_vocab((tree,), terminal_policy="value")

        self.assertNotIn("terminal:x", structural_vocab)
        self.assertIn("terminal:x", lexical_vocab)
        self.assertIn("terminal:1", lexical_vocab)

    def test_node_input_modes_control_prefix_information(self) -> None:
        tree = self._tree()

        label_only = _root_to_node_tokens(tree, 4, input_mode="label_only")
        label_depth = _root_to_node_tokens(tree, 4, input_mode="label_depth")
        label_depth_prefix = _root_to_node_tokens(tree, 4, input_mode="label_depth_prefix")

        self.assertEqual(label_only, ("node:Literal",))
        self.assertEqual(label_depth, ("node:Literal", "depth:2"))
        self.assertIn("edge:child_0", label_depth_prefix)
        self.assertIn("depth:2", label_depth_prefix)

    def test_build_vocab_respects_node_input_mode_depth_tokens(self) -> None:
        tree = self._tree()

        label_only_vocab = build_raw_ast_token_vocab((tree,), node_input_mode="label_only")
        label_depth_vocab = build_raw_ast_token_vocab((tree,), node_input_mode="label_depth")

        self.assertNotIn("depth:2", label_only_vocab)
        self.assertIn("depth:2", label_depth_vocab)

    def test_encode_method_returns_uniform_lca_product_measure(self) -> None:
        tree = self._tree()
        vocab = build_raw_ast_token_vocab((tree,))
        model = RawASTCode2Hyp(vocab, dim=4, manifold="poincare", max_paths=4)

        measure = model.encode_method(tree)

        self.assertEqual(measure.points.ndim, 3)
        self.assertEqual(measure.points.shape[1], 3)
        self.assertEqual(measure.left_branch.shape, measure.right_branch.shape)
        self.assertEqual(measure.left_branch.shape[0], measure.points.shape[0])
        self.assertAlmostEqual(float(measure.mass.sum()), 1.0, places=6)
        self.assertTrue(torch.all(measure.mass > 0.0))

    def test_encode_product_measure_skips_branch_encoder(self) -> None:
        tree = self._tree()
        vocab = build_raw_ast_token_vocab((tree,))
        model = RawASTCode2Hyp(vocab, dim=4, manifold="euclidean", max_paths=4)

        def fail_if_called(*args, **kwargs):
            raise AssertionError("branch encoder must not run for point-only Stage A measures")

        model._encode_branch = fail_if_called
        measure = model.encode_product_measure(tree)

        self.assertEqual(measure.points.shape[1], 3)
        self.assertIsNone(measure.side_features)
        self.assertAlmostEqual(float(measure.mass.sum()), 1.0, places=6)

    def test_encode_method_supports_single_point_path_object(self) -> None:
        tree = self._tree()
        vocab = build_raw_ast_token_vocab((tree,))
        model = RawASTCode2Hyp(vocab, dim=4, manifold="euclidean", max_paths=4, path_object_mode="single_point")

        measure = model.encode_method(tree)

        self.assertEqual(measure.points.ndim, 3)
        self.assertEqual(measure.points.shape[1], 1)
        self.assertEqual(measure.path_object_mode, "single_point")

    def test_anchor_modes_change_only_the_lca_role(self) -> None:
        tree = self._tree()
        vocab = build_raw_ast_token_vocab((tree,))
        model = RawASTCode2Hyp(vocab, dim=4, manifold="euclidean", max_paths=4, anchor_mode="true_lca")

        true_lca = model.encode_method(tree)
        model.anchor_mode = "zero_anchor"
        zero_anchor = model.encode_method(tree)
        model.anchor_mode = "root_anchor"
        root_anchor = model.encode_method(tree)

        torch.testing.assert_close(true_lca.points[:, 1:], zero_anchor.points[:, 1:])
        torch.testing.assert_close(true_lca.points[:, 1:], root_anchor.points[:, 1:])
        torch.testing.assert_close(zero_anchor.points[:, 0], torch.zeros_like(zero_anchor.points[:, 0]))
        torch.testing.assert_close(root_anchor.points[:, 0], root_anchor.points[0, 0].expand_as(root_anchor.points[:, 0]))

    def test_depth_matched_shuffled_anchor_is_deterministic(self) -> None:
        tree = self._tree()
        vocab = build_raw_ast_token_vocab((tree,))
        model = RawASTCode2Hyp(
            vocab,
            dim=4,
            manifold="euclidean",
            max_paths=4,
            anchor_mode="depth_matched_shuffled",
        )

        first = model.encode_method(tree)
        second = model.encode_method(tree)

        torch.testing.assert_close(first.points, second.points)

    def test_method_distance_is_symmetric_and_zero_on_identical_measure(self) -> None:
        tree = self._tree()
        vocab = build_raw_ast_token_vocab((tree,))
        model = RawASTCode2Hyp(vocab, dim=4, manifold="euclidean", max_paths=4)

        measure = model.encode_method(tree)
        same = model.method_distance(measure, measure, sinkhorn_iterations=40)

        self.assertLess(abs(float(same.detach())), 1e-5)

    def test_centroid_method_distance_is_symmetric_and_zero_on_identical_measure(self) -> None:
        tree = self._tree()
        vocab = build_raw_ast_token_vocab((tree,))
        model = RawASTCode2Hyp(vocab, dim=4, manifold="euclidean", max_paths=4, method_aggregation="centroid")

        measure = model.encode_method(tree)
        same = model.method_distance(measure, measure)

        self.assertLess(abs(float(same.detach())), 1e-5)

    def test_path_cost_preserves_left_right_branch_order(self) -> None:
        vocab = {"<pad>": 0, "<unk>": 1}
        model = RawASTCode2Hyp(vocab, dim=2, manifold="euclidean")
        points = torch.zeros((1, 3, 2), dtype=torch.float32)
        left = RawASTMethodMeasure(
            points=points,
            left_branch=torch.tensor([[1.0, 0.0]]),
            right_branch=torch.tensor([[0.0, 1.0]]),
            mass=torch.ones(1),
            manifold="euclidean",
        )
        swapped = RawASTMethodMeasure(
            points=points,
            left_branch=torch.tensor([[0.0, 1.0]]),
            right_branch=torch.tensor([[1.0, 0.0]]),
            mass=torch.ones(1),
            manifold="euclidean",
        )

        cost = model._path_cost_matrix(left, swapped)

        self.assertGreater(float(cost.item()), 0.0)

    def test_unoriented_path_cost_identifies_reversed_lca_product_path(self) -> None:
        vocab = {"<pad>": 0, "<unk>": 1}
        model = RawASTCode2Hyp(vocab, dim=2, manifold="euclidean", path_cost_orientation="unoriented")
        left = RawASTMethodMeasure(
            points=torch.tensor([[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]], dtype=torch.float32),
            left_branch=torch.tensor([[1.0, 0.0]]),
            right_branch=torch.tensor([[0.0, 1.0]]),
            mass=torch.ones(1),
            manifold="euclidean",
        )
        reversed_path = RawASTMethodMeasure(
            points=torch.tensor([[[0.0, 0.0], [0.0, 1.0], [1.0, 0.0]]], dtype=torch.float32),
            left_branch=torch.tensor([[0.0, 1.0]]),
            right_branch=torch.tensor([[1.0, 0.0]]),
            mass=torch.ones(1),
            manifold="euclidean",
        )

        cost = model._path_cost_matrix(left, reversed_path)

        self.assertAlmostEqual(float(cost.item()), 0.0, places=6)

    def test_unoriented_centroid_distance_identifies_reversed_lca_product_path(self) -> None:
        vocab = {"<pad>": 0, "<unk>": 1}
        model = RawASTCode2Hyp(
            vocab,
            dim=2,
            manifold="euclidean",
            method_aggregation="centroid",
            path_cost_orientation="unoriented",
        )
        left = RawASTMethodMeasure(
            points=torch.tensor([[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]], dtype=torch.float32),
            left_branch=torch.tensor([[1.0, 0.0]]),
            right_branch=torch.tensor([[0.0, 1.0]]),
            mass=torch.ones(1),
            manifold="euclidean",
        )
        reversed_path = RawASTMethodMeasure(
            points=torch.tensor([[[0.0, 0.0], [0.0, 1.0], [1.0, 0.0]]], dtype=torch.float32),
            left_branch=torch.tensor([[0.0, 1.0]]),
            right_branch=torch.tensor([[1.0, 0.0]]),
            mass=torch.ones(1),
            manifold="euclidean",
        )

        distance = model.method_distance(left, reversed_path)

        self.assertAlmostEqual(float(distance.item()), 0.0, places=6)

    def test_poincare_and_euclidean_controls_share_api(self) -> None:
        tree = self._tree()
        vocab = build_raw_ast_token_vocab((tree,))
        euclidean = RawASTCode2Hyp(vocab, dim=4, manifold="euclidean", max_paths=4)
        poincare = RawASTCode2Hyp(vocab, dim=4, manifold="poincare", max_paths=4)

        euclidean_measure = euclidean.encode_method(tree)
        poincare_measure = poincare.encode_method(tree)

        self.assertEqual(euclidean_measure.points.shape, poincare_measure.points.shape)
        self.assertEqual(euclidean_measure.left_branch.shape, poincare_measure.left_branch.shape)

    def test_model_uses_configured_terminal_policy(self) -> None:
        tree = parse_java_ast_tree("class Demo { int value(int x) { return x + 1; } }")
        vocab = build_raw_ast_token_vocab((tree,), terminal_policy="value")
        model = RawASTCode2Hyp(vocab, dim=4, manifold="euclidean", max_paths=4, terminal_policy="value")

        measure = model.encode_method(tree)

        self.assertEqual(measure.points.shape[1], 3)

    def test_model_supports_label_only_node_input_mode(self) -> None:
        tree = self._tree()
        vocab = build_raw_ast_token_vocab((tree,), node_input_mode="label_only")
        model = RawASTCode2Hyp(vocab, dim=4, manifold="euclidean", max_paths=4, node_input_mode="label_only")

        measure = model.encode_method(tree)

        self.assertEqual(measure.points.shape[1], 3)

    def test_training_loss_exposes_retrieval_edge_and_gromov_terms(self) -> None:
        tree = self._tree()
        vocab = build_raw_ast_token_vocab((tree,))
        model = RawASTCode2Hyp(vocab, dim=4, manifold="poincare", max_paths=4)

        loss = model.training_loss(((tree, tree, tree),), margin=0.1, sinkhorn_iterations=20)

        self.assertIn("loss", loss)
        self.assertIn("retrieval", loss)
        self.assertIn("edge", loss)
        self.assertIn("gromov_lca", loss)
        self.assertIn("gromov_lca_mean_abs_residual", loss)
        self.assertIn("branch_length", loss)
        self.assertIn("reversal", loss)
        self.assertTrue(torch.isfinite(loss["loss"]))
        self.assertTrue(torch.isfinite(loss["gromov_lca_mean_abs_residual"]))

        neutral = model.training_loss(
            ((tree, tree, tree),),
            margin=0.1,
            sinkhorn_iterations=20,
            lambda_retrieval=0.0,
        )
        expected_neutral = 0.1 * (
            neutral["edge"]
            + neutral["gromov_lca"]
            + neutral["branch_length"]
            + neutral["reversal"]
        )
        torch.testing.assert_close(neutral["loss"], expected_neutral.to(neutral["loss"].dtype))

        diagnostics = model.gromov_lca_diagnostics(tree)
        self.assertGreaterEqual(diagnostics["path_count"], 1.0)
        self.assertGreaterEqual(diagnostics["mean_abs_residual"], 0.0)
        self.assertGreaterEqual(diagnostics["max_abs_residual"], diagnostics["mean_abs_residual"])


if __name__ == "__main__":
    unittest.main()
