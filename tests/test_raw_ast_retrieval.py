from __future__ import annotations

import unittest

from geometry_profile_research.java_raw_ast import parse_java_ast_tree
from geometry_profile_research.python_raw_ast import parse_python_ast_tree
from geometry_profile_research.raw_ast import find_nodes_by_label, terminal_to_terminal_paths
from geometry_profile_research.raw_ast_retrieval import (
    RawASTRetrievalItem,
    alpha_rename_tree,
    build_retrieval_triples,
    callable_subtrees,
    make_positive_tree,
    retrieval_item_trees,
    select_hard_negative,
    structural_gap,
    structural_noop_tree,
    subtree_as_tree,
    terminal_jaccard_similarity,
)


JAVA_ANCHOR = """
class Demo {
    int absLike(int x) {
        if (x > 0) {
            return x;
        }
        return -x;
    }
}
"""

JAVA_LEXICALLY_SIMILAR_DIFFERENT_STRUCTURE = """
class Demo {
    int absLike(int x) {
        while (x > 0) {
            x = x - 1;
        }
        return x;
    }
}
"""

JAVA_UNRELATED = """
class Other {
    String join(String left, String right) {
        return left + right;
    }
}
"""

PYTHON_ANCHOR = """
def abs_like(x):
    if x > 0:
        return x
    return -x
"""

PYTHON_TOP_LEVEL_PROGRAM = """
total = 0
for value in values:
    if value > 0:
        total += value
print(total)
"""


class RawASTRetrievalTests(unittest.TestCase):
    def test_alpha_rename_preserves_topology_and_replaces_identifier_tokens(self) -> None:
        tree = parse_java_ast_tree(JAVA_ANCHOR)

        renamed = alpha_rename_tree(tree)

        self.assertEqual(renamed.root_id, tree.root_id)
        self.assertEqual(renamed.parent_by_node, tree.parent_by_node)
        self.assertEqual(renamed.children_by_node, tree.children_by_node)
        self.assertEqual(renamed.labels, tree.labels)
        original_tokens = _terminal_tokens(tree)
        renamed_tokens = _terminal_tokens(renamed)
        self.assertIn("x", original_tokens)
        self.assertNotIn("x", renamed_tokens)
        self.assertIn("id_0", renamed_tokens)
        self.assertIn(">", renamed_tokens)
        self.assertIn("0", renamed_tokens)

    def test_similarity_and_gap_separate_lexical_and_structural_axes(self) -> None:
        anchor = parse_java_ast_tree(JAVA_ANCHOR)
        similar_structure = alpha_rename_tree(anchor)
        different_structure = parse_java_ast_tree(JAVA_LEXICALLY_SIMILAR_DIFFERENT_STRUCTURE)

        self.assertGreater(terminal_jaccard_similarity(anchor, different_structure), 0.35)
        self.assertEqual(structural_gap(anchor, similar_structure), 0.0)
        self.assertGreater(structural_gap(anchor, different_structure), 0.05)

    def test_select_hard_negative_prefers_lexically_similar_structurally_different_tree(self) -> None:
        anchor = RawASTRetrievalItem("anchor", parse_java_ast_tree(JAVA_ANCHOR))
        hard = RawASTRetrievalItem("hard", parse_java_ast_tree(JAVA_LEXICALLY_SIMILAR_DIFFERENT_STRUCTURE))
        unrelated = RawASTRetrievalItem("unrelated", parse_java_ast_tree(JAVA_UNRELATED))

        selected = select_hard_negative(anchor, (unrelated, hard), min_structural_gap=0.05)

        self.assertEqual(selected.item_id, "hard")

    def test_build_retrieval_triples_returns_safe_positive_and_hard_negative(self) -> None:
        anchor = RawASTRetrievalItem("anchor", parse_java_ast_tree(JAVA_ANCHOR))
        hard = RawASTRetrievalItem("hard", parse_java_ast_tree(JAVA_LEXICALLY_SIMILAR_DIFFERENT_STRUCTURE))
        unrelated = RawASTRetrievalItem("unrelated", parse_java_ast_tree(JAVA_UNRELATED))

        triples = build_retrieval_triples((anchor, hard, unrelated), min_structural_gap=0.05)

        self.assertEqual(len(triples), 3)
        first_anchor, first_positive, first_negative = triples[0]
        self.assertIs(first_anchor, anchor.tree)
        self.assertEqual(first_positive.parent_by_node, anchor.tree.parent_by_node)
        self.assertIsNot(first_positive, anchor.tree)
        self.assertIs(first_negative, hard.tree)

    def test_structural_noop_positive_changes_structure_but_preserves_terminal_tokens(self) -> None:
        tree = callable_subtrees(parse_python_ast_tree(PYTHON_ANCHOR))[0]

        positive = structural_noop_tree(tree)

        self.assertGreater(len(positive.preorder()), len(tree.preorder()))
        self.assertGreater(structural_gap(tree, positive), 0.0)
        self.assertEqual(terminal_jaccard_similarity(tree, positive), 1.0)
        self.assertIn("SyntheticNoOp", positive.labels.values())
        self.assertNotEqual(_path_label_sequences(tree), _path_label_sequences(positive))

    def test_build_retrieval_triples_supports_structural_positive_mode(self) -> None:
        anchor = RawASTRetrievalItem("anchor", parse_java_ast_tree(JAVA_ANCHOR))
        hard = RawASTRetrievalItem("hard", parse_java_ast_tree(JAVA_LEXICALLY_SIMILAR_DIFFERENT_STRUCTURE))
        unrelated = RawASTRetrievalItem("unrelated", parse_java_ast_tree(JAVA_UNRELATED))

        triples = build_retrieval_triples(
            (anchor, hard, unrelated),
            min_structural_gap=0.05,
            positive_mode="alpha_structural_noop",
        )

        _, first_positive, _ = triples[0]
        self.assertGreater(structural_gap(anchor.tree, first_positive), 0.0)
        self.assertNotEqual(_terminal_tokens(anchor.tree), _terminal_tokens(first_positive))

    def test_make_positive_tree_rejects_unknown_mode(self) -> None:
        tree = parse_java_ast_tree(JAVA_ANCHOR)

        with self.assertRaises(ValueError):
            make_positive_tree(tree, mode="unknown")

    def test_callable_subtrees_extract_method_level_trees(self) -> None:
        tree = parse_java_ast_tree(JAVA_ANCHOR)
        method_node = find_nodes_by_label(tree, "MethodDeclaration")[0]

        subtree = subtree_as_tree(tree, method_node)
        callables = callable_subtrees(tree)

        self.assertEqual(subtree.root_id, method_node)
        self.assertEqual(subtree.labels[subtree.root_id], "MethodDeclaration")
        self.assertNotIn(tree.root_id, subtree.parent_by_node)
        self.assertEqual(len(callables), 1)
        self.assertEqual(callables[0].labels[callables[0].root_id], "MethodDeclaration")

    def test_callable_subtrees_support_python_functions(self) -> None:
        tree = parse_python_ast_tree(PYTHON_ANCHOR)

        callables = callable_subtrees(tree)

        self.assertEqual(len(callables), 1)
        self.assertEqual(callables[0].labels[callables[0].root_id], "FunctionDef")
        self.assertEqual(callables[0].attributes[callables[0].root_id]["name"], "abs_like")

    def test_retrieval_item_trees_can_use_module_scope_for_top_level_programs(self) -> None:
        tree = parse_python_ast_tree(PYTHON_TOP_LEVEL_PROGRAM)

        self.assertEqual(callable_subtrees(tree), ())
        self.assertEqual(retrieval_item_trees(tree, item_scope="callable"), ())
        module_items = retrieval_item_trees(tree, item_scope="module")
        fallback_items = retrieval_item_trees(tree, item_scope="callable_or_module")

        self.assertEqual(len(module_items), 1)
        self.assertIs(module_items[0], tree)
        self.assertEqual(fallback_items, module_items)
        self.assertGreaterEqual(len(terminal_to_terminal_paths(module_items[0], max_paths=8)), 2)


def _terminal_tokens(tree):
    return {
        tree.attributes[node].get("terminal_token")
        for node in find_nodes_by_label(tree, "TerminalToken")
    }


def _path_label_sequences(tree):
    return tuple(
        tuple(tree.labels.get(node, "") for node in path.nodes)
        for path in terminal_to_terminal_paths(tree, max_paths=8)
    )


if __name__ == "__main__":
    unittest.main()
