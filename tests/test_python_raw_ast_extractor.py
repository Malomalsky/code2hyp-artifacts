from __future__ import annotations

import unittest

from geometry_profile_research.python_raw_ast import parse_python_ast_tree
from geometry_profile_research.raw_ast import find_nodes_by_label, gromov_lca_depth, leaf_node_ids


PYTHON_FUNCTION = """
def abs_like(x):
    if x > 0:
        return x
    return -x
"""


class PythonRawAstExtractorTests(unittest.TestCase):
    def test_parse_python_source_builds_raw_ast_with_function_scope(self) -> None:
        tree = parse_python_ast_tree(PYTHON_FUNCTION)

        function_nodes = find_nodes_by_label(tree, "FunctionDef")
        return_nodes = find_nodes_by_label(tree, "Return")
        leaves = leaf_node_ids(tree)

        self.assertEqual(tree.labels[tree.root_id], "Module")
        self.assertEqual(len(function_nodes), 1)
        self.assertEqual(len(return_nodes), 2)
        self.assertGreater(len(leaves), 0)
        self.assertEqual(tree.attributes[function_nodes[0]]["name"], "abs_like")

        first_return, second_return = return_nodes
        ancestor = tree.lca(first_return, second_return)
        self.assertEqual(tree.labels[ancestor], "FunctionDef")
        self.assertEqual(gromov_lca_depth(tree, first_return, second_return), float(tree.depth(ancestor)))

    def test_parse_python_source_preserves_edge_and_terminal_metadata(self) -> None:
        tree = parse_python_ast_tree(PYTHON_FUNCTION)

        terminal_nodes = find_nodes_by_label(tree, "TerminalToken")
        terminal_tokens = {
            tree.attributes[node].get("terminal_token")
            for node in terminal_nodes
        }

        self.assertIn("abs_like", terminal_tokens)
        self.assertIn("x", terminal_tokens)
        self.assertIn("0", terminal_tokens)
        for node in tree.preorder():
            self.assertIn("node_type", tree.attributes[node])
            if node != tree.root_id:
                self.assertIn("edge_type", tree.attributes[node])
                self.assertIn("child_index", tree.attributes[node])
        self.assertIn("source_span", tree.attributes[find_nodes_by_label(tree, "FunctionDef")[0]])


if __name__ == "__main__":
    unittest.main()
