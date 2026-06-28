from __future__ import annotations

import unittest

from geometry_profile_research.java_raw_ast import leaf_node_ids, parse_java_ast_tree
from geometry_profile_research.raw_ast import find_nodes_by_label, gromov_lca_depth


JAVA_METHOD = """
class Demo {
    int absLike(int x) {
        if (x > 0) {
            return x;
        }
        return -x;
    }
}
"""


class JavaRawAstExtractorTests(unittest.TestCase):
    def test_parse_java_source_builds_raw_ast_with_true_lca_relations(self) -> None:
        tree = parse_java_ast_tree(JAVA_METHOD)

        method_nodes = find_nodes_by_label(tree, "MethodDeclaration")
        return_nodes = find_nodes_by_label(tree, "ReturnStatement")
        leaves = leaf_node_ids(tree)

        self.assertEqual(tree.labels[tree.root_id], "CompilationUnit")
        self.assertEqual(len(method_nodes), 1)
        self.assertEqual(len(return_nodes), 2)
        self.assertGreater(len(leaves), 0)
        self.assertEqual(tree.attributes[method_nodes[0]]["name"], "absLike")

        first_return, second_return = return_nodes
        ancestor = tree.lca(first_return, second_return)
        self.assertEqual(tree.labels[ancestor], "MethodDeclaration")
        self.assertEqual(gromov_lca_depth(tree, first_return, second_return), float(tree.depth(ancestor)))
        self.assertEqual(
            tree.path_between(first_return, second_return).length,
            tree.tree_distance(first_return, second_return),
        )

    def test_parse_java_source_preserves_raw_ast_edge_and_terminal_metadata(self) -> None:
        tree = parse_java_ast_tree(JAVA_METHOD)

        non_root = [node for node in tree.preorder() if node != tree.root_id]
        terminal_nodes = find_nodes_by_label(tree, "TerminalToken")

        self.assertGreater(len(terminal_nodes), 0)
        self.assertTrue(any(tree.attributes[node].get("terminal_token") == "absLike" for node in terminal_nodes))
        for node in non_root:
            self.assertIn("edge_type", tree.attributes[node])
            self.assertIn("child_index", tree.attributes[node])
        for node in tree.preorder():
            self.assertIn("node_type", tree.attributes[node])
        method_node = find_nodes_by_label(tree, "MethodDeclaration")[0]
        self.assertIn("source_span", tree.attributes[method_node])


if __name__ == "__main__":
    unittest.main()
