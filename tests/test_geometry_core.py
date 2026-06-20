import math
import unittest

from geometry_profile_research.embeddings import (
    angular_distance,
    euclidean_distance,
    path_to_poincare,
    poincare_distance,
)
from geometry_profile_research.analysis import geometry_profile_for_paths
from geometry_profile_research.graphs import build_file_tree_graph
from geometry_profile_research.metrics import (
    all_pairs_shortest_paths,
    distortion_summary,
    gromov_hyperbolicity,
)


class FileTreeGraphTests(unittest.TestCase):
    def test_build_file_tree_graph_preserves_prefix_hierarchy(self):
        graph = build_file_tree_graph(
            [
                "src/utils/io.py",
                "src/utils/path.py",
                "tests/test_io.py",
            ]
        )

        self.assertIn("", graph.nodes)
        self.assertIn("src", graph.nodes)
        self.assertIn("src/utils", graph.nodes)
        self.assertIn("src/utils/io.py", graph.nodes)
        self.assertEqual(graph.depth(""), 0)
        self.assertEqual(graph.depth("src/utils/io.py"), 3)
        self.assertIn("src/utils/io.py", graph.neighbors("src/utils"))
        self.assertIn("src/utils/path.py", graph.neighbors("src/utils"))


class HyperbolicEmbeddingTests(unittest.TestCase):
    def test_deeper_paths_have_larger_poincare_norm(self):
        shallow = path_to_poincare("src")
        deep = path_to_poincare("src/utils/io.py")

        self.assertGreater(math.hypot(*deep), math.hypot(*shallow))
        self.assertLess(math.hypot(*deep), 1.0)

    def test_shared_prefix_paths_are_angularly_closer(self):
        io = path_to_poincare("src/utils/io.py")
        path = path_to_poincare("src/utils/path.py")
        test = path_to_poincare("tests/test_io.py")

        self.assertLess(angular_distance(io, path), angular_distance(io, test))


class GeometryMetricTests(unittest.TestCase):
    def test_tree_has_zero_gromov_hyperbolicity(self):
        graph = build_file_tree_graph(
            [
                "src/a.py",
                "src/b.py",
                "tests/test_a.py",
                "docs/readme.md",
            ]
        )
        distances = all_pairs_shortest_paths(graph)
        result = gromov_hyperbolicity(distances)

        self.assertAlmostEqual(result.delta, 0.0)
        self.assertAlmostEqual(result.delta_norm, 0.0)

    def test_distortion_is_zero_for_exact_line_embedding(self):
        graph = build_file_tree_graph(["a/b/c.py"])
        distances = all_pairs_shortest_paths(graph)
        embedding = {
            "": (0.0, 0.0),
            "a": (1.0, 0.0),
            "a/b": (2.0, 0.0),
            "a/b/c.py": (3.0, 0.0),
        }

        summary = distortion_summary(distances, embedding, euclidean_distance)

        self.assertAlmostEqual(summary.median_relative_error, 0.0)
        self.assertAlmostEqual(summary.stress, 0.0)

    def test_hyperbolic_distance_is_monotonic_with_depth_on_same_branch(self):
        root = path_to_poincare("")
        src = path_to_poincare("src")
        deep = path_to_poincare("src/utils/io.py")

        self.assertGreater(poincare_distance(root, deep), poincare_distance(root, src))

    def test_geometry_profile_reports_branching_shape(self):
        profile = geometry_profile_for_paths(
            [
                "src/a.py",
                "src/b.py",
                "tests/test_a.py",
            ],
            assume_tree_hyperbolicity=True,
        )

        self.assertEqual(profile.max_branching_factor, 2)
        self.assertGreater(profile.mean_branching_factor, 1.0)
        self.assertAlmostEqual(profile.leaf_fraction, 0.5)
        self.assertGreater(profile.branching_entropy, 0.0)


if __name__ == "__main__":
    unittest.main()
