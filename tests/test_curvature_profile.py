import unittest

from geometry_profile_research.curvature import (
    ball_growth_profile,
    forman_ricci_curvature,
    local_probability_measure,
    ollivier_ricci_curvature,
    summarize_curvature,
    wasserstein_distance,
)
from geometry_profile_research.graphs import SimpleGraph


class CurvatureProfileTests(unittest.TestCase):
    def test_forman_curvature_on_path_edges(self):
        graph = SimpleGraph()
        graph.add_edge("a", "b")
        graph.add_edge("b", "c")

        curvatures = forman_ricci_curvature(graph)

        self.assertEqual(curvatures[("a", "b")], 1.0)
        self.assertEqual(curvatures[("b", "c")], 1.0)

    def test_forman_curvature_detects_star_center_bottleneck(self):
        graph = SimpleGraph()
        for leaf in ["a", "b", "c", "d"]:
            graph.add_edge("center", leaf)

        curvatures = forman_ricci_curvature(graph)
        summary = summarize_curvature(curvatures.values(), threshold=0.05)

        self.assertTrue(all(value < 0.0 for value in curvatures.values()))
        self.assertEqual(summary["negative_mass"], 1.0)
        self.assertEqual(summary["positive_mass"], 0.0)

    def test_ball_growth_profile_reports_mean_ball_sizes(self):
        graph = SimpleGraph()
        graph.add_edge("a", "b")
        graph.add_edge("b", "c")

        profile = ball_growth_profile(graph, radii=(0, 1, 2))

        self.assertEqual(profile["ball_size_mean_r0"], 1.0)
        self.assertAlmostEqual(profile["ball_size_mean_r1"], 7 / 3)
        self.assertEqual(profile["ball_size_mean_r2"], 3.0)

    def test_local_probability_measure_respects_idleness(self):
        graph = SimpleGraph()
        graph.add_edge("a", "b")
        graph.add_edge("a", "c")

        measure = local_probability_measure(graph, "a", idleness=0.25)

        self.assertEqual(measure["a"], 0.25)
        self.assertEqual(measure["b"], 0.375)
        self.assertEqual(measure["c"], 0.375)
        self.assertAlmostEqual(sum(measure.values()), 1.0)

    def test_wasserstein_distance_solves_simple_transport_exactly(self):
        distance = wasserstein_distance(
            {"a": 1.0},
            {"b": 1.0},
            {("a", "b"): 2.0},
        )

        self.assertEqual(distance, 2.0)

    def test_ollivier_curvature_separates_triangle_from_path_edge(self):
        path = SimpleGraph()
        path.add_edge("a", "b")
        path.add_edge("b", "c")
        triangle = SimpleGraph()
        triangle.add_edge("a", "b")
        triangle.add_edge("b", "c")
        triangle.add_edge("a", "c")

        path_curvature = ollivier_ricci_curvature(path, idleness=0.0)
        triangle_curvature = ollivier_ricci_curvature(triangle, idleness=0.0)

        self.assertAlmostEqual(path_curvature[("a", "b")], 0.0)
        self.assertAlmostEqual(triangle_curvature[("a", "b")], 0.5)
        self.assertGreater(triangle_curvature[("a", "b")], path_curvature[("a", "b")])


if __name__ == "__main__":
    unittest.main()
