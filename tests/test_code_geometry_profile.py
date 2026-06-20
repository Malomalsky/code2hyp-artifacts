import unittest

from geometry_profile_research.code_geometry_profile import code_geometry_profile
from geometry_profile_research.program_views import extract_python_program_views


class CodeGeometryProfileTests(unittest.TestCase):
    def test_ast_profile_contains_size_curvature_and_growth(self):
        views = extract_python_program_views("def f(x):\n    return x + 1\n")

        profile = code_geometry_profile(views.ast.graph)

        self.assertGreater(profile["node_count"], 0)
        self.assertGreater(profile["edge_count"], 0)
        self.assertIn("forman_mean", profile)
        self.assertIn("forman_negative_mass", profile)
        self.assertIn("ball_size_mean_r1", profile)
        self.assertNotIn("ollivier_mean", profile)

    def test_profile_can_include_ollivier_curvature_opt_in(self):
        views = extract_python_program_views("if x:\n    y = 1\nelse:\n    y = 2\n")

        profile = code_geometry_profile(views.ast.graph, include_ollivier=True)

        self.assertIn("ollivier_mean", profile)
        self.assertIn("ollivier_negative_mass", profile)
        self.assertIn("ollivier_positive_mass", profile)


if __name__ == "__main__":
    unittest.main()
