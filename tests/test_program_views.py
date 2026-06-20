import unittest

from geometry_profile_research.program_views import extract_python_program_views


class ProgramViewsTests(unittest.TestCase):
    def test_extract_python_program_views_exposes_ast(self):
        views = extract_python_program_views("x = 1\nprint(x)\n")

        self.assertIn("ast", views.available_views())
        self.assertGreater(len(views.ast.graph.nodes), 0)
        self.assertEqual(views.ast.kind, "ast")

    def test_unsupported_views_are_not_faked(self):
        views = extract_python_program_views("x = 1\n")

        self.assertNotIn("cfg", views.available_views())
        self.assertIsNone(views.cfg)
        self.assertIsNone(views.dfg)
        self.assertIsNone(views.cpg)


if __name__ == "__main__":
    unittest.main()
