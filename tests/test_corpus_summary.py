import unittest

from geometry_profile_research.aggregate import summarize_geometry_profiles
from geometry_profile_research.analysis import geometry_profile_for_paths


class CorpusSummaryTests(unittest.TestCase):
    def test_summarize_geometry_profiles_reports_distribution(self):
        profiles = [
            geometry_profile_for_paths(["src/a.py", "tests/test_a.py"]),
            geometry_profile_for_paths(["src/a/b/c.py", "src/a/b/d.py"]),
        ]

        summary = summarize_geometry_profiles(profiles)

        self.assertEqual(summary["profile_count"], 2)
        self.assertIn("geometry_advantage", summary)
        self.assertIn("median", summary["geometry_advantage"])
        self.assertIn("euclidean_stress", summary)
        self.assertIn("hyperbolic_stress", summary)
        self.assertGreaterEqual(summary["geometry_advantage"]["median"], -1.0)
        self.assertLessEqual(summary["geometry_advantage"]["median"], 1.0)


if __name__ == "__main__":
    unittest.main()
