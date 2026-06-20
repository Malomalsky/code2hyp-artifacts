import unittest

from geometry_profile_research.experiment_summary import summarize_metric_series


class ExperimentSummaryTests(unittest.TestCase):
    def test_summarize_metric_series_reports_mean_std_and_range(self):
        summary = summarize_metric_series([1.0, 2.0, 3.0])

        self.assertEqual(summary["count"], 3)
        self.assertEqual(summary["mean"], 2.0)
        self.assertAlmostEqual(summary["std"], 0.8164965809)
        self.assertEqual(summary["min"], 1.0)
        self.assertEqual(summary["max"], 3.0)

    def test_summarize_metric_series_handles_empty_series(self):
        summary = summarize_metric_series([])

        self.assertEqual(summary["count"], 0)
        self.assertEqual(summary["mean"], 0.0)
        self.assertEqual(summary["std"], 0.0)


if __name__ == "__main__":
    unittest.main()
