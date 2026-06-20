import unittest

from geometry_profile_research.task_characterization import (
    benjamini_hochberg_q_values,
    bootstrap_spearman_ci,
    leave_one_out_spearman,
    spearman_permutation_p_value,
    spearman_correlation,
    summarize_numeric_features_by_label,
)


class TaskCharacterizationTests(unittest.TestCase):
    def test_summarize_numeric_features_by_label_reports_mean_and_spread(self):
        rows = summarize_numeric_features_by_label(
            labels=[0, 0, 1],
            vectors=[
                {"nodes": 10.0, "depth": 2.0},
                {"nodes": 14.0, "depth": 4.0},
                {"nodes": 5.0, "depth": 1.0},
            ],
        )

        task_zero = next(row for row in rows if row["task_label"] == "0")
        self.assertEqual(task_zero["n"], 2)
        self.assertEqual(task_zero["nodes_mean"], 12.0)
        self.assertEqual(task_zero["nodes_std"], 2.0)
        self.assertEqual(task_zero["depth_min"], 2.0)
        self.assertEqual(task_zero["depth_max"], 4.0)

    def test_spearman_correlation_handles_monotonic_relationship(self):
        corr = spearman_correlation([1.0, 2.0, 3.0], [10.0, 20.0, 30.0])

        self.assertAlmostEqual(corr, 1.0)

    def test_spearman_correlation_handles_reverse_relationship(self):
        corr = spearman_correlation([1.0, 2.0, 3.0], [30.0, 20.0, 10.0])

        self.assertAlmostEqual(corr, -1.0)

    def test_spearman_permutation_p_value_detects_strong_relationship(self):
        p_value = spearman_permutation_p_value(
            [1.0, 2.0, 3.0, 4.0, 5.0],
            [1.0, 2.0, 3.0, 4.0, 5.0],
            iterations=500,
            seed=7,
        )

        self.assertGreaterEqual(p_value, 0.0)
        self.assertLess(p_value, 0.05)

    def test_bootstrap_spearman_ci_returns_ordered_interval(self):
        low, high = bootstrap_spearman_ci(
            [1.0, 2.0, 3.0, 4.0, 5.0],
            [1.0, 2.0, 3.0, 4.0, 5.0],
            iterations=200,
            seed=7,
        )

        self.assertLessEqual(low, high)
        self.assertGreater(high, 0.0)

    def test_leave_one_out_spearman_reports_sensitivity_range(self):
        summary = leave_one_out_spearman(
            [1.0, 2.0, 3.0, 4.0],
            [1.0, 2.0, 3.0, 10.0],
            labels=["a", "b", "c", "d"],
        )

        self.assertEqual(summary["n"], 4)
        self.assertLessEqual(summary["rho_min"], summary["rho_max"])
        self.assertIn("omit_label_at_min", summary)

    def test_benjamini_hochberg_q_values_are_monotone_by_rank(self):
        q_values = benjamini_hochberg_q_values([0.01, 0.02, 0.50])

        self.assertEqual(len(q_values), 3)
        self.assertLessEqual(q_values[0], q_values[1])
        self.assertLessEqual(q_values[1], q_values[2])
        self.assertLessEqual(q_values[0], 0.05)


if __name__ == "__main__":
    unittest.main()
