from __future__ import annotations

import unittest

from geometry_profile_research.task_geometry_analysis import (
    add_holm_correction,
    compute_metric_effect_sizes,
    compute_permutation_tests,
    compute_residual_effect_sizes,
    permutation_p_value_eta_squared,
    residualize_metric,
    summarize_task_geometry,
    zscore_task_means,
)


class TaskGeometryAnalysisTests(unittest.TestCase):
    def test_summarize_task_geometry_groups_by_task(self) -> None:
        rows = [
            {"task_id": 0, "node_count": 10, "ollivier_mean": -0.4},
            {"task_id": 0, "node_count": 14, "ollivier_mean": -0.2},
            {"task_id": 1, "node_count": 30, "ollivier_mean": 0.1},
        ]

        summary = summarize_task_geometry(rows, ["node_count", "ollivier_mean"])

        self.assertEqual([row["task_id"] for row in summary], [0, 1])
        self.assertEqual(summary[0]["n"], 2)
        self.assertAlmostEqual(summary[0]["node_count_mean"], 12.0)
        self.assertAlmostEqual(summary[0]["ollivier_mean_mean"], -0.3)
        self.assertAlmostEqual(summary[1]["node_count_mean"], 30.0)

    def test_effect_size_is_zero_when_task_means_are_equal(self) -> None:
        rows = [
            {"task_id": 0, "node_count": 10},
            {"task_id": 0, "node_count": 20},
            {"task_id": 1, "node_count": 10},
            {"task_id": 1, "node_count": 20},
        ]

        effects = compute_metric_effect_sizes(rows, ["node_count"])

        self.assertAlmostEqual(effects[0]["eta_squared_task"], 0.0)

    def test_effect_size_is_one_when_only_between_task_variance_exists(self) -> None:
        rows = [
            {"task_id": 0, "node_count": 10},
            {"task_id": 0, "node_count": 10},
            {"task_id": 1, "node_count": 20},
            {"task_id": 1, "node_count": 20},
        ]

        effects = compute_metric_effect_sizes(rows, ["node_count"])

        self.assertAlmostEqual(effects[0]["eta_squared_task"], 1.0)

    def test_zscore_task_means_standardizes_each_metric(self) -> None:
        summary = [
            {"task_id": 0, "node_count_mean": 10.0},
            {"task_id": 1, "node_count_mean": 20.0},
            {"task_id": 2, "node_count_mean": 30.0},
        ]

        zscores = zscore_task_means(summary, ["node_count"])

        self.assertEqual([row["task_id"] for row in zscores], [0, 1, 2])
        self.assertLess(zscores[0]["node_count"], 0)
        self.assertAlmostEqual(zscores[1]["node_count"], 0.0)
        self.assertGreater(zscores[2]["node_count"], 0)

    def test_residualize_metric_removes_linear_size_effect(self) -> None:
        rows = [
            {"task_id": 0, "node_count": 1.0, "curvature": 7.0},
            {"task_id": 0, "node_count": 2.0, "curvature": 9.0},
            {"task_id": 1, "node_count": 3.0, "curvature": 11.0},
            {"task_id": 1, "node_count": 4.0, "curvature": 13.0},
        ]

        residuals, r_squared, coefficients = residualize_metric(
            rows,
            response_metric="curvature",
            covariates=["node_count"],
        )

        self.assertAlmostEqual(r_squared, 1.0)
        self.assertAlmostEqual(coefficients[0], 5.0, places=5)
        self.assertAlmostEqual(coefficients[1], 2.0, places=5)
        for residual in residuals:
            self.assertAlmostEqual(residual, 0.0, places=5)

    def test_residual_effect_disappears_when_task_signal_is_size_only(self) -> None:
        rows = [
            {"task_id": 0, "node_count": 1.0, "curvature": 7.0},
            {"task_id": 0, "node_count": 2.0, "curvature": 9.0},
            {"task_id": 1, "node_count": 5.0, "curvature": 15.0},
            {"task_id": 1, "node_count": 6.0, "curvature": 17.0},
        ]

        effects = compute_residual_effect_sizes(
            rows,
            response_metrics=["curvature"],
            covariates=["node_count"],
        )

        self.assertAlmostEqual(effects[0]["covariate_r_squared"], 1.0)
        self.assertAlmostEqual(effects[0]["eta_squared_task_residual"], 0.0)

    def test_residual_effect_remains_when_task_offset_is_not_size(self) -> None:
        rows = [
            {"task_id": 0, "node_count": 1.0, "curvature": 7.0},
            {"task_id": 0, "node_count": 2.0, "curvature": 9.0},
            {"task_id": 1, "node_count": 1.0, "curvature": 17.0},
            {"task_id": 1, "node_count": 2.0, "curvature": 19.0},
        ]

        effects = compute_residual_effect_sizes(
            rows,
            response_metrics=["curvature"],
            covariates=["node_count"],
        )

        self.assertGreater(effects[0]["eta_squared_task_residual"], 0.99)

    def test_permutation_test_detects_clear_group_separation(self) -> None:
        values = [0.0] * 8 + [10.0] * 8
        groups = [0] * 8 + [1] * 8

        observed, p_value = permutation_p_value_eta_squared(
            values,
            groups,
            permutations=199,
            rng=__import__("random").Random(7),
        )

        self.assertAlmostEqual(observed, 1.0)
        self.assertLessEqual(p_value, 0.01)

    def test_compute_permutation_tests_includes_raw_and_residual_rows(self) -> None:
        rows = [
            {
                "task_id": 0,
                "node_count": 1.0,
                "ball_size_mean_r3": 1.0,
                "curvature": 0.0,
            },
            {
                "task_id": 0,
                "node_count": 2.0,
                "ball_size_mean_r3": 1.0,
                "curvature": 0.0,
            },
            {
                "task_id": 1,
                "node_count": 1.0,
                "ball_size_mean_r3": 1.0,
                "curvature": 5.0,
            },
            {
                "task_id": 1,
                "node_count": 2.0,
                "ball_size_mean_r3": 1.0,
                "curvature": 5.0,
            },
        ]

        result = compute_permutation_tests(
            rows,
            metrics=["curvature"],
            residual_response_metrics=["curvature"],
            covariates=["node_count"],
            permutations=19,
            seed=3,
        )

        self.assertEqual({row["analysis"] for row in result}, {"raw", "residual"})
        self.assertEqual(len(result), 2)
        for row in result:
            self.assertIn("p_value", row)
            self.assertIn("p_value_holm", row)

    def test_holm_correction_is_monotone_and_not_smaller_than_raw_p(self) -> None:
        rows = [
            {"metric": "a", "p_value": 0.01},
            {"metric": "b", "p_value": 0.02},
            {"metric": "c", "p_value": 0.50},
        ]

        corrected = add_holm_correction(rows)

        for row in corrected:
            self.assertGreaterEqual(row["p_value_holm"], row["p_value"])
        self.assertLessEqual(corrected[0]["p_value_holm"], corrected[1]["p_value_holm"])
        self.assertLessEqual(corrected[1]["p_value_holm"], corrected[2]["p_value_holm"])


if __name__ == "__main__":
    unittest.main()
