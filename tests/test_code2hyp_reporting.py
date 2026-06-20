from __future__ import annotations

import unittest

from geometry_profile_research.code2hyp_reporting import (
    multi_objective_variant_selection,
    paired_metric_comparison,
    pareto_frontier,
    summarize_pilot_runs,
)


class Code2HypReportingTests(unittest.TestCase):
    def test_summarize_pilot_runs_groups_variants_and_reports_means(self) -> None:
        result = {
            "runs": [
                {
                    "variant": "B1_euclidean",
                    "validation_f1": 0.10,
                    "validation_structural_loss": 0.80,
                    "validation_structural_rank_loss": 0.20,
                    "validation_structural_spearman": -0.30,
                    "validation_structural_neighbor_overlap_at_1": 0.25,
                },
                {
                    "variant": "B1_euclidean",
                    "validation_f1": 0.20,
                    "validation_structural_loss": 0.60,
                    "validation_structural_rank_loss": 0.10,
                    "validation_structural_spearman": -0.10,
                    "validation_structural_neighbor_overlap_at_1": 0.75,
                },
                {
                    "variant": "B3_product",
                    "validation_f1": 0.30,
                    "validation_structural_loss": 0.15,
                    "validation_structural_rank_loss": 0.40,
                    "validation_structural_spearman": 0.20,
                    "validation_structural_neighbor_overlap_at_1": 1.00,
                },
            ]
        }

        summaries = summarize_pilot_runs(result)

        self.assertEqual([summary["variant"] for summary in summaries], ["B1_euclidean", "B3_product"])
        self.assertEqual(summaries[0]["n"], 2)
        self.assertAlmostEqual(summaries[0]["validation_f1_mean"], 0.15)
        self.assertAlmostEqual(summaries[0]["validation_structural_loss_mean"], 0.70)
        self.assertAlmostEqual(summaries[0]["validation_structural_spearman_mean"], -0.20)
        self.assertAlmostEqual(summaries[0]["validation_structural_neighbor_overlap_at_1_mean"], 0.50)
        self.assertEqual(summaries[1]["n"], 1)
        self.assertAlmostEqual(summaries[1]["validation_f1_std"], 0.0)

    def test_summarize_pilot_runs_supports_legacy_neighbor_recall_key(self) -> None:
        result = {
            "runs": [
                {
                    "variant": "B1_euclidean",
                    "validation_structural_neighbor_recall_at_1": 0.25,
                },
                {
                    "variant": "B1_euclidean",
                    "validation_structural_neighbor_recall_at_1": 0.75,
                },
            ]
        }

        summaries = summarize_pilot_runs(result)

        self.assertAlmostEqual(summaries[0]["validation_structural_neighbor_overlap_at_1_mean"], 0.50)

    def test_paired_metric_comparison_matches_runs_by_model_seed(self) -> None:
        result = {
            "runs": [
                {"variant": "B4", "model_seed": 1, "validation_f1": 0.30},
                {"variant": "B_tree", "model_seed": 1, "validation_f1": 0.10},
                {"variant": "B4", "model_seed": 2, "validation_f1": 0.40},
                {"variant": "B_tree", "model_seed": 2, "validation_f1": 0.25},
                {"variant": "B4", "model_seed": 3, "validation_f1": 0.50},
                {"variant": "B_tree", "model_seed": 3, "validation_f1": 0.35},
                {"variant": "B4", "model_seed": 999, "validation_f1": 0.99},
            ]
        }

        comparison = paired_metric_comparison(result, "B4", "B_tree", metric_key="validation_f1")

        self.assertEqual(comparison["n"], 3)
        self.assertEqual(comparison["pairing_key"], "model_seed")
        self.assertEqual(comparison["paired_keys"], [1, 2, 3])
        self.assertEqual(comparison["positive_deltas"], 3)
        self.assertEqual(comparison["negative_deltas"], 0)
        self.assertEqual(comparison["evidence_status"], "exploratory_low_power")

    def test_paired_metric_comparison_supports_legacy_neighbor_recall_key(self) -> None:
        result = {
            "runs": [
                {
                    "variant": "B31",
                    "model_seed": 1,
                    "validation_structural_neighbor_recall_at_1": 0.60,
                },
                {
                    "variant": "B1",
                    "model_seed": 1,
                    "validation_structural_neighbor_recall_at_1": 0.30,
                },
            ]
        }

        comparison = paired_metric_comparison(
            result,
            "B31",
            "B1",
            metric_key="validation_structural_neighbor_overlap_at_1",
        )

        self.assertEqual(comparison["n"], 1)
        self.assertAlmostEqual(comparison["mean_delta"], 0.30)

    def test_paired_metric_comparison_rejects_missing_common_pairs(self) -> None:
        result = {
            "runs": [
                {"variant": "B4", "model_seed": 1, "validation_f1": 0.30},
                {"variant": "B_tree", "model_seed": 2, "validation_f1": 0.10},
            ]
        }

        with self.assertRaises(ValueError):
            paired_metric_comparison(result, "B4", "B_tree", metric_key="validation_f1")

    def test_pareto_frontier_keeps_non_dominated_f1_spearman_tradeoffs(self) -> None:
        rows = [
            {"variant": "B31", "validation_f1_mean": 0.20, "validation_structural_spearman_mean": 0.16},
            {"variant": "B32", "validation_f1_mean": 0.15, "validation_structural_spearman_mean": 0.29},
            {"variant": "weak", "validation_f1_mean": 0.12, "validation_structural_spearman_mean": 0.10},
        ]

        frontier = pareto_frontier(
            rows,
            objectives=(
                ("validation_f1_mean", "max"),
                ("validation_structural_spearman_mean", "max"),
            ),
        )

        self.assertEqual([row["variant"] for row in frontier], ["B31", "B32"])

    def test_multi_objective_variant_selection_scores_normalized_compromise(self) -> None:
        result = {
            "runs": [
                {
                    "variant": "B31",
                    "validation_f1": 0.20,
                    "validation_structural_loss": 0.13,
                    "validation_structural_rank_loss": 0.31,
                    "validation_structural_spearman": 0.16,
                },
                {
                    "variant": "B32",
                    "validation_f1": 0.15,
                    "validation_structural_loss": 0.45,
                    "validation_structural_rank_loss": 0.07,
                    "validation_structural_spearman": 0.29,
                },
                {
                    "variant": "weak",
                    "validation_f1": 0.10,
                    "validation_structural_loss": 0.50,
                    "validation_structural_rank_loss": 0.50,
                    "validation_structural_spearman": 0.05,
                },
            ],
        }

        selection = multi_objective_variant_selection(
            result,
            objectives=(
                ("validation_f1_mean", "max", 0.5),
                ("validation_structural_spearman_mean", "max", 0.5),
            ),
        )

        self.assertEqual(selection["best"]["variant"], "B32")
        self.assertEqual([row["variant"] for row in selection["pareto_frontier"]], ["B32", "B31"])
        self.assertEqual(selection["ranked"][0]["variant"], "B32")
        self.assertGreater(selection["ranked"][0]["multi_objective_score"], selection["ranked"][-1]["multi_objective_score"])
        self.assertTrue(selection["ranked"][0]["pareto_frontier"])
        self.assertFalse(selection["ranked"][-1]["pareto_frontier"])

        f1_preserving_selection = multi_objective_variant_selection(
            result,
            objectives=(
                ("validation_f1_mean", "max", 0.7),
                ("validation_structural_spearman_mean", "max", 0.3),
            ),
        )

        self.assertEqual(f1_preserving_selection["best"]["variant"], "B31")


if __name__ == "__main__":
    unittest.main()
