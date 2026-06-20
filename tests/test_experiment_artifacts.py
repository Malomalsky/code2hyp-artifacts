import unittest

from geometry_profile_research.experiment_artifacts import (
    classify_experiment,
    extract_delta_rows,
    extract_inventory_row,
    extract_metric_rows,
    extract_task_delta_rows,
    extract_task_metric_rows,
    summarize_delta_rows,
    summarize_task_delta_rows,
)


class ExperimentArtifactTests(unittest.TestCase):
    def test_extract_metric_rows_from_confirmatory_split(self):
        payload = {
            "dataset": {"test_per_task": 2},
            "parameters": {"baseline_kind": "transition_count", "selected_markov_weight": 0.9},
            "test_results": {
                "baseline": {"top1_accuracy": 0.9, "mrr": 0.91, "map": 0.7, "recall@10": 0.2},
                "candidate": {"top1_accuracy": 0.92, "mrr": 0.93, "map": 0.72, "recall@10": 0.21},
            },
        }

        rows = extract_metric_rows("confirmatory.json", payload)

        self.assertEqual(classify_experiment("confirmatory.json", payload), "confirmatory_split")
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["scope"], "test")
        self.assertEqual(rows[0]["method"], "baseline")
        self.assertEqual(rows[1]["method"], "candidate")
        self.assertEqual(rows[1]["map"], 0.72)

    def test_extract_metric_rows_from_confirmatory_feature_sweep(self):
        payload = {
            "dataset": {"test_per_task": 2},
            "parameters": {"baseline_kind": "transition_count"},
            "baseline": {
                "test": {"top1_accuracy": 0.9, "mrr": 0.91, "map": 0.7, "recall@10": 0.2},
            },
            "feature_set_results": {
                "length_only": {
                    "selected_markov_weight": 0.9,
                    "selected_geometry_weight": 0.1,
                    "test_results": {
                        "candidate": {"top1_accuracy": 0.91, "mrr": 0.92, "map": 0.73, "recall@10": 0.21},
                        "paired_tests_candidate_minus_baseline": {
                            "map": {
                                "mean_delta": 0.03,
                                "bootstrap_ci95": [0.02, 0.04],
                                "permutation_p_one_sided": 0.001,
                            }
                        },
                    },
                }
            },
        }

        metric_rows = extract_metric_rows("feature_sweep.json", payload)
        delta_rows = extract_delta_rows("feature_sweep.json", payload)

        self.assertEqual(classify_experiment("feature_sweep.json", payload), "confirmatory_feature_sweep")
        self.assertEqual(metric_rows[0]["method"], "baseline")
        self.assertEqual(metric_rows[1]["method"], "candidate_length_only")
        self.assertEqual(metric_rows[1]["feature_set"], "length_only")
        self.assertEqual(delta_rows[0]["comparison"], "candidate_length_only_minus_baseline")
        self.assertEqual(delta_rows[0]["mean_delta"], 0.03)

    def test_extract_task_rows_from_confirmatory_feature_sweep(self):
        payload = {
            "dataset": {"split_seed": 101},
            "parameters": {"baseline_kind": "transition_count"},
            "baseline": {
                "test_by_task": {
                    "0": {"queries": 2, "map": 0.7, "recall@10": 0.2},
                },
            },
            "feature_set_results": {
                "length_only": {
                    "test_results": {
                        "candidate_by_task": {
                            "0": {"queries": 2, "map": 0.8, "recall@10": 0.25},
                        },
                    },
                },
            },
        }

        metric_rows = extract_task_metric_rows("feature_sweep.json", payload)
        delta_rows = extract_task_delta_rows("feature_sweep.json", payload)

        self.assertEqual(len(metric_rows), 2)
        self.assertEqual(metric_rows[0]["method"], "baseline")
        self.assertEqual(metric_rows[0]["task_label"], "0")
        self.assertEqual(metric_rows[1]["method"], "candidate_length_only")
        self.assertEqual(delta_rows[0]["feature_set"], "length_only")
        self.assertEqual(delta_rows[0]["metric"], "map")
        self.assertAlmostEqual(delta_rows[0]["mean_delta"], 0.1)

    def test_extract_delta_rows_from_confirmatory_split(self):
        payload = {
            "test_results": {
                "paired_tests_candidate_minus_baseline": {
                    "map": {
                        "mean_delta": 0.01,
                        "bootstrap_ci95": [0.005, 0.015],
                        "permutation_p_one_sided": 0.001,
                    }
                }
            }
        }

        rows = extract_delta_rows("confirmatory.json", payload)

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["comparison"], "candidate_minus_baseline")
        self.assertEqual(rows[0]["metric"], "map")
        self.assertEqual(rows[0]["mean_delta"], 0.01)

    def test_extract_metric_rows_from_methods_payload(self):
        payload = {
            "records": {"valid": 10},
            "parameters": {"baseline_kind": "flat_markov", "markov_weight": 0.85},
            "methods": {
                "M2": {"metrics": {"top1_accuracy": 0.8, "mrr": 0.9, "map": 0.5, "recall@10": 0.2}},
                "M4": {"metrics": {"top1_accuracy": 0.9, "mrr": 0.95, "map": 0.6, "recall@10": 0.3}},
            },
        }

        rows = extract_metric_rows("ablation.json", payload)

        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["experiment_type"], "method_comparison")
        self.assertEqual(rows[1]["method"], "M4")

    def test_extract_inventory_row_marks_confirmatory_experiment(self):
        payload = {
            "dataset": {"validation_per_task": 20, "test_per_task": 50, "split_seed": 101},
            "records": {"validation": 220, "test": 550},
            "parameters": {
                "baseline_kind": "transition_count",
                "feature_set_name": "all",
                "selected_markov_weight": 0.9,
                "selected_geometry_weight": 0.1,
            },
            "test_results": {
                "baseline": {"map": 0.74},
                "candidate": {"map": 0.75},
            },
        }

        row = extract_inventory_row("confirmatory.json", payload)

        self.assertEqual(row["role"], "confirmatory")
        self.assertEqual(row["records"], 550)
        self.assertEqual(row["baseline_kind"], "transition_count")
        self.assertEqual(row["article_use"], "primary evidence")

    def test_extract_inventory_row_counts_multiseed_records(self):
        payload = {
            "dataset": {"limit_per_task": 20, "seeds": [13, 29, 43]},
            "parameters": {"baseline_kind": "transition_count", "markov_weight": 0.9},
            "method_summary": {},
            "runs": [
                {"records": {"valid": 220}},
                {"records": {"valid": 220}},
                {"records": {"valid": 220}},
            ],
        }

        row = extract_inventory_row("multiseed.json", payload)

        self.assertEqual(row["records"], 660)
        self.assertEqual(row["role"], "exploratory")
        self.assertEqual(row["article_use"], "robustness check")

    def test_extract_inventory_row_infers_flat_markov_for_old_multiseed(self):
        payload = {
            "dataset": {"limit_per_task": 20, "seeds": [13, 29, 43]},
            "parameters": {"markov_weight": 0.85},
            "method_summary": {},
            "runs": [{"records": {"valid": 220}}],
        }

        row = extract_inventory_row("dta_multiseed_ablation_limit20_w085.json", payload)

        self.assertEqual(row["baseline_kind"], "flat_markov_jsd")

    def test_classify_residual_feature_sweep(self):
        payload = {
            "protocol": {"status": "confirmatory_residual_feature_set_control"},
            "feature_set_results": {},
        }

        row = extract_inventory_row("residual_sweep.json", payload)

        self.assertEqual(classify_experiment("residual_sweep.json", payload), "confirmatory_residual_sweep")
        self.assertEqual(row["article_use"], "residual feature control")

    def test_summarize_delta_rows_groups_feature_sets_across_splits(self):
        rows = [
            {
                "experiment_file": "dta_confirmatory_feature_sweep_seed101.json",
                "sample_seed": 101,
                "feature_set": "length_only",
                "metric": "map",
                "mean_delta": 0.02,
                "p_one_sided": 0.001,
            },
            {
                "experiment_file": "dta_confirmatory_feature_sweep_seed202.json",
                "sample_seed": 202,
                "feature_set": "length_only",
                "metric": "map",
                "mean_delta": 0.04,
                "p_one_sided": 0.020,
            },
            {
                "experiment_file": "dta_confirmatory_feature_sweep_seed202.json",
                "sample_seed": 202,
                "feature_set": "shape",
                "metric": "map",
                "mean_delta": 0.01,
                "p_one_sided": 0.200,
            },
        ]

        summary = summarize_delta_rows(
            rows,
            file_contains="dta_confirmatory_feature_sweep",
            metrics={"map"},
        )

        self.assertEqual(len(summary), 2)
        length_row = next(row for row in summary if row["feature_set"] == "length_only")
        self.assertEqual(length_row["n_splits"], 2)
        self.assertEqual(length_row["split_seeds"], "101,202")
        self.assertAlmostEqual(length_row["mean_delta_mean"], 0.03)
        self.assertAlmostEqual(length_row["mean_delta_min"], 0.02)
        self.assertAlmostEqual(length_row["mean_delta_max"], 0.04)
        self.assertEqual(length_row["significant_splits"], 2)

    def test_summarize_task_delta_rows_groups_by_task_feature_and_metric(self):
        rows = [
            {
                "experiment_file": "dta_confirmatory_feature_sweep_seed101.json",
                "sample_seed": 101,
                "task_label": "0",
                "feature_set": "length_only",
                "metric": "map",
                "mean_delta": 0.01,
            },
            {
                "experiment_file": "dta_confirmatory_feature_sweep_seed202.json",
                "sample_seed": 202,
                "task_label": "0",
                "feature_set": "length_only",
                "metric": "map",
                "mean_delta": 0.03,
            },
            {
                "experiment_file": "dta_confirmatory_feature_sweep_seed202.json",
                "sample_seed": 202,
                "task_label": "1",
                "feature_set": "length_only",
                "metric": "map",
                "mean_delta": -0.02,
            },
        ]

        summary = summarize_task_delta_rows(
            rows,
            file_contains="dta_confirmatory_feature_sweep",
            metrics={"map"},
        )

        task_zero = next(row for row in summary if row["task_label"] == "0")
        self.assertEqual(task_zero["n_splits"], 2)
        self.assertEqual(task_zero["split_seeds"], "101,202")
        self.assertAlmostEqual(task_zero["mean_delta_mean"], 0.02)
        self.assertAlmostEqual(task_zero["mean_delta_min"], 0.01)
        self.assertAlmostEqual(task_zero["mean_delta_max"], 0.03)


if __name__ == "__main__":
    unittest.main()
