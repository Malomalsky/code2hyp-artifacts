import unittest

from geometry_profile_research.retrieval import (
    apply_zscore_scaler,
    apply_residualizer,
    combine_distance_matrices,
    evaluate_retrieval,
    evaluate_retrieval_by_label,
    fit_residualizer,
    fit_zscore_scaler,
    jensen_shannon_divergence,
    jensen_shannon_distance,
    median_nonzero_distance,
    pairwise_distances,
    paired_bootstrap_ci,
    paired_permutation_p_value,
    per_query_retrieval_records,
    per_query_retrieval_scores,
    rowwise_markov_jsd,
    zscore_feature_vectors,
)


class JensenShannonTests(unittest.TestCase):
    def test_jsd_is_zero_for_identical_sparse_distributions(self):
        distribution = {"a": 0.25, "b": 0.75}

        distance = jensen_shannon_divergence(distribution, distribution)

        self.assertEqual(distance, 0.0)

    def test_jsd_is_symmetric_and_bounded_with_log2(self):
        left = {"a": 1.0}
        right = {"b": 1.0}

        forward = jensen_shannon_divergence(left, right)
        backward = jensen_shannon_divergence(right, left)

        self.assertAlmostEqual(forward, backward)
        self.assertGreater(forward, 0.0)
        self.assertLessEqual(forward, 1.0)

    def test_jensen_shannon_distance_is_square_root_of_divergence(self):
        left = {"a": 1.0}
        right = {"b": 1.0}

        divergence = jensen_shannon_divergence(left, right)
        distance = jensen_shannon_distance(left, right)

        self.assertAlmostEqual(distance * distance, divergence)

    def test_rowwise_markov_jsd_averages_parent_state_divergences(self):
        left = {
            "Module": {"FunctionDef": 1.0},
            "Return": {"Name": 0.5, "Constant": 0.5},
        }
        right = {
            "Module": {"FunctionDef": 1.0},
            "Return": {"Name": 1.0},
        }

        distance = rowwise_markov_jsd(left, right)

        self.assertGreater(distance, 0.0)
        self.assertLess(distance, jensen_shannon_divergence({"a": 1.0}, {"b": 1.0}))


class FeatureScalingTests(unittest.TestCase):
    def test_zscore_feature_vectors_standardizes_columns(self):
        vectors = [
            {"depth": 1.0, "nodes": 10.0},
            {"depth": 2.0, "nodes": 20.0},
            {"depth": 3.0, "nodes": 30.0},
        ]

        scaled = zscore_feature_vectors(vectors)

        self.assertEqual(set(scaled[0]), {"depth", "nodes"})
        self.assertAlmostEqual(sum(row["depth"] for row in scaled), 0.0)
        self.assertAlmostEqual(sum(row["nodes"] for row in scaled), 0.0)

    def test_fit_and_apply_zscore_scaler_keeps_validation_parameters(self):
        validation = [{"x": 1.0}, {"x": 3.0}]
        test = [{"x": 5.0}]

        scaler = fit_zscore_scaler(validation)
        scaled_test = apply_zscore_scaler(test, scaler)

        self.assertEqual(scaled_test[0]["x"], 3.0)

    def test_residualizer_removes_linear_length_effect_on_validation(self):
        controls = [{"length": 1.0}, {"length": 2.0}, {"length": 3.0}]
        targets = [{"depth": 3.0}, {"depth": 5.0}, {"depth": 7.0}]

        residualizer = fit_residualizer(controls, targets)
        residuals = apply_residualizer(controls, targets, residualizer)

        self.assertEqual(set(residuals[0]), {"depth"})
        for residual in residuals:
            self.assertAlmostEqual(residual["depth"], 0.0, places=8)

    def test_residualizer_applies_validation_coefficients_to_test(self):
        validation_controls = [{"length": 1.0}, {"length": 2.0}, {"length": 3.0}]
        validation_targets = [{"depth": 3.0}, {"depth": 5.0}, {"depth": 7.0}]
        test_controls = [{"length": 4.0}]
        test_targets = [{"depth": 12.0}]

        residualizer = fit_residualizer(validation_controls, validation_targets)
        residuals = apply_residualizer(test_controls, test_targets, residualizer)

        self.assertAlmostEqual(residuals[0]["depth"], 3.0)


class RetrievalEvaluationTests(unittest.TestCase):
    def test_evaluate_retrieval_reports_map_mrr_and_recall(self):
        labels = ["A", "A", "B", "B"]
        distances = [
            [0.0, 0.1, 0.8, 0.9],
            [0.1, 0.0, 0.7, 0.8],
            [0.9, 0.8, 0.0, 0.2],
            [0.8, 0.9, 0.2, 0.0],
        ]

        metrics = evaluate_retrieval(labels, distances, k_values=(1, 2))

        self.assertEqual(metrics["queries"], 4)
        self.assertEqual(metrics["top1_accuracy"], 1.0)
        self.assertEqual(metrics["mrr"], 1.0)
        self.assertEqual(metrics["map"], 1.0)
        self.assertEqual(metrics["recall@1"], 1.0)
        self.assertEqual(metrics["recall@2"], 1.0)

    def test_per_query_scores_keep_query_level_values(self):
        labels = ["A", "A", "B", "B"]
        distances = [
            [0.0, 0.1, 0.8, 0.9],
            [0.1, 0.0, 0.7, 0.8],
            [0.9, 0.8, 0.0, 0.2],
            [0.8, 0.9, 0.2, 0.0],
        ]

        scores = per_query_retrieval_scores(labels, distances, k_values=(1, 2))

        self.assertEqual(len(scores), 4)
        self.assertEqual(scores[0]["top1"], 1.0)
        self.assertEqual(scores[0]["reciprocal_rank"], 1.0)
        self.assertEqual(scores[0]["average_precision"], 1.0)
        self.assertEqual(scores[0]["recall@1"], 1.0)

    def test_per_query_records_keep_query_identity_and_skip_singletons(self):
        labels = ["A", "A", "B"]
        distances = [
            [0.0, 0.2, 0.8],
            [0.2, 0.0, 0.7],
            [0.8, 0.7, 0.0],
        ]

        records = per_query_retrieval_records(labels, distances, k_values=(1,))

        self.assertEqual([record["query_index"] for record in records], [0, 1])
        self.assertEqual([record["query_label"] for record in records], ["A", "A"])
        self.assertEqual(records[0]["top1"], 1.0)
        self.assertNotIn("B", {record["query_label"] for record in records})

    def test_evaluate_retrieval_by_label_reports_task_level_metrics(self):
        labels = ["A", "A", "B", "B", "C"]
        distances = [
            [0.0, 0.1, 0.8, 0.9, 0.5],
            [0.1, 0.0, 0.7, 0.8, 0.5],
            [0.9, 0.8, 0.0, 0.2, 0.5],
            [0.8, 0.9, 0.2, 0.0, 0.5],
            [0.5, 0.5, 0.5, 0.5, 0.0],
        ]

        by_label = evaluate_retrieval_by_label(labels, distances, k_values=(1, 2))

        self.assertEqual(set(by_label), {"A", "B"})
        self.assertEqual(by_label["A"]["queries"], 2)
        self.assertEqual(by_label["A"]["map"], 1.0)
        self.assertEqual(by_label["B"]["recall@1"], 1.0)
        self.assertNotIn("C", by_label)

    def test_paired_bootstrap_ci_and_permutation_use_matched_scores(self):
        baseline = [0.1, 0.2, 0.3, 0.4, 0.2, 0.3, 0.4, 0.5]
        improved = [0.2, 0.3, 0.4, 0.5, 0.3, 0.4, 0.5, 0.6]

        low, high = paired_bootstrap_ci(
            baseline,
            improved,
            iterations=200,
            seed=7,
        )
        p_value = paired_permutation_p_value(
            baseline,
            improved,
            iterations=200,
            seed=7,
        )

        self.assertGreater(low, 0.0)
        self.assertGreater(high, 0.0)
        self.assertLess(p_value, 0.1)

    def test_pairwise_distances_are_symmetric(self):
        vectors = [{"x": 0.0}, {"x": 2.0}, {"x": 5.0}]

        distances = pairwise_distances(
            vectors,
            lambda left, right: abs(left["x"] - right["x"]),
        )

        self.assertEqual(distances[0][0], 0.0)
        self.assertEqual(distances[0][2], distances[2][0])
        self.assertEqual(distances[0][2], 5.0)

    def test_combine_distance_matrices_scales_before_weighting(self):
        left = [
            [0.0, 1.0, 10.0],
            [1.0, 0.0, 5.0],
            [10.0, 5.0, 0.0],
        ]
        right = [
            [0.0, 100.0, 50.0],
            [100.0, 0.0, 25.0],
            [50.0, 25.0, 0.0],
        ]

        combined = combine_distance_matrices(left, right, left_weight=0.5)

        self.assertEqual(combined[0][0], 0.0)
        self.assertAlmostEqual(combined[0][1], combined[1][0])
        self.assertGreater(combined[0][1], 0.0)

    def test_combine_distance_matrices_accepts_precomputed_scales(self):
        left = [
            [0.0, 2.0],
            [2.0, 0.0],
        ]
        right = [
            [0.0, 10.0],
            [10.0, 0.0],
        ]

        combined = combine_distance_matrices(
            left,
            right,
            left_weight=0.5,
            left_scale=2.0,
            right_scale=10.0,
        )

        self.assertEqual(combined[0][1], 1.0)

    def test_median_nonzero_distance_ignores_diagonal_and_zeros(self):
        matrix = [
            [0.0, 2.0, 0.0],
            [2.0, 0.0, 8.0],
            [0.0, 8.0, 0.0],
        ]

        self.assertEqual(median_nonzero_distance(matrix), 5.0)


if __name__ == "__main__":
    unittest.main()
