from __future__ import annotations

import unittest

from scripts.summarize_code2hyp_paper_benchmark import (
    CODE2SEQ_JAVA_SMALL_BASELINES,
    build_paper_benchmark_markdown,
)


class Code2HypPaperBenchmarkScriptTests(unittest.TestCase):
    def test_external_table_contains_code2seq_java_small_f1(self) -> None:
        code2seq = [row for row in CODE2SEQ_JAVA_SMALL_BASELINES if row[0] == "code2seq"]

        self.assertEqual(code2seq, [("code2seq", 50.64, 37.40, 43.02, "Alon et al. 2019; code2seq Table 1")])

    def test_build_paper_benchmark_markdown_separates_external_and_local_results(self) -> None:
        result = {
            "dataset": {
                "train_records": 10,
                "validation_records_loaded": 4,
                "validation_records_after_known_target_filter": 3,
            },
            "evaluation": {
                "split": "test",
            },
            "training": {
                "epochs": 2,
                "batch_size": 4,
                "model_seeds": [101],
                "metric": "target-subtoken micro precision/recall/F1",
            },
            "runs": [
                {
                    "variant": "B39_code2vec_context_transform_baseline",
                    "validation_precision": 0.10,
                    "validation_recall": 0.20,
                    "validation_f1": 0.133333,
                    "validation_structural_spearman": -0.10,
                    "validation_structural_normalized_stress": 0.80,
                    "validation_structural_neighbor_overlap_at_3": 0.40,
                    "curvature": 1.0,
                    "product_attention_bias_weight": 0.0,
                },
                {
                    "variant": "B36_code2hyp_product_frechet_neighbor",
                    "validation_precision": 0.20,
                    "validation_recall": 0.30,
                    "validation_f1": 0.24,
                    "validation_structural_spearman": 0.50,
                    "validation_structural_normalized_stress": 0.20,
                    "validation_structural_neighbor_overlap_at_3": 0.80,
                    "curvature": 0.9,
                    "product_attention_bias_weight": 0.0,
                },
                {
                    "variant": "B51_code2vec_context_transform_l1_distance_control",
                    "validation_precision": 0.16,
                    "validation_recall": 0.24,
                    "validation_f1": 0.192,
                    "validation_structural_spearman": 0.65,
                    "validation_structural_normalized_stress": 0.32,
                    "validation_structural_neighbor_overlap_at_3": 0.68,
                    "validation_structural_neighbor_exact_overlap_at_3": 0.58,
                    "curvature": 1.0,
                    "product_attention_bias_weight": 0.0,
                },
                {
                    "variant": "B49_code2hyp_context_transform_product_bias_near_euclidean",
                    "validation_precision": 0.15,
                    "validation_recall": 0.25,
                    "validation_f1": 0.1875,
                    "validation_structural_spearman": 0.70,
                    "validation_structural_normalized_stress": 0.30,
                    "validation_structural_neighbor_overlap_at_3": 0.70,
                    "validation_structural_neighbor_exact_overlap_at_3": 0.60,
                    "validation_poincare_frechet_residual_mean": 0.01,
                    "validation_poincare_context_radius_ratio_max": 0.12,
                    "validation_poincare_context_near_boundary_rate": 0.0,
                    "curvature": 1e-4,
                    "product_attention_bias_weight": 0.2,
                },
            ],
        }

        markdown = build_paper_benchmark_markdown(result)

        self.assertIn("External Java-small literature baselines", markdown)
        self.assertIn("| code2seq | 50.64 | 37.40 | 43.02 |", markdown)
        self.assertIn("Local Code2Hyp controlled results", markdown)
        self.assertIn("Edit Spearman", markdown)
        self.assertIn("Jaccard stress", markdown)
        self.assertIn("Karcher residual", markdown)
        self.assertIn("Radius max", markdown)
        self.assertIn(
            "| B39 matched code2vec-style baseline | 10.00 | 20.00 | 13.33 | -0.1000 | n/a | n/a | 0.8000 |",
            markdown,
        )
        self.assertIn(
            "| Code2Hyp B36 product-Frechet + neighbor | 20.00 | 30.00 | 24.00 | 0.5000 | n/a | n/a | 0.2000 |",
            markdown,
        )
        self.assertIn(
            "| B51 L1 structural-distance + distance loss | 16.00 | 24.00 | 19.20 | 0.6500 | n/a | n/a | 0.3200 |",
            markdown,
        )
        self.assertIn(
            "| B49 same code path, near-Euclidean curvature | 15.00 | 25.00 | 18.75 | 0.7000 | n/a | n/a | 0.3000 |",
            markdown,
        )
        self.assertIn("| 0.6000 | 0.0100 | 0.1200 | 0.0000 | 0.0001 | 0.2000 |", markdown)
        self.assertIn("Do not compare the local subset run as a direct SOTA claim", markdown)

    def test_build_paper_benchmark_markdown_is_backward_compatible_without_stress(self) -> None:
        result = {
            "runs": [
                {
                    "variant": "B39_code2vec_context_transform_baseline",
                    "validation_precision": 0.10,
                    "validation_recall": 0.20,
                    "validation_f1": 0.133333,
                    "validation_structural_spearman": -0.10,
                    "validation_structural_neighbor_overlap_at_3": 0.40,
                    "curvature": 1.0,
                    "product_attention_bias_weight": 0.0,
                }
            ]
        }

        markdown = build_paper_benchmark_markdown(result)

        self.assertIn(
            "| B39 matched code2vec-style baseline | 10.00 | 20.00 | 13.33 | -0.1000 | n/a | n/a | n/a |",
            markdown,
        )


if __name__ == "__main__":
    unittest.main()
