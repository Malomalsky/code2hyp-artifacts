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
            ],
        }

        markdown = build_paper_benchmark_markdown(result)

        self.assertIn("External Java-small literature baselines", markdown)
        self.assertIn("| code2seq | 50.64 | 37.40 | 43.02 |", markdown)
        self.assertIn("Local Code2Hyp controlled results", markdown)
        self.assertIn("| B39 matched code2vec-style baseline | 10.00 | 20.00 | 13.33 | -0.1000 | 0.8000 |", markdown)
        self.assertIn("| Code2Hyp B36 product-Frechet + neighbor | 20.00 | 30.00 | 24.00 | 0.5000 | 0.2000 |", markdown)
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

        self.assertIn("| B39 matched code2vec-style baseline | 10.00 | 20.00 | 13.33 | -0.1000 | n/a |", markdown)


if __name__ == "__main__":
    unittest.main()
