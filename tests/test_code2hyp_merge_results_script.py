from __future__ import annotations

import unittest

from scripts.merge_code2hyp_benchmark_results import merge_results


def _result(variant: str, seed: int, *, lexical_ablation: str = "original") -> dict:
    return {
        "dataset": {
            "train_path": "train.c2s",
            "validation_path": "test.c2s",
            "train_records": 25_000,
            "validation_records_loaded": 8192,
            "validation_records_after_known_target_filter": 6642,
            "target_subtoken_vocab_size": 4097,
            "token_vocab_size": 65395,
            "ast_node_vocab_size": 279,
            "path_encoder": "gru",
            "representation_transform": "identity",
            "lexical_ablation": lexical_ablation,
        },
        "training": {
            "epochs": 5,
            "batch_size": 128,
            "learning_rate": 0.003,
            "use_positive_weighting": True,
            "max_positive_weight": 7.0,
            "curvature": 1.0,
            "metric": "target-subtoken micro precision/recall/F1",
            "model_seeds": [seed],
            "variant_filter": [variant],
        },
        "evaluation": {"split": "test"},
        "runs": [{"variant": variant, "model_seed": seed, "validation_f1": 0.1}],
    }


class Code2HypMergeResultsScriptTests(unittest.TestCase):
    def test_merge_results_combines_runs_and_updates_metadata(self) -> None:
        merged = merge_results_from_dicts(
            _result("B36_code2hyp_product_frechet_neighbor", 101),
            _result("B6_euclidean_metric_code2vec", 202),
        )

        self.assertEqual(len(merged["runs"]), 2)
        self.assertEqual(merged["training"]["model_seeds"], [101, 202])
        self.assertEqual(
            merged["training"]["variant_filter"],
            ["B36_code2hyp_product_frechet_neighbor", "B6_euclidean_metric_code2vec"],
        )
        self.assertEqual(merged["merge"]["n_runs"], 2)

    def test_merge_results_rejects_incompatible_lexical_ablation(self) -> None:
        with self.assertRaisesRegex(ValueError, "dataset.lexical_ablation"):
            merge_results_from_dicts(
                _result("B36_code2hyp_product_frechet_neighbor", 101),
                _result("B6_euclidean_metric_code2vec", 202, lexical_ablation="structural_only"),
            )

    def test_merge_results_rejects_duplicate_variant_seed(self) -> None:
        with self.assertRaisesRegex(ValueError, "duplicate run"):
            merge_results_from_dicts(
                _result("B36_code2hyp_product_frechet_neighbor", 101),
                _result("B36_code2hyp_product_frechet_neighbor", 101),
            )


def merge_results_from_dicts(*results: dict) -> dict:
    import json
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmpdir:
        paths = []
        for index, result in enumerate(results):
            path = Path(tmpdir) / f"result_{index}.json"
            path.write_text(json.dumps(result), encoding="utf-8")
            paths.append(path)
        return merge_results(paths)


if __name__ == "__main__":
    unittest.main()
