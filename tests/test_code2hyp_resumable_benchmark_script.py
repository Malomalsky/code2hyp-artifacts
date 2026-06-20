from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.run_code2hyp_resumable_benchmark import (
    completed_run_keys,
    mark_complete,
    merge_single_run_result,
    write_json_atomic,
)


def _single_result(variant: str, seed: int, f1: float) -> dict:
    return {
        "experiment": "real_code2hyp_multilabel_pilot",
        "dataset": {"train_records": 10},
        "training": {"epochs": 1},
        "runs": [
            {
                "variant": variant,
                "model_seed": seed,
                "validation_f1": f1,
            }
        ],
    }


class Code2HypResumableBenchmarkScriptTests(unittest.TestCase):
    def test_merge_single_run_result_appends_and_deduplicates_by_variant_seed(self) -> None:
        first = merge_single_run_result(None, _single_result("B39", 101, 0.1))
        second = merge_single_run_result(first, _single_result("B36", 101, 0.2))
        duplicate = merge_single_run_result(second, _single_result("B39", 101, 0.3))

        self.assertEqual(completed_run_keys(duplicate), {("B39", 101), ("B36", 101)})
        self.assertEqual(len(duplicate["runs"]), 2)
        self.assertEqual(duplicate["resumable_benchmark"]["completed_runs"], 2)

    def test_mark_complete_reports_partial_or_complete_status(self) -> None:
        result = merge_single_run_result(None, _single_result("B39", 101, 0.1))

        self.assertEqual(mark_complete(result, expected_runs=2)["resumable_benchmark"]["status"], "partial")
        self.assertEqual(mark_complete(result, expected_runs=1)["resumable_benchmark"]["status"], "complete")

    def test_mark_complete_updates_full_run_metadata(self) -> None:
        result = merge_single_run_result(None, _single_result("B39", 101, 0.1))

        marked = mark_complete(
            result,
            expected_runs=1,
            model_seeds=(101, 202, 303),
            variant_filter=("B39", "B36"),
            eval_split="test",
        )

        self.assertEqual(marked["training"]["model_seeds"], [101, 202, 303])
        self.assertEqual(marked["training"]["variant_filter"], ["B39", "B36"])
        self.assertEqual(marked["evaluation"]["split"], "test")

    def test_write_json_atomic_writes_valid_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "result.json"

            write_json_atomic(output, {"runs": [{"variant": "B39", "model_seed": 101}]})

            self.assertEqual(json.loads(output.read_text(encoding="utf-8"))["runs"][0]["variant"], "B39")
            self.assertFalse(output.with_suffix(".json.tmp").exists())


if __name__ == "__main__":
    unittest.main()
