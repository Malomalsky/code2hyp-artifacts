"""Small checks for the fixed benchmark's selection and completion boundary."""

import unittest
import json
from pathlib import Path
import tempfile
import numpy as np
from e2e_common_20261014 import select_queries, order, rerank, summary, validate_times


class BenchmarkTests(unittest.TestCase):
    def test_selection_is_order_independent_and_one_per_problem(self):
        rows = [{"id": str(i), "problem": str(i // 3)} for i in range(12)]
        first = select_queries(rows)
        self.assertEqual([rows[i]["id"] for i in first],
                         [rows[::-1][i]["id"] for i in select_queries(rows[::-1])])
        self.assertEqual(len({rows[i]["problem"] for i in first}), 4)

    def test_ties_and_unchanged_tail(self):
        tie = np.array([b"z", b"a", b"b"])
        dense = order(np.array([1., 1., 0.]), tie)
        np.testing.assert_array_equal(dense, [1, 0, 2])
        np.testing.assert_array_equal(rerank(dense, np.array([0., 2.]), tie), [0, 1, 2])

    def test_summary_requires_three_complete_passes(self):
        values = np.full((3, 4), .2)
        report = summary(values)
        self.assertAlmostEqual(report["mean_seconds"], .2)
        self.assertAlmostEqual(report["p95_seconds"], .2)
        for bad in (values[:2], np.array([[np.nan]]), np.zeros((3, 4))):
            with self.assertRaises(ValueError):
                validate_times(bad, 4)

    def test_missing_complete_receipt_cannot_be_evaluated(self):
        from evaluate_e2e_20261014 import require_complete
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(FileNotFoundError):
                require_complete(Path(folder), Path(folder))

    def test_partial_audit_rejects_tampered_worker_files(self):
        from audit_e2e_partial_20261015 import verify_worker
        from e2e_common_20261014 import sha
        with tempfile.TemporaryDirectory() as name:
            folder = Path(name)
            data = folder / "requests.jsonl"
            data.write_text("original")
            (folder / "index_receipt.json").write_text("{}")
            (folder / "memory.json").write_text("{}")
            report = {"status": "complete", "method": "BM25", "host_id": "h", "binding": "b",
                      "requests": 1200, "quality_queries": 3200, "training_performed": False,
                      "java_test_accessed": False, "files": {"requests.jsonl": sha(data)},
                      "index": {}, "memory": {}}
            (folder / "complete.json").write_text(json.dumps(report))
            verify_worker(folder, "BM25", "h", "b")
            data.write_text("changed")
            with self.assertRaises(AssertionError):
                verify_worker(folder, "BM25", "h", "b")


if __name__ == "__main__":
    unittest.main()
