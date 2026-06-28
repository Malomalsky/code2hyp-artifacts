from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.summarize_dta_task_transfer import format_markdown, summarize_dta_task_transfer


def _run(project: str, geometry: str, curvature: float, seed: int, mrr: float) -> dict[str, object]:
    return {
        "project": project,
        "geometry": geometry,
        "curvature": curvature,
        "seed": seed,
        "mrr": mrr,
        "recall_at_5": mrr / 2.0,
        "mean_rank": 1.0 / max(mrr, 1e-6),
    }


class SummarizeDTATaskTransferTests(unittest.TestCase):
    def test_reports_task_macro_deltas_against_euclidean(self) -> None:
        payload = {
            "status": "complete",
            "completed_runs": 8,
            "expected_runs": 8,
            "runs": [
                _run("task-a", "euclidean", 1.0, 1, 0.10),
                _run("task-a", "euclidean", 1.0, 2, 0.20),
                _run("task-a", "poincare", 1.0, 1, 0.30),
                _run("task-a", "poincare", 1.0, 2, 0.40),
                _run("task-b", "euclidean", 1.0, 1, 0.20),
                _run("task-b", "euclidean", 1.0, 2, 0.30),
                _run("task-b", "poincare", 1.0, 1, 0.25),
                _run("task-b", "poincare", 1.0, 2, 0.35),
            ],
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "dta.json"
            path.write_text(json.dumps(payload), encoding="utf-8")

            summary = summarize_dta_task_transfer(path, bootstrap_samples=50, seed=7)
            markdown = format_markdown(summary)

        macro = summary["macro_delta_rows"][0]
        self.assertEqual(macro["n_tasks"], 2)
        self.assertEqual(macro["positive_tasks"], 2)
        self.assertAlmostEqual(macro["macro_delta_mrr"], 0.125)
        self.assertIn("Unit of external uncertainty", markdown)
        self.assertIn("task-a", markdown)


if __name__ == "__main__":
    unittest.main()
