from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.summarize_raw_ast_curvature_controls import format_markdown, summarize_curvature_controls


def _run(project: str, geometry: str, curvature: float, seed: int, mrr: float) -> dict[str, object]:
    diagnostics = {"manifold": geometry}
    if geometry == "poincare":
        diagnostics.update(
            {
                "curvature": curvature,
                "sqrt_curvature_norm_mean": curvature,
                "sqrt_curvature_norm_max": curvature * 2,
                "near_boundary_fraction": 0.0,
                "projection_active_fraction": 0.0,
            }
        )
    return {
        "project": project,
        "node_input_mode": "label_only",
        "geometry": geometry,
        "curvature": curvature,
        "dim": 4,
        "seed": seed,
        "recall_at_1": mrr,
        "mrr": mrr,
        "margin_mean": mrr - 0.5,
        "geometry_diagnostics": diagnostics,
    }


class SummarizeRawASTCurvatureControlsTests(unittest.TestCase):
    def test_summarizer_keeps_poincare_curvature_controls_separate(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "matrix.json"
            input_path.write_text(
                json.dumps(
                    {
                        "status": "complete",
                        "completed_runs": 3,
                        "expected_runs": 3,
                        "runs": [
                            _run("toy", "euclidean", 1.0, 11, 0.40),
                            _run("toy", "poincare", 1e-4, 11, 0.45),
                            _run("toy", "poincare", 1.0, 11, 0.55),
                        ],
                    }
                ),
                encoding="utf-8",
            )

            summary = summarize_curvature_controls(input_path)
            markdown = format_markdown(summary)

        curvatures = {row["curvature"] for row in summary["paired_deltas"]}
        deltas = {row["curvature"]: row["mean_delta_mrr"] for row in summary["paired_deltas"]}
        self.assertEqual(curvatures, {1e-4, 1.0})
        self.assertAlmostEqual(deltas[1e-4], 0.05)
        self.assertAlmostEqual(deltas[1.0], 0.15)
        self.assertIn("Positive MRR seeds", markdown)


if __name__ == "__main__":
    unittest.main()
