from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.summarize_raw_ast_factor_matrix import format_markdown, summarize_factor_matrix


def _run(
    *,
    path_object_mode: str,
    method_aggregation: str,
    geometry: str,
    curvature: float,
    seed: int,
    mrr: float,
    path_cost_orientation: str = "directed",
) -> dict[str, object]:
    return {
        "project": "toy",
        "node_input_mode": "label_only",
        "path_object_mode": path_object_mode,
        "method_aggregation": method_aggregation,
        "path_cost_orientation": path_cost_orientation,
        "geometry": geometry,
        "curvature": curvature,
        "dim": 4,
        "seed": seed,
        "recall_at_1": mrr,
        "ndcg_at_3": mrr,
        "mrr": mrr,
        "margin_mean": mrr - 0.5,
        "geometry_diagnostics": {"manifold": geometry},
    }


class SummarizeRawASTFactorMatrixTests(unittest.TestCase):
    def test_summarizer_reports_factor_contrasts(self) -> None:
        runs = [
            _run(path_object_mode="single_point", method_aggregation="centroid", geometry="euclidean", curvature=1.0, seed=11, mrr=0.40),
            _run(path_object_mode="single_point", method_aggregation="centroid", geometry="poincare", curvature=1.0, seed=11, mrr=0.45),
            _run(path_object_mode="lca_product", method_aggregation="centroid", geometry="euclidean", curvature=1.0, seed=11, mrr=0.55),
            _run(path_object_mode="lca_product", method_aggregation="measure", geometry="euclidean", curvature=1.0, seed=11, mrr=0.60),
            _run(path_object_mode="lca_product", method_aggregation="measure", geometry="poincare", curvature=1.0, seed=11, mrr=0.70),
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "matrix.json"
            input_path.write_text(
                json.dumps({"status": "partial", "completed_runs": len(runs), "expected_runs": 12, "runs": runs}),
                encoding="utf-8",
            )

            summary = summarize_factor_matrix(input_path)
            markdown = format_markdown(summary)

        curvature_rows = summary["curvature_deltas"]
        path_rows = summary["path_object_deltas"]
        aggregation_rows = summary["aggregation_deltas"]
        full_rows = summary["full_model_deltas"]
        self.assertTrue(_has_delta(curvature_rows, 0.05))
        self.assertTrue(_has_delta(path_rows, 0.15))
        self.assertTrue(_has_delta(aggregation_rows, 0.05))
        self.assertAlmostEqual(full_rows[0]["mean_delta_mrr"], 0.30)
        self.assertIn("Full Code2Hyp-v1 deltas", markdown)

    def test_summarizer_reports_orientation_contrast_without_mixing_cells(self) -> None:
        runs = [
            _run(
                path_object_mode="lca_product",
                method_aggregation="measure",
                path_cost_orientation="directed",
                geometry="euclidean",
                curvature=1.0,
                seed=11,
                mrr=0.50,
            ),
            _run(
                path_object_mode="lca_product",
                method_aggregation="measure",
                path_cost_orientation="unoriented",
                geometry="euclidean",
                curvature=1.0,
                seed=11,
                mrr=0.56,
            ),
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "matrix.json"
            input_path.write_text(
                json.dumps({"status": "partial", "completed_runs": len(runs), "expected_runs": 2, "runs": runs}),
                encoding="utf-8",
            )

            summary = summarize_factor_matrix(input_path)
            markdown = format_markdown(summary)

        self.assertEqual(len(summary["means"]), 2)
        self.assertEqual(len(summary["orientation_deltas"]), 1)
        self.assertAlmostEqual(summary["orientation_deltas"][0]["mean_delta_mrr"], 0.06)
        self.assertIn("Orientation deltas", markdown)


if __name__ == "__main__":
    unittest.main()


def _has_delta(rows: list[dict[str, object]], expected: float) -> bool:
    return any(abs(float(row["mean_delta_mrr"]) - expected) < 1e-9 for row in rows)
