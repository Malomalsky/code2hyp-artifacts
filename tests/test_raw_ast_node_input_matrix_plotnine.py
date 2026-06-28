from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.plot_raw_ast_node_input_matrix_plotnine import build_paired_delta_rows, plot_node_input_matrix_deltas


class RawASTNodeInputMatrixPlotnineTests(unittest.TestCase):
    def test_build_paired_delta_rows_pairs_poincare_against_euclidean(self) -> None:
        payload = {
            "runs": [
                {
                    "project": "p1",
                    "node_input_mode": "label_only",
                    "geometry": "euclidean",
                    "dim": 4,
                    "seed": 1,
                    "mrr": 0.30,
                    "recall_at_1": 0.20,
                    "margin_mean": -0.10,
                },
                {
                    "project": "p1",
                    "node_input_mode": "label_only",
                    "geometry": "poincare",
                    "dim": 4,
                    "seed": 1,
                    "mrr": 0.40,
                    "recall_at_1": 0.25,
                    "margin_mean": -0.05,
                },
            ]
        }

        rows = build_paired_delta_rows([payload])

        self.assertEqual(len(rows), 3)
        self.assertEqual({row["metric"] for row in rows}, {"MRR", "Recall@1", "Mean margin"})
        mrr_row = next(row for row in rows if row["metric"] == "MRR")
        self.assertAlmostEqual(mrr_row["delta"], 0.10)
        self.assertEqual(mrr_row["node_input_mode"], "label_only")

    def test_plot_node_input_matrix_deltas_writes_png_and_pdf(self) -> None:
        payload = {
            "runs": [
                {
                    "project": "p1",
                    "node_input_mode": "label_only",
                    "geometry": "euclidean",
                    "dim": 4,
                    "seed": 1,
                    "mrr": 0.30,
                    "recall_at_1": 0.20,
                    "margin_mean": -0.10,
                },
                {
                    "project": "p1",
                    "node_input_mode": "label_only",
                    "geometry": "poincare",
                    "dim": 4,
                    "seed": 1,
                    "mrr": 0.40,
                    "recall_at_1": 0.25,
                    "margin_mean": -0.05,
                },
                {
                    "project": "p1",
                    "node_input_mode": "label_depth_prefix",
                    "geometry": "euclidean",
                    "dim": 4,
                    "seed": 1,
                    "mrr": 0.42,
                    "recall_at_1": 0.30,
                    "margin_mean": 0.01,
                },
                {
                    "project": "p1",
                    "node_input_mode": "label_depth_prefix",
                    "geometry": "poincare",
                    "dim": 4,
                    "seed": 1,
                    "mrr": 0.37,
                    "recall_at_1": 0.25,
                    "margin_mean": -0.02,
                },
            ]
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "matrix.json"
            output_prefix = Path(tmpdir) / "matrix_deltas"
            input_path.write_text(json.dumps(payload), encoding="utf-8")

            png_path, pdf_path = plot_node_input_matrix_deltas(
                inputs=(input_path,),
                output_prefix=output_prefix,
            )

            self.assertTrue(png_path.exists())
            self.assertTrue(pdf_path.exists())
            self.assertGreater(png_path.stat().st_size, 0)
            self.assertGreater(pdf_path.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
