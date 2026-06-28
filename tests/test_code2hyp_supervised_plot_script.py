from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.plot_code2hyp_supervised_plotnine import build_supervised_metric_rows, plot_supervised_results


class Code2HypSupervisedPlotScriptTests(unittest.TestCase):
    def test_build_supervised_metric_rows_extracts_validation_metrics(self) -> None:
        payload = {
            "runs": [
                {
                    "variant": "euclidean",
                    "validation_f1": 0.10,
                    "validation_fixed_top3_f1": 0.08,
                    "validation_precision": 0.11,
                    "validation_recall": 0.09,
                }
            ]
        }

        rows = build_supervised_metric_rows(payload)

        self.assertEqual({row["metric"] for row in rows}, {"Oracle-top-k F1", "Fixed-top-3 F1", "Precision", "Recall"})
        self.assertEqual(rows[0]["variant"], "euclidean")

    def test_plot_supervised_results_writes_png_and_pdf(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            input_path = root / "result.json"
            output_prefix = root / "figure"
            input_path.write_text(
                json.dumps(
                    {
                        "runs": [
                            {
                                "variant": "euclidean",
                                "validation_f1": 0.10,
                                "validation_fixed_top3_f1": 0.08,
                                "validation_precision": 0.11,
                                "validation_recall": 0.09,
                            },
                            {
                                "variant": "poincare",
                                "validation_f1": 0.12,
                                "validation_fixed_top3_f1": 0.07,
                                "validation_precision": 0.13,
                                "validation_recall": 0.11,
                            },
                        ]
                    }
                ),
                encoding="utf-8",
            )

            plot_supervised_results(input_path=input_path, output_prefix=output_prefix)

            self.assertTrue(output_prefix.with_suffix(".png").exists())
            self.assertTrue(output_prefix.with_suffix(".pdf").exists())


if __name__ == "__main__":
    unittest.main()
