from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.plot_raw_ast_code2hyp_retrieval_plotnine import (
    build_metric_rows,
    plot_retrieval_results,
)


class RawASTCode2HypRetrievalPlotnineTests(unittest.TestCase):
    def test_build_metric_rows_flattens_retrieval_json_for_grammar_of_graphics(self) -> None:
        payload = _payload("euclidean", "python", mrr=0.75, margin=0.12)

        rows = build_metric_rows((("Euclidean", payload),))

        self.assertTrue(any(row["metric"] == "MRR" and row["value"] == 0.75 for row in rows))
        self.assertTrue(any(row["metric"] == "Mean margin" and row["value"] == 0.12 for row in rows))
        self.assertTrue(all(row["variant"] == "Euclidean" for row in rows))
        self.assertTrue(all(row["language"] == "python" for row in rows))

    def test_plot_retrieval_results_writes_png_and_pdf(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            left = root / "euclidean.json"
            right = root / "poincare.json"
            left.write_text(json.dumps(_payload("euclidean", "python", mrr=0.75, margin=0.12)), encoding="utf-8")
            right.write_text(json.dumps(_payload("poincare", "python", mrr=0.83, margin=0.18)), encoding="utf-8")
            output_prefix = root / "retrieval_plot"

            plot_retrieval_results(
                inputs=(("Euclidean", left), ("Poincare", right)),
                output_prefix=output_prefix,
            )

            self.assertGreater(output_prefix.with_suffix(".png").stat().st_size, 0)
            self.assertGreater(output_prefix.with_suffix(".pdf").stat().st_size, 0)


def _payload(geometry: str, language: str, *, mrr: float, margin: float) -> dict:
    return {
        "config": {
            "geometry": geometry,
            "language": language,
            "dim": 4,
            "curvature": 1.0,
            "terminal_policy": "class",
        },
        "item_count": 12,
        "vocab_size": 300,
        "metrics": {
            "mrr": mrr,
            "recall_at_1": mrr,
            "recall_at_3": 1.0,
            "positive_distance_mean": 0.0,
            "nearest_negative_distance_mean": margin,
            "margin_mean": margin,
            "margin_min": 0.01,
        },
    }


if __name__ == "__main__":
    unittest.main()
