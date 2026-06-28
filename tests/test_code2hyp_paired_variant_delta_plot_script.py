from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.plot_code2hyp_paired_variant_deltas import plot_paired_variant_deltas


class Code2HypPairedVariantDeltaPlotScriptTests(unittest.TestCase):
    def test_plot_paired_variant_deltas_writes_png_and_pdf(self) -> None:
        payload = {
            "runs": [
                {
                    "variant": "B85",
                    "model_seed": 101,
                    "validation_f1": 0.10,
                    "validation_fixed_top3_f1": 0.12,
                    "validation_structural_spearman": 0.20,
                    "validation_method_transport_spearman": 0.30,
                    "validation_method_transport_normalized_stress": 0.20,
                },
                {
                    "variant": "B85",
                    "model_seed": 103,
                    "validation_f1": 0.20,
                    "validation_fixed_top3_f1": 0.15,
                    "validation_structural_spearman": 0.30,
                    "validation_method_transport_spearman": 0.40,
                    "validation_method_transport_normalized_stress": 0.18,
                },
                {
                    "variant": "B86",
                    "model_seed": 101,
                    "validation_f1": 0.12,
                    "validation_fixed_top3_f1": 0.12,
                    "validation_structural_spearman": 0.35,
                    "validation_method_transport_spearman": 0.70,
                    "validation_method_transport_normalized_stress": 0.12,
                },
                {
                    "variant": "B86",
                    "model_seed": 103,
                    "validation_f1": 0.23,
                    "validation_fixed_top3_f1": 0.16,
                    "validation_structural_spearman": 0.45,
                    "validation_method_transport_spearman": 0.80,
                    "validation_method_transport_normalized_stress": 0.11,
                },
            ]
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "runs.json"
            output_prefix = Path(tmpdir) / "paired_deltas"
            input_path.write_text(json.dumps(payload), encoding="utf-8")

            png_path, pdf_path = plot_paired_variant_deltas(
                [input_path],
                baseline="B85",
                candidate="B86",
                output_prefix=output_prefix,
            )

            self.assertTrue(png_path.exists())
            self.assertTrue(pdf_path.exists())
            self.assertGreater(png_path.stat().st_size, 0)
            self.assertGreater(pdf_path.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
