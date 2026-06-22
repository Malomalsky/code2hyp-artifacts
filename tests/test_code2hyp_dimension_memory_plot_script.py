from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts.plot_code2hyp_dimension_memory_ablation import plot_dimension_ablation


class Code2HypDimensionMemoryPlotScriptTests(unittest.TestCase):
    def test_plot_dimension_ablation_writes_png_and_pdf(self) -> None:
        csv_text = """structural_dim,variant,variant_label,n_seeds,token_dim,representation_dim,parameter_count,parameter_memory_mib_float32,activation_proxy_kib_per_example_float32,validation_f1_mean,validation_f1_sd,validation_structural_spearman_mean,validation_structural_spearman_sd,validation_structural_normalized_stress_mean,validation_structural_normalized_stress_sd,validation_structural_neighbor_overlap_at_3_mean,validation_structural_neighbor_overlap_at_3_sd
4,B44,Code2Hyp B44,3,32,68,1000,0.01,8,0.1,0.01,0.5,0.02,0.4,0.03,0.6,0.04
4,B46,Euclidean B46,3,32,68,995,0.01,8,0.1,0.01,0.1,0.02,0.7,0.03,0.4,0.04
8,B44,Code2Hyp B44,3,32,72,1200,0.02,8,0.2,0.01,0.8,0.02,0.2,0.03,0.8,0.04
8,B46,Euclidean B46,3,32,72,1195,0.02,8,0.1,0.01,0.2,0.02,0.6,0.03,0.5,0.04
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "summary.csv"
            output_prefix = Path(tmpdir) / "figure"
            input_path.write_text(csv_text, encoding="utf-8")

            png_path, pdf_path = plot_dimension_ablation(input_path, output_prefix)

            self.assertTrue(png_path.exists())
            self.assertTrue(pdf_path.exists())
            self.assertGreater(png_path.stat().st_size, 0)
            self.assertGreater(pdf_path.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
