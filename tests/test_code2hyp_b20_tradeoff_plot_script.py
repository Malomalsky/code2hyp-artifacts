from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.plot_code2hyp_b20_tradeoff import plot_b20_tradeoff


class Code2HypB20TradeoffPlotScriptTests(unittest.TestCase):
    def test_plot_b20_tradeoff_writes_png_and_pdf(self) -> None:
        variants = (
            "B4_hyperbolic_code2vec",
            "B8_hyperbolic_frechet_code2vec",
            "B19_hyperbolic_path_mp_rank_annealed",
            "B20_hyperbolic_path_mp_rank_delayed",
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            original_path = root / "original.json"
            structural_path = root / "structural.json"
            output_prefix = root / "b20_tradeoff"
            for path, spearman_offset in ((original_path, 0.3), (structural_path, -0.1)):
                payload = {
                    "runs": [
                        {
                            "variant": variant,
                            "validation_f1": 0.12 + 0.01 * index,
                            "validation_structural_spearman": spearman_offset + 0.02 * index,
                            "validation_structural_loss": 0.20 - 0.01 * index,
                            "validation_structural_rank_loss": 0.50 + 0.02 * index,
                        }
                        for index, variant in enumerate(variants)
                    ]
                }
                path.write_text(json.dumps(payload), encoding="utf-8")

            plot_b20_tradeoff(original_path, structural_path, output_prefix)

            self.assertTrue(output_prefix.with_suffix(".png").exists())
            self.assertTrue(output_prefix.with_suffix(".pdf").exists())


if __name__ == "__main__":
    unittest.main()
