from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.plot_code2hyp_schedule_sweep import plot_schedule_sweep


class Code2HypScheduleSweepPlotScriptTests(unittest.TestCase):
    def test_plot_schedule_sweep_writes_png_and_pdf(self) -> None:
        variants = (
            "B4_hyperbolic_code2vec",
            "B17_hyperbolic_path_mp_code2vec",
            "B21_hyperbolic_path_mp_rank_cosine",
            "B22_hyperbolic_path_mp_rank_warmup_decay",
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            inputs: list[tuple[str, str, Path]] = []
            for row_label in ("512 / 3 epochs", "256 / 5 epochs"):
                for column_label in ("Original", "Structural only"):
                    path = root / f"{row_label}_{column_label}.json".replace(" ", "_").replace("/", "-")
                    payload = {
                        "runs": [
                            {
                                "variant": variant,
                                "validation_f1": 0.10 + 0.01 * variant_index,
                                "validation_structural_spearman": -0.20 + 0.05 * variant_index,
                                "validation_structural_loss": 0.30 - 0.01 * variant_index,
                                "validation_structural_rank_loss": 0.50 + 0.01 * variant_index,
                            }
                            for variant_index, variant in enumerate(variants)
                        ]
                    }
                    path.write_text(json.dumps(payload), encoding="utf-8")
                    inputs.append((row_label, column_label, path))

            output_prefix = root / "schedule_sweep"
            plot_schedule_sweep(tuple(inputs), output_prefix)

            self.assertTrue(output_prefix.with_suffix(".png").exists())
            self.assertTrue(output_prefix.with_suffix(".pdf").exists())


if __name__ == "__main__":
    unittest.main()
