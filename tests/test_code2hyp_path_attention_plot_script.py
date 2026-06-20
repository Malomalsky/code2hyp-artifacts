from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.plot_code2hyp_path_attention import VARIANT_ORDER, plot_path_attention


class Code2HypPathAttentionPlotScriptTests(unittest.TestCase):
    def test_variant_order_includes_depth_attention_candidate(self) -> None:
        self.assertIn("B25_hyperbolic_path_depth_attention_mp_code2vec", VARIANT_ORDER)
        self.assertIn("B26_hyperbolic_path_depth_attention_mp_rank_annealed", VARIANT_ORDER)
        self.assertIn("B27_hyperbolic_path_attention_mp_monotone", VARIANT_ORDER)
        self.assertIn("B28_hyperbolic_path_attention_mp_tree_distance", VARIANT_ORDER)
        self.assertIn("B29_hyperbolic_path_dual_attention_mp_separated", VARIANT_ORDER)
        self.assertIn("B30_hyperbolic_path_dual_attention_mp_rank_separated", VARIANT_ORDER)
        self.assertIn("B31_hyperbolic_path_dual_attention_mp_soft_rank", VARIANT_ORDER)
        self.assertIn("B32_lorentz_path_dual_attention_mp_soft_rank", VARIANT_ORDER)
        self.assertIn("B34_hyperbolic_path_dual_attention_mp_adaptive_rank", VARIANT_ORDER)

    def test_plot_path_attention_writes_png_and_pdf(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            inputs = []
            for label, regime in (("512 / 3 epochs", "Original"), ("512 / 3 epochs", "Structural only")):
                path = root / f"{regime.replace(' ', '_').lower()}.json"
                path.write_text(
                    json.dumps(
                        {
                            "runs": [
                                {
                                    "variant": "B17_hyperbolic_path_mp_code2vec",
                                    "model_seed": 101,
                                    "validation_f1": 0.18,
                                    "validation_structural_loss": 0.12,
                                    "validation_structural_rank_loss": 0.30,
                                    "validation_structural_spearman": 0.40,
                                },
                                {
                                    "variant": "B23_hyperbolic_path_attention_mp_code2vec",
                                    "model_seed": 101,
                                    "validation_f1": 0.19,
                                    "validation_structural_loss": 0.13,
                                    "validation_structural_rank_loss": 0.28,
                                    "validation_structural_spearman": 0.30,
                                },
                                {
                                    "variant": "B24_hyperbolic_path_attention_mp_rank_annealed",
                                    "model_seed": 101,
                                    "validation_f1": 0.20,
                                    "validation_structural_loss": 0.11,
                                    "validation_structural_rank_loss": 0.25,
                                    "validation_structural_spearman": 0.35,
                                },
                                {
                                    "variant": "B25_hyperbolic_path_depth_attention_mp_code2vec",
                                    "model_seed": 101,
                                    "validation_f1": 0.21,
                                    "validation_structural_loss": 0.10,
                                    "validation_structural_rank_loss": 0.22,
                                    "validation_structural_spearman": 0.37,
                                },
                                {
                                    "variant": "B26_hyperbolic_path_depth_attention_mp_rank_annealed",
                                    "model_seed": 101,
                                    "validation_f1": 0.22,
                                    "validation_structural_loss": 0.09,
                                    "validation_structural_rank_loss": 0.21,
                                    "validation_structural_spearman": 0.38,
                                },
                                {
                                    "variant": "B27_hyperbolic_path_attention_mp_monotone",
                                    "model_seed": 101,
                                    "validation_f1": 0.23,
                                    "validation_structural_loss": 0.08,
                                    "validation_structural_rank_loss": 0.20,
                                    "validation_structural_spearman": 0.39,
                                },
                                {
                                    "variant": "B28_hyperbolic_path_attention_mp_tree_distance",
                                    "model_seed": 101,
                                    "validation_f1": 0.24,
                                    "validation_structural_loss": 0.07,
                                    "validation_structural_rank_loss": 0.19,
                                    "validation_structural_spearman": 0.41,
                                },
                                {
                                    "variant": "B29_hyperbolic_path_dual_attention_mp_separated",
                                    "model_seed": 101,
                                    "validation_f1": 0.25,
                                    "validation_structural_loss": 0.06,
                                    "validation_structural_rank_loss": 0.18,
                                    "validation_structural_spearman": 0.42,
                                },
                                {
                                    "variant": "B30_hyperbolic_path_dual_attention_mp_rank_separated",
                                    "model_seed": 101,
                                    "validation_f1": 0.26,
                                    "validation_structural_loss": 0.05,
                                    "validation_structural_rank_loss": 0.17,
                                    "validation_structural_spearman": 0.43,
                                },
                                {
                                    "variant": "B31_hyperbolic_path_dual_attention_mp_soft_rank",
                                    "model_seed": 101,
                                    "validation_f1": 0.27,
                                    "validation_structural_loss": 0.04,
                                    "validation_structural_rank_loss": 0.16,
                                    "validation_structural_spearman": 0.44,
                                },
                                {
                                    "variant": "B32_lorentz_path_dual_attention_mp_soft_rank",
                                    "model_seed": 101,
                                    "validation_f1": 0.28,
                                    "validation_structural_loss": 0.03,
                                    "validation_structural_rank_loss": 0.15,
                                    "validation_structural_spearman": 0.45,
                                },
                                {
                                    "variant": "B34_hyperbolic_path_dual_attention_mp_adaptive_rank",
                                    "model_seed": 101,
                                    "validation_f1": 0.29,
                                    "validation_structural_loss": 0.02,
                                    "validation_structural_rank_loss": 0.14,
                                    "validation_structural_spearman": 0.46,
                                },
                            ]
                        }
                    ),
                    encoding="utf-8",
                )
                inputs.append((label, regime, path))

            output_prefix = root / "path_attention"
            plot_path_attention(tuple(inputs), output_prefix)

            self.assertTrue(output_prefix.with_suffix(".png").exists())
            self.assertTrue(output_prefix.with_suffix(".pdf").exists())


if __name__ == "__main__":
    unittest.main()
