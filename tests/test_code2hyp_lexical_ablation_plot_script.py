from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.plot_code2hyp_lexical_ablation import (
    VARIANT_ORDER,
    plot_lexical_ablation_metrics,
)


class Code2HypLexicalAblationPlotScriptTests(unittest.TestCase):
    def test_plot_lexical_ablation_metrics_writes_png_and_pdf(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            inputs = []
            for mode_index, mode_label in enumerate(("Original", "Obfuscated", "Structural only")):
                path = root / f"{mode_index}.json"
                payload = {
                    "runs": [
                        {
                            "variant": variant,
                            "validation_f1": 0.10 + mode_index * 0.01 + variant_index * 0.001,
                            "validation_structural_loss": 0.80 - mode_index * 0.02 - variant_index * 0.001,
                            "validation_structural_rank_loss": 0.20 + variant_index * 0.001,
                            "validation_structural_spearman": -0.30 + mode_index * 0.02 + variant_index * 0.01,
                        }
                        for variant_index, variant in enumerate(VARIANT_ORDER)
                    ]
                }
                path.write_text(json.dumps(payload), encoding="utf-8")
                inputs.append((mode_label, path))

            output_prefix = root / "lexical_ablation"
            plot_lexical_ablation_metrics(tuple(inputs), output_prefix)

            self.assertTrue(output_prefix.with_suffix(".png").exists())
            self.assertTrue(output_prefix.with_suffix(".pdf").exists())

    def test_plot_variant_order_contains_primary_geometry_controls(self) -> None:
        self.assertIn("B4_hyperbolic_code2vec", VARIANT_ORDER)
        self.assertIn("B6_euclidean_metric_code2vec", VARIANT_ORDER)
        self.assertIn("B17_hyperbolic_path_mp_code2vec", VARIANT_ORDER)
        self.assertIn("B18_hyperbolic_path_mp_struct_rank", VARIANT_ORDER)
        self.assertIn("B19_hyperbolic_path_mp_rank_annealed", VARIANT_ORDER)
        self.assertIn("B20_hyperbolic_path_mp_rank_delayed", VARIANT_ORDER)
        self.assertIn("B15_lorentz_product_code2vec", VARIANT_ORDER)
        self.assertIn("B16_factorized_product_three_metric_rank", VARIANT_ORDER)
        self.assertIn("B14_bounded_euclidean_metric_code2vec", VARIANT_ORDER)
        self.assertIn("B_tree_euclidean_lca_bias", VARIANT_ORDER)

    def test_plot_lexical_ablation_metrics_accepts_focused_variant_subset(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            inputs = []
            focused_variants = (
                "B4_hyperbolic_code2vec",
                "B20_hyperbolic_path_mp_rank_delayed",
            )
            for mode_index, mode_label in enumerate(("Original", "Obfuscated", "Structural only")):
                path = root / f"focused_{mode_index}.json"
                payload = {
                    "runs": [
                        {
                            "variant": variant,
                            "validation_f1": 0.10 + mode_index * 0.01 + variant_index * 0.001,
                            "validation_structural_loss": 0.80 - mode_index * 0.02 - variant_index * 0.001,
                            "validation_structural_rank_loss": 0.20 + variant_index * 0.001,
                            "validation_structural_spearman": -0.30 + mode_index * 0.02 + variant_index * 0.01,
                        }
                        for variant_index, variant in enumerate(focused_variants)
                    ]
                }
                path.write_text(json.dumps(payload), encoding="utf-8")
                inputs.append((mode_label, path))

            output_prefix = root / "focused_lexical_ablation"
            plot_lexical_ablation_metrics(tuple(inputs), output_prefix)

            self.assertTrue(output_prefix.with_suffix(".png").exists())
            self.assertTrue(output_prefix.with_suffix(".pdf").exists())


if __name__ == "__main__":
    unittest.main()
