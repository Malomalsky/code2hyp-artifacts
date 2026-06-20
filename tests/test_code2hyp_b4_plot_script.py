from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.plot_code2hyp_b4_pilot import DEFAULT_INPUT, VARIANT_ORDER, plot_b4_pilot_metrics


class Code2HypB4PlotScriptTests(unittest.TestCase):
    def test_plot_b4_pilot_metrics_writes_png_and_pdf(self) -> None:
        variants = (
            "B1_euclidean",
            "B2_product_fixed_curvature",
            "B3_product",
            "B4_hyperbolic_code2vec",
            "B4T_hyperbolic_code2vec_trainable_curvature",
            "B8_hyperbolic_frechet_code2vec",
            "B17_hyperbolic_path_mp_code2vec",
            "B18_hyperbolic_path_mp_struct_rank",
            "B19_hyperbolic_path_mp_rank_annealed",
            "B20_hyperbolic_path_mp_rank_delayed",
            "B9_lorentz_code2vec",
            "B15_lorentz_product_code2vec",
            "B10_factorized_product_code2vec",
            "B11_factorized_product_struct_rank",
            "B12_factorized_product_learned_metric_rank",
            "B16_factorized_product_three_metric_rank",
            "B13_factorized_product_channel_mixer_rank",
            "B7_hyperbolic_attention_only",
            "B5_euclidean_struct_loss",
            "B6_euclidean_metric_code2vec",
            "B14_bounded_euclidean_metric_code2vec",
            "B_tree_euclidean_lca_bias",
        )
        result = {
            "runs": [
                {
                    "variant": variant,
                    "validation_f1": 0.1 + index * 0.01,
                    "validation_structural_loss": 0.8 - index * 0.1,
                    "validation_structural_rank_loss": 0.2 + index * 0.01,
                    "validation_structural_spearman": -0.3 + index * 0.1,
                }
                for index, variant in enumerate(variants)
            ]
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "pilot.json"
            output_prefix = Path(tmpdir) / "figure"
            input_path.write_text(json.dumps(result), encoding="utf-8")

            plot_b4_pilot_metrics(input_path, output_prefix)

            self.assertTrue(output_prefix.with_suffix(".png").exists())
            self.assertTrue(output_prefix.with_suffix(".pdf").exists())

    def test_plot_variant_order_includes_b4_trainable_curvature_control(self) -> None:
        self.assertIn("B4T_hyperbolic_code2vec_trainable_curvature", VARIANT_ORDER)
        self.assertIn("B8_hyperbolic_frechet_code2vec", VARIANT_ORDER)
        self.assertIn("B17_hyperbolic_path_mp_code2vec", VARIANT_ORDER)
        self.assertIn("B18_hyperbolic_path_mp_struct_rank", VARIANT_ORDER)
        self.assertIn("B19_hyperbolic_path_mp_rank_annealed", VARIANT_ORDER)
        self.assertIn("B20_hyperbolic_path_mp_rank_delayed", VARIANT_ORDER)
        self.assertIn("B9_lorentz_code2vec", VARIANT_ORDER)
        self.assertIn("B15_lorentz_product_code2vec", VARIANT_ORDER)
        self.assertIn("B10_factorized_product_code2vec", VARIANT_ORDER)
        self.assertIn("B11_factorized_product_struct_rank", VARIANT_ORDER)
        self.assertIn("B12_factorized_product_learned_metric_rank", VARIANT_ORDER)
        self.assertIn("B16_factorized_product_three_metric_rank", VARIANT_ORDER)
        self.assertIn("B13_factorized_product_channel_mixer_rank", VARIANT_ORDER)
        self.assertIn("B7_hyperbolic_attention_only", VARIANT_ORDER)
        self.assertIn("B6_euclidean_metric_code2vec", VARIANT_ORDER)
        self.assertIn("B14_bounded_euclidean_metric_code2vec", VARIANT_ORDER)
        self.assertIn("B_tree_euclidean_lca_bias", VARIANT_ORDER)

    def test_default_input_uses_hyperbolic_path_message_passing_rank_artifact(self) -> None:
        self.assertIn("focused_b20", str(DEFAULT_INPUT))

    def test_plot_b4_pilot_metrics_accepts_focused_variant_subset(self) -> None:
        result = {
            "runs": [
                {
                    "variant": variant,
                    "validation_f1": 0.1 + index * 0.01,
                    "validation_structural_loss": 0.8 - index * 0.1,
                    "validation_structural_rank_loss": 0.2 + index * 0.01,
                    "validation_structural_spearman": -0.3 + index * 0.1,
                }
                for index, variant in enumerate(
                    (
                        "B4_hyperbolic_code2vec",
                        "B20_hyperbolic_path_mp_rank_delayed",
                    )
                )
            ]
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "focused.json"
            output_prefix = Path(tmpdir) / "focused_figure"
            input_path.write_text(json.dumps(result), encoding="utf-8")

            plot_b4_pilot_metrics(input_path, output_prefix)

            self.assertTrue(output_prefix.with_suffix(".png").exists())
            self.assertTrue(output_prefix.with_suffix(".pdf").exists())


if __name__ == "__main__":
    unittest.main()
