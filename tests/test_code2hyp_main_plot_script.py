from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.plot_code2hyp_java_small_pilot import DEFAULT_INPUTS, VARIANT_LABELS, plot_pilot_metrics


class Code2HypMainPlotScriptTests(unittest.TestCase):
    def test_main_plot_variant_labels_include_hyperbolic_code2vec_controls(self) -> None:
        self.assertIn("B4_hyperbolic_code2vec", VARIANT_LABELS)
        self.assertIn("B4T_hyperbolic_code2vec_trainable_curvature", VARIANT_LABELS)
        self.assertIn("B8_hyperbolic_frechet_code2vec", VARIANT_LABELS)
        self.assertIn("B17_hyperbolic_path_mp_code2vec", VARIANT_LABELS)
        self.assertIn("B18_hyperbolic_path_mp_struct_rank", VARIANT_LABELS)
        self.assertIn("B19_hyperbolic_path_mp_rank_annealed", VARIANT_LABELS)
        self.assertIn("B20_hyperbolic_path_mp_rank_delayed", VARIANT_LABELS)
        self.assertIn("B9_lorentz_code2vec", VARIANT_LABELS)
        self.assertIn("B15_lorentz_product_code2vec", VARIANT_LABELS)
        self.assertIn("B10_factorized_product_code2vec", VARIANT_LABELS)
        self.assertIn("B11_factorized_product_struct_rank", VARIANT_LABELS)
        self.assertIn("B12_factorized_product_learned_metric_rank", VARIANT_LABELS)
        self.assertIn("B16_factorized_product_three_metric_rank", VARIANT_LABELS)
        self.assertIn("B13_factorized_product_channel_mixer_rank", VARIANT_LABELS)
        self.assertIn("B7_hyperbolic_attention_only", VARIANT_LABELS)
        self.assertIn("B6_euclidean_metric_code2vec", VARIANT_LABELS)
        self.assertIn("B14_bounded_euclidean_metric_code2vec", VARIANT_LABELS)
        self.assertIn("B_tree_euclidean_lca_bias", VARIANT_LABELS)

    def test_default_inputs_include_current_and_legacy_control_artifacts(self) -> None:
        self.assertIn("focused_b20", str(DEFAULT_INPUTS[0][1]))
        self.assertIn("with_b14_evalfix_3seeds", str(DEFAULT_INPUTS[1][1]))

    def test_plot_pilot_metrics_writes_outputs_for_two_dataset_sizes(self) -> None:
        variants = tuple(VARIANT_LABELS)
        base_result = {
            "runs": [
                {
                    "variant": variant,
                    "validation_f1": 0.1 + index * 0.01,
                    "validation_structural_loss": 0.8 - index * 0.05,
                    "validation_structural_rank_loss": 0.2 + index * 0.01,
                    "validation_structural_spearman": -0.3 + index * 0.1,
                }
                for index, variant in enumerate(variants)
            ]
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            first_path = Path(tmpdir) / "first.json"
            second_path = Path(tmpdir) / "second.json"
            output_prefix = Path(tmpdir) / "pilot"
            first_path.write_text(json.dumps(base_result), encoding="utf-8")
            second_path.write_text(json.dumps(base_result), encoding="utf-8")

            plot_pilot_metrics((("1k", first_path), ("4k", second_path)), output_prefix)

            self.assertTrue(output_prefix.with_suffix(".png").exists())
            self.assertTrue(output_prefix.with_suffix(".pdf").exists())


if __name__ == "__main__":
    unittest.main()
