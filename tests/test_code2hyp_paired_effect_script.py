from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.summarize_code2hyp_paired_effects import DEFAULT_RIGHT_VARIANTS, build_paired_effects_markdown


class Code2HypPairedEffectScriptTests(unittest.TestCase):
    def test_build_paired_effects_markdown_reports_requested_comparisons(self) -> None:
        result = {
            "runs": [
                {"variant": "B4", "model_seed": 1, "validation_f1": 0.30},
                {"variant": "B6", "model_seed": 1, "validation_f1": 0.20},
                {"variant": "B_tree", "model_seed": 1, "validation_f1": 0.10},
                {"variant": "B4", "model_seed": 2, "validation_f1": 0.40},
                {"variant": "B6", "model_seed": 2, "validation_f1": 0.25},
                {"variant": "B_tree", "model_seed": 2, "validation_f1": 0.20},
                {"variant": "B4", "model_seed": 3, "validation_f1": 0.50},
                {"variant": "B6", "model_seed": 3, "validation_f1": 0.30},
                {"variant": "B_tree", "model_seed": 3, "validation_f1": 0.35},
            ]
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "pilot.json"
            input_path.write_text(json.dumps(result), encoding="utf-8")
            markdown = build_paired_effects_markdown(
                input_path,
                left_variant="B4",
                right_variants=("B6", "B_tree"),
                metric_key="validation_f1",
            )

        self.assertIn("| B4 | B6 | validation_f1 | 3 | +0.1500 |", markdown)
        self.assertIn("| B4 | B_tree | validation_f1 | 3 | +0.1833 |", markdown)
        self.assertIn("sign-test p", markdown)

    def test_default_right_variants_include_b7_attention_only_ablation(self) -> None:
        self.assertIn("B7_hyperbolic_attention_only", DEFAULT_RIGHT_VARIANTS)
        self.assertIn("B8_hyperbolic_frechet_code2vec", DEFAULT_RIGHT_VARIANTS)
        self.assertIn("B17_hyperbolic_path_mp_code2vec", DEFAULT_RIGHT_VARIANTS)
        self.assertIn("B18_hyperbolic_path_mp_struct_rank", DEFAULT_RIGHT_VARIANTS)
        self.assertIn("B19_hyperbolic_path_mp_rank_annealed", DEFAULT_RIGHT_VARIANTS)
        self.assertIn("B20_hyperbolic_path_mp_rank_delayed", DEFAULT_RIGHT_VARIANTS)
        self.assertIn("B21_hyperbolic_path_mp_rank_cosine", DEFAULT_RIGHT_VARIANTS)
        self.assertIn("B22_hyperbolic_path_mp_rank_warmup_decay", DEFAULT_RIGHT_VARIANTS)
        self.assertIn("B23_hyperbolic_path_attention_mp_code2vec", DEFAULT_RIGHT_VARIANTS)
        self.assertIn("B24_hyperbolic_path_attention_mp_rank_annealed", DEFAULT_RIGHT_VARIANTS)
        self.assertIn("B25_hyperbolic_path_depth_attention_mp_code2vec", DEFAULT_RIGHT_VARIANTS)
        self.assertIn("B26_hyperbolic_path_depth_attention_mp_rank_annealed", DEFAULT_RIGHT_VARIANTS)
        self.assertIn("B27_hyperbolic_path_attention_mp_monotone", DEFAULT_RIGHT_VARIANTS)
        self.assertIn("B28_hyperbolic_path_attention_mp_tree_distance", DEFAULT_RIGHT_VARIANTS)
        self.assertIn("B29_hyperbolic_path_dual_attention_mp_separated", DEFAULT_RIGHT_VARIANTS)
        self.assertIn("B30_hyperbolic_path_dual_attention_mp_rank_separated", DEFAULT_RIGHT_VARIANTS)
        self.assertIn("B31_hyperbolic_path_dual_attention_mp_soft_rank", DEFAULT_RIGHT_VARIANTS)
        self.assertIn("B32_lorentz_path_dual_attention_mp_soft_rank", DEFAULT_RIGHT_VARIANTS)
        self.assertIn("B34_hyperbolic_path_dual_attention_mp_adaptive_rank", DEFAULT_RIGHT_VARIANTS)
        self.assertIn("B9_lorentz_code2vec", DEFAULT_RIGHT_VARIANTS)
        self.assertIn("B15_lorentz_product_code2vec", DEFAULT_RIGHT_VARIANTS)
        self.assertIn("B10_factorized_product_code2vec", DEFAULT_RIGHT_VARIANTS)
        self.assertIn("B11_factorized_product_struct_rank", DEFAULT_RIGHT_VARIANTS)
        self.assertIn("B12_factorized_product_learned_metric_rank", DEFAULT_RIGHT_VARIANTS)
        self.assertIn("B16_factorized_product_three_metric_rank", DEFAULT_RIGHT_VARIANTS)
        self.assertIn("B13_factorized_product_channel_mixer_rank", DEFAULT_RIGHT_VARIANTS)
        self.assertIn("B14_bounded_euclidean_metric_code2vec", DEFAULT_RIGHT_VARIANTS)


if __name__ == "__main__":
    unittest.main()
