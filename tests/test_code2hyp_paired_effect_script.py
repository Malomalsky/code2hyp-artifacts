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

    def test_default_right_variants_cover_final_factorial_controls(self) -> None:
        self.assertEqual(
            DEFAULT_RIGHT_VARIANTS,
            (
                "B47_code2vec_context_transform_distance_control",
                "B50_code2vec_context_transform_l1_baseline",
                "B51_code2vec_context_transform_l1_distance_control",
                "B48_code2hyp_context_transform_product_bias_no_struct",
                "B49_code2hyp_context_transform_product_bias_near_euclidean",
                "B36_code2hyp_product_frechet_neighbor",
                "B39_code2vec_context_transform_baseline",
            ),
        )


if __name__ == "__main__":
    unittest.main()
