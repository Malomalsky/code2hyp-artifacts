from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.plot_code2hyp_distance_levels import plot_distance_levels


class Code2HypDistanceLevelPlotScriptTests(unittest.TestCase):
    def test_plot_distance_levels_writes_png_and_pdf(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            input_path = tmp_path / "result.json"
            output_prefix = tmp_path / "distance_levels"
            input_path.write_text(
                json.dumps(
                    {
                        "runs": [
                            {
                                "variant": "B39_code2vec_context_transform_baseline",
                                "validation_structural_prefix_distance_level_summary": [
                                    {
                                        "target_distance": 2.0,
                                        "pair_count": 10,
                                        "model_distance_mean": 0.5,
                                        "model_distance_std": 0.1,
                                    },
                                    {
                                        "target_distance": 4.0,
                                        "pair_count": 12,
                                        "model_distance_mean": 0.7,
                                        "model_distance_std": 0.1,
                                    },
                                ],
                            },
                            {
                                "variant": "B44_code2hyp_context_transform_product_bias_frechet",
                                "validation_structural_prefix_distance_level_summary": [
                                    {
                                        "target_distance": 2.0,
                                        "pair_count": 10,
                                        "model_distance_mean": 0.4,
                                        "model_distance_std": 0.1,
                                    },
                                    {
                                        "target_distance": 4.0,
                                        "pair_count": 12,
                                        "model_distance_mean": 0.9,
                                        "model_distance_std": 0.1,
                                    },
                                ],
                            },
                            {
                                "variant": "B51_code2vec_context_transform_l1_distance_control",
                                "validation_structural_prefix_distance_level_summary": [
                                    {
                                        "target_distance": 2.0,
                                        "pair_count": 10,
                                        "model_distance_mean": 0.35,
                                        "model_distance_std": 0.05,
                                    },
                                    {
                                        "target_distance": 4.0,
                                        "pair_count": 12,
                                        "model_distance_mean": 0.65,
                                        "model_distance_std": 0.05,
                                    },
                                ],
                            },
                            {
                                "variant": "B49_code2hyp_context_transform_product_bias_near_euclidean",
                                "validation_structural_prefix_distance_level_summary": [
                                    {
                                        "target_distance": 2.0,
                                        "pair_count": 10,
                                        "model_distance_mean": 0.3,
                                        "model_distance_std": 0.05,
                                    },
                                    {
                                        "target_distance": 4.0,
                                        "pair_count": 12,
                                        "model_distance_mean": 0.5,
                                        "model_distance_std": 0.05,
                                    },
                                ],
                            },
                        ]
                    }
                ),
                encoding="utf-8",
            )

            plot_distance_levels(input_path, output_prefix)

            self.assertTrue(output_prefix.with_suffix(".png").exists())
            self.assertTrue(output_prefix.with_suffix(".pdf").exists())


if __name__ == "__main__":
    unittest.main()
