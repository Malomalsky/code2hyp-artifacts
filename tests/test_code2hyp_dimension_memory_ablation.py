from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.summarize_code2hyp_dimension_memory_ablation import build_markdown, load_results, summarize


class Code2HypDimensionMemoryAblationTests(unittest.TestCase):
    def test_summary_reports_parameter_memory_and_thresholds(self) -> None:
        payload = {
            "training": {
                "token_dim": 32,
                "structural_dim": 8,
                "representation_dim": 72,
            },
            "dimension_memory_ablation": {
                "activation_proxy_bytes_per_example": 8640,
            },
            "runs": [
                {
                    "variant": "B44_code2hyp_context_transform_product_bias_frechet",
                    "model_seed": 101,
                    "parameter_count": 1000,
                    "validation_f1": 0.1,
                    "validation_structural_spearman": 0.8,
                    "validation_structural_normalized_stress": 0.15,
                    "validation_structural_neighbor_overlap_at_3": 0.9,
                },
                {
                    "variant": "B44_code2hyp_context_transform_product_bias_frechet",
                    "model_seed": 202,
                    "parameter_count": 1000,
                    "validation_f1": 0.2,
                    "validation_structural_spearman": 0.9,
                    "validation_structural_normalized_stress": 0.13,
                    "validation_structural_neighbor_overlap_at_3": 0.8,
                },
            ],
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "dim8.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            rows = load_results([path])

        summary = summarize(rows)
        markdown = build_markdown(summary, spearman_threshold=0.7, stress_threshold=0.2)

        self.assertEqual(summary[0]["parameter_count"], 1000)
        self.assertAlmostEqual(summary[0]["parameter_memory_mib_float32"], 4000 / (1024 * 1024))
        self.assertIn("first reaches AST Spearman >= 0.70 at structural_dim=8", markdown)
        self.assertIn("must not be reported as measured RAM or VRAM", markdown)


if __name__ == "__main__":
    unittest.main()
