from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.plot_code2hyp_structural_retrieval import VARIANT_ORDER, plot_structural_retrieval


class Code2HypStructuralRetrievalPlotScriptTests(unittest.TestCase):
    def test_variant_order_contains_main_structural_retrieval_variants(self) -> None:
        self.assertEqual(
            VARIANT_ORDER,
            (
                "B1_euclidean",
                "B4_hyperbolic_code2vec",
                "B8_hyperbolic_frechet_code2vec",
                "B29_hyperbolic_path_dual_attention_mp_separated",
                "B31_hyperbolic_path_dual_attention_mp_soft_rank",
                "B34_hyperbolic_path_dual_attention_mp_adaptive_rank",
            ),
        )

    def test_plot_structural_retrieval_writes_png_and_pdf(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            inputs = []
            for regime in ("Original", "Structural only"):
                path = root / f"{regime.replace(' ', '_').lower()}.json"
                path.write_text(
                    json.dumps(
                        {
                            "runs": [
                                {
                                    "variant": "B1_euclidean",
                                    "model_seed": 101,
                                    "validation_f1": 0.08,
                                    "validation_structural_spearman": -0.10,
                                    "validation_structural_neighbor_overlap_at_3": 0.30,
                                },
                                {
                                    "variant": "B8_hyperbolic_frechet_code2vec",
                                    "model_seed": 101,
                                    "validation_f1": 0.12,
                                    "validation_structural_spearman": 0.35,
                                    "validation_structural_neighbor_overlap_at_3": 0.50,
                                },
                            ]
                        }
                    ),
                    encoding="utf-8",
                )
                inputs.append((regime, path))

            output_prefix = root / "structural_retrieval"
            plot_structural_retrieval(tuple(inputs), output_prefix)

            self.assertTrue(output_prefix.with_suffix(".png").exists())
            self.assertTrue(output_prefix.with_suffix(".pdf").exists())


if __name__ == "__main__":
    unittest.main()
