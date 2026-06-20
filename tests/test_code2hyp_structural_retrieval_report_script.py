from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.summarize_code2hyp_structural_retrieval import build_structural_retrieval_markdown


class Code2HypStructuralRetrievalReportScriptTests(unittest.TestCase):
    def test_build_structural_retrieval_markdown_reports_local_neighborhood_overlap(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            original_path = root / "original.json"
            structural_path = root / "structural.json"
            original_path.write_text(
                json.dumps(
                    {
                        "runs": [
                            {
                                "variant": "B1_euclidean",
                                "validation_f1": 0.10,
                                "validation_structural_spearman": -0.10,
                                "validation_structural_neighbor_overlap_at_1": 0.25,
                                "validation_structural_neighbor_overlap_at_3": 0.30,
                            },
                            {
                                "variant": "B4_hyperbolic_code2vec",
                                "validation_f1": 0.20,
                                "validation_structural_spearman": 0.40,
                                "validation_structural_neighbor_overlap_at_1": 0.35,
                                "validation_structural_neighbor_overlap_at_3": 0.50,
                            },
                        ]
                    }
                ),
                encoding="utf-8",
            )
            structural_path.write_text(
                json.dumps(
                    {
                        "runs": [
                            {
                                "variant": "B1_euclidean",
                                "validation_f1": 0.08,
                                "validation_structural_spearman": -0.20,
                                "validation_structural_neighbor_overlap_at_1": 0.20,
                                "validation_structural_neighbor_overlap_at_3": 0.25,
                            },
                            {
                                "variant": "B29_hyperbolic_path_dual_attention_mp_separated",
                                "validation_f1": 0.18,
                                "validation_structural_spearman": 0.30,
                                "validation_structural_neighbor_overlap_at_1": 0.55,
                                "validation_structural_neighbor_overlap_at_3": 0.60,
                            },
                        ]
                    }
                ),
                encoding="utf-8",
            )

            markdown = build_structural_retrieval_markdown(
                (("Original", original_path), ("Structural only", structural_path))
            )

        self.assertIn("| Original | B4_hyperbolic_code2vec | 1 | 0.2000 | +0.4000 | 0.3500 | 0.5000 |", markdown)
        self.assertIn(
            "| Structural only | B29_hyperbolic_path_dual_attention_mp_separated | 1 | 0.1800 | +0.3000 | 0.5500 | 0.6000 |",
            markdown,
        )
        self.assertIn("| Regime | Variant | n | F1 | Spearman | Overlap@1 | Overlap@3 |", markdown)
        self.assertIn("Best Overlap@1 variant: `B29_hyperbolic_path_dual_attention_mp_separated`", markdown)
        self.assertIn("Best Overlap@3 variant: `B29_hyperbolic_path_dual_attention_mp_separated`", markdown)


if __name__ == "__main__":
    unittest.main()
