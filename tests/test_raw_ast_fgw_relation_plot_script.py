from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.plot_raw_ast_fgw_relation_ablation import plot_relation_ablation


class RawAstFgwRelationPlotScriptTests(unittest.TestCase):
    def test_plot_relation_ablation_writes_png_pdf_and_summary_csv(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_paths: list[Path] = []
            for relation, structural, feature, top1 in (
                ("endpoint", 0.58, 0.70, 0.06),
                ("lca_depth", 0.94, 0.56, 0.31),
            ):
                path = root / f"{relation}.json"
                path.write_text(
                    json.dumps(
                        {
                            "config": {"structural_relation": relation},
                            "method_count": 32,
                            "pair_count": 496,
                            "spearman_against_fgw": {
                                "centroid": 0.30,
                                "gw_structure": structural,
                                "ot_feature": feature,
                            },
                            "retrieval_overlap_at_1": {
                                "centroid": 0.20,
                                "gw_structure": top1,
                                "ot_feature": 0.50,
                                "fgw": 1.0,
                            },
                        }
                    ),
                    encoding="utf-8",
                )
                input_paths.append(path)
            output_stem = root / "figure"

            written = plot_relation_ablation(tuple(input_paths), output_stem)

            self.assertEqual(written["png"].name, "figure.png")
            self.assertEqual(written["pdf"].name, "figure.pdf")
            self.assertEqual(written["csv"].name, "figure_summary.csv")
            for output_path in written.values():
                self.assertTrue(output_path.exists())
                self.assertGreater(output_path.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
