from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.summarize_raw_ast_fgw_project_suite import summarize_project_suite


class RawAstFgwProjectSuiteSummaryTests(unittest.TestCase):
    def test_summarize_project_suite_writes_tables_report_and_figure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            inputs: list[Path] = []
            for project, relation, structural, feature in (
                ("libgdx", "lca_depth", 0.90, 0.70),
                ("libgdx", "lca_anchored_product", 0.60, 0.84),
                ("hadoop", "lca_depth", 0.96, 0.66),
                ("hadoop", "lca_anchored_product", 0.64, 0.85),
            ):
                output_path = root / f"{project}_{relation}.json"
                output_path.write_text(
                    json.dumps(
                        {
                            "config": {
                                "sources": [f"data/code2seq_java_small_raw/extracted/java-small/test/{project}"],
                                "structural_relation": relation,
                                "alpha": 0.75,
                            },
                            "method_count": 32,
                            "pair_count": 496,
                            "spearman_against_fgw": {
                                "centroid": 0.33,
                                "gw_structure": structural,
                                "ot_feature": feature,
                            },
                            "retrieval_overlap_at_1": {
                                "centroid": 0.30,
                                "gw_structure": 0.20,
                                "ot_feature": 0.70,
                                "fgw": 1.0,
                            },
                            "retrieval_overlap_at_3": {
                                "centroid": 0.40,
                                "gw_structure": 0.30,
                                "ot_feature": 0.80,
                                "fgw": 1.0,
                            },
                        }
                    ),
                    encoding="utf-8",
                )
                inputs.append(output_path)

            written = summarize_project_suite(tuple(inputs), root / "suite")

            for output_path in written.values():
                self.assertTrue(output_path.exists())
                self.assertGreater(output_path.stat().st_size, 0)

            rows = written["rows_csv"].read_text(encoding="utf-8")
            self.assertIn("libgdx,lca_depth", rows)
            self.assertIn("hadoop,lca_anchored_product", rows)

            summary = json.loads(written["json"].read_text(encoding="utf-8"))

        self.assertAlmostEqual(
            summary["relations"]["lca_depth"]["gw_structure_spearman_mean"],
            0.93,
        )
        self.assertGreater(
            summary["relations"]["lca_depth"]["gw_minus_feature_spearman_mean"],
            0.0,
        )


if __name__ == "__main__":
    unittest.main()
