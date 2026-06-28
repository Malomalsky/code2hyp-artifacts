from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.analyze_raw_ast_fgw_cross_relation import analyze_cross_relation_transfer


class RawAstFgwCrossRelationScriptTests(unittest.TestCase):
    def test_analyze_cross_relation_transfer_writes_matrix_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            inputs: list[Path] = []
            for relation, structural_values, target_values in (
                ("endpoint", [0.1, 0.2, 0.3], [0.1, 0.2, 0.3]),
                ("lca_depth", [0.3, 0.2, 0.1], [0.3, 0.2, 0.1]),
            ):
                path = root / f"{relation}.json"
                pairs = [
                    {
                        "left": 0,
                        "right": 1,
                        "gw_structure": structural_values[0],
                        "fgw": target_values[0],
                    },
                    {
                        "left": 0,
                        "right": 2,
                        "gw_structure": structural_values[1],
                        "fgw": target_values[1],
                    },
                    {
                        "left": 1,
                        "right": 2,
                        "gw_structure": structural_values[2],
                        "fgw": target_values[2],
                    },
                ]
                path.write_text(
                    json.dumps(
                        {
                            "config": {"structural_relation": relation},
                            "method_count": 3,
                            "pair_count": 3,
                            "pairs": pairs,
                        }
                    ),
                    encoding="utf-8",
                )
                inputs.append(path)
            output_stem = root / "cross"

            written = analyze_cross_relation_transfer(tuple(inputs), output_stem)

            for output_path in written.values():
                self.assertTrue(output_path.exists())
                self.assertGreater(output_path.stat().st_size, 0)
            matrix = json.loads(written["json"].read_text(encoding="utf-8"))

        self.assertAlmostEqual(matrix["spearman_matrix"]["endpoint"]["endpoint"], 1.0)
        self.assertAlmostEqual(matrix["spearman_matrix"]["endpoint"]["lca_depth"], -1.0)


if __name__ == "__main__":
    unittest.main()
