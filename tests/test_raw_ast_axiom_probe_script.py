from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "scripts" / "run_raw_ast_axiom_probe.py"


class RawAstAxiomProbeScriptTests(unittest.TestCase):
    def test_cli_writes_reproducible_json_payload(self) -> None:
        payload = {
            "status": "ok",
            "paths": [
                {"start": 1, "end": 3, "lca": 2, "length": 2, "lca_depth": 1},
                {"start": 4, "end": 6, "lca": 5, "length": 4, "lca_depth": 2},
            ],
            "order_records": [
                {"ancestor": 1, "descendant": 2, "label": 1, "is_direct_edge": True, "tree_distance": 1},
                {"ancestor": 2, "descendant": 3, "label": 1, "is_direct_edge": True, "tree_distance": 1},
                {"ancestor": 4, "descendant": 5, "label": 1, "is_direct_edge": True, "tree_distance": 1},
                {"ancestor": 5, "descendant": 6, "label": 1, "is_direct_edge": True, "tree_distance": 1},
            ],
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "relations.jsonl"
            output_path = Path(tmpdir) / "axiom_probe.json"
            input_path.write_text(json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8")

            completed = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--input",
                    str(input_path),
                    "--output",
                    str(output_path),
                    "--dims",
                    "2",
                    "--seeds",
                    "101",
                    "--geometries",
                    "euclidean",
                    "--epochs",
                    "1",
                    "--depth-weight",
                    "0.25",
                ],
                cwd=PROJECT_ROOT,
                check=True,
                capture_output=True,
                text=True,
            )

            result = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertIn("wrote", completed.stdout)
        self.assertEqual(result["config"]["dims"], [2])
        self.assertEqual(result["config"]["seeds"], [101])
        self.assertEqual(result["config"]["geometries"], ["euclidean"])
        self.assertEqual(result["config"]["depth_weight"], 0.25)
        self.assertEqual(result["node_count"], 6)
        self.assertEqual(len(result["runs"]), 1)
        self.assertEqual(result["runs"][0]["geometry"], "euclidean")
        self.assertIn("eval_length_spearman", result["runs"][0])
        self.assertIn("eval_lca_radial_depth_spearman_mean", result["summary"][0])
        self.assertIn("summary", result)


if __name__ == "__main__":
    unittest.main()
