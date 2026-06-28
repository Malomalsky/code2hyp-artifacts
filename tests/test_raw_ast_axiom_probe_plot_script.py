from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "scripts" / "plot_raw_ast_axiom_probe.py"


class RawAstAxiomProbePlotScriptTests(unittest.TestCase):
    def test_plot_script_writes_png_and_pdf(self) -> None:
        payload = {
            "summary": [
                {
                    "geometry": "poincare",
                    "dim": 2,
                    "eval_length_spearman_mean": 0.2,
                    "eval_lca_depth_spearman_mean": 0.1,
                    "eval_lca_radial_depth_spearman_mean": 0.5,
                },
                {
                    "geometry": "euclidean",
                    "dim": 2,
                    "eval_length_spearman_mean": 0.1,
                    "eval_lca_depth_spearman_mean": 0.2,
                    "eval_lca_radial_depth_spearman_mean": 0.4,
                },
            ]
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "probe.json"
            output_prefix = Path(tmpdir) / "figure"
            input_path.write_text(json.dumps(payload), encoding="utf-8")

            subprocess.run(
                [sys.executable, str(SCRIPT), "--input", str(input_path), "--output-prefix", str(output_prefix)],
                cwd=PROJECT_ROOT,
                check=True,
                capture_output=True,
                text=True,
            )

            self.assertTrue(output_prefix.with_suffix(".png").exists())
            self.assertTrue(output_prefix.with_suffix(".pdf").exists())


if __name__ == "__main__":
    unittest.main()
