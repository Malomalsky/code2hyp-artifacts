from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "scripts" / "summarize_code2hyp_paired_variant_deltas.py"


class Code2HypPairedVariantDeltaScriptTests(unittest.TestCase):
    def test_reports_metric_deltas_for_matching_seeds(self) -> None:
        payload = {
            "runs": [
                {
                    "variant": "B85",
                    "model_seed": 101,
                    "validation_f1": 0.10,
                    "validation_structural_spearman": 0.20,
                    "validation_method_aggregate_spearman": 0.25,
                    "validation_method_transport_spearman": 0.30,
                    "validation_structural_neighbor_overlap_at_3": 0.35,
                },
                {
                    "variant": "B85",
                    "model_seed": 103,
                    "validation_f1": 0.20,
                    "validation_structural_spearman": 0.30,
                    "validation_method_aggregate_spearman": 0.35,
                    "validation_method_transport_spearman": 0.40,
                    "validation_structural_neighbor_overlap_at_3": 0.45,
                },
                {
                    "variant": "B86",
                    "model_seed": 101,
                    "validation_f1": 0.15,
                    "validation_structural_spearman": 0.25,
                    "validation_method_aggregate_spearman": 0.50,
                    "validation_method_transport_spearman": 0.60,
                    "validation_structural_neighbor_overlap_at_3": 0.30,
                },
                {
                    "variant": "B86",
                    "model_seed": 103,
                    "validation_f1": 0.25,
                    "validation_structural_spearman": 0.45,
                    "validation_method_aggregate_spearman": 0.70,
                    "validation_method_transport_spearman": 0.80,
                    "validation_structural_neighbor_overlap_at_3": 0.35,
                },
                {
                    "variant": "B86",
                    "model_seed": 107,
                    "validation_f1": 0.35,
                    "validation_structural_spearman": 0.55,
                    "validation_method_aggregate_spearman": 0.75,
                    "validation_method_transport_spearman": 0.90,
                    "validation_structural_neighbor_overlap_at_3": 0.40,
                },
            ]
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "runs.json"
            output_path = Path(tmpdir) / "paired.md"
            input_path.write_text(json.dumps(payload), encoding="utf-8")

            subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--inputs",
                    str(input_path),
                    "--baseline",
                    "B85",
                    "--candidate",
                    "B86",
                    "--output",
                    str(output_path),
                ],
                cwd=PROJECT_ROOT,
                check=True,
                capture_output=True,
                text=True,
            )

            text = output_path.read_text(encoding="utf-8")

        self.assertIn("Matched seeds: `101, 103`", text)
        self.assertIn("Unmatched candidate seeds ignored: `107`", text)
        self.assertIn("| validation_f1 | 2 | 0.0500 +- 0.0000 | 0.0500 | 0.0500 |", text)
        self.assertIn("| validation_structural_spearman | 2 | 0.1000 +- 0.0707 | 0.0500 | 0.1500 |", text)
        self.assertIn("| validation_method_aggregate_spearman | 2 | 0.3000 +- 0.0707 | 0.2500 | 0.3500 |", text)
        self.assertIn("| validation_method_transport_spearman | 2 | 0.3500 +- 0.0707 | 0.3000 | 0.4000 |", text)
        self.assertIn("| validation_structural_neighbor_overlap_at_3 | 2 | -0.0750 +- 0.0354 | -0.1000 | -0.0500 |", text)


if __name__ == "__main__":
    unittest.main()
