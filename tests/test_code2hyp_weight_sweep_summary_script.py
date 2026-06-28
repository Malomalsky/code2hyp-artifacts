from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "scripts" / "summarize_code2hyp_weight_sweep.py"


class Code2HypWeightSweepSummaryScriptTests(unittest.TestCase):
    def test_summarizes_multiple_result_files_by_structural_weight(self) -> None:
        first = {
            "runs": [
                {
                    "variant": "B85",
                    "model_seed": 101,
                    "structural_regularizer": "relation_conditioned_lca_axiom",
                    "structural_loss_weight": 0.0,
                    "validation_f1": 0.10,
                    "validation_fixed_top3_f1": 0.11,
                    "validation_structural_spearman": 0.20,
                    "validation_structural_edit_spearman": 0.30,
                    "validation_structural_jaccard_spearman": 0.40,
                    "validation_structural_normalized_stress": 0.50,
                    "validation_method_aggregate_spearman": 0.45,
                    "validation_method_aggregate_normalized_stress": 0.55,
                    "validation_method_transport_spearman": 0.60,
                    "validation_method_transport_normalized_stress": 0.70,
                    "validation_method_transport_edit_spearman": 0.62,
                    "validation_method_transport_edit_normalized_stress": 0.72,
                    "validation_structural_neighbor_overlap_at_3": 0.35,
                },
                {
                    "variant": "B85",
                    "model_seed": 202,
                    "structural_regularizer": "relation_conditioned_lca_axiom",
                    "structural_loss_weight": 0.0,
                    "validation_f1": 0.20,
                    "validation_fixed_top3_f1": 0.21,
                    "validation_structural_spearman": 0.30,
                    "validation_structural_edit_spearman": 0.40,
                    "validation_structural_jaccard_spearman": 0.50,
                    "validation_structural_normalized_stress": 0.60,
                    "validation_method_aggregate_spearman": 0.55,
                    "validation_method_aggregate_normalized_stress": 0.65,
                    "validation_method_transport_spearman": 0.80,
                    "validation_method_transport_normalized_stress": 0.90,
                    "validation_method_transport_edit_spearman": 0.82,
                    "validation_method_transport_edit_normalized_stress": 0.92,
                    "validation_structural_neighbor_overlap_at_3": 0.45,
                },
            ]
        }
        second = {
            "runs": [
                {
                    "variant": "B86",
                    "model_seed": 101,
                    "structural_regularizer": "method_transport",
                    "structural_loss_weight": 0.01,
                    "validation_f1": 0.30,
                    "validation_fixed_top3_f1": 0.31,
                    "validation_structural_spearman": 0.70,
                    "validation_structural_edit_spearman": 0.80,
                    "validation_structural_jaccard_spearman": 0.90,
                    "validation_structural_normalized_stress": 0.25,
                    "validation_method_aggregate_spearman": 0.75,
                    "validation_method_aggregate_normalized_stress": 0.15,
                    "validation_method_transport_spearman": 0.95,
                    "validation_method_transport_normalized_stress": 0.05,
                    "validation_method_transport_edit_spearman": 0.96,
                    "validation_method_transport_edit_normalized_stress": 0.06,
                    "validation_structural_neighbor_overlap_at_3": 0.85,
                },
            ]
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            input_a = Path(tmpdir) / "a.json"
            input_b = Path(tmpdir) / "b.json"
            output = Path(tmpdir) / "summary.md"
            input_a.write_text(json.dumps(first), encoding="utf-8")
            input_b.write_text(json.dumps(second), encoding="utf-8")

            subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--inputs",
                    str(input_a),
                    str(input_b),
                    "--output",
                    str(output),
                ],
                cwd=PROJECT_ROOT,
                check=True,
                capture_output=True,
                text=True,
            )

            text = output.read_text(encoding="utf-8")

        self.assertIn("| B85 | relation_conditioned_lca_axiom | 0.0000 | 2 | 0.1500", text)
        self.assertIn("| B86 | method_transport | 0.0100 | 1 | 0.3000", text)
        self.assertIn("validation_structural_spearman", text)
        self.assertIn("validation_method_aggregate_spearman", text)
        self.assertIn("validation_method_transport_spearman", text)
        self.assertIn("validation_method_transport_edit_spearman", text)
        self.assertIn("validation_structural_neighbor_overlap_at_3", text)
        self.assertIn("0.7000 +- 0.1414", text)
        self.assertIn("0.5000 +- 0.0707", text)
        self.assertIn("0.4000 +- 0.0707", text)
        self.assertIn("0.7200 +- 0.1414", text)
        self.assertIn("n/a", text)


if __name__ == "__main__":
    unittest.main()
