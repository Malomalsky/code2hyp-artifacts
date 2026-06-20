from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.select_code2hyp_multi_objective import build_multi_objective_markdown, main


class Code2HypMultiObjectiveScriptTests(unittest.TestCase):
    def test_build_multi_objective_markdown_marks_best_and_pareto_rows(self) -> None:
        result = {
            "runs": [
                {
                    "variant": "B31",
                    "validation_f1": 0.20,
                    "validation_structural_loss": 0.13,
                    "validation_structural_rank_loss": 0.31,
                    "validation_structural_spearman": 0.16,
                },
                {
                    "variant": "B32",
                    "validation_f1": 0.15,
                    "validation_structural_loss": 0.45,
                    "validation_structural_rank_loss": 0.07,
                    "validation_structural_spearman": 0.29,
                },
                {
                    "variant": "weak",
                    "validation_f1": 0.10,
                    "validation_structural_loss": 0.50,
                    "validation_structural_rank_loss": 0.50,
                    "validation_structural_spearman": 0.05,
                },
            ],
        }

        markdown = build_multi_objective_markdown(
            result,
            f1_weight=0.5,
            spearman_weight=0.5,
        )

        self.assertIn("Multi-objective Code2Hyp selection", markdown)
        self.assertIn("B32", markdown)
        self.assertIn("best", markdown)
        self.assertIn("pareto", markdown)
        self.assertIn("weak", markdown)

    def test_main_writes_markdown_report(self) -> None:
        result = {
            "runs": [
                {
                    "variant": "B31",
                    "validation_f1": 0.20,
                    "validation_structural_loss": 0.13,
                    "validation_structural_rank_loss": 0.31,
                    "validation_structural_spearman": 0.16,
                },
                {
                    "variant": "B32",
                    "validation_f1": 0.15,
                    "validation_structural_loss": 0.45,
                    "validation_structural_rank_loss": 0.07,
                    "validation_structural_spearman": 0.29,
                },
            ],
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "pilot.json"
            output_path = Path(tmpdir) / "selection.md"
            input_path.write_text(json.dumps(result), encoding="utf-8")

            main([str(input_path), "--output", str(output_path)])

            self.assertTrue(output_path.exists())
            self.assertIn("Multi-objective Code2Hyp selection", output_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
