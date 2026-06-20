from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.plot_code2hyp_final_controls import parse_labeled_input, plot_final_controls


class Code2HypFinalControlsPlotScriptTests(unittest.TestCase):
    def test_plot_final_controls_writes_png_and_pdf(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            inputs: list[tuple[str, Path]] = []
            for mode_index, mode in enumerate(("Original", "Record-obfuscated", "Structural only")):
                path = root / f"{mode_index}.json"
                path.write_text(
                    json.dumps(
                        {
                            "runs": [
                                {
                                    "variant": "B39_code2vec_context_transform_baseline",
                                    "model_seed": 101,
                                    "validation_f1": 0.10 + mode_index * 0.01,
                                    "validation_structural_spearman": -0.10,
                                    "validation_structural_normalized_stress": 0.80,
                                    "validation_structural_neighbor_overlap_at_3": 0.40,
                                },
                                {
                                    "variant": "B36_code2hyp_product_frechet_neighbor",
                                    "model_seed": 101,
                                    "validation_f1": 0.20 + mode_index * 0.01,
                                    "validation_structural_spearman": 0.50,
                                    "validation_structural_normalized_stress": 0.25,
                                    "validation_structural_neighbor_overlap_at_3": 0.80,
                                },
                            ]
                        }
                    ),
                    encoding="utf-8",
                )
                inputs.append((mode, path))

            output_prefix = root / "final_controls"

            plot_final_controls(tuple(inputs), output_prefix)

            self.assertTrue(output_prefix.with_suffix(".png").exists())
            self.assertTrue(output_prefix.with_suffix(".pdf").exists())

    def test_parse_labeled_input_accepts_label_equals_path(self) -> None:
        label, path = parse_labeled_input("Original=outputs/result.json")

        self.assertEqual(label, "Original")
        self.assertEqual(path, Path("outputs/result.json"))


if __name__ == "__main__":
    unittest.main()
