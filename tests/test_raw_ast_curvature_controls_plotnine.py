from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.plot_raw_ast_curvature_controls_plotnine import build_curvature_delta_rows, plot_curvature_controls


def _run(project: str, geometry: str, curvature: float, seed: int, mrr: float) -> dict[str, object]:
    diagnostics = {"manifold": geometry}
    if geometry == "poincare":
        diagnostics.update(
            {
                "curvature": curvature,
                "sqrt_curvature_norm_mean": curvature,
                "sqrt_curvature_norm_max": curvature * 2,
                "near_boundary_fraction": 0.0,
                "projection_active_fraction": 0.0,
            }
        )
    return {
        "project": project,
        "node_input_mode": "label_only",
        "geometry": geometry,
        "curvature": curvature,
        "dim": 4,
        "seed": seed,
        "recall_at_1": mrr,
        "mrr": mrr,
        "margin_mean": mrr - 0.5,
        "geometry_diagnostics": diagnostics,
    }


class RawASTCurvatureControlsPlotnineTests(unittest.TestCase):
    def test_plot_curvature_controls_keeps_near_zero_and_unit_curvature_separate(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            input_path = root / "matrix.json"
            input_path.write_text(
                json.dumps(
                    {
                        "runs": [
                            _run("toy-a", "euclidean", 1.0, 11, 0.40),
                            _run("toy-a", "poincare", 1e-4, 11, 0.45),
                            _run("toy-a", "poincare", 1.0, 11, 0.55),
                            _run("toy-b", "euclidean", 1.0, 11, 0.60),
                            _run("toy-b", "poincare", 1e-4, 11, 0.62),
                            _run("toy-b", "poincare", 1.0, 11, 0.65),
                        ]
                    }
                ),
                encoding="utf-8",
            )

            payload = json.loads(input_path.read_text(encoding="utf-8"))
            rows = build_curvature_delta_rows((payload,))
            outputs = plot_curvature_controls(inputs=(input_path,), output_prefix=root / "fig")
            output_sizes = [path.stat().st_size for path in outputs if path.exists()]

            self.assertEqual({row["curvature_label"] for row in rows}, {"H(c=1e-4)", "H(c=1)"})
            self.assertEqual(len(output_sizes), len(outputs))
            self.assertTrue(all(size > 0 for size in output_sizes))


if __name__ == "__main__":
    unittest.main()
