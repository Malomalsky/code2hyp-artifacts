from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.plot_raw_ast_synthetic_factor_probe_plotnine import plot_synthetic_factor_probe


def _row(case: str, dim: int, geometry: str, path_object_mode: str, stress: float, rho: float) -> dict[str, object]:
    return {
        "case": case,
        "expected": "toy",
        "geometry": geometry,
        "curvature": 1.0 if geometry == "poincare" else 0.0,
        "path_object_mode": path_object_mode,
        "dim": dim,
        "node_count": 7,
        "path_count": 6,
        "node_stress": 0.1,
        "node_spearman": 0.9,
        "path_stress": stress,
        "path_spearman": rho,
    }


class RawASTSyntheticFactorProbePlotnineTests(unittest.TestCase):
    def test_plot_synthetic_factor_probe_writes_figures(self) -> None:
        rows = [
            _row("toy", 4, "euclidean", "single_point", 0.40, 0.50),
            _row("toy", 4, "euclidean", "lca_product", 0.25, 0.70),
            _row("toy", 4, "poincare", "single_point", 0.42, 0.45),
            _row("toy", 4, "poincare", "lca_product", 0.10, 0.90),
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            input_path = root / "synthetic.json"
            input_path.write_text(json.dumps({"rows": rows}), encoding="utf-8")

            outputs = plot_synthetic_factor_probe(input_path=input_path, output_prefix=root / "synthetic")
            sizes = [path.stat().st_size for path in outputs]

        self.assertEqual(len(outputs), 4)
        self.assertTrue(all(size > 0 for size in sizes))


if __name__ == "__main__":
    unittest.main()
