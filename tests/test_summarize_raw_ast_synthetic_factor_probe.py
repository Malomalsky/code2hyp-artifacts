from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.summarize_raw_ast_synthetic_factor_probe import format_markdown, summarize_synthetic_factor_probe


def _row(
    *,
    case: str,
    dim: int,
    geometry: str,
    path_object_mode: str,
    stress: float,
    rho: float,
    curvature: float | None = None,
) -> dict[str, object]:
    return {
        "case": case,
        "expected": "toy",
        "geometry": geometry,
        "curvature": (1.0 if curvature is None and geometry == "poincare" else 0.0 if curvature is None else curvature),
        "path_object_mode": path_object_mode,
        "dim": dim,
        "node_count": 7,
        "path_count": 6,
        "node_stress": 0.1,
        "node_spearman": 0.9,
        "path_stress": stress,
        "path_spearman": rho,
    }


class SummarizeRawASTSyntheticFactorProbeTests(unittest.TestCase):
    def test_summarizer_reports_lca_curvature_and_full_deltas(self) -> None:
        rows = [
            _row(case="toy", dim=4, geometry="euclidean", path_object_mode="single_point", stress=0.40, rho=0.50),
            _row(case="toy", dim=4, geometry="euclidean", path_object_mode="lca_product", stress=0.25, rho=0.70),
            _row(case="toy", dim=4, geometry="poincare", path_object_mode="single_point", stress=0.39, rho=0.55, curvature=1e-4),
            _row(case="toy", dim=4, geometry="poincare", path_object_mode="lca_product", stress=0.20, rho=0.75, curvature=1e-4),
            _row(case="toy", dim=4, geometry="poincare", path_object_mode="single_point", stress=0.42, rho=0.45),
            _row(case="toy", dim=4, geometry="poincare", path_object_mode="lca_product", stress=0.10, rho=0.90),
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "synthetic.json"
            input_path.write_text(json.dumps({"rows": rows}), encoding="utf-8")

            summary = summarize_synthetic_factor_probe(input_path)
            markdown = format_markdown(summary)

        lca_delta = _find_delta(summary["lca_product_deltas"], factor="euclidean")
        curvature_delta = _find_delta(summary["curvature_deltas"], factor="single_point", curvature=1.0)
        near_zero_delta = _find_delta(summary["curvature_deltas"], factor="single_point", curvature=1e-4)
        self.assertAlmostEqual(lca_delta["delta_path_stress"], 0.15)
        self.assertAlmostEqual(curvature_delta["delta_path_stress"], -0.02)
        self.assertAlmostEqual(near_zero_delta["delta_path_stress"], 0.01)
        full_delta = _find_delta(summary["full_model_deltas"], factor="full", curvature=1.0)
        self.assertAlmostEqual(full_delta["delta_path_stress"], 0.30)
        self.assertIn("LCA-product path-object effect", markdown)


def _find_delta(rows: list[dict[str, object]], *, factor: str, curvature: float | None = None) -> dict[str, object]:
    for row in rows:
        if row["factor"] == factor and (curvature is None or abs(float(row["curvature"]) - curvature) < 1e-12):
            return row
    raise AssertionError(f"missing factor {factor!r} curvature={curvature!r}")


if __name__ == "__main__":
    unittest.main()
