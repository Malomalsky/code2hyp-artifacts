from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.plot_raw_ast_factor_matrix_plotnine import plot_factor_matrix


def _run(
    *,
    project: str,
    path_object_mode: str,
    method_aggregation: str,
    geometry: str,
    curvature: float,
    seed: int,
    mrr: float,
    path_cost_orientation: str = "directed",
) -> dict[str, object]:
    return {
        "project": project,
        "node_input_mode": "label_only",
        "path_object_mode": path_object_mode,
        "method_aggregation": method_aggregation,
        "path_cost_orientation": path_cost_orientation,
        "geometry": geometry,
        "curvature": curvature,
        "dim": 4,
        "seed": seed,
        "recall_at_1": mrr,
        "ndcg_at_3": mrr,
        "mrr": mrr,
        "margin_mean": mrr - 0.5,
        "geometry_diagnostics": {"manifold": geometry},
    }


class RawASTFactorMatrixPlotnineTests(unittest.TestCase):
    def test_plot_factor_matrix_writes_heatmap_and_delta_figures(self) -> None:
        runs = [
            _run(project="toy", path_object_mode="single_point", method_aggregation="centroid", geometry="euclidean", curvature=1.0, seed=11, mrr=0.40),
            _run(project="toy", path_object_mode="single_point", method_aggregation="measure", geometry="poincare", curvature=1.0, seed=11, mrr=0.45),
            _run(project="toy", path_object_mode="lca_product", method_aggregation="centroid", geometry="poincare", curvature=1.0, seed=11, mrr=0.55),
            _run(project="toy", path_object_mode="lca_product", method_aggregation="measure", geometry="euclidean", curvature=1.0, seed=11, mrr=0.60),
            _run(project="toy", path_object_mode="lca_product", method_aggregation="measure", geometry="poincare", curvature=1.0, seed=11, mrr=0.70),
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            input_path = root / "matrix.json"
            input_path.write_text(
                json.dumps({"status": "partial", "completed_runs": len(runs), "expected_runs": 12, "runs": runs}),
                encoding="utf-8",
            )

            outputs = plot_factor_matrix(input_path=input_path, output_prefix=root / "factor")
            sizes = [path.stat().st_size for path in outputs]

        self.assertEqual(len(outputs), 4)
        self.assertTrue(all(size > 0 for size in sizes))

    def test_plot_factor_matrix_handles_orientation_factor(self) -> None:
        runs = [
            _run(
                project="toy",
                path_object_mode="lca_product",
                method_aggregation="measure",
                path_cost_orientation="directed",
                geometry="euclidean",
                curvature=1.0,
                seed=11,
                mrr=0.50,
            ),
            _run(
                project="toy",
                path_object_mode="lca_product",
                method_aggregation="measure",
                path_cost_orientation="unoriented",
                geometry="euclidean",
                curvature=1.0,
                seed=11,
                mrr=0.57,
            ),
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            input_path = root / "orientation_matrix.json"
            input_path.write_text(
                json.dumps({"status": "partial", "completed_runs": len(runs), "expected_runs": 2, "runs": runs}),
                encoding="utf-8",
            )

            outputs = plot_factor_matrix(input_path=input_path, output_prefix=root / "orientation_factor")
            sizes = [path.stat().st_size for path in outputs]

        self.assertEqual(len(outputs), 4)
        self.assertTrue(all(size > 0 for size in sizes))


if __name__ == "__main__":
    unittest.main()
