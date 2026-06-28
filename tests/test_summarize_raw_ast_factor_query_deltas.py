from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.summarize_raw_ast_factor_query_deltas import format_markdown, summarize_factor_query_deltas


def _run(
    *,
    root: Path,
    name: str,
    path_object_mode: str,
    method_aggregation: str,
    geometry: str,
    curvature: float,
    ranks: tuple[int, int, int],
    path_cost_orientation: str = "directed",
) -> dict[str, object]:
    query_records = [
        {
            "anchor_id": f"q{index}",
            "rank": rank,
            "margin": 1.0 / rank - 0.25,
            "positive_distance": float(rank),
            "nearest_negative_distance": float(rank) - 0.5,
        }
        for index, rank in enumerate(ranks)
    ]
    path = root / f"{name}.json"
    path.write_text(json.dumps({"query_records": query_records}), encoding="utf-8")
    return {
        "project": "toy",
        "node_input_mode": "label_only",
        "path_object_mode": path_object_mode,
        "method_aggregation": method_aggregation,
        "path_cost_orientation": path_cost_orientation,
        "geometry": geometry,
        "curvature": curvature,
        "dim": 4,
        "seed": 11,
        "output_path": str(path),
    }


class SummarizeRawASTFactorQueryDeltasTests(unittest.TestCase):
    def test_query_delta_summary_reports_path_object_bootstrap_interval(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            runs = [
                _run(
                    root=root,
                    name="single",
                    path_object_mode="single_point",
                    method_aggregation="centroid",
                    geometry="euclidean",
                    curvature=1.0,
                    ranks=(3, 4, 5),
                ),
                _run(
                    root=root,
                    name="lca",
                    path_object_mode="lca_product",
                    method_aggregation="centroid",
                    geometry="euclidean",
                    curvature=1.0,
                    ranks=(1, 2, 4),
                ),
            ]
            matrix = root / "matrix.json"
            matrix.write_text(json.dumps({"runs": runs}), encoding="utf-8")

            summary = summarize_factor_query_deltas(matrix, bootstrap_samples=50, seed=7)
            markdown = format_markdown(summary)

        rows = summary["path_object_deltas"]
        aggregate_rows = summary["path_object_deltas_aggregated"]
        self.assertEqual(len(rows), 1)
        self.assertEqual(len(aggregate_rows), 1)
        self.assertGreater(rows[0]["mean_delta_reciprocal_rank"], 0.0)
        self.assertGreater(aggregate_rows[0]["mean_delta_reciprocal_rank"], 0.0)
        self.assertEqual(rows[0]["positive_queries"], 3)
        self.assertIn("Aggregated query-level path-object deltas", markdown)
        self.assertNotIn("{project}", markdown)
        self.assertNotIn("{node_input_mode}", markdown)
        self.assertNotIn("{cell}", markdown)
        self.assertNotIn("{contrast}", markdown)

    def test_query_delta_summary_reports_orientation_contrast(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            runs = [
                _run(
                    root=root,
                    name="directed",
                    path_object_mode="lca_product",
                    method_aggregation="measure",
                    path_cost_orientation="directed",
                    geometry="euclidean",
                    curvature=1.0,
                    ranks=(3, 4, 5),
                ),
                _run(
                    root=root,
                    name="unoriented",
                    path_object_mode="lca_product",
                    method_aggregation="measure",
                    path_cost_orientation="unoriented",
                    geometry="euclidean",
                    curvature=1.0,
                    ranks=(1, 2, 4),
                ),
            ]
            matrix = root / "matrix.json"
            matrix.write_text(json.dumps({"runs": runs}), encoding="utf-8")

            summary = summarize_factor_query_deltas(matrix, bootstrap_samples=50, seed=7)
            markdown = format_markdown(summary)

        rows = summary["orientation_deltas"]
        aggregate_rows = summary["orientation_deltas_aggregated"]
        self.assertEqual(len(rows), 1)
        self.assertEqual(len(aggregate_rows), 1)
        self.assertGreater(rows[0]["mean_delta_reciprocal_rank"], 0.0)
        self.assertIn("Aggregated query-level orientation deltas", markdown)
        self.assertIn("unoriented - directed", markdown)


if __name__ == "__main__":
    unittest.main()
