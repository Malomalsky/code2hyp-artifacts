from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import torch

from geometry_profile_research.raw_ast_synthetic_geometry import (
    metric_stress,
    spearman_correlation,
    synthetic_path_distortion_rows,
    synthetic_tree_suite,
)
from scripts.run_raw_ast_synthetic_factor_probe import run_synthetic_factor_probe


class RawASTSyntheticGeometryTests(unittest.TestCase):
    def test_synthetic_suite_contains_mechanistic_tree_families(self) -> None:
        names = {case.name for case in synthetic_tree_suite()}

        self.assertEqual(
            names,
            {
                "comb_chain",
                "star",
                "balanced_binary",
                "repeated_labels",
                "two_axis_product",
            },
        )

    def test_metric_stress_is_scale_invariant_on_matching_distances(self) -> None:
        target = torch.tensor(
            [
                [0.0, 1.0, 2.0],
                [1.0, 0.0, 3.0],
                [2.0, 3.0, 0.0],
            ]
        )
        represented = target * 4.0

        self.assertLess(metric_stress(target, represented), 1e-6)

    def test_spearman_correlation_is_bounded(self) -> None:
        value = spearman_correlation(torch.tensor([1.0, 2.0, 3.0]), torch.tensor([3.0, 2.0, 1.0]))

        self.assertGreaterEqual(value, -1.0)
        self.assertLessEqual(value, 1.0)

    def test_synthetic_rows_cover_geometry_and_path_object_factors(self) -> None:
        case = synthetic_tree_suite()[0]

        rows = synthetic_path_distortion_rows(cases=(case,), dims=(2,), steps=2, max_paths=6, poincare_curvatures=(1e-4, 1.0))

        factors = {(row["geometry"], row["curvature"], row["path_object_mode"]) for row in rows}
        self.assertEqual(
            factors,
            {
                ("euclidean", 0.0, "single_point"),
                ("euclidean", 0.0, "lca_product"),
                ("poincare", 1e-4, "single_point"),
                ("poincare", 1e-4, "lca_product"),
                ("poincare", 1.0, "single_point"),
                ("poincare", 1.0, "lca_product"),
            },
        )
        for row in rows:
            self.assertIn("path_stress", row)
            self.assertIn("path_spearman", row)
            self.assertGreaterEqual(float(row["path_stress"]), 0.0)
            self.assertGreaterEqual(float(row["path_spearman"]), -1.0)
            self.assertLessEqual(float(row["path_spearman"]), 1.0)

    def test_runner_writes_json_and_csv_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_json = Path(tmpdir) / "synthetic.json"
            output_csv = Path(tmpdir) / "synthetic.csv"

            payload = run_synthetic_factor_probe(
                output_json=output_json,
                output_csv=output_csv,
                dims=(2,),
                steps=1,
                max_paths=5,
                poincare_curvatures=(1e-4, 1.0),
                seed=11,
            )

            loaded = json.loads(output_json.read_text(encoding="utf-8"))
            csv_size = output_csv.stat().st_size

        self.assertEqual(payload["experiment"], "raw_ast_synthetic_factor_probe")
        self.assertEqual(loaded["experiment"], "raw_ast_synthetic_factor_probe")
        self.assertTrue(payload["rows"])
        self.assertIn("path_stress", payload["rows"][0])
        self.assertGreater(csv_size, 0)


if __name__ == "__main__":
    unittest.main()
