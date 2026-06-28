from __future__ import annotations

import unittest
from pathlib import Path

from scripts.run_code2hyp_curvature_sweep import curvature_output_path, parse_args, parse_curvature_grid


class Code2HypCurvatureSweepScriptTests(unittest.TestCase):
    def test_parse_curvature_grid_accepts_positive_comma_separated_values(self) -> None:
        self.assertEqual(parse_curvature_grid("0.1, 0.3,1,3.0"), (0.1, 0.3, 1.0, 3.0))

    def test_parse_curvature_grid_rejects_non_positive_values(self) -> None:
        with self.assertRaises(ValueError):
            parse_curvature_grid("0.1,0")

    def test_curvature_output_path_uses_safe_decimal_token(self) -> None:
        path = curvature_output_path(Path("outputs/sweep"), curvature=0.3)

        self.assertEqual(path, Path("outputs/sweep/curvature_c0p3.json"))

    def test_parse_args_accepts_neighbor_distribution_regularizer(self) -> None:
        args = parse_args(["--structural-regularizer", "neighbor_distribution"])

        self.assertEqual(args.structural_regularizer, "neighbor_distribution")

    def test_parse_args_accepts_relation_specific_regularizer(self) -> None:
        args = parse_args(["--structural-regularizer", "distance_edit"])

        self.assertEqual(args.structural_regularizer, "distance_edit")


if __name__ == "__main__":
    unittest.main()
