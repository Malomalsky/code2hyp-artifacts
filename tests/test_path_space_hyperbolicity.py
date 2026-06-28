from __future__ import annotations

import unittest

from geometry_profile_research.path_space_hyperbolicity import (
    four_point_delta,
    line_tree_distance_matrix,
    matrix_max_four_point_delta,
    product_grid_l1_distance_matrix,
    upper_triangle_records_to_distance_matrix,
)


class PathSpaceHyperbolicityTests(unittest.TestCase):
    def test_four_point_delta_is_zero_for_tree_line_metric(self) -> None:
        distances = line_tree_distance_matrix(6)

        self.assertEqual(matrix_max_four_point_delta(distances), 0.0)

    def test_product_grid_l1_metric_has_linearly_growing_delta(self) -> None:
        distances = product_grid_l1_distance_matrix(side_length=5)

        self.assertEqual(matrix_max_four_point_delta(distances), 5.0)

    def test_four_point_delta_uses_largest_two_sums(self) -> None:
        delta = four_point_delta(ab=2.0, ac=2.0, ad=4.0, bc=4.0, bd=2.0, cd=2.0)

        self.assertEqual(delta, 2.0)

    def test_upper_triangle_records_reconstruct_distance_matrix(self) -> None:
        records = (
            {"left_index": 0, "right_index": 1, "distance": 2},
            {"left_index": 0, "right_index": 2, "distance": 5},
            {"left_index": 1, "right_index": 2, "distance": 3},
        )

        distances = upper_triangle_records_to_distance_matrix(records, path_count=3, distance_key="distance")

        self.assertEqual(distances, ((0.0, 2.0, 5.0), (2.0, 0.0, 3.0), (5.0, 3.0, 0.0)))


if __name__ == "__main__":
    unittest.main()
