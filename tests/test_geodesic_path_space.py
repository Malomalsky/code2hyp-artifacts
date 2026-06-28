from __future__ import annotations

import unittest

import torch

from geometry_profile_research.geodesic_path_space import (
    poincare_directed_endpoint_product_distance,
    poincare_geodesic_discrete_hausdorff_distance,
    poincare_geodesic_points,
    poincare_gromov_product_at_origin,
    poincare_unoriented_endpoint_product_distance,
)


class GeodesicPathSpaceTests(unittest.TestCase):
    def test_geodesic_points_include_endpoints(self) -> None:
        start = torch.tensor([[0.10, 0.0]], dtype=torch.float64)
        end = torch.tensor([[0.45, 0.0]], dtype=torch.float64)

        points = poincare_geodesic_points(start, end, curvature=1.0, num_points=5)

        self.assertEqual(points.shape, (1, 5, 2))
        self.assertTrue(torch.allclose(points[:, 0], start, atol=1e-10))
        self.assertTrue(torch.allclose(points[:, -1], end, atol=1e-10))

    def test_directed_endpoint_product_distance_keeps_orientation(self) -> None:
        left_start = torch.tensor([[0.10, 0.0]], dtype=torch.float64)
        left_end = torch.tensor([[0.60, 0.0]], dtype=torch.float64)
        right_same = torch.tensor([[0.10, 0.0]], dtype=torch.float64)
        right_end = torch.tensor([[0.60, 0.0]], dtype=torch.float64)

        same = poincare_directed_endpoint_product_distance(left_start, left_end, right_same, right_end, curvature=1.0)
        reversed_distance = poincare_directed_endpoint_product_distance(
            left_start,
            left_end,
            right_end,
            right_same,
            curvature=1.0,
        )

        self.assertAlmostEqual(float(same.item()), 0.0, places=10)
        self.assertGreater(float(reversed_distance.item()), 0.0)

    def test_unoriented_endpoint_product_distance_is_reversal_invariant(self) -> None:
        start = torch.tensor([[0.10, 0.0]], dtype=torch.float64)
        end = torch.tensor([[0.60, 0.0]], dtype=torch.float64)

        distance = poincare_unoriented_endpoint_product_distance(start, end, end, start, curvature=1.0)

        self.assertAlmostEqual(float(distance.item()), 0.0, places=10)

    def test_discrete_hausdorff_distance_is_zero_for_reversed_same_segment(self) -> None:
        start = torch.tensor([[0.10, 0.0]], dtype=torch.float64)
        end = torch.tensor([[0.60, 0.0]], dtype=torch.float64)

        distance = poincare_geodesic_discrete_hausdorff_distance(
            start,
            end,
            end,
            start,
            curvature=1.0,
            num_points=9,
        )

        self.assertLess(float(distance.item()), 1e-8)

    def test_gromov_product_at_origin_recovers_radial_prefix_on_same_ray(self) -> None:
        nearer = torch.tensor([[0.25, 0.0]], dtype=torch.float64)
        farther = torch.tensor([[0.60, 0.0]], dtype=torch.float64)
        origin = torch.zeros_like(nearer)

        product = poincare_gromov_product_at_origin(nearer, farther, curvature=1.0)
        expected = poincare_directed_endpoint_product_distance(origin, origin, origin, nearer, curvature=1.0)

        self.assertAlmostEqual(float(product.item()), float(expected.item()), places=8)


if __name__ == "__main__":
    unittest.main()
