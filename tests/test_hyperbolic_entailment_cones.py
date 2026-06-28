from __future__ import annotations

import unittest

import torch

from geometry_profile_research.code2hyp_torch import torch_poincare_entailment_cone_energy


class HyperbolicEntailmentConeTests(unittest.TestCase):
    def test_radial_descendant_has_zero_cone_energy(self) -> None:
        apex = torch.tensor([[0.35, 0.0]], dtype=torch.float64)
        descendant = torch.tensor([[0.70, 0.0]], dtype=torch.float64)

        energy = torch_poincare_entailment_cone_energy(apex, descendant, curvature=1.0, cone_k=0.1)

        self.assertLess(float(energy.item()), 1e-8)

    def test_lateral_point_outside_narrow_cone_has_positive_energy(self) -> None:
        apex = torch.tensor([[0.35, 0.0]], dtype=torch.float64)
        lateral = torch.tensor([[0.35, 0.30]], dtype=torch.float64)

        energy = torch_poincare_entailment_cone_energy(apex, lateral, curvature=1.0, cone_k=0.05)

        self.assertGreater(float(energy.item()), 0.25)

    def test_curvature_scaling_preserves_unit_ball_cone_energy(self) -> None:
        unit_apex = torch.tensor([[0.35, 0.0]], dtype=torch.float64)
        unit_point = torch.tensor([[0.35, 0.30]], dtype=torch.float64)
        scaled_apex = unit_apex / 2.0
        scaled_point = unit_point / 2.0

        unit_energy = torch_poincare_entailment_cone_energy(unit_apex, unit_point, curvature=1.0, cone_k=0.05)
        scaled_energy = torch_poincare_entailment_cone_energy(scaled_apex, scaled_point, curvature=4.0, cone_k=0.05)

        self.assertAlmostEqual(float(unit_energy.item()), float(scaled_energy.item()), places=10)

    def test_cone_energy_is_differentiable_and_finite(self) -> None:
        apex = torch.tensor([[0.35, 0.0]], dtype=torch.float64, requires_grad=True)
        point = torch.tensor([[0.35, 0.30]], dtype=torch.float64, requires_grad=True)

        energy = torch_poincare_entailment_cone_energy(apex, point, curvature=1.0, cone_k=0.05)
        energy.sum().backward()

        self.assertTrue(torch.isfinite(energy).all())
        self.assertTrue(torch.isfinite(apex.grad).all())
        self.assertTrue(torch.isfinite(point.grad).all())


if __name__ == "__main__":
    unittest.main()
