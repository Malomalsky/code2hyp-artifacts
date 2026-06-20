from __future__ import annotations

import unittest

from geometry_profile_research.code2hyp_smoke import build_code2hyp_smoke_report


class Code2HypSmokeReportTests(unittest.TestCase):
    def test_smoke_report_contains_matched_capacity_evidence(self) -> None:
        report = build_code2hyp_smoke_report(seed=17)

        self.assertEqual(report["batch_size"], 2)
        self.assertEqual(report["label_vocab_size"], 7)
        self.assertEqual(report["variants"]["B1_euclidean"]["logits_shape"], [2, 7])
        self.assertEqual(report["variants"]["B3_product_trainable_curvature"]["logits_shape"], [2, 7])
        self.assertEqual(
            report["variants"]["B1_euclidean"]["representation_shape"],
            report["variants"]["B3_product_trainable_curvature"]["representation_shape"],
        )
        self.assertEqual(report["parameter_delta_B3_minus_B1"], 1)
        self.assertLess(report["relative_parameter_overhead_B3_vs_B1"], 0.03)
        self.assertGreater(report["variants"]["B3_product_trainable_curvature"]["curvature"], 0.0)


if __name__ == "__main__":
    unittest.main()
