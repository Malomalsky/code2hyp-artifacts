from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.evaluate_code2hyp_claim_gates import evaluate_claim_gates, format_markdown


def _row(project: str, *, contrast: str, cell: str, mean: float, low: float, high: float) -> dict[str, object]:
    return {
        "project": project,
        "node_input_mode": "label_only",
        "cell": cell,
        "contrast": contrast,
        "n_seeds": 3,
        "n_queries": 192,
        "mean_delta_reciprocal_rank": mean,
        "ci_low_delta_reciprocal_rank": low,
        "ci_high_delta_reciprocal_rank": high,
        "positive_queries": 40,
        "mean_delta_margin": 0.1,
    }


class EvaluateCode2HypClaimGatesTests(unittest.TestCase):
    def test_supported_product_measure_claim_is_reported(self) -> None:
        payload = {
            "path_object_deltas_aggregated": [
                _row("p1", contrast="LCA-product - single point", cell="path=lca_product", mean=0.05, low=0.02, high=0.08),
                _row("p2", contrast="LCA-product - single point", cell="path=lca_product", mean=0.06, low=0.03, high=0.09),
            ],
            "aggregation_deltas_aggregated": [
                _row("p1", contrast="measure - centroid", cell="method=measure", mean=0.04, low=0.01, high=0.07),
                _row("p2", contrast="measure - centroid", cell="method=measure", mean=0.05, low=0.02, high=0.08),
            ],
            "curvature_deltas_aggregated": [
                _row(
                    "p1",
                    contrast="Poincare c=1 - Euclidean",
                    cell="path=lca_product; method=measure; geometry=poincare c=1",
                    mean=0.02,
                    low=0.005,
                    high=0.04,
                )
            ],
            "full_model_deltas_aggregated": [
                _row(
                    "p1",
                    contrast="Poincare LCA-product measure - Euclidean single-point centroid",
                    cell="path=lca_product; method=measure",
                    mean=0.1,
                    low=0.06,
                    high=0.14,
                ),
                _row(
                    "p2",
                    contrast="Poincare LCA-product measure - Euclidean single-point centroid",
                    cell="path=lca_product; method=measure",
                    mean=0.2,
                    low=0.12,
                    high=0.28,
                ),
            ],
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "deltas.json"
            path.write_text(json.dumps(payload), encoding="utf-8")

            result = evaluate_claim_gates(path, min_projects=2)
            markdown = format_markdown(result)

        self.assertEqual(result["overall_claim_status"], "representation_claims_supported_under_matched_geometry_controls")
        gates = {gate["name"]: gate for gate in result["gates"]}
        self.assertEqual(gates["H1_LCA_product_path_object"]["decision"], "supported")
        self.assertEqual(gates["H2_measure_over_paths"]["decision"], "supported")
        self.assertIn("full_model_contrast", markdown)

    def test_missing_rows_are_not_evaluable(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "empty.json"
            path.write_text(json.dumps({}), encoding="utf-8")

            result = evaluate_claim_gates(path, min_projects=2)

        self.assertEqual(result["overall_claim_status"], "continue_experiments_before_claiming_main_result")
        self.assertTrue(all(gate["decision"] == "not_evaluable" for gate in result["gates"]))


if __name__ == "__main__":
    unittest.main()
