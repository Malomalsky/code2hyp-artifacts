from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from geometry_profile_research.raw_ast_axiom_probe import (
    AxiomProbeResult,
    load_axiom_probe_dataset,
    run_axiom_probe,
    spearman_correlation,
)


class RawAstAxiomProbeTests(unittest.TestCase):
    def test_spearman_correlation_reports_monotone_rank_agreement(self) -> None:
        target = [1.0, 2.0, 3.0, 4.0]

        self.assertAlmostEqual(spearman_correlation([1.0, 2.0, 3.0, 4.0], target), 1.0)
        self.assertAlmostEqual(spearman_correlation([4.0, 3.0, 2.0, 1.0], target), -1.0)
        self.assertEqual(spearman_correlation([1.0, 1.0, 1.0, 1.0], target), 0.0)

    def test_loader_uses_direct_edges_and_splits_path_axioms(self) -> None:
        payload = {
            "status": "ok",
            "paths": [
                {"start": 1, "end": 3, "lca": 2, "length": 2, "lca_depth": 1},
                {"start": 4, "end": 6, "lca": 5, "length": 4, "lca_depth": 2},
            ],
            "order_records": [
                {
                    "ancestor": 1,
                    "ancestor_depth": 0,
                    "descendant": 2,
                    "descendant_depth": 1,
                    "label": 1,
                    "is_direct_edge": True,
                    "tree_distance": 1,
                },
                {
                    "ancestor": 1,
                    "ancestor_depth": 0,
                    "descendant": 3,
                    "descendant_depth": 2,
                    "label": 1,
                    "is_direct_edge": False,
                    "tree_distance": 2,
                },
            ],
        }
        with TemporaryDirectory() as directory:
            path = Path(directory) / "relations.jsonl"
            path.write_text(json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8")

            dataset = load_axiom_probe_dataset(path)

        self.assertEqual(dataset.node_count, 6)
        self.assertEqual(dataset.train_edge_left.numel(), 1)
        self.assertEqual(dataset.train_node_depth.numel(), 3)
        self.assertEqual(dataset.train_path_start.numel(), 1)
        self.assertEqual(dataset.eval_path_start.numel(), 1)
        self.assertEqual(dataset.train_path_length.tolist(), [2.0])
        self.assertEqual(dataset.eval_path_length.tolist(), [4.0])

    def test_result_serializes_numeric_diagnostics(self) -> None:
        result = AxiomProbeResult(
            geometry="poincare",
            dim=4,
            seed=101,
            node_count=12,
            train_edge_count=8,
            train_path_count=10,
            eval_path_count=9,
            eval_length_spearman=0.75,
            eval_lca_depth_spearman=0.5,
            eval_length_stress=0.2,
            eval_lca_depth_stress=0.3,
            eval_additivity_residual_mean=0.4,
            eval_lca_radial_depth_spearman=0.6,
            train_edge_distance_mean=1.1,
        )

        payload = result.as_dict()

        self.assertEqual(payload["dim"], 4)
        self.assertEqual(payload["geometry"], "poincare")
        self.assertEqual(payload["eval_lca_depth_spearman"], 0.5)
        self.assertEqual(payload["eval_lca_radial_depth_spearman"], 0.6)
        self.assertEqual(payload["train_edge_distance_mean"], 1.1)

    def test_probe_can_run_matched_euclidean_control(self) -> None:
        payload = {
            "status": "ok",
            "paths": [
                {"start": 1, "end": 3, "lca": 2, "length": 2, "lca_depth": 1},
                {"start": 4, "end": 6, "lca": 5, "length": 4, "lca_depth": 2},
            ],
            "order_records": [
                {
                    "ancestor": 1,
                    "ancestor_depth": 0,
                    "descendant": 2,
                    "descendant_depth": 1,
                    "label": 1,
                    "is_direct_edge": True,
                    "tree_distance": 1,
                },
                {
                    "ancestor": 2,
                    "ancestor_depth": 1,
                    "descendant": 3,
                    "descendant_depth": 2,
                    "label": 1,
                    "is_direct_edge": True,
                    "tree_distance": 1,
                },
                {
                    "ancestor": 4,
                    "ancestor_depth": 0,
                    "descendant": 5,
                    "descendant_depth": 1,
                    "label": 1,
                    "is_direct_edge": True,
                    "tree_distance": 1,
                },
                {
                    "ancestor": 5,
                    "ancestor_depth": 1,
                    "descendant": 6,
                    "descendant_depth": 2,
                    "label": 1,
                    "is_direct_edge": True,
                    "tree_distance": 1,
                },
            ],
        }
        with TemporaryDirectory() as directory:
            path = Path(directory) / "relations.jsonl"
            path.write_text(json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8")
            dataset = load_axiom_probe_dataset(path)

        result = run_axiom_probe(dataset, geometry="euclidean", dim=2, seed=101, epochs=1)

        self.assertEqual(result.geometry, "euclidean")
        self.assertEqual(result.eval_path_count, 1)


if __name__ == "__main__":
    unittest.main()
