from __future__ import annotations

import unittest

from geometry_profile_research.raw_ast_order_probe import (
    OrderProbeRecord,
    binary_auc,
    split_order_probe_records,
)


class RawAstOrderProbeTests(unittest.TestCase):
    def test_binary_auc_reports_pairwise_ranking_quality(self) -> None:
        labels = [1, 1, 0, 0]
        scores = [0.9, 0.8, 0.2, 0.1]

        self.assertEqual(binary_auc(scores, labels), 1.0)
        self.assertEqual(binary_auc([0.5, 0.5, 0.5, 0.5], labels), 0.5)
        self.assertEqual(binary_auc([0.1, 0.2, 0.8, 0.9], labels), 0.0)

    def test_split_uses_direct_edges_for_training_and_transitive_pairs_for_evaluation(self) -> None:
        records = (
            OrderProbeRecord(scope_index=0, ancestor=0, descendant=1, label=1, is_direct_edge=True, tree_distance=1),
            OrderProbeRecord(scope_index=0, ancestor=1, descendant=2, label=1, is_direct_edge=True, tree_distance=1),
            OrderProbeRecord(scope_index=0, ancestor=0, descendant=2, label=1, is_direct_edge=False, tree_distance=2),
            OrderProbeRecord(scope_index=0, ancestor=2, descendant=0, label=0, is_direct_edge=False, tree_distance=2),
            OrderProbeRecord(scope_index=0, ancestor=2, descendant=1, label=0, is_direct_edge=False, tree_distance=1),
        )

        split = split_order_probe_records(records)

        self.assertEqual([(record.ancestor, record.descendant) for record in split.train_positive], [(0, 1), (1, 2)])
        self.assertEqual([(record.ancestor, record.descendant) for record in split.eval_positive], [(0, 2)])
        self.assertEqual(len(split.train_negative), 1)
        self.assertEqual(len(split.eval_negative), 1)


if __name__ == "__main__":
    unittest.main()
