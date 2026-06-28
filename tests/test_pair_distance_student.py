from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.run_pair_distance_student import fit_pair_distance_student


class PairDistanceStudentTests(unittest.TestCase):
    def test_fits_nonnegative_distance_student_on_heldout_pairs(self) -> None:
        pairs = []
        for index in range(12):
            lca = float(index % 4)
            feature = float(index // 4)
            teacher = 2.0 * lca + 0.5 * feature
            pairs.append(
                {
                    "left": index % 4,
                    "right": (index + 1) % 4,
                    "teacher_fgw": teacher,
                    "lca_only_sinkhorn": lca,
                    "feature_ot": feature,
                    "centroid": float(index % 3),
                    "endpoint_product_sinkhorn": float(index % 2),
                    "lca_product_sinkhorn": float(index % 5),
                }
            )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "pairs.json"
            path.write_text(json.dumps({"pairs": pairs, "method_count": 4}), encoding="utf-8")
            result = fit_pair_distance_student(
                path,
                feature_names=("lca_only_sinkhorn", "feature_ot", "centroid"),
                train_fraction=0.67,
                split_seed=1,
                epochs=200,
                learning_rate=0.05,
            )

        self.assertEqual(result["config"]["target"], "teacher_fgw")
        self.assertGreater(result["test"]["spearman"], 0.5)
        self.assertEqual(len(result["weights"]), 3)
        self.assertGreaterEqual(result["weights"]["lca_only_sinkhorn"], 0.0)
        self.assertGreaterEqual(result["weights"]["feature_ot"], 0.0)
        self.assertIn("claim_boundary", result)


if __name__ == "__main__":
    unittest.main()
