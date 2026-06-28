from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts.run_lca_path_measure_weight_grid import run_lca_path_measure_weight_grid


class LcaPathMeasureWeightGridTests(unittest.TestCase):
    def test_searches_relation_specific_product_weights(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "A.java").write_text(
                """
                class A {
                    int add(int a, int b) {
                        int c = a + b;
                        return c;
                    }
                    int max(int a, int b) {
                        if (a > b) {
                            return a;
                        }
                        return b;
                    }
                }
                """,
                encoding="utf-8",
            )
            (root / "B.java").write_text(
                """
                class B {
                    int mul(int a, int b) {
                        int c = a * b;
                        return c;
                    }
                    boolean positive(int value) {
                        return value > 0;
                    }
                }
                """,
                encoding="utf-8",
            )

            result = run_lca_path_measure_weight_grid(
                (root,),
                weight_grid=((1.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
                max_files=2,
                max_methods=4,
                max_paths_per_method=8,
                min_paths_per_method=3,
                teacher_relation="lca_depth",
                pair_limit=6,
                alpha=0.75,
                epsilon=0.05,
                angle_mode="label_hash",
                gw_iterations=2,
                sinkhorn_iterations=60,
                train_fraction=0.5,
                split_seed=7,
            )

        self.assertEqual(result["config"]["teacher_relation"], "lca_depth")
        self.assertEqual(result["config"]["angle_mode"], "label_hash")
        self.assertEqual(len(result["weight_results"]), 2)
        self.assertIn("best_by_spearman", result)
        self.assertIn("lca_weight", result["best_by_spearman"])
        self.assertIn("spearman_against_teacher", result["best_by_spearman"])
        self.assertIn("heldout_selection", result)
        self.assertIn("selected_by_train", result["heldout_selection"])
        self.assertIn("test_spearman", result["heldout_selection"]["selected_by_train"])
        self.assertEqual(result["claim_boundary"]["teacher"], "raw-AST FGW")


if __name__ == "__main__":
    unittest.main()
