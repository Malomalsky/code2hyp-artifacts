from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.run_lca_path_measure_student_benchmark import run_lca_path_measure_student_benchmark
from scripts.run_learned_poincare_path_student import fit_learned_poincare_path_student


class LearnedPoincarePathStudentTests(unittest.TestCase):
    def test_trains_shared_label_depth_encoder_smoke(self) -> None:
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
                    int min(int a, int b) {
                        if (a < b) {
                            return a;
                        }
                        return b;
                    }
                }
                """,
                encoding="utf-8",
            )
            pair_payload = run_lca_path_measure_student_benchmark(
                (root,),
                max_files=1,
                max_methods=3,
                max_paths_per_method=6,
                min_paths_per_method=3,
                teacher_relation="lca_depth",
                alpha=0.75,
                epsilon=0.05,
                angle_mode="label_hash",
                gw_iterations=2,
                sinkhorn_iterations=40,
            )
            pair_json = root / "pairs.json"
            pair_json.write_text(json.dumps(pair_payload), encoding="utf-8")

            result = fit_learned_poincare_path_student(
                (root,),
                pair_json=pair_json,
                max_files=1,
                max_methods=3,
                max_paths_per_method=6,
                min_paths_per_method=3,
                teacher_relation="lca_depth",
                epochs=3,
                batch_size=2,
                sinkhorn_iterations=20,
                split_seed=11,
                gromov_loss_weight=0.1,
                use_prefix_encoder=True,
                geometry="euclidean",
                node_input_mode="label_only",
            )

        self.assertEqual(result["config"]["teacher_relation"], "lca_depth")
        self.assertEqual(result["config"]["geometry"], "euclidean")
        self.assertEqual(result["config"]["node_input_mode"], "label_only")
        self.assertEqual(result["method_count"], 3)
        self.assertIn("train", result)
        self.assertIn("test", result)
        self.assertIn("factor_weights", result)
        self.assertGreaterEqual(result["factor_weights"]["lca_weight"], 0.0)
        self.assertEqual(result["config"]["gromov_loss_weight"], 0.1)
        self.assertTrue(result["config"]["use_prefix_encoder"])
        self.assertIn("gromov_alignment", result)
        self.assertIn("claim_boundary", result)


if __name__ == "__main__":
    unittest.main()
