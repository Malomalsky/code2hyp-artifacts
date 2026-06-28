from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts.run_lca_path_measure_student_benchmark import run_lca_path_measure_student_benchmark


class LcaPathMeasureStudentBenchmarkTests(unittest.TestCase):
    def test_runs_student_benchmark_against_raw_ast_fgw_teacher(self) -> None:
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

            result = run_lca_path_measure_student_benchmark(
                (root,),
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
            )

        self.assertGreaterEqual(result["method_count"], 2)
        self.assertGreater(result["pair_count"], 0)
        self.assertEqual(result["config"]["teacher_relation"], "lca_depth")
        self.assertEqual(result["config"]["angle_mode"], "label_hash")
        self.assertIn("lca_product_sinkhorn", result["spearman_against_teacher"])
        self.assertIn("endpoint_product_sinkhorn", result["spearman_against_teacher"])
        self.assertIn("feature_ot", result["spearman_against_teacher"])
        self.assertIn("centroid", result["spearman_against_teacher"])
        self.assertIn("claim_boundary", result)
        self.assertEqual(result["claim_boundary"]["teacher"], "raw-AST FGW")


if __name__ == "__main__":
    unittest.main()
