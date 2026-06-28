from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts.run_raw_ast_fgw_benchmark import collect_raw_ast_method_spaces, run_raw_ast_fgw_benchmark


class RawAstFgwBenchmarkScriptTests(unittest.TestCase):
    def test_collects_real_java_method_spaces_and_runs_fgw_benchmark(self) -> None:
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

            spaces = collect_raw_ast_method_spaces(
                (root,),
                max_files=2,
                max_methods=4,
                max_paths_per_method=8,
                min_paths_per_method=3,
            )
            result = run_raw_ast_fgw_benchmark(
                (root,),
                max_files=2,
                max_methods=4,
                max_paths_per_method=8,
                min_paths_per_method=3,
                structural_relation="edge_jaccard",
                pair_limit=6,
                alpha=0.5,
                epsilon=0.05,
                gw_iterations=2,
                sinkhorn_iterations=60,
            )

        self.assertGreaterEqual(len(spaces), 2)
        self.assertEqual(result["method_count"], len(spaces))
        self.assertEqual(result["config"]["structural_relation"], "edge_jaccard")
        self.assertTrue(all(method["structural_relation"] == "edge_jaccard" for method in result["methods"]))
        self.assertGreater(result["pair_count"], 0)
        self.assertIn("distance_summary", result)
        self.assertIn("fgw", result["distance_summary"])
        self.assertIn("retrieval_overlap_at_1", result)
        self.assertIn("ot_feature", result["retrieval_overlap_at_1"])
        self.assertIn("mean_plan_entropy", result)
        self.assertLess(result["max_marginal_residual"], 1e-3)

    def test_benchmark_accepts_lca_anchored_and_composite_structural_relations(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "C.java").write_text(
                """
                class C {
                    int choose(int a, int b) {
                        if (a > b) {
                            return a - b;
                        }
                        return b - a;
                    }
                    int clamp(int value) {
                        if (value < 0) {
                            return 0;
                        }
                        if (value > 10) {
                            return 10;
                        }
                        return value;
                    }
                }
                """,
                encoding="utf-8",
            )
            lca_result = run_raw_ast_fgw_benchmark(
                (root,),
                max_files=1,
                max_methods=2,
                max_paths_per_method=8,
                min_paths_per_method=3,
                structural_relation="lca_anchored_product",
                alpha=0.75,
                epsilon=0.05,
                gw_iterations=2,
                sinkhorn_iterations=60,
            )
            composite_result = run_raw_ast_fgw_benchmark(
                (root,),
                max_files=1,
                max_methods=2,
                max_paths_per_method=8,
                min_paths_per_method=3,
                structural_relation="multi_endpoint_lca_edge",
                alpha=0.75,
                epsilon=0.05,
                gw_iterations=2,
                sinkhorn_iterations=60,
            )

        self.assertEqual(lca_result["config"]["structural_relation"], "lca_anchored_product")
        self.assertEqual(composite_result["config"]["structural_relation"], "multi_endpoint_lca_edge")
        self.assertEqual(lca_result["method_count"], composite_result["method_count"])
        self.assertIn("fgw_structure_term", lca_result["pairs"][0])
        self.assertIn("fgw_structure_term", composite_result["pairs"][0])

    def test_collects_reproducible_seeded_method_sample_instead_of_prefix_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "Sample.java").write_text(
                """
                class Sample {
                    int first(int a, int b) {
                        int c = a + b;
                        return c;
                    }
                    int second(int a, int b) {
                        int c = a - b;
                        return c;
                    }
                    int third(int a, int b) {
                        int c = a * b;
                        return c;
                    }
                    int fourth(int a, int b) {
                        if (b == 0) {
                            return a;
                        }
                        return a / b;
                    }
                }
                """,
                encoding="utf-8",
            )

            prefix_spaces = collect_raw_ast_method_spaces(
                (root,),
                max_files=1,
                max_methods=2,
                max_paths_per_method=8,
                min_paths_per_method=3,
            )
            sampled_once = collect_raw_ast_method_spaces(
                (root,),
                max_files=1,
                max_methods=2,
                max_paths_per_method=8,
                min_paths_per_method=3,
                sample_seed=1,
            )
            sampled_twice = collect_raw_ast_method_spaces(
                (root,),
                max_files=1,
                max_methods=2,
                max_paths_per_method=8,
                min_paths_per_method=3,
                sample_seed=1,
            )

        prefix_names = [space.scope_name for space in prefix_spaces]
        sampled_names = [space.scope_name for space in sampled_once]
        self.assertEqual(sampled_names, [space.scope_name for space in sampled_twice])
        self.assertNotEqual(prefix_names, sampled_names)
        self.assertEqual(len(sampled_names), 2)


if __name__ == "__main__":
    unittest.main()
