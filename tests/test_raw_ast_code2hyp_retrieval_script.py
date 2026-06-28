from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.run_raw_ast_code2hyp_retrieval import run_retrieval_experiment


JAVA_A = """
class A {
    int absLike(int x) {
        if (x > 0) {
            return x;
        }
        return -x;
    }
}
"""

JAVA_B = """
class B {
    int absLike(int x) {
        while (x > 0) {
            x = x - 1;
        }
        return x;
    }
}
"""

JAVA_C = """
class C {
    String join(String left, String right) {
        return left + right;
    }
}
"""

PYTHON_D = """
def clamp(x):
    if x < 0:
        return 0
    return x
"""

PYTHON_E = """
def countdown(x):
    while x > 0:
        x = x - 1
    return x
"""


class RawASTCode2HypRetrievalScriptTests(unittest.TestCase):
    def test_retrieval_runner_writes_reproducible_metrics_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            for name, source in {"A.java": JAVA_A, "B.java": JAVA_B, "C.java": JAVA_C}.items():
                (root / name).write_text(source, encoding="utf-8")
            (root / "D.py").write_text(PYTHON_D, encoding="utf-8")
            output = root / "retrieval.json"

            payload = run_retrieval_experiment(
                sources=(root,),
                output_path=output,
                language="auto",
                geometry="euclidean",
                dim=4,
                epochs=1,
                learning_rate=0.01,
                max_files=4,
                max_methods=4,
                max_paths=4,
                seed=123,
                sinkhorn_iterations=10,
                sinkhorn_epsilon=0.07,
                terminal_policy="class",
                path_object_mode="single_point",
                method_aggregation="centroid",
                path_cost_orientation="unoriented",
                curvature=0.5,
                positive_mode="alpha_structural_noop",
            )

            saved = json.loads(output.read_text(encoding="utf-8"))

        self.assertEqual(payload["config"]["geometry"], "euclidean")
        self.assertEqual(saved["config"]["seed"], 123)
        self.assertEqual(saved["config"]["language"], "auto")
        self.assertEqual(saved["config"]["terminal_policy"], "class")
        self.assertEqual(saved["config"]["path_object_mode"], "single_point")
        self.assertEqual(saved["config"]["method_aggregation"], "centroid")
        self.assertEqual(saved["config"]["path_cost_orientation"], "unoriented")
        self.assertEqual(saved["config"]["curvature"], 0.5)
        self.assertEqual(saved["config"]["sinkhorn_epsilon"], 0.07)
        self.assertEqual(saved["config"]["positive_mode"], "alpha_structural_noop")
        self.assertEqual(saved["transport"]["type"], "weighted_centroid_distance")
        self.assertEqual(saved["transport"]["epsilon"], 0.07)
        self.assertFalse(saved["transport"]["debiased"])
        self.assertEqual(saved["transport"]["path_object_mode"], "single_point")
        self.assertEqual(saved["transport"]["method_aggregation"], "centroid")
        self.assertEqual(saved["transport"]["path_cost_orientation"], "unoriented")
        self.assertEqual(saved["geometry_diagnostics"]["manifold"], "euclidean")
        self.assertEqual(saved["item_count"], 4)
        self.assertEqual({item["language"] for item in saved["items"]}, {"java", "python"})
        self.assertEqual(len(saved["training_history"]), 1)
        self.assertIn("branch_length", saved["training_history"][0])
        self.assertIn("reversal", saved["training_history"][0])
        self.assertEqual(len(saved["hard_negatives"]), 4)
        self.assertIn("lexical_similarity", saved["hard_negatives"][0])
        self.assertIn("structural_gap", saved["hard_negatives"][0])
        self.assertIn("recall_at_1", saved["metrics"])
        self.assertIn("ndcg_at_3", saved["metrics"])
        self.assertIn("mrr", saved["metrics"])
        self.assertIn("positive_distance_mean", saved["metrics"])
        self.assertIn("nearest_negative_distance_mean", saved["metrics"])
        self.assertIn("margin_mean", saved["metrics"])
        self.assertTrue(0.0 <= saved["metrics"]["mrr"] <= 1.0)
        self.assertTrue(0.0 <= saved["metrics"]["ndcg_at_3"] <= 1.0)
        self.assertEqual(len(saved["query_records"]), saved["item_count"])
        self.assertIn("rank", saved["query_records"][0])
        self.assertIn("top_candidates", saved["query_records"][0])

    def test_retrieval_runner_supports_python_only_mode(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "D.py").write_text(PYTHON_D, encoding="utf-8")
            (root / "E.py").write_text(PYTHON_E, encoding="utf-8")
            output = root / "python_retrieval.json"

            payload = run_retrieval_experiment(
                sources=(root,),
                output_path=output,
                language="python",
                geometry="euclidean",
                dim=4,
                epochs=1,
                learning_rate=0.01,
                max_files=2,
                max_methods=2,
                max_paths=4,
                seed=321,
                sinkhorn_iterations=10,
            )

        self.assertEqual(payload["config"]["language"], "python")
        self.assertEqual(payload["item_count"], 2)
        self.assertEqual({item["language"] for item in payload["items"]}, {"python"})

    def test_retrieval_runner_reports_poincare_curvature_diagnostics(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "A.java").write_text(JAVA_A, encoding="utf-8")
            (root / "B.java").write_text(JAVA_B, encoding="utf-8")
            output = root / "poincare_retrieval.json"

            payload = run_retrieval_experiment(
                sources=(root,),
                output_path=output,
                language="java",
                geometry="poincare",
                dim=2,
                epochs=0,
                max_files=2,
                max_methods=2,
                max_paths=4,
                seed=123,
                sinkhorn_iterations=5,
                curvature=1e-4,
                terminal_policy="class",
            )

        diagnostics = payload["geometry_diagnostics"]
        self.assertEqual(diagnostics["manifold"], "poincare")
        self.assertEqual(diagnostics["curvature"], 1e-4)
        self.assertIn("sqrt_curvature_norm_mean", diagnostics)
        self.assertIn("sqrt_curvature_norm_max", diagnostics)
        self.assertIn("near_boundary_fraction", diagnostics)
        self.assertIn("projection_active_fraction", diagnostics)
        self.assertLess(diagnostics["sqrt_curvature_norm_max"], 1.0)


if __name__ == "__main__":
    unittest.main()
