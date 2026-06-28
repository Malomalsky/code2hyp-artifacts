from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.run_raw_ast_node_input_matrix import MatrixProject, _parse_projects, run_node_input_matrix


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
    int countdown(int x) {
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


class RawASTNodeInputMatrixScriptTests(unittest.TestCase):
    def test_default_project_points_to_raw_ast_source_tree(self) -> None:
        default_project = _parse_projects(None)[0]

        self.assertIn("code2seq_java_small_raw", str(default_project.source))
        self.assertIn("validation", str(default_project.source))

    def test_matrix_runner_writes_summary_and_per_run_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            project = root / "project"
            project.mkdir()
            for name, source in {"A.java": JAVA_A, "B.java": JAVA_B, "C.java": JAVA_C}.items():
                (project / name).write_text(source, encoding="utf-8")

            output = root / "matrix.json"
            runs_dir = root / "runs"
            summary = run_node_input_matrix(
                projects=(MatrixProject("toy-java", project),),
                output_path=output,
                runs_dir=runs_dir,
                node_input_modes=("label_only", "label_depth"),
                geometries=("euclidean",),
                dims=(2,),
                seeds=(11, 12),
                language="java",
                epochs=1,
                learning_rate=0.01,
                max_files=3,
                max_methods=3,
                max_paths=4,
                sinkhorn_iterations=5,
                terminal_policy="class",
                positive_mode="alpha_structural_noop",
                resume=False,
            )

            saved = json.loads(output.read_text(encoding="utf-8"))
            run_paths_exist = all(Path(run["output_path"]).exists() for run in saved["runs"])

            self.assertEqual(summary["experiment"], "raw_ast_node_input_matrix")
            self.assertEqual(saved["status"], "complete")
            self.assertEqual(saved["expected_runs"], 4)
            self.assertEqual(saved["completed_runs"], 4)
            self.assertEqual(saved["config"]["path_cost_orientation"], "directed")
            self.assertEqual(len(saved["runs"]), 4)
            self.assertEqual({run["node_input_mode"] for run in saved["runs"]}, {"label_only", "label_depth"})
            self.assertEqual({run["geometry"] for run in saved["runs"]}, {"euclidean"})
            self.assertEqual({run["seed"] for run in saved["runs"]}, {11, 12})
            self.assertTrue(run_paths_exist)
            self.assertTrue(all("mrr" in run for run in saved["runs"]))
            self.assertTrue(all("recall_at_1" in run for run in saved["runs"]))
            self.assertTrue(all("ndcg_at_3" in run for run in saved["runs"]))
            self.assertTrue(all(run["query_record_count"] > 0 for run in saved["runs"]))

    def test_matrix_runner_names_unoriented_runs_separately(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            project = root / "project"
            project.mkdir()
            for name, source in {"A.java": JAVA_A, "B.java": JAVA_B, "C.java": JAVA_C}.items():
                (project / name).write_text(source, encoding="utf-8")

            summary = run_node_input_matrix(
                projects=(MatrixProject("toy-java", project),),
                output_path=root / "matrix.json",
                runs_dir=root / "runs",
                node_input_modes=("label_only",),
                geometries=("euclidean",),
                dims=(2,),
                seeds=(11,),
                language="java",
                epochs=0,
                max_files=3,
                max_methods=3,
                max_paths=4,
                sinkhorn_iterations=5,
                terminal_policy="class",
                positive_mode="alpha_structural_noop",
                path_cost_orientation="unoriented",
                resume=False,
            )

        run = summary["runs"][0]
        self.assertEqual(summary["config"]["path_cost_orientation"], "unoriented")
        self.assertEqual(run["path_cost_orientation"], "unoriented")
        self.assertIn("unoriented_euclidean", Path(run["output_path"]).name)

    def test_matrix_runner_expands_path_cost_orientation_controls(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            project = root / "project"
            project.mkdir()
            for name, source in {"A.java": JAVA_A, "B.java": JAVA_B, "C.java": JAVA_C}.items():
                (project / name).write_text(source, encoding="utf-8")

            summary = run_node_input_matrix(
                projects=(MatrixProject("toy-java", project),),
                output_path=root / "matrix.json",
                runs_dir=root / "runs",
                node_input_modes=("label_only",),
                geometries=("euclidean",),
                dims=(2,),
                seeds=(11,),
                language="java",
                epochs=0,
                max_files=3,
                max_methods=3,
                max_paths=4,
                sinkhorn_iterations=5,
                terminal_policy="class",
                positive_mode="alpha_structural_noop",
                path_cost_orientations=("directed", "unoriented"),
                resume=False,
            )

        self.assertEqual(summary["expected_runs"], 2)
        self.assertEqual(summary["config"]["path_cost_orientation"], None)
        self.assertEqual(summary["config"]["path_cost_orientations"], ["directed", "unoriented"])
        self.assertEqual({run["path_cost_orientation"] for run in summary["runs"]}, {"directed", "unoriented"})
        self.assertEqual(len({run["output_path"] for run in summary["runs"]}), 2)

    def test_matrix_runner_expands_path_object_and_aggregation_controls(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            project = root / "project"
            project.mkdir()
            for name, source in {"A.java": JAVA_A, "B.java": JAVA_B, "C.java": JAVA_C}.items():
                (project / name).write_text(source, encoding="utf-8")

            summary = run_node_input_matrix(
                projects=(MatrixProject("toy-java", project),),
                output_path=root / "matrix.json",
                runs_dir=root / "runs",
                node_input_modes=("label_only",),
                path_object_modes=("single_point", "lca_product"),
                method_aggregations=("centroid", "measure"),
                geometries=("euclidean",),
                dims=(2,),
                seeds=(11,),
                language="java",
                epochs=0,
                max_files=3,
                max_methods=3,
                max_paths=4,
                sinkhorn_iterations=5,
                terminal_policy="class",
                positive_mode="alpha_structural_noop",
                resume=False,
            )

        self.assertEqual(summary["expected_runs"], 4)
        self.assertEqual(len(summary["runs"]), 4)
        self.assertEqual({run["path_object_mode"] for run in summary["runs"]}, {"single_point", "lca_product"})
        self.assertEqual({run["method_aggregation"] for run in summary["runs"]}, {"centroid", "measure"})

    def test_matrix_runner_expands_poincare_curvature_controls_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            project = root / "project"
            project.mkdir()
            for name, source in {"A.java": JAVA_A, "B.java": JAVA_B, "C.java": JAVA_C}.items():
                (project / name).write_text(source, encoding="utf-8")

            output = root / "matrix.json"
            summary = run_node_input_matrix(
                projects=(MatrixProject("toy-java", project),),
                output_path=output,
                runs_dir=root / "runs",
                node_input_modes=("label_only",),
                geometries=("euclidean", "poincare"),
                curvatures=(1e-4, 1.0),
                dims=(2,),
                seeds=(11,),
                language="java",
                epochs=0,
                max_files=3,
                max_methods=3,
                max_paths=4,
                sinkhorn_iterations=5,
                terminal_policy="class",
                positive_mode="alpha_structural_noop",
                resume=False,
            )

        self.assertEqual(summary["expected_runs"], 3)
        self.assertEqual(len(summary["runs"]), 3)
        euclidean_runs = [run for run in summary["runs"] if run["geometry"] == "euclidean"]
        poincare_runs = [run for run in summary["runs"] if run["geometry"] == "poincare"]
        self.assertEqual(len(euclidean_runs), 1)
        self.assertEqual({run["curvature"] for run in poincare_runs}, {1e-4, 1.0})
        self.assertTrue(all("geometry_diagnostics" in run for run in summary["runs"]))

    def test_matrix_runner_resumes_existing_per_run_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            project = root / "project"
            project.mkdir()
            for name, source in {"A.java": JAVA_A, "B.java": JAVA_B, "C.java": JAVA_C}.items():
                (project / name).write_text(source, encoding="utf-8")

            output = root / "matrix.json"
            runs_dir = root / "runs"
            first = run_node_input_matrix(
                projects=(MatrixProject("toy-java", project),),
                output_path=output,
                runs_dir=runs_dir,
                node_input_modes=("label_only",),
                geometries=("euclidean",),
                dims=(2,),
                seeds=(11,),
                language="java",
                epochs=1,
                max_files=3,
                max_methods=3,
                max_paths=4,
                sinkhorn_iterations=5,
                terminal_policy="class",
                positive_mode="alpha_structural_noop",
                resume=True,
            )
            run_path = Path(first["runs"][0]["output_path"])
            before_mtime_ns = run_path.stat().st_mtime_ns

            second = run_node_input_matrix(
                projects=(MatrixProject("toy-java", project),),
                output_path=output,
                runs_dir=runs_dir,
                node_input_modes=("label_only",),
                geometries=("euclidean",),
                dims=(2,),
                seeds=(11,),
                language="java",
                epochs=1,
                max_files=3,
                max_methods=3,
                max_paths=4,
                sinkhorn_iterations=5,
                terminal_policy="class",
                positive_mode="alpha_structural_noop",
                resume=True,
            )

            self.assertEqual(len(second["runs"]), 1)
            self.assertTrue(second["runs"][0]["resumed"])
            self.assertEqual(run_path.stat().st_mtime_ns, before_mtime_ns)


if __name__ == "__main__":
    unittest.main()
