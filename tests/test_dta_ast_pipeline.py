import ast
import csv
import tempfile
import unittest
from pathlib import Path

from geometry_profile_research.ast_features import (
    ast_markov_probabilities,
    ast_markov_rows,
    ast_node_histogram,
    ast_transition_counts,
    ast_root_paths,
    build_ast_graph,
)
from geometry_profile_research.analysis import geometry_profile_for_ast_source
from geometry_profile_research.dta import (
    DtaRecord,
    load_dta_records,
    stratified_sample_records,
    stratified_validation_test_split,
)
from geometry_profile_research.metrics import all_pairs_shortest_paths


class DtaLoaderTests(unittest.TestCase):
    def test_load_dta_records_from_task_csv_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_dir = Path(tmpdir)
            for task_id, rows in {
                "00": ["def main(x):\n    return x + 1\n"],
                "01": ["def main(x):\n    return x * 2\n", "def main(x):\n    return x - 2\n"],
            }.items():
                path = dataset_dir / f"task-{task_id}.csv"
                with path.open("w", encoding="utf-8", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=["code"])
                    writer.writeheader()
                    for code in rows:
                        writer.writerow({"code": code})

            records = load_dta_records(dataset_dir)

        self.assertEqual(len(records), 3)
        self.assertEqual([record.task_id for record in records], [0, 1, 1])
        self.assertEqual(records[0].record_id, "task-00:0")
        self.assertIn("return x + 1", records[0].code)

    def test_load_dta_records_supports_limit_per_task(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_dir = Path(tmpdir)
            path = dataset_dir / "task-10.csv"
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["code"])
                writer.writeheader()
                for value in range(5):
                    writer.writerow({"code": f"def main():\n    return {value}\n"})

            records = load_dta_records(dataset_dir, limit_per_task=2)

        self.assertEqual(len(records), 2)
        self.assertEqual([record.row_index for record in records], [0, 1])

    def test_stratified_sample_records_is_seeded_and_balanced(self):
        records = [
            DtaRecord(
                record_id=f"task-{task_id:02d}:{row}",
                task_id=task_id,
                row_index=row,
                source_file=f"task-{task_id:02d}.csv",
                code="def main():\n    return 1\n",
            )
            for task_id in (0, 1)
            for row in range(5)
        ]

        first = stratified_sample_records(records, per_task=2, seed=42)
        second = stratified_sample_records(records, per_task=2, seed=42)
        third = stratified_sample_records(records, per_task=2, seed=43)

        self.assertEqual([record.record_id for record in first], [record.record_id for record in second])
        self.assertNotEqual([record.record_id for record in first], [record.record_id for record in third])
        self.assertEqual([record.task_id for record in first].count(0), 2)
        self.assertEqual([record.task_id for record in first].count(1), 2)

    def test_stratified_validation_test_split_is_disjoint_and_balanced(self):
        records = [
            DtaRecord(
                record_id=f"task-{task_id:02d}:{row}",
                task_id=task_id,
                row_index=row,
                source_file=f"task-{task_id:02d}.csv",
                code="def main():\n    return 1\n",
            )
            for task_id in (0, 1)
            for row in range(10)
        ]

        split = stratified_validation_test_split(
            records,
            validation_per_task=3,
            test_per_task=4,
            seed=11,
        )

        validation_ids = {record.record_id for record in split["validation"]}
        test_ids = {record.record_id for record in split["test"]}
        self.assertFalse(validation_ids & test_ids)
        self.assertEqual([record.task_id for record in split["validation"]].count(0), 3)
        self.assertEqual([record.task_id for record in split["validation"]].count(1), 3)
        self.assertEqual([record.task_id for record in split["test"]].count(0), 4)
        self.assertEqual([record.task_id for record in split["test"]].count(1), 4)


class AstFeatureTests(unittest.TestCase):
    def test_build_ast_graph_preserves_parent_child_distances(self):
        code = "def main(x):\n    y = x + 1\n    return y\n"

        graph = build_ast_graph(code)
        distances = all_pairs_shortest_paths(graph)
        root = "0:Module"

        self.assertIn(root, graph.nodes)
        self.assertTrue(any(node.endswith(":FunctionDef") for node in graph.nodes))
        self.assertTrue(any(node.endswith(":Return") for node in graph.nodes))
        self.assertEqual(graph.depth(root), 0)
        self.assertGreater(max(distances[root].values()), 2)

    def test_ast_node_histogram_counts_ast_types(self):
        code = "def main(x):\n    return x + 1\n"

        histogram = ast_node_histogram(code)

        self.assertEqual(histogram["Module"], 1)
        self.assertEqual(histogram["FunctionDef"], 1)
        self.assertEqual(histogram["Return"], 1)
        self.assertEqual(histogram["BinOp"], 1)

    def test_ast_markov_probabilities_are_row_stochastic(self):
        code = "def main(x):\n    if x > 0:\n        return x\n    return -x\n"

        probabilities = ast_markov_probabilities(code)
        parents = {parent for parent, _ in probabilities}

        self.assertIn("Module", parents)
        for parent in parents:
            row_sum = sum(
                probability
                for (row_parent, _), probability in probabilities.items()
                if row_parent == parent
            )
            self.assertAlmostEqual(row_sum, 1.0)

    def test_ast_markov_rows_preserve_parent_rows(self):
        code = "def main(x):\n    return x + 1\n"

        rows = ast_markov_rows(code)

        self.assertIn("Module", rows)
        for row in rows.values():
            self.assertAlmostEqual(sum(row.values()), 1.0)

    def test_ast_transition_counts_are_not_row_normalized(self):
        code = "def main(x):\n    a = x + 1\n    b = x + 2\n    return a + b\n"

        counts = ast_transition_counts(code)

        self.assertTrue(all(isinstance(value, int) for value in counts.values()))
        self.assertGreater(sum(counts.values()), len(counts))

    def test_ast_features_raise_syntax_error_for_invalid_code(self):
        with self.assertRaises(SyntaxError):
            ast.parse("def broken(:\n")
        with self.assertRaises(SyntaxError):
            build_ast_graph("def broken(:\n")

    def test_ast_root_paths_encode_depth_and_node_type(self):
        code = "def main(x):\n    return x + 1\n"

        paths = ast_root_paths(code)

        self.assertIn("Module", paths)
        self.assertTrue(any("FunctionDef" in path for path in paths))
        self.assertTrue(any("Return" in path for path in paths))
        self.assertGreater(max(path.count("/") for path in paths), 2)

    def test_geometry_profile_for_ast_source_uses_ast_tree(self):
        code = "def main(x):\n    if x > 0:\n        return x\n    return -x\n"

        profile = geometry_profile_for_ast_source(code)

        self.assertGreater(profile.node_count, 0)
        self.assertGreater(profile.max_depth, 2)
        self.assertEqual(profile.hyperbolicity.delta, 0.0)
        self.assertGreater(profile.euclidean.pairs, 0)


if __name__ == "__main__":
    unittest.main()
