from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from scripts.materialize_task_labeled_corpus import materialize_task_labeled_corpus


class MaterializeTaskLabeledCorpusTests(unittest.TestCase):
    def test_generic_python_layout_filters_and_writes_task_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            input_root = root / "raw"
            output_dir = root / "materialized"
            task_a = input_root / "task-alpha"
            task_b = input_root / "task-beta"
            task_a.mkdir(parents=True)
            task_b.mkdir(parents=True)
            for index in range(3):
                (task_a / f"a{index}.py").write_text(
                    f"def solve_{index}(x):\n    value = x + {index}\n    return value * 2\n",
                    encoding="utf-8",
                )
                (task_b / f"b{index}.py").write_text(
                    f"def solve_{index}(items):\n    total = 0\n    for item in items:\n        total += item\n    return total + {index}\n",
                    encoding="utf-8",
                )
            (task_a / "broken.py").write_text("def broken(:\n    pass\n", encoding="utf-8")

            payload = materialize_task_labeled_corpus(
                input_root=input_root,
                output_dir=output_dir,
                layout="generic",
                language="python",
                max_files_per_task=2,
                min_files_per_task=2,
                min_paths=1,
                seed=7,
            )

            manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(payload["task_count"], 2)
            self.assertEqual(payload["file_count"], 4)
            self.assertEqual(manifest["task_count"], 2)
            self.assertEqual(len(manifest["factor_matrix_task_args"]), 6)
            for task in manifest["tasks"]:
                self.assertEqual(task["selected_files"], 2)
                self.assertTrue(Path(task["path"]).is_dir())
            written_files = sorted(output_dir.glob("task-*/*.py"))
            self.assertEqual(len(written_files), 4)

    def test_codenet_layout_discovers_language_subdirectories(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            input_root = root / "Project_CodeNet"
            problem = input_root / "data" / "p00001" / "Python"
            other_language = input_root / "data" / "p00001" / "Java"
            problem.mkdir(parents=True)
            other_language.mkdir(parents=True)
            for index in range(2):
                (problem / f"s{index}.py").write_text(f"def f{index}():\n    return {index}\n", encoding="utf-8")
            (other_language / "Ignored.java").write_text("class Ignored {}", encoding="utf-8")

            payload = materialize_task_labeled_corpus(
                input_root=input_root,
                output_dir=root / "out",
                layout="codenet",
                language="python",
                max_files_per_task=2,
                min_files_per_task=2,
                validate_parse=False,
            )

            self.assertEqual(payload["task_count"], 1)
            self.assertEqual(payload["tasks"][0]["source_task_label"], "p00001")
            self.assertEqual(payload["file_count"], 2)
            self.assertIn("run_dta_factor_matrix.py", payload["recommended_factor_matrix_command"])

    def test_table_layout_materializes_csv_rows_with_source_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            input_path = root / "hf_export.csv"
            output_dir = root / "out"
            with input_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=["id", "problem_id", "code"])
                writer.writeheader()
                for task in ("p00001", "p00002"):
                    for index in range(2):
                        writer.writerow(
                            {
                                "id": f"{task}_{index}",
                                "problem_id": task,
                                "code": f"def solve_{task}_{index}(x):\n    return x + {index}\n",
                            }
                        )

            payload = materialize_task_labeled_corpus(
                input_root=input_path,
                output_dir=output_dir,
                layout="table",
                language="python",
                table_code_column="code",
                table_label_column="problem_id",
                table_id_column="id",
                max_files_per_task=2,
                min_files_per_task=2,
                min_paths=1,
                seed=1,
            )

            manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(payload["task_count"], 2)
            self.assertEqual(payload["file_count"], 4)
            self.assertEqual(manifest["table_label_column"], "problem_id")
            self.assertTrue(all("#p0000" in row["source_path"] for row in manifest["files"]))

    def test_table_layout_materializes_jsonl_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            input_path = root / "export.jsonl"
            rows = [
                {"label": "0", "code": "def f0():\n    return 0\n"},
                {"label": "0", "code": "def f1():\n    return 1\n"},
                {"label": "1", "code": "def g0():\n    return 0\n"},
                {"label": "1", "code": "def g1():\n    return 1\n"},
            ]
            input_path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")

            payload = materialize_task_labeled_corpus(
                input_root=input_path,
                output_dir=root / "out",
                layout="table",
                language="python",
                max_files_per_task=2,
                min_files_per_task=2,
                min_paths=1,
            )

            self.assertEqual(payload["task_count"], 2)
            self.assertEqual(payload["file_count"], 4)

    def test_table_layout_supports_module_scope_for_top_level_programs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            input_path = root / "bugnet_like.jsonl"
            rows = []
            for task in ("p02658", "p02924"):
                for index in range(2):
                    code = (
                        f"n = {index + 2}\n"
                        "total = 0\n"
                        "for value in range(n):\n"
                        "    if value % 2 == 0:\n"
                        "        total += value\n"
                        "print(total)\n"
                    )
                    rows.append({"problem_id": task, "pass": code})
            input_path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")

            with self.assertRaises(ValueError):
                materialize_task_labeled_corpus(
                    input_root=input_path,
                    output_dir=root / "callable_out",
                    layout="table",
                    language="python",
                    table_code_column="pass",
                    table_label_column="problem_id",
                    max_files_per_task=2,
                    min_files_per_task=2,
                    min_paths=2,
                )

            payload = materialize_task_labeled_corpus(
                input_root=input_path,
                output_dir=root / "module_out",
                layout="table",
                language="python",
                table_code_column="pass",
                table_label_column="problem_id",
                max_files_per_task=2,
                min_files_per_task=2,
                min_paths=2,
                item_scope="module",
            )

            self.assertEqual(payload["item_scope"], "module")
            self.assertEqual(payload["task_count"], 2)
            self.assertEqual(payload["file_count"], 4)


if __name__ == "__main__":
    unittest.main()
