from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.extract_java_raw_ast_relations import (
    extract_java_raw_ast_relations,
    write_java_raw_ast_relations_jsonl,
)


JAVA_SOURCE = """
class Demo {
    int absLike(int x) {
        if (x > 0) {
            return x;
        }
        return -x;
    }
}
"""

JAVA_TWO_METHODS = """
class Demo {
    int first(int x) {
        return x + 1;
    }

    int second(int x) {
        return x - 1;
    }
}
"""

JAVA_CONSTRUCTOR_THEN_METHOD = """
class Demo {
    Demo() {
    }

    int value() {
        return 1;
    }
}
"""


class ExtractJavaRawAstRelationsScriptTests(unittest.TestCase):
    def test_extracts_json_friendly_raw_ast_relation_targets_from_java_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            source_path = Path(tmpdir) / "Demo.java"
            source_path.write_text(JAVA_SOURCE, encoding="utf-8")

            payloads = extract_java_raw_ast_relations(
                (source_path,),
                max_files=1,
                max_paths_per_file=4,
                max_records_per_file=3,
            )

        self.assertEqual(len(payloads), 1)
        payload = payloads[0]
        self.assertEqual(payload["status"], "ok")
        self.assertEqual(payload["source_path"], str(source_path))
        self.assertGreater(payload["node_count"], 0)
        self.assertEqual(payload["path_count"], 4)
        self.assertEqual(payload["record_count"], 3)
        self.assertEqual(payload["records"][0]["left_index"], 0)
        self.assertIn("oriented_endpoint_distance", payload["records"][0])
        self.assertIn("edge_jaccard_distance", payload["records"][0])
        self.assertIn("left_lca_depth", payload["records"][0])
        self.assertIn("start_id", payload["paths"][0])
        self.assertIn("end_id", payload["paths"][0])
        self.assertIn("lca_id", payload["paths"][0])
        self.assertIn("left_branch_node_ids", payload["paths"][0])
        self.assertIn("right_branch_node_ids", payload["paths"][0])
        self.assertIn("directed_edge_types", payload["paths"][0])
        self.assertGreater(len(payload["paths"][0]["directed_edge_types"]), 0)

    def test_callable_scope_emits_one_payload_per_method_with_method_name(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            source_path = Path(tmpdir) / "Demo.java"
            source_path.write_text(JAVA_TWO_METHODS, encoding="utf-8")

            payloads = extract_java_raw_ast_relations(
                (source_path,),
                scope="callable",
                max_files=1,
                max_paths_per_file=4,
                max_records_per_file=3,
            )

        self.assertEqual([payload["scope_name"] for payload in payloads], ["first", "second"])
        self.assertEqual([payload["scope_label"] for payload in payloads], ["MethodDeclaration", "MethodDeclaration"])
        self.assertTrue(all(payload["status"] == "ok" for payload in payloads))
        self.assertTrue(all(payload["source_path"] == str(source_path) for payload in payloads))

    def test_callable_scope_preserves_ast_preorder_for_constructors_and_methods(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            source_path = Path(tmpdir) / "Demo.java"
            source_path.write_text(JAVA_CONSTRUCTOR_THEN_METHOD, encoding="utf-8")

            payloads = extract_java_raw_ast_relations(
                (source_path,),
                scope="callable",
                max_files=1,
                max_paths_per_file=4,
                max_records_per_file=3,
            )

        self.assertEqual([payload["scope_label"] for payload in payloads], ["ConstructorDeclaration", "MethodDeclaration"])
        self.assertEqual([payload["scope_name"] for payload in payloads], ["Demo", "value"])

    def test_optional_order_targets_are_emitted_for_entailment_experiments(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            source_path = Path(tmpdir) / "Demo.java"
            source_path.write_text(JAVA_SOURCE, encoding="utf-8")

            payloads = extract_java_raw_ast_relations(
                (source_path,),
                scope="callable",
                max_files=1,
                max_paths_per_file=4,
                max_records_per_file=3,
                include_order_relations=True,
                max_order_records_per_scope=10,
            )

        payload = payloads[0]
        self.assertEqual(payload["status"], "ok")
        self.assertIn("order_record_count", payload)
        self.assertGreater(payload["order_record_count"], 0)
        self.assertEqual(len(payload["order_records"]), payload["order_record_count"])
        self.assertIn("ancestor", payload["order_records"][0])
        self.assertIn("descendant", payload["order_records"][0])
        self.assertIn("label", payload["order_records"][0])
        self.assertTrue(any(record["label"] == 1 for record in payload["order_records"]))

    def test_writer_keeps_parse_errors_as_auditable_jsonl_records(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            bad_source_path = Path(tmpdir) / "Broken.java"
            output_path = Path(tmpdir) / "relations.jsonl"
            bad_source_path.write_text("class Broken {", encoding="utf-8")

            write_java_raw_ast_relations_jsonl(
                (bad_source_path,),
                output_path,
                max_files=1,
                max_paths_per_file=4,
                max_records_per_file=3,
            )

            lines = output_path.read_text(encoding="utf-8").splitlines()

        self.assertEqual(len(lines), 1)
        payload = json.loads(lines[0])
        self.assertEqual(payload["status"], "error")
        self.assertEqual(payload["source_path"], str(bad_source_path))
        self.assertIn("error", payload)


if __name__ == "__main__":
    unittest.main()
