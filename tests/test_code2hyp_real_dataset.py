from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from geometry_profile_research.code2hyp_real_dataset import (
    ArchiveStatus,
    Code2SeqPreprocessedInventory,
    java_small_preprocessed_spec,
    inspect_archive_status,
    write_dataset_manifest,
)


class Code2HypRealDatasetTests(unittest.TestCase):
    def test_java_small_spec_points_to_real_code2seq_preprocessed_archive(self) -> None:
        spec = java_small_preprocessed_spec()

        self.assertEqual(spec.name, "code2seq-java-small-preprocessed")
        self.assertEqual(
            spec.url,
            "https://s3.amazonaws.com/code2seq/datasets/java-small-preprocessed.tar.gz",
        )
        self.assertEqual(spec.expected_bytes, 479_663_374)
        self.assertEqual(spec.archive_name, "java-small-preprocessed.tar.gz")
        self.assertIn("code2seq", spec.source_repository)

    def test_inventory_detects_expected_split_files_without_synthetic_data(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "java-small.train.c2s").write_text("train a,A,b\n", encoding="utf-8")
            (root / "java-small.val.c2s").write_text("val a,A,b\n", encoding="utf-8")
            (root / "java-small.test.c2s").write_text("test a,A,b\n", encoding="utf-8")

            inventory = Code2SeqPreprocessedInventory.from_directory(root)

        self.assertTrue(inventory.has_all_required_splits)
        self.assertEqual(inventory.split_paths["train"].name, "java-small.train.c2s")
        self.assertEqual(inventory.split_line_counts["train"], 1)

    def test_manifest_records_source_and_rejects_synthetic_evidence_boundary(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "manifest.md"

            write_dataset_manifest(output, java_small_preprocessed_spec())

            manifest = output.read_text(encoding="utf-8")

        self.assertIn("Real dataset source", manifest)
        self.assertIn("java-small-preprocessed.tar.gz", manifest)
        self.assertIn("Synthetic datasets are not valid research evidence", manifest)

    def test_archive_status_checks_expected_size_without_opening_archive(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            archive = Path(tmpdir) / "archive.tar.gz"
            archive.write_bytes(b"abc")

            status = inspect_archive_status(
                archive,
                expected_bytes=3,
            )

        self.assertEqual(status, ArchiveStatus(path=archive, exists=True, bytes=3, size_matches=True))


if __name__ == "__main__":
    unittest.main()
