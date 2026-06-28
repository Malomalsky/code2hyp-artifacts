from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from scripts.materialize_dta_python_sources import materialize_dta_python_sources


class MaterializeDTAPythonSourcesTests(unittest.TestCase):
    def test_materializer_writes_deduplicated_parseable_python_files_and_manifest(self) -> None:
        valid_code = "def main(x):\n    return x + 1\n"
        duplicate_code = valid_code
        invalid_code = "def broken(:\n    pass\n"
        no_callable = "x = 1\n"
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            input_dir = root / "input"
            output_dir = root / "output"
            input_dir.mkdir()
            with (input_dir / "task-00.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=["code"])
                writer.writeheader()
                writer.writerow({"code": valid_code})
                writer.writerow({"code": duplicate_code})
                writer.writerow({"code": invalid_code})
                writer.writerow({"code": no_callable})

            payload = materialize_dta_python_sources(
                input_dir=input_dir,
                output_dir=output_dir,
                max_records_per_task=4,
                min_paths=1,
            )

            files = list((output_dir / "task-00").glob("*.py"))
            manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(len(files), 1)
            self.assertEqual(payload["summary"][0]["written"], 1)
            self.assertEqual(payload["summary"][0]["skipped_duplicates"], 1)
            self.assertEqual(payload["summary"][0]["parse_errors"], 1)
            self.assertEqual(payload["summary"][0]["skipped_no_callable"], 1)
            self.assertEqual(len(manifest["files"]), 1)


if __name__ == "__main__":
    unittest.main()
