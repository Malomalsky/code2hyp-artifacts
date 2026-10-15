"""Small checks for the portable export and fixed candidate binding."""

import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent / "source"))
from test_e2e_continuation_20261015 import ReplayTests
from test_e2e_20261014 import BenchmarkTests
from source_normalization import normalize_python_source


class ExportTests(unittest.TestCase):
    def test_original_source_normalization(self):
        normalized = normalize_python_source(b"x = 1 \t\r\n\r\n")
        self.assertTrue(normalized.decode_ok)
        self.assertEqual(normalized.text, "x = 1\n")


if __name__ == "__main__":
    unittest.main()
