import json
import tempfile
import unittest
from pathlib import Path

from geometry_profile_research.analysis import geometry_profile_for_paths
from geometry_profile_research.io import extract_paths_from_text


class PathExtractionTests(unittest.TestCase):
    def test_extract_paths_from_free_form_text(self):
        path_text = """
changed: a/src/utils/io.py -> b/src/utils/io.py
tests/test_io.py
previous location: src/legacy/reader.py
new location: src/io/reader.py
"""

        paths = extract_paths_from_text(path_text)

        self.assertEqual(
            paths,
            [
                "src/io/reader.py",
                "src/legacy/reader.py",
                "src/utils/io.py",
                "tests/test_io.py",
            ],
        )

    def test_extract_paths_ignores_dev_null_and_duplicates(self):
        path_text = """
/dev/null
b/src/new_module.py
a/src/new_module.py
"""

        self.assertEqual(extract_paths_from_text(path_text), ["src/new_module.py"])


class GeometryProfileTests(unittest.TestCase):
    def test_geometry_profile_reports_core_metrics(self):
        profile = geometry_profile_for_paths(
            [
                "src/utils/io.py",
                "src/utils/path.py",
                "src/services/auth/token.py",
                "tests/test_io.py",
                "docs/readme.md",
            ]
        )

        self.assertGreater(profile.node_count, 0)
        self.assertGreater(profile.edge_count, 0)
        self.assertEqual(profile.hyperbolicity.delta, 0.0)
        self.assertGreater(profile.euclidean.pairs, 0)
        self.assertEqual(profile.euclidean.pairs, profile.hyperbolic.pairs)
        self.assertGreaterEqual(profile.geometry_advantage, -1.0)
        self.assertLessEqual(profile.geometry_advantage, 1.0)

    def test_geometry_profile_is_json_serializable(self):
        profile = geometry_profile_for_paths(["src/a/b/c.py", "src/a/b/d.py"])

        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "profile.json"
            output.write_text(json.dumps(profile.to_dict(), ensure_ascii=False), encoding="utf-8")
            loaded = json.loads(output.read_text(encoding="utf-8"))

        self.assertEqual(loaded["node_count"], profile.node_count)
        self.assertIn("hyperbolicity", loaded)
        self.assertIn("euclidean", loaded)
        self.assertIn("hyperbolic", loaded)


if __name__ == "__main__":
    unittest.main()
