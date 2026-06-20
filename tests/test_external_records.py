import unittest

from geometry_profile_research.external import group_records_by_repo


class ExternalRecordGroupingTests(unittest.TestCase):
    def test_group_records_by_repo_keeps_paths_and_languages(self):
        records = [
            {"repo_name": "org/a", "path": "src/main.py", "language": "Python"},
            {"repo_name": "org/a", "path": "tests/test_main.py", "language": "Python"},
            {"repo_name": "org/b", "path": "src/Main.java", "language": "Java"},
        ]

        repos = group_records_by_repo(records)

        self.assertEqual(sorted(repos), ["org/a", "org/b"])
        self.assertEqual(repos["org/a"].paths, ["src/main.py", "tests/test_main.py"])
        self.assertEqual(repos["org/a"].languages, {"Python"})
        self.assertEqual(repos["org/b"].paths, ["src/Main.java"])
        self.assertEqual(repos["org/b"].languages, {"Java"})

    def test_group_records_by_repo_drops_incomplete_rows(self):
        records = [
            {"repo_name": "org/a", "path": "src/main.py"},
            {"repo_name": "", "path": "src/missing_repo.py"},
            {"repo_name": "org/a", "path": ""},
            {"path": "src/no_repo.py"},
        ]

        repos = group_records_by_repo(records)

        self.assertEqual(list(repos), ["org/a"])
        self.assertEqual(repos["org/a"].paths, ["src/main.py"])


if __name__ == "__main__":
    unittest.main()
