"""Execution-port checks; no neural inference."""
import ast
from pathlib import Path
import unittest
import coderank_colab as old_encoder
import coderank_kaggle_20261010 as new_encoder
from test_root_tfidf_evaluate_20261010 import FusionTests
import root_tfidf_evaluate_20261010 as old
import root_tfidf_evaluate_kaggle_20261010 as new
import inspect
import numpy as np


class KagglePortTests(unittest.TestCase):
    def test_encoder_settings_and_local_guard(self):
        for key, value in old_encoder.SETTINGS.items():
            if key != "source_sha256":
                self.assertEqual(new_encoder.SETTINGS[key], value)
        self.assertEqual(new_encoder.SETTINGS["execution_platform"], "Kaggle")
        with self.assertRaisesRegex(AssertionError, "Kaggle only"):
            new_encoder.run(Path("absent"), Path("absent"), "full")

    def test_analysis_code_unchanged(self):
        for name in ("mixed", "run"):
            before = inspect.getsource(getattr(old, name))
            after = inspect.getsource(getattr(new, name))
            after = after.replace('"evaluation_kaggle"', '"evaluation"')
            after = after.replace('"kaggle_evaluation_code_receipt.json"', '"evaluation_code_receipt.json"')
            after = after.replace('"test_root_tfidf_kaggle_20261010.py"', '"test_root_tfidf_evaluate_20261010.py"')
            self.assertEqual(ast.dump(ast.parse(before)), ast.dump(ast.parse(after)))
        d = np.array([.1, .1, -.4])
        aux = {n: np.array([.9, .2, 0.]) for n in {c for c, _ in old.HYBRIDS.values()} | {"lexical"}}
        a, b = old.mixed(d, aux, np.array([b"z", b"a", b"m"])), new.mixed(d, aux, np.array([b"z", b"a", b"m"]))
        for key in a:
            np.testing.assert_array_equal(a[key], b[key])


if __name__ == "__main__":
    unittest.main()
