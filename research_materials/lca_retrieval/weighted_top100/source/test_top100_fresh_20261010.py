import unittest
import hashlib
import json
import tempfile
from unittest.mock import patch
from pathlib import Path
import numpy as np
from top100_fresh_evaluate_20261010 import rankings
from top100_fresh_encode_20261010 import encoder, load_sources
from two_stage_20261010 import order
from bm25_20261010 import hybrid


class FreshTests(unittest.TestCase):
    def test_encoder_membership_and_source_binding(self):
        source = "pass\n"
        rows = [{"cluster_id": str(p), "role": role, "source_relpath": f"{p}/{role}/{i}",
                 "source": source, "d0": hashlib.sha256(source.encode()).hexdigest()}
                for p in range(400) for role in ("query", "gallery") for i in range(8)]
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/"source.jsonl"
            path.write_text("\n".join(map(json.dumps, rows)))
            with patch.object(encoder, "SOURCE_SHA", encoder.sha(path)):
                self.assertEqual(len(load_sources(path)), 6400)
                path.write_text("modified")
                with self.assertRaises(AssertionError):
                    load_sources(path)

    def test_frozen_weights_and_candidate_tail(self):
        rng = np.random.default_rng(4)
        dense, bm = rng.random((2, 160))
        features = {n: rng.random(160) for n in ("nodes", "root", "idf")}
        tie = np.arange(160)[::-1]
        actual = rankings(dense, features, bm, tie)
        prefix = order(dense, tie)[:100]
        for name, alpha in (("nodes", .50), ("root", .25), ("idf", .20)):
            fused = (1-alpha)*dense + alpha*features[name]
            np.testing.assert_array_equal(actual[name+"_full"], order(fused, tie))
            np.testing.assert_array_equal(actual[name+"_100"][:100], prefix[order(fused[prefix], tie[prefix])])
            np.testing.assert_array_equal(actual[name+"_100"][100:], actual["D"][100:])
        np.testing.assert_array_equal(actual["D_BM25"], order(hybrid(dense, bm, .10), tie))

    def test_ties_and_existing_local_guard(self):
        actual = rankings(np.ones(120), {n: np.ones(120) for n in ("nodes", "root", "idf")}, np.ones(120), np.arange(120)[::-1])
        for ranking in actual.values():
            np.testing.assert_array_equal(ranking, np.arange(120)[::-1])
        with patch("pathlib.Path.is_dir", return_value=False):
            with self.assertRaisesRegex(AssertionError, "Kaggle only"):
                encoder.run(Path("absent"), Path("absent"), "full")


if __name__ == "__main__":
    unittest.main()
