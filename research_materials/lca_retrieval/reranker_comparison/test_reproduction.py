import unittest
import numpy as np
from verify_results import effect
from kernels import order, rerank, query_metrics


class ReproductionTests(unittest.TestCase):
    def test_ties_and_unchanged_tail(self):
        paths = np.array([b"b", b"a", b"c", b"d"])
        ranking = order(np.array([1., 1., 0., -1.]), paths)
        np.testing.assert_array_equal(ranking, [1, 0, 2, 3])
        np.testing.assert_array_equal(rerank(ranking, np.array([0., 2.]), paths), [0, 1, 2, 3])

    def test_known_metrics_and_ties(self):
        values = query_metrics(np.arange(8), np.ones(8, dtype=bool))[0]
        np.testing.assert_array_equal(values, [1., 1., 1.])
        row = effect(np.array([.2, .3]), np.array([.1, .3]), np.array([[0, 1], [1, 0]]))
        self.assertEqual(row["problem_counts"], {"better": 1, "worse": 0, "tied": 1})
        self.assertAlmostEqual(row["mean"], .05)


if __name__ == "__main__":
    unittest.main()
