"""Small tests for tie handling, top-budget preservation and metric definitions."""
import unittest
import numpy as np
from verify_results import limited, metrics, sorted_scores


class ReproductionTests(unittest.TestCase):
    def test_ties_and_tail(self):
        ids = np.array([f"g{i:03d}".encode() for i in range(120)][::-1])
        ties = np.argsort(ids)
        dense = np.ones(120)
        root = np.arange(120, dtype=float)
        base = sorted_scores(dense, ties)
        np.testing.assert_array_equal(limited(dense, root, 0., ties), base)
        ranked = limited(dense, root, .25, ties)
        np.testing.assert_array_equal(ranked[100:], base[100:])
        self.assertEqual(set(ranked[:100]), set(base[:100]))

    def test_metrics(self):
        common, native = metrics(np.arange(120), [0, 2])
        self.assertAlmostEqual(common[0], (1+.5)/(1+1/np.log2(3)))
        np.testing.assert_array_equal(common[1:], [1., 1.])
        self.assertAlmostEqual(native[0], (1+2/3)/2)
        common, _ = metrics(np.arange(120), [11, 119])
        np.testing.assert_array_equal(common, [0., 0., .5])
        with self.assertRaises(AssertionError):
            metrics(np.array([0, 0]), [0])


if __name__ == "__main__":
    unittest.main()
