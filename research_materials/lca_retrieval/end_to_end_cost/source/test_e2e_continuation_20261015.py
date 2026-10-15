"""Candidate identity survives floating-point permutations without score reassignment."""

import unittest
import numpy as np
from continue_e2e_20261015 import cached_ranking


class ReplayTests(unittest.TestCase):
    def test_scores_follow_saved_ids_not_recomputed_order(self):
        tie = np.array([b"a", b"b", b"c", b"d"])
        candidates, pair = np.array([1, 0]), np.array([3., 1.])
        for dense in (np.array([1., 1., .5, .1]), np.array([1., 1.00001, .5, .1])):
            np.testing.assert_array_equal(cached_ranking(dense, candidates, pair, tie), [1, 0, 2, 3])
            np.testing.assert_array_equal(cached_ranking(dense, candidates[::-1], pair[::-1], tie), [1, 0, 2, 3])

    def test_fixed_membership_and_tie_breaking(self):
        ranked = cached_ranking(np.array([1., 2., 3., 4.]), np.array([0, 2]),
                                np.array([1., 1.]), np.array([b"z", b"b", b"a", b"d"]))
        np.testing.assert_array_equal(ranked, [2, 0, 3, 1])

    def test_invalid_bindings_fail(self):
        dense, tie = np.array([1., 2., 3.]), np.array([b"a", b"b", b"c"])
        for candidates, scores in (([0, 0], [1., 2.]), ([-1, 0], [1., 2.]),
                                   ([0, 3], [1., 2.]), ([0, 1], [1.]),
                                   ([0, 1], [np.nan, 2.])):
            with self.assertRaises(ValueError):
                cached_ranking(dense, np.array(candidates), np.array(scores), tie)


if __name__ == "__main__":
    unittest.main()
