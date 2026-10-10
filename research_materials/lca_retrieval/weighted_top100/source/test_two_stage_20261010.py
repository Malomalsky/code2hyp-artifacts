"""Small exact fixtures for the operational reranking study."""

import unittest
import numpy as np
from scipy.sparse import csr_matrix
from two_stage_20261010 import order, rerank, query_metrics, serving


class TwoStageTests(unittest.TestCase):
    def test_ties_and_tail(self):
        tie = np.array([b"z", b"a", b"m", b"b"])
        dense = np.array([1., 1., .5, .4])
        ranking = order(dense, tie)
        np.testing.assert_array_equal(ranking, [1, 0, 2, 3])
        actual = rerank(ranking, np.array([.2, .3]), tie)
        np.testing.assert_array_equal(actual, [0, 1, 2, 3])
        np.testing.assert_array_equal(rerank(ranking, np.ones(2), tie), ranking)

    def test_full_and_zero_weight_equivalence(self):
        generator = np.random.default_rng(7)
        dense, structural = generator.random((2, 20))
        tie = np.arange(20)[::-1]
        ranking = order(dense, tie)
        fused = .8*dense + .2*structural
        np.testing.assert_array_equal(rerank(ranking, fused[ranking], tie), order(fused, tie))
        np.testing.assert_array_equal(rerank(ranking, dense[ranking[:10]], tie), ranking)

    def test_relevant_outside_prefix_and_metric_denominator(self):
        ranked = np.arange(16)
        relevant = np.array([False]*8 + [True]*8)
        metrics, first = query_metrics(ranked, relevant)
        np.testing.assert_allclose(metrics, [0., 1/9, 1.])
        self.assertEqual(first, 9)
        relevant = np.array([True, False, True] + [False]*6 + [True]*6 + [False])
        metrics, _ = query_metrics(ranked, relevant)
        np.testing.assert_allclose(metrics, [(1+2/3)/8, 1., 1.])

    def test_actual_serving_scorer(self):
        generator = np.random.default_rng(8)
        qv, gv = generator.random((1, 4), dtype=np.float32), generator.random((20, 4), dtype=np.float32)
        qf = {"candidate": csr_matrix(generator.random((1, 6)))}
        gf = {"candidate": csr_matrix(generator.random((20, 6)))}
        tie = np.arange(20)[::-1]
        dense = (qv[0] @ gv.T).astype(np.float64)
        fused = .8*dense + .2*(qf["candidate"] @ gf["candidate"].T).toarray().ravel()
        for k in ("full", 10):
            actual, timing, _ = serving(0, "candidate", k, qv, gv, qf, gf, {"candidate": gf["candidate"].T}, tie)
            expected = order(fused, tie) if k == "full" else rerank(order(dense, tie), fused[order(dense, tie)[:k]], tie)
            np.testing.assert_array_equal(actual, expected[:10])
            np.testing.assert_allclose(timing[:4].sum(), timing[4], atol=1e-10)


if __name__ == "__main__":
    unittest.main()
