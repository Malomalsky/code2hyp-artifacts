import unittest
import numpy as np
from scipy.sparse import csr_matrix
from bm25_20261010 import weighted_gallery, minmax, hybrid, K1, B


class BM25Tests(unittest.TestCase):
    def test_direct_formula_and_repeated_query_terms(self):
        counts = np.array([[2., 1., 0.], [1., 0., 4.], [0., 0., 0.]])
        query = np.array([3., 1., 2.])
        actual = query @ weighted_gallery(csr_matrix(counts)).toarray().T
        expected = []
        for document in counts:
            value = 0.
            for term, qtf in enumerate(query):
                df = sum(counts[:, term] > 0)
                tf = document[term]
                value += qtf * np.log(1+(3-df+.5)/(df+.5)) * (K1+1)*tf / (
                    tf+K1*(1-B+B*document.sum()/counts.sum(axis=1).mean()))
            expected.append(value)
        np.testing.assert_allclose(actual, expected, rtol=1e-14, atol=1e-14)
        self.assertEqual(actual[-1], 0.)

    def test_scaling_constant_and_endpoints(self):
        d, b = np.array([-.5, .5, 0.]), np.array([0., 20., 10.])
        np.testing.assert_array_equal(minmax(np.ones(3)), np.zeros(3))
        np.testing.assert_allclose(minmax(d), [0, 1, .5])
        np.testing.assert_allclose(hybrid(d, b, .2), [0, 1, .5])
        np.testing.assert_array_equal(hybrid(d, b, 0), d)
        np.testing.assert_array_equal(hybrid(d, b, 1), b)
        np.testing.assert_allclose(hybrid(d, b*100, .2), hybrid(d, b, .2))

    def test_empty_gallery_rejected(self):
        with self.assertRaises(ValueError):
            weighted_gallery(csr_matrix((3, 0)))


if __name__ == "__main__":
    unittest.main()
