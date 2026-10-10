"""Small independent checks of tie handling, tail preservation and AP@8."""

import tempfile
from pathlib import Path
import numpy as np
from scipy.sparse import csr_matrix
from kernels import weighted_gallery, hybrid, transform, group_counts


def self_check():
    from verify_results import ranked, limited, metrics, sha, arrays
    ties = np.array([1, 0, *range(2, 16)])
    assert ranked(np.zeros(16), ties).tolist() == ties.tolist()
    dense = np.arange(16)
    score = np.zeros(16)
    score[2] = 1
    result = limited(dense, score, ties, 4)
    assert result.tolist() == [2, 1, 0, 3, *range(4, 16)]
    labels = np.r_[np.zeros(8), np.ones(8)]
    assert metrics(np.arange(16), labels, 0) == ([1., 1., 1.], 1)
    assert metrics(np.arange(16), labels, 1) == ([0., 1 / 9, 1.], 9)
    np.testing.assert_array_equal(hybrid(np.ones(2), np.ones(2), .1), np.zeros(2))
    counts = csr_matrix([[1., 4.], [0., 0.]])
    root = transform(counts, [0], .5, False).toarray()
    np.testing.assert_allclose(root, [[1 / np.sqrt(5), 2 / np.sqrt(5)], [0, 0]])
    np.testing.assert_array_equal(group_counts(counts, np.array([0, 0])).toarray(), [[5], [0]])
    assert np.isfinite(weighted_gallery(counts).data).all()
    try:
        weighted_gallery(csr_matrix((2, 2)))
    except ValueError:
        pass
    else:
        raise AssertionError("Empty lexical gallery was accepted")
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "fixture"
        path.write_bytes(b"original")
        before = sha(path)
        path.write_bytes(b"modified")
        assert sha(path) != before
        packed = Path(directory) / "fixture.npz"
        np.savez_compressed(packed, values=np.arange(4))
        unpacked = arrays(packed)
        packed.unlink()
        np.testing.assert_array_equal(unpacked["values"], np.arange(4))
    print("PASS: sorting, AP@8, candidate tail, transforms and hash self-checks", flush=True)


if __name__ == "__main__":
    self_check()
