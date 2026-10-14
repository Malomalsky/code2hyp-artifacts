import numpy as np
from scipy.sparse import csr_matrix

K1, B = 1.2, .75

def normalized(matrix):
    norms = np.sqrt(matrix.multiply(matrix).sum(axis=1)).A1
    return matrix.multiply(1 / np.where(norms > 0, norms, 1)[:, None]).tocsr()

def group_counts(counts, group):
    assert len(group) == counts.shape[1] and (group >= 0).all()
    membership = csr_matrix((np.ones(len(group)), (np.arange(len(group)), group)),
                            shape=(len(group), int(group.max()) + 1))
    return (counts @ membership).tocsr()

def transform(counts, gallery, beta, idf):
    assert beta in (1., .5) and np.isfinite(counts.data).all() and (counts.data >= 0).all()
    values = counts.copy().astype(np.float64)
    values.data **= beta
    if idf:
        df = (counts[gallery] > 0).sum(axis=0).A1
        assert (df > 0).all(), "Gallery-only vocabulary required"
        values = values.multiply(1 + np.log((len(gallery) + 1) / (df + 1)))
    return normalized(values)

def weighted_gallery(counts):
    counts = counts.tocsr().astype(np.float64)
    assert np.isfinite(counts.data).all() and (counts.data > 0).all()
    lengths = counts.sum(axis=1).A1
    if len(lengths) == 0 or lengths.mean() == 0:
        raise ValueError("BM25 requires a nonempty gallery with some tokens")
    df = (counts > 0).sum(axis=0).A1
    idf = np.log1p((len(lengths) - df + .5) / (df + .5))
    norm = K1 * (1 - B + B * lengths / lengths.mean())
    weighted = counts.copy()
    weighted.data = (idf[counts.indices] * (K1 + 1) * counts.data
                     / (counts.data + np.repeat(norm, np.diff(counts.indptr))))
    return weighted

def minmax(scores):
    scores = np.asarray(scores, dtype=np.float64)
    assert np.isfinite(scores).all()
    span = scores.max() - scores.min()
    return (scores - scores.min()) / span if span > 0 else np.zeros_like(scores)

def hybrid(dense, lexical, alpha):
    if alpha == 0:
        return dense
    if alpha == 1:
        return lexical
    return (1-alpha) * minmax(dense) + alpha * minmax(lexical)

def order(scores, tie):
    return np.lexsort((tie, -scores))

def rerank(dense_order, prefix_scores, tie):
    k = len(prefix_scores)
    prefix = dense_order[:k]
    return np.concatenate((prefix[order(prefix_scores, tie[prefix])], dense_order[k:]))

def query_metrics(ranking, relevant):
    hits = relevant[ranking]
    first = int(np.flatnonzero(hits)[0]) + 1
    ap = np.sum(np.cumsum(hits[:8]) * hits[:8] / np.arange(1, 9)) / 8
    return np.array([ap, 1 / first, float(first <= 10)]), first

def summarize(values, queries):
    problems, groups = np.unique([r["cluster_id"] for r in queries], return_inverse=True)
    assert len(problems) == 400 and (np.bincount(groups) == 8).all()
    methods = {}
    for name, scores in values.items():
        means = np.bincount(groups, weights=scores[:, 0]) / 8
        methods[name] = {**dict(zip(("MAP_at_8", "MRR", "Success_at_10"), scores.mean(axis=0).tolist())),
                         "problem_ap8": dict(zip(problems, means.tolist()))}
    return methods, problems

