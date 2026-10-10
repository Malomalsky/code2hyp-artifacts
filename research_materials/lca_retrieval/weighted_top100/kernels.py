import numpy as np
from scipy.sparse import csr_matrix

K1, B = 1.2, .75

def normalized(matrix):
    norms = np.sqrt(matrix.multiply(matrix).sum(axis=1)).A1
    return matrix.multiply(1 / np.where(norms > 0, norms, 1)[:, None]).tocsr()

def ranks(values, tie):
    assert np.isfinite(values).all()
    order = np.lexsort((tie, values))
    result = np.empty(len(order), dtype=np.int64)
    result[order] = np.arange(1, len(order) + 1)
    return result

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

def choose_folds(values, folds):
    choices, held = [], np.empty(len(values))
    for fold in range(int(folds.max()) + 1):
        test = folds == fold
        assert test.any() and (~test).any()
        choice = int(np.argmax(values[~test].mean(axis=0)))
        choices.append(choice)
        held[test] = values[test, choice]
    return choices, held

GRID = [0., .005, .01, .02, .05, .10, .15, .20, .25, .30, .35, .40,
        .45, .50, .55, .60, .65, .70, .75, .80, .85, .90, .95, 1.]

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

HYBRIDS = {"candidate": ("ret_root_idf", .20), "previous_tfidf": ("ret_raw_idf", .15),
           "matched_mask": ("mask_root_idf", .20), "simpler_root": ("ret_root", .25),
           "node_counts": ("nodes", .50), "fixed_weight_control": ("ret_root_idf", .15)}

def mixed(dense, auxiliary, tie):
    values = {"D": dense, **{n: (1 - a) * dense + a * auxiliary[c] for n, (c, a) in HYBRIDS.items()}}
    values["lexical_dense_rrf"] = 1 / (60 + ranks(-dense, tie)) + 1 / (60 + ranks(-auxiliary["lexical"], tie))
    values.update({"standalone_" + n: v for n, v in auxiliary.items()})
    return values

def contrast(values, indices):
    return {"mean": float(values.mean()), "percentile_95_interval": np.quantile(
        values[indices].mean(axis=1), [.025, .975]).tolist()}

