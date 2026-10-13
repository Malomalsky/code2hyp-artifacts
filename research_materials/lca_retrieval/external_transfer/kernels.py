import numpy as np

K1, B = 1.2, .75

def normalized(matrix):
    norms = np.sqrt(matrix.multiply(matrix).sum(axis=1)).A1
    return matrix.multiply(1 / np.where(norms > 0, norms, 1)[:, None]).tocsr()

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

METHODS = ("D", "BM25", "D_BM25", "nodes_100", "root_full", "root_100")

def rankings(d, root, nodes, bm, tie):
    dense = order(d, tie)
    prefix = dense[:100]
    fused = .75 * d + .25 * root
    return {"D": dense, "BM25": order(bm, tie), "D_BM25": order(hybrid(d, bm, .10), tie),
        "nodes_100": rerank(dense, .50 * d[prefix] + .50 * nodes[prefix], tie),
        "root_full": order(fused, tie), "root_100": rerank(dense, fused[prefix], tie)}

def interval(value, sampled):
    return {"mean": float(value), "percentile_95_interval": np.quantile(sampled, [.025, .975]).tolist()}

def crossed_contrasts(problem_scores):
    rng = np.random.Generator(np.random.PCG64(20261012))
    indices = {c: rng.integers(problem_scores[c+"_CodeRank"].shape[1],
        size=(20000, problem_scores[c+"_CodeRank"].shape[1])) for c in ("CodeContests", "CPRet")}
    result = {}
    for reference in ("D", "D_BM25"):
        delta, sampled = {}, {}
        for key, values in problem_scores.items():
            corpus = key.split("_")[0]
            delta[key] = values[METHODS.index("root_100"), :, 0] - values[METHODS.index(reference), :, 0]
            sampled[key] = delta[key][indices[corpus]].mean(axis=1)
        effects = {k: interval(v.mean(), sampled[k]) for k, v in delta.items()}
        pairs = {"model_difference_in_"+c: (c+"_Qwen", c+"_CodeRank") for c in ("CodeContests", "CPRet")}
        pairs.update({"corpus_difference_for_"+m: ("CPRet_"+m, "CodeContests_"+m) for m in ("CodeRank", "Qwen")})
        for name, (a, b) in pairs.items():
            effects[name] = interval(delta[a].mean()-delta[b].mean(), sampled[a]-sampled[b])
        signs = {"CPRet_Qwen": 1, "CPRet_CodeRank": -1, "CodeContests_Qwen": -1, "CodeContests_CodeRank": 1}
        effects["interaction"] = interval(sum(s*delta[k].mean() for k, s in signs.items()),
                                           sum(s*sampled[k] for k, s in signs.items()))
        result["root100_minus_"+reference] = effects
    return result

