"""Frozen-weight top-K reranking and actual warm-index CPU timing."""

import os
for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[name] = "1"

from root_tfidf_evaluate_kaggle_20261010 import load_inputs, HYBRIDS
from root_tfidf_external_20261010 import OUT as INPUT, ROOT, now
from rank_fusion_followup import sha, write
from pathlib import Path
import hashlib
import json
import platform
import sys
import time
import numpy as np
import scipy

OUT = ROOT / "followup_results/two_stage_20261010"
OLD = INPUT / "evaluation_kaggle"
ENCODING = ROOT.parents[2] / "code2hyp_informatics_artifacts/local_execution/kaggle_root_tfidf_20261010/output_v1/root_tfidf_encoding"
FAMILIES = ("candidate", "simpler_root")
KS = (50, 100, 500)
METHODS = [("D", None, None)] + [(f"{family}_{k}", family, k)
    for family in FAMILIES for k in ("full", *KS)]
OLD_SHA = "e0ce71481fc89bd8a2cfca0500473efa165cd9d1eda4eb2ff6909afd2d747121"


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


def prepare():
    assert sha(OLD / "result.json") == OLD_SHA
    paths = [Path(__file__), ROOT / "test_two_stage_20261010.py",
             ROOT / "two_stage_protocol_20261010.md", OLD / "result.json",
             ROOT / "root_tfidf_evaluate_kaggle_20261010.py",
             ROOT / "rank_fusion_followup.py", ROOT / "root_tfidf_external_20261010.py",
             INPUT / "verification_kaggle.json", INPUT / "features.json",
             INPUT / "selection_seal.json", INPUT / "metadata.jsonl",
             ROOT / "followup_results/two_stage_resources_20261010.json",
             ENCODING / "encoding_receipt.json", ENCODING / "embeddings.npy"]
    paths += [INPUT / (HYBRIDS[f][0] + "_features.npz") for f in FAMILIES]
    old = json.loads((OLD / "result.json").read_text())
    for name in ("D", *FAMILIES):
        path = OLD / (name + ".npy")
        assert sha(path) == old["methods"][name]["negative_score_sha256"]
        paths.append(path)
    bindings = {str(p): sha(p) for p in paths}
    OUT.mkdir(exist_ok=True)
    receipt = OUT / "design_receipt.json"
    if receipt.exists():
        assert json.loads(receipt.read_text())["bindings"] == bindings
    else:
        assert not list(OUT.iterdir())
        write(receipt, {"fixed_before_two_stage_run_utc": now(), "bindings": bindings,
            "status": "exploratory_previously_evaluated_cohort", "methods": METHODS,
            "primary_K": 100, "timing_queries": 512, "timing_repetitions": 5})
    return receipt


def data():
    rows, queries, gallery, vectors, features = load_inputs(ENCODING)
    qi = np.array([i for i, row in enumerate(rows) if row["role"] == "query"])
    gi = np.array([i for i, row in enumerate(rows) if row["role"] == "gallery"])
    qf = {f: features[HYBRIDS[f][0]][qi].tocsr() for f in FAMILIES}
    gf = {f: features[HYBRIDS[f][0]][gi].tocsr() for f in FAMILIES}
    ties = np.array([r["source_relpath"].encode("utf-8") for r in gallery])
    return queries, gallery, vectors[qi], vectors[gi], qf, gf, ties


def serving(i, family, k, qv, gv, qf, gf, transposed, tie):
    """Return top ten; component clocks include candidate gathering and sorting."""
    start = time.perf_counter_ns()
    dense = (qv[i] @ gv.T).astype(np.float64)
    t1 = time.perf_counter_ns()
    if family is None:
        top = order(dense, tie)[:10]
        t2 = time.perf_counter_ns()
        return top, np.array([t1-start, t2-t1, 0, 0, t2-start]) / 1e6, dense
    ids = None if k == "full" else order(dense, tie)[:k]
    t2 = time.perf_counter_ns()
    structure = (qf[family].getrow(i) @ (transposed[family] if ids is None else gf[family][ids].T)).toarray().ravel()
    t3 = time.perf_counter_ns()
    alpha = HYBRIDS[family][1]
    fused = (1-alpha) * (dense if ids is None else dense[ids]) + alpha * structure
    top = order(fused, tie if ids is None else tie[ids])[:10]
    if ids is not None:
        top = ids[top]
    t4 = time.perf_counter_ns()
    return top, np.array([t1-start, t2-t1, t3-t2, t4-t3, t4-start]) / 1e6, dense


def run():
    receipt = prepare()
    assert not (OUT / "quality.npz").exists(), "Existing run: use --check, never overwrite"
    queries, gallery, qv, gv, qf, gf, tie = data()
    transposed = {f: g.T for f, g in gf.items()}
    old = json.loads((OLD / "result.json").read_text())
    saved = {n: np.load(OLD / (n + ".npy"), mmap_mode="r", allow_pickle=False) for n in ("D", *FAMILIES)}
    labels = np.array([r["cluster_id"] for r in gallery])
    problem_ids, groups = np.unique([r["cluster_id"] for r in queries], return_inverse=True)
    assert len(problem_ids) == 650 and np.all(np.bincount(groups) == 8)
    metrics = np.empty((len(METHODS), len(queries), 3))
    top10 = np.empty((len(METHODS), len(queries), 10), dtype=np.int32)
    first = np.empty((len(METHODS), len(queries)), dtype=np.int32)
    recall = np.empty((len(KS), len(queries)))
    coverage = np.empty((len(FAMILIES), len(KS), len(queries)))
    for i, query in enumerate(queries):
        dense = -saved["D"][i]
        dorder = order(dense, tie)
        relevant = labels == query["cluster_id"]
        assert relevant.sum() == 8
        full = {}
        for f in FAMILIES:
            s = (qf[f].getrow(i) @ transposed[f]).toarray().ravel()
            a = HYBRIDS[f][1]
            full[f] = (1-a)*dense + a*s
            np.testing.assert_allclose(full[f], -saved[f][i], rtol=0, atol=1e-12)
        for j, (_, f, k) in enumerate(METHODS):
            ranking = dorder if f is None else (order(full[f], tie) if k == "full" else rerank(dorder, full[f][dorder[:k]], tie))
            metrics[j, i], first[j, i] = query_metrics(ranking, relevant)
            top10[j, i] = ranking[:10]
        for ki, k in enumerate(KS):
            recall[ki, i] = relevant[dorder[:k]].sum() / 8
            for fi, f in enumerate(FAMILIES):
                coverage[fi, ki, i] = np.isin(order(full[f], tie)[:8], dorder[:k]).mean()
        if (i+1) % 1000 == 0:
            print(f"Quality: {i+1}/{len(queries)}", flush=True)
    aggregate = metrics.mean(axis=1)
    for j, (_, f, k) in enumerate(METHODS):
        if f is None or k == "full":
            previous = old["methods"][f or "D"]["metrics"]
            np.testing.assert_allclose(aggregate[j], [previous[key] for key in
                ("problem_macro_map_at_r", "mrr", "recall_at_10")], atol=1e-12, rtol=0)
    np.savez_compressed(OUT / "quality.npz", metrics=metrics, top10=top10, first=first,
        recall=recall, coverage=coverage, groups=groups)

    selected = np.array(sorted(range(len(queries)), key=lambda i:
        hashlib.sha256(queries[i]["source_relpath"].encode()).hexdigest())[:512])
    # shortcut: warm exact search at 5200 items; benchmark ANN separately before scale claims.
    for i in selected[:16]:
        for _, f, k in METHODS:
            serving(i, f, k, qv, gv, qf, gf, transposed, tie)
    timings = np.empty((5, len(selected), len(METHODS), 5))
    differences = np.zeros(len(METHODS), dtype=int)
    max_dense_error = 0.
    generator = np.random.Generator(np.random.PCG64(20261010))
    for rep in range(5):
        for si, i in enumerate(selected):
            for j in generator.permutation(len(METHODS)):
                _, f, k = METHODS[j]
                result, timings[rep, si, j], dense = serving(i, f, k, qv, gv, qf, gf, transposed, tie)
                if rep == 0:
                    differences[j] += not np.array_equal(result, top10[j, i])
                    max_dense_error = max(max_dense_error, float(np.max(np.abs(dense + saved["D"][i]))))
        print(f"Runtime repetition {rep+1}/5 complete", flush=True)
    np.savez_compressed(OUT / "timing.npz", milliseconds=timings, query_indices=selected)
    methods = {}
    for j, (name, f, k) in enumerate(METHODS):
        means = timings[:, :, j, :].mean(axis=1)
        full_index = next((x for x, (_, family, cutoff) in enumerate(METHODS) if family == f and cutoff == "full"), 0)
        denominator = aggregate[full_index, 0] - aggregate[0, 0]
        methods[name] = {"MAP_at_8": float(aggregate[j, 0]), "MRR": float(aggregate[j, 1]),
            "Success_at_10": float(aggregate[j, 2]), "delta_MAP_vs_dense": float(aggregate[j, 0]-aggregate[0, 0]),
            "delta_MAP_vs_full": float(aggregate[j, 0]-aggregate[full_index, 0]),
            "fraction_full_MAP_gain_retained": float((aggregate[j, 0]-aggregate[0, 0])/denominator) if denominator > 0 else None,
            "problem_MAP": {p: float(metrics[j, groups == ix, 0].mean()) for ix, p in enumerate(problem_ids)},
            "timing_ms": {"per_pass_mean_total": means[:, 4].tolist(), "median_pass_mean_total": float(np.median(means[:, 4])),
                "individual_median_total": float(np.median(timings[:, :, j, 4])),
                "individual_p95_total": float(np.quantile(timings[:, :, j, 4], .95)),
                "mean_components_dense_select_structure_sort_total": means.mean(axis=0).tolist()},
            "runtime_top10_different_queries_of_512": int(differences[j])}
    write(OUT / "result.json", {"status": "completed_exploratory_two_stage", "completed_utc": now(),
        "design_sha256": sha(receipt), "methods": methods,
        "candidate_sets": {str(k): {"recall_at_K": float(recall[ki].mean()),
            "any_relevant_fraction": float((recall[ki] > 0).mean()),
            "full_hybrid_top8_coverage": {f: float(coverage[fi, ki].mean()) for fi, f in enumerate(FAMILIES)}} for ki, k in enumerate(KS)},
        "timing": {"queries": 512, "repetitions": 5, "threads": 1, "max_dense_absolute_error_vs_saved": max_dense_error,
            "scope": "warm precomputed representations; actual CPU dot products; excludes encoding and feature extraction",
            "platform": platform.platform(), "python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__},
        "files_sha256": {n: sha(OUT / n) for n in ("quality.npz", "timing.npz")},
        "training_performed": False, "java_test_accessed": False, "no_new_parameter_selection": True})
    print(json.dumps({n: {k: v for k, v in m.items() if k not in ("problem_MAP", "timing_ms")} for n, m in methods.items()}, indent=2), flush=True)


def check():
    receipt = prepare()
    result = json.loads((OUT / "result.json").read_text())
    assert result["design_sha256"] == sha(receipt)
    for name, expected in result["files_sha256"].items():
        assert sha(OUT / name) == expected
    from rank_fusion_followup import metadata
    _, queries, gallery = metadata(INPUT)
    labels = np.array([r["cluster_id"] for r in gallery])
    tie_order = np.argsort([r["source_relpath"].encode() for r in gallery])
    saved = {n: np.load(OLD / (n + ".npy"), mmap_mode="r", allow_pickle=False) for n in ("D", *FAMILIES)}
    quality = np.load(OUT / "quality.npz", allow_pickle=False)
    for i, query in enumerate(queries):
        dorder = tie_order[np.argsort(saved["D"][i, tie_order], kind="stable")]
        for j, (_, f, k) in enumerate(METHODS):
            if f is None:
                ranked = dorder
            elif k == "full":
                ranked = tie_order[np.argsort(saved[f][i, tie_order], kind="stable")]
            else:
                prefix = np.sort(dorder[:k])
                source_ties = {idx: gallery[idx]["source_relpath"].encode() for idx in prefix}
                prefix = sorted(prefix, key=lambda idx: (saved[f][i, idx], source_ties[idx]))
                ranked = np.r_[prefix, dorder[k:]]
            np.testing.assert_array_equal(ranked[:10], quality["top10"][j, i])
            positions = np.flatnonzero(labels[ranked] == query["cluster_id"]) + 1
            assert len(positions) == 8 and positions[0] == quality["first"][j, i]
            ap = sum((hit+1)/rank for hit, rank in enumerate(positions) if rank <= 8)/8
            np.testing.assert_allclose([ap, 1/positions[0], float(positions[0] <= 10)], quality["metrics"][j, i], atol=1e-12, rtol=0)
        for ki, k in enumerate(KS):
            assert quality["recall"][ki, i] == np.count_nonzero(labels[dorder[:k]] == query["cluster_id"])/8
            for fi, f in enumerate(FAMILIES):
                full_top8 = tie_order[np.argsort(saved[f][i, tie_order], kind="stable")][:8]
                assert quality["coverage"][fi, ki, i] == len(set(full_top8) & set(dorder[:k]))/8
    timing = np.load(OUT / "timing.npz", allow_pickle=False)
    values = timing["milliseconds"]
    assert values.shape == (5, 512, 9, 5) and np.isfinite(values).all() and np.all(values >= 0)
    np.testing.assert_allclose(values[..., :4].sum(axis=-1), values[..., 4], atol=1e-10, rtol=0)
    for j, (name, _, _) in enumerate(METHODS):
        m = result["methods"][name]
        np.testing.assert_allclose(quality["metrics"][j].mean(axis=0), [m[x] for x in ("MAP_at_8", "MRR", "Success_at_10")], atol=1e-12, rtol=0)
        np.testing.assert_allclose(values[:, :, j, 4].mean(axis=1), m["timing_ms"]["per_pass_mean_total"], rtol=0, atol=1e-12)
    write(OUT / "verification.json", {"status": "passed", "checked_utc": now(), "result_sha256": sha(OUT / "result.json"),
        "check": "Every query metric and top10 reconstructed from frozen full matrices with a separate sort/AP calculation; timing components checked"})
    print("PASS: all 46800 query/method rankings and metrics independently reconstructed", flush=True)


if __name__ == "__main__":
    if sys.argv[1:] == ["--prepare"]:
        print("Frozen: " + sha(prepare()), flush=True)
    elif sys.argv[1:] == ["--run"]:
        run()
    elif sys.argv[1:] == ["--check"]:
        check()
    else:
        raise ValueError("Use --prepare, --run or --check")
