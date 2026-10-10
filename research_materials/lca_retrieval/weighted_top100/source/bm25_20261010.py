"""One fixed BM25 comparator; reuse sources and dense scores, without training."""

from rank_fusion_followup import ROOT, metadata, sha, write
from followup_baselines import lexical_token_stream, vectorize
from conditional_structure_pilot_20261010 import GRID, ap8, problem_means
from two_stage_20261010 import query_metrics
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import platform
import resource
import sys
import time
import numpy as np
from scipy.sparse import load_npz, save_npz

OUT = ROOT / "followup_results/bm25_20261010"
OLD = ROOT / "followup_results/fresh_fusion"
OLD_SCORES = ROOT / "followup_results/coderank_colab_20261007/evaluation"
NEW = ROOT / "followup_results/root_tfidf_external_20261010"
NEW_SCORES = NEW / "evaluation_kaggle"
SPLIT = ROOT / "followup_results/conditional_structure_20261010/design_receipt.json"
REFERENCES = ("candidate", "simpler_root", "node_counts")
K1, B = 1.2, .75


def now():
    return datetime.now(timezone.utc).isoformat()


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


def prepare():
    anchors = {
        OLD_SCORES / "result.json": "d25fe73921b328604e9bf317af2a7e9c02c0b67b85ed22b9f7e7679da7f73524",
        NEW_SCORES / "result.json": "e0ce71481fc89bd8a2cfca0500473efa165cd9d1eda4eb2ff6909afd2d747121"}
    for path, digest in anchors.items():
        assert sha(path) == digest, path
    expected = dict(anchors)
    for folder in (OLD, NEW):
        seal = folder / "selection_seal.json"
        expected[seal] = sha(seal)
        for name in ("selected_programs.jsonl", "metadata.jsonl"):
            expected[folder / name] = json.loads(seal.read_text())["sha256"][name]
    for folder, names, key in ((OLD_SCORES, ("D",), "score_sha256"),
                              (NEW_SCORES, ("D", *REFERENCES), "negative_score_sha256")):
        result = json.loads((folder / "result.json").read_text())
        for name in names:
            expected[folder / (name + ".npy")] = result["methods"][name][key]
    old_design = json.loads(SPLIT.read_text())
    helpers = [ROOT / n for n in ("rank_fusion_followup.py", "followup_baselines.py",
                                  "reviewer_reproduction/verify_results.py")]
    helpers.append(ROOT.parents[2] / "code2hyp_informatics_artifacts/geometry_profile_research/codenet_eligibility.py")
    for path in helpers:
        expected[path] = old_design["bindings"][str(path)]
    for path, digest in expected.items():
        assert sha(path) == digest, path
    paths = [Path(__file__), ROOT / "test_bm25_20261010.py", ROOT / "bm25_protocol_20261010.md",
             SPLIT, ROOT / "conditional_structure_pilot_20261010.py", ROOT / "two_stage_20261010.py",
             ROOT / "followup_results/bm25_resources_20261010.json"]
    expected.update({p: sha(p) for p in paths})
    design = {"bindings": {str(p): h for p, h in expected.items()}, "k1": K1, "b": B,
              "weight_grid": GRID, "query_term_weight": "raw_count", "normalization": "per_query_full_gallery_minmax",
              "calibration_problems": old_design["calibration_problems"],
              "validation_problems": old_design["validation_problems"], "threads": 1,
              "status": "exploratory_both_cohorts_previously_analysed"}
    OUT.mkdir(exist_ok=True)
    receipt = OUT / "design_receipt.json"
    if receipt.exists():
        previous = json.loads(receipt.read_text())
        assert all(previous[k] == v for k, v in design.items())
    else:
        assert not list(OUT.iterdir()), "Inspect partial outputs before continuing"
        write(receipt, {**design, "fixed_before_bm25_scores_utc": now()})
    return design


def build(folder, name):
    target = OUT / name
    target.mkdir()
    started = time.perf_counter()
    rows, queries, gallery = metadata(folder)
    programs = [json.loads(line) for line in (folder / "selected_programs.jsonl").read_text().splitlines()]
    assert [{k: r[k] for k in ("source_relpath", "role", "cluster_id")} for r in programs] == [
        {k: r[k] for k in ("source_relpath", "role", "cluster_id")} for r in rows]
    assert all(hashlib.sha256(r["source"].encode()).hexdigest() == r["d0"] for r in programs)
    bags = [Counter(lexical_token_stream(r["source"])) for r in programs]
    qi = [i for i, r in enumerate(rows) if r["role"] == "query"]
    gi = [i for i, r in enumerate(rows) if r["role"] == "gallery"]
    assert len(qi) == len(gi) == 5200
    keys = sorted(set().union(*(set(bags[i]) for i in gi)))
    vocab = {k: i for i, k in enumerate(keys)}
    q = vectorize([bags[i] for i in qi], vocab)
    g = vectorize([bags[i] for i in gi], vocab)
    q.sort_indices()
    g.sort_indices()
    save_npz(target / "query_counts.npz", q)
    save_npz(target / "gallery_counts.npz", g)
    write(target / "vocabulary.json", keys)
    names = ("query_counts.npz", "gallery_counts.npz", "vocabulary.json")
    write(target / "features.json", {"sha256": {n: sha(target/n) for n in names},
        "vocabulary_size": len(keys), "gallery_mean_length": float(g.sum(axis=1).mean()),
        "query_oov_occurrences": sum(v for i in qi for k, v in bags[i].items() if k not in vocab),
        "query_total_occurrences": sum(sum(bags[i].values()) for i in qi),
        "seconds": time.perf_counter()-started})
    print(f"{name}: tokenized {len(rows)} programs, vocabulary {len(keys)}", flush=True)
    return queries, gallery, q, weighted_gallery(g).T.tocsr()


def calibrate(data, design):
    queries, gallery, q, gt = data
    cal = [i for i, r in enumerate(queries) if r["cluster_id"] in set(design["calibration_problems"])]
    assert len(cal) == 2600
    dense = np.load(OLD_SCORES / "D.npy", mmap_mode="r", allow_pickle=False)
    tie = np.array([r["source_relpath"].encode() for r in gallery])
    labels = np.array([r["cluster_id"] for r in gallery])
    values = np.empty((len(GRID), len(cal)))
    for ci, i in enumerate(cal):
        bm = (q.getrow(i) @ gt).toarray().ravel()
        d = -dense[i]
        relevant = np.flatnonzero(labels == queries[i]["cluster_id"])
        for j, alpha in enumerate(GRID):
            values[j, ci] = ap8(hybrid(d, bm, alpha), tie, relevant)
        if (ci+1) % 650 == 0:
            print(f"Calibration: {ci+1}/{len(cal)}", flush=True)
    means = [float(np.mean(list(problem_means(v, queries, cal).values()))) for v in values]
    chosen = GRID[int(np.argmax(means))]
    np.savez_compressed(OUT / "calibration_ap8.npz", values=values, query_indices=cal)
    write(OUT / "calibration.json", {"fixed_before_evaluation_utc": now(), "alpha": chosen,
        "weight_grid": GRID, "MAP_at_8_by_weight": means, "queries": len(cal),
        "design_sha256": sha(OUT / "design_receipt.json"),
        "query_values_sha256": sha(OUT / "calibration_ap8.npz")})
    print(f"Frozen BM25 weight: {chosen}; development MAP@8={max(means):.9f}", flush=True)
    return chosen


def evaluate(name, data, alpha, indices):
    queries, gallery, q, gt = data
    source = OLD_SCORES if name == "old" else NEW_SCORES
    saved = {n: np.load(source / (n + ".npy"), mmap_mode="r", allow_pickle=False)
             for n in (("D",) if name == "old" else ("D", *REFERENCES))}
    methods = ["D", "BM25", "D_BM25", *(REFERENCES if name == "later" else ())]
    tie = np.array([r["source_relpath"].encode() for r in gallery])
    labels = np.array([r["cluster_id"] for r in gallery])
    quality = np.empty((len(methods), len(indices), 3))
    top10 = np.empty((len(methods), len(indices), 10), dtype=np.int32)
    for ii, i in enumerate(indices):
        bm = (q.getrow(i) @ gt).toarray().ravel()
        scores = {n: -m[i] for n, m in saved.items()}
        scores.update(BM25=bm, D_BM25=hybrid(scores["D"], bm, alpha))
        for mi, method in enumerate(methods):
            ranking = np.lexsort((tie, -scores[method]))
            quality[mi, ii], _ = query_metrics(ranking, labels == queries[i]["cluster_id"])
            top10[mi, ii] = ranking[:10]
    np.savez_compressed(OUT / name / "quality.npz", metrics=quality, top10=top10, query_indices=indices)
    entries = {}
    for mi, method in enumerate(methods):
        entries[method] = dict(zip(("MAP_at_8", "MRR", "Success_at_10"), quality[mi].mean(axis=0).tolist()))
        entries[method]["problem_ap8"] = problem_means(quality[mi, :, 0], queries, indices)
        if method in saved:
            previous = json.loads((source / "result.json").read_text())["methods"][method]["metrics"]
            np.testing.assert_allclose(quality[mi, :, 0], [previous["query_scores"][queries[i]["source_relpath"]]
                for i in indices], atol=1e-12, rtol=0)
        print(name, method, {k: v for k, v in entries[method].items() if k != "problem_ap8"}, flush=True)
    differences = {}
    for method in methods[1:]:
        for ref in ("D", "D_BM25"):
            if method == ref:
                continue
            a, b = entries[method]["problem_ap8"], entries[ref]["problem_ap8"]
            delta = np.array([a[p]-b[p] for p in sorted(a)])
            differences[method + "_minus_" + ref] = {"mean_MAP_difference": float(delta.mean()),
                "better_problems": int((delta > 1e-12).sum()), "worse_problems": int((delta < -1e-12).sum()),
                "tied_problems": int((np.abs(delta) <= 1e-12).sum())}
    return {"method_order": methods, "methods": entries, "differences": differences,
            "quality_sha256": sha(OUT / name / "quality.npz"),
            "features_sha256": sha(OUT / name / "features.json"), "queries": len(indices)}


def run():
    design = prepare()
    assert not (OUT / "old").exists(), "Existing run: use --check; never overwrite"
    started = time.perf_counter()
    old = build(OLD, "old")
    alpha = calibrate(old, design)
    val = [i for i, r in enumerate(old[0]) if r["cluster_id"] in set(design["validation_problems"])]
    assert len(val) == 2600
    outputs = {"old_validation": evaluate("old", old, alpha, val)}
    later = build(NEW, "later")
    assert not {r["cluster_id"] for r in old[0]} & {r["cluster_id"] for r in later[0]}
    outputs["later_reused"] = evaluate("later", later, alpha, list(range(5200)))
    write(OUT / "result.json", {"completed_utc": now(), "status": "completed_exploratory_BM25_comparison",
        "design_sha256": sha(OUT / "design_receipt.json"), "calibration_sha256": sha(OUT / "calibration.json"),
        "alpha": alpha, "cohorts": outputs, "seconds_total": time.perf_counter()-started,
        "peak_rss_bytes_macos": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "platform": platform.platform(), "numpy": np.__version__, "python": platform.python_version(),
        "neural_training": False, "neural_inference": False, "java_test_accessed": False,
        "limitations": "Reused cohorts; one tokenizer and BM25 setting; no confirmatory inference or serving-latency claim."})


def verify():
    prepare()
    result = json.loads((OUT / "result.json").read_text())
    calibration = json.loads((OUT / "calibration.json").read_text())
    assert sha(OUT / "design_receipt.json") == result["design_sha256"] == calibration["design_sha256"]
    assert sha(OUT / "calibration.json") == result["calibration_sha256"]
    assert sha(OUT / "calibration_ap8.npz") == calibration["query_values_sha256"]
    alpha = calibration["alpha"]
    assert alpha == result["alpha"] == GRID[int(np.argmax(calibration["MAP_at_8_by_weight"]))]
    checked, scalar_pairs = 0, 0
    for name, folder, source, key in (("old", OLD, OLD_SCORES, "old_validation"),
                                     ("later", NEW, NEW_SCORES, "later_reused")):
        target = OUT / name
        entry = result["cohorts"][key]
        assert sha(target / "quality.npz") == entry["quality_sha256"]
        assert sha(target / "features.json") == entry["features_sha256"]
        for filename, digest in json.loads((target / "features.json").read_text())["sha256"].items():
            assert sha(target / filename) == digest
        _, queries, gallery = metadata(folder)
        q, g = (load_npz(target / (n + "_counts.npz")) for n in ("query", "gallery"))
        gt = weighted_gallery(g).T.tocsr()
        quality = np.load(target / "quality.npz", allow_pickle=False)
        tie_order = np.argsort(np.array([r["source_relpath"].encode() for r in gallery]))
        labels = np.array([r["cluster_id"] for r in gallery])
        saved = {n: np.load(source / (n + ".npy"), mmap_mode="r", allow_pickle=False)
                 for n in entry["method_order"] if n not in ("BM25", "D_BM25")}
        for ii, i in enumerate(quality["query_indices"]):
            bm = (q.getrow(i) @ gt).toarray().ravel()
            scores = {n: -v[i] for n, v in saved.items()}
            scores.update(BM25=bm, D_BM25=hybrid(scores["D"], bm, alpha))
            for mi, method in enumerate(entry["method_order"]):
                order = tie_order[np.argsort(-scores[method][tie_order], kind="stable")]
                positions = np.flatnonzero(labels[order] == queries[i]["cluster_id"]) + 1
                at8 = positions[positions <= 8]
                metrics = [sum((j+1)/p for j, p in enumerate(at8))/8, 1/positions[0], float(positions[0] <= 10)]
                np.testing.assert_array_equal(order[:10], quality["top10"][mi, ii])
                np.testing.assert_allclose(metrics, quality["metrics"][mi, ii], rtol=0, atol=1e-12)
                checked += 1
        for mi, method in enumerate(entry["method_order"]):
            expected = entry["methods"][method]
            np.testing.assert_allclose(quality["metrics"][mi].mean(axis=0),
                [expected[k] for k in ("MAP_at_8", "MRR", "Success_at_10")], atol=1e-12, rtol=0)
            assert problem_means(quality["metrics"][mi, :, 0], queries, quality["query_indices"]) == expected["problem_ap8"]
        lengths = g.sum(axis=1).A1
        df = (g > 0).sum(axis=0).A1
        for i in range(16):
            bm = (q.getrow(i) @ gt).toarray().ravel()
            for j in (0, (i*73+1) % len(gallery), int(np.flatnonzero(labels == queries[i]["cluster_id"])[0]), int(bm.argmax())):
                qb, gb = dict(zip(q.getrow(i).indices, q.getrow(i).data)), dict(zip(g.getrow(j).indices, g.getrow(j).data))
                scalar = sum(v * np.log(1+(len(gallery)-df[t]+.5)/(df[t]+.5)) * (K1+1)*gb[t]
                    / (gb[t]+K1*(1-B+B*lengths[j]/lengths.mean())) for t, v in qb.items() if t in gb)
                np.testing.assert_allclose(scalar, bm[j], atol=1e-10, rtol=1e-12)
                scalar_pairs += 1
        if name == "old":
            cal = np.load(OUT / "calibration_ap8.npz", allow_pickle=False)
            for ci, i in enumerate(cal["query_indices"]):
                bm = (q.getrow(i) @ gt).toarray().ravel()
                for wi, weight in enumerate(GRID):
                    score = hybrid(-saved["D"][i], bm, weight)
                    top = tie_order[np.argsort(-score[tie_order], kind="stable")][:8]
                    hit = labels[top] == queries[i]["cluster_id"]
                    ap = np.sum(np.cumsum(hit)*hit/np.arange(1, 9))/8
                    np.testing.assert_allclose(ap, cal["values"][wi, ci], atol=1e-12, rtol=0)
            np.testing.assert_allclose(cal["values"].mean(axis=1), calibration["MAP_at_8_by_weight"], atol=1e-12, rtol=0)
        print(f"Verified {name}: rankings, metrics, scalar BM25", flush=True)
    receipt = {"status": "passed", "result_sha256": sha(OUT / "result.json"),
        "evaluation_query_method_checks": checked, "calibration_query_weight_checks": 2600*len(GRID),
        "scalar_BM25_pairs": scalar_pairs, "input_and_output_hashes": "passed"}
    if (OUT / "verification.json").exists():
        assert json.loads((OUT / "verification.json").read_text()) == receipt
    else:
        write(OUT / "verification.json", receipt)
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    if sys.argv[1:] == ["--run"]:
        run()
    elif sys.argv[1:] != ["--check"]:
        raise SystemExit("Usage: bm25_20261010.py --run|--check")
    verify()
