"""Offline numerical replay, with an independent rank/metric implementation."""
import hashlib
import json
import os
from pathlib import Path
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[key] = "1"
import numpy as np
import scipy
from scipy.sparse import load_npz
from kernels import METHODS, rankings, crossed_contrasts, weighted_gallery, choose_folds, transform

ROOT = Path(__file__).resolve().parent
KEYS = ("nDCG@10", "MRR@10", "Recall@100")


def read(path):
    return json.loads((ROOT / path).read_text())


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def close(a, b):
    np.testing.assert_allclose(a, b, rtol=0, atol=1e-12)


def arrays(path):
    with np.load(ROOT / path, allow_pickle=False) as archive:
        return {k: archive[k] for k in archive.files}


def manifest():
    if not __debug__:
        raise RuntimeError("Do not use Python -O: assertions are required")
    entries = read("MANIFEST.json")
    for name, entry in entries.items():
        relative = Path(name)
        assert not relative.is_absolute() and ".." not in relative.parts
        path = ROOT / relative
        assert path.is_file() and not path.is_symlink()
        assert path.stat().st_size == entry["bytes"] and sha(path) == entry["sha256"], name
    for name, item in read("PROVENANCE.json")["files"].items():
        if name != "extracted_functions" and item["export"] == "original_bytes":
            assert sha(ROOT / name) == item["original_sha256"]
    print(f"PASS manifest: {len(entries)} files", flush=True)


def cohort(name):
    meta = read(f"data/{name}/cohort.json")
    ids = meta["ids"]
    assert len(set(ids["query"] + ids["gallery"])) == len(ids["query"]) + len(ids["gallery"])
    assert len(meta["groups"]) == len(meta["positive_gallery_indices"]) == len(ids["query"])
    for rel in meta["positive_gallery_indices"]:
        assert rel and len(set(rel)) == len(rel) and min(rel) >= 0 and max(rel) < len(ids["gallery"])
    return meta


def sorted_scores(scores, tie_order):
    return tie_order[np.argsort(-scores[tie_order], kind="stable")]


def limited(dense, structure, alpha, ties):
    base = sorted_scores(dense, ties)
    head = base[:100]
    head = head[np.argsort(np.argsort(ties)[head], kind="stable")]
    scores = (1-alpha)*dense[head].astype(float) + alpha*structure[head]
    return np.r_[head[np.argsort(-scores, kind="stable")], base[100:]]


def metrics(order, positive):
    assert len(order) == len(np.unique(order)) and min(order) == 0 and max(order) == len(order)-1
    inverse = np.empty(len(order), dtype=int)
    inverse[order] = np.arange(1, len(order)+1)
    positions = np.sort(inverse[positive])
    top = positions[positions <= 10]
    ndcg = np.sum(1/np.log2(top+1)) / np.sum(1/np.log2(np.arange(2, min(10, len(positive))+2)))
    common = [ndcg, 1/positions[0] if positions[0] <= 10 else 0., np.count_nonzero(positions <= 100)/len(positions)]
    ap = np.sum(np.arange(1, np.count_nonzero(positions <= 8)+1)/positions[positions <= 8])/len(positive)
    return np.array(common), np.array([ap, 1/positions[0], float(positions[0] <= 10)])


def grouped(quality, groups):
    groups = np.asarray(groups)
    return np.stack([quality[:, groups == p].mean(axis=1) for p in np.unique(groups)], axis=1)


def embeddings(name, meta, dense):
    folder = ROOT / "data" / name
    receipt = read(f"data/{name}/encoding_receipt.json")
    for item in ("embeddings", "token_lengths"):
        assert sha(folder / (item + ".npy")) == receipt[item + "_sha256"]
    vectors = np.load(folder / "embeddings.npy", allow_pickle=False)
    lengths = np.load(folder / "token_lengths.npy", allow_pickle=False)
    assert vectors.dtype == np.float32 and np.isfinite(vectors).all()
    assert len(vectors) == receipt["programs"] == len(lengths)
    assert lengths.dtype.kind in "iu" and (lengths > 0).all()
    assert int((lengths > receipt["settings"]["max_length"]).sum()) == receipt["truncated_programs"]
    assert not receipt["training_performed"] and not receipt["java_test_accessed"]
    np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), 1., rtol=0, atol=1e-5)
    qi, gi = (meta["positions"][r] for r in ("query", "gallery"))
    # shortcut: preserve historical float32 score ties; encoding requires upstream inputs.
    np.testing.assert_allclose(vectors[qi] @ vectors[gi].T, dense, rtol=0, atol=1e-6)
    return receipt


def crossed():
    results, values = {}, {}
    for corpus in ("CodeContests", "CPRet"):
        meta = cohort(corpus)
        qi, gi = (meta["positions"][r] for r in ("query", "gallery"))
        folder = ROOT / "data" / corpus
        auxiliary = {n: (load_npz(folder/(n+".npz"))[qi] @ load_npz(folder/(n+".npz"))[gi].T).toarray()
                     for n in ("root", "nodes")}
        if corpus == "CPRet":
            lexical = load_npz(folder/"lexical.npz")
            q, g = lexical[qi], lexical[gi]
        else:
            q, g = (load_npz(folder/(r+"_counts.npz")) for r in ("query", "gallery"))
        auxiliary["bm"] = (q @ weighted_gallery(g).T).toarray()
        tie = np.array([s.encode() for s in meta["ids"]["gallery"]])
        ties = np.argsort(tie, kind="stable")
        for model in ("CodeRank", "Qwen"):
            name = corpus + "_" + model
            result = read(f"data/{name}/result.json")
            saved = arrays(f"data/{name}/arrays.npz")
            assert sha(ROOT/f"data/{name}/arrays.npz") == result["arrays_sha256"]
            assert sha(ROOT/"data/crossed/analysis_receipt.json") == result["analysis_sha256"]
            dense = np.load(ROOT/f"data/{name}/dense.npy", allow_pickle=False)
            receipt = embeddings(name, meta, dense)
            assert sha(ROOT/f"data/{name}/encoding_receipt.json") == result["encoding"]["receipt_sha256"]
            assert receipt["embeddings_sha256"] == result["encoding"]["embedding_sha256"]
            quality = np.empty_like(saved["quality"])
            native = np.empty_like(saved["native"])
            for i, positives in enumerate(meta["positive_gallery_indices"]):
                d, b = dense[i].astype(float), auxiliary["bm"][i]
                scale = lambda v: (v-v.min())/(v.max()-v.min()) if v.max() > v.min() else np.zeros_like(v)
                expected_orders = rankings(d, auxiliary["root"][i], auxiliary["nodes"][i], b, tie)
                independent = {"D": sorted_scores(d, ties), "BM25": sorted_scores(b, ties),
                    "D_BM25": sorted_scores(.9*scale(d)+.1*scale(b), ties),
                    "root_full": sorted_scores(.75*d+.25*auxiliary["root"][i], ties),
                    "nodes_100": limited(d, auxiliary["nodes"][i], .5, ties),
                    "root_100": limited(d, auxiliary["root"][i], .25, ties)}
                for j, method in enumerate(METHODS):
                    order = independent[method]
                    np.testing.assert_array_equal(order, expected_orders[method])
                    np.testing.assert_array_equal(order, saved["rankings"][j, i])
                    quality[j, i], n = metrics(order, positives)
                    if corpus == "CodeContests":
                        native[j, i] = n
            close(quality, saved["quality"])
            by_problem = grouped(quality, meta["groups"])
            close(by_problem, saved["by_problem"])
            assert result["problems"] == sorted(set(meta["groups"]))
            for j, method in enumerate(METHODS):
                close(by_problem[j].mean(axis=0), [result["methods"][method][k] for k in KEYS])
            if corpus == "CodeContests":
                close(native, saved["native"])
                for j, method in enumerate(METHODS):
                    close(native[j].mean(axis=0), [result["native_metrics"][method][k] for k in ("MAP_at_8", "MRR", "Success_at_10")])
            values[name] = by_problem
            results[name] = {m: dict(zip(KEYS, by_problem[j].mean(axis=0).tolist())) for j, m in enumerate(METHODS)}
            print("PASS", name, "all six full rankings and metrics", flush=True)
    contrasts = crossed_contrasts(values)
    reference = read("data/crossed/summary.json")
    for name, result in results.items():
        assert sha(ROOT/f"data/{name}/result.json") == reference["cell_result_sha256"][name]
        for method in METHODS:
            close(list(result[method].values()), [reference["cells"][name][method][k] for k in KEYS])
    for comparison, cells in contrasts.items():
        for name, effect in cells.items():
            close(effect["mean"], reference["contrasts"][comparison][name]["mean"])
            close(effect["percentile_95_interval"], reference["contrasts"][comparison][name]["percentile_95_interval"])
    return {"cells": results, "contrasts": contrasts}


def development():
    folder = ROOT/"data/development"
    result, design = read("data/development/result.json"), read("data/development/design_receipt.json")
    assert sha(folder/"development.npz") == result["arrays_sha256"]
    assert sha(folder/"design_receipt.json") == result["design_sha256"]
    saved, meta = arrays("data/development/development.npz"), cohort("development")
    embeddings("development", meta, saved["dense"])
    structure = np.load(folder/"structure.npy", mmap_mode="r", allow_pickle=False)
    control = np.load(folder/"control_dense.npy", mmap_mode="r", allow_pickle=False)
    ties = np.argsort([s.encode() for s in meta["ids"]["gallery"]], kind="stable")
    grid = result["grid"]
    assert grid == design["grid"] and meta["ids"]["query"] == design["query_ids"]
    actual = np.empty_like(saved["query_grid"])
    control_ap = np.empty_like(saved["control_ap"])
    for i, positive in enumerate(meta["positive_gallery_indices"]):
        for j, alpha in enumerate(grid):
            ranking = limited(saved["dense"][i].astype(float), structure[i], alpha, ties)
            actual[i, j] = metrics(ranking, positive)[1][0]
        for j, alpha in enumerate((0., .25)):
            control_ap[i, j] = metrics(limited(control[i], structure[i], alpha, ties), positive)[1][0]
    close(actual, saved["query_grid"])
    close(control_ap, saved["control_ap"])
    close(control_ap.mean(axis=0), result["coderank_dense_fixed025_map8"])
    table = grouped(actual.T[:, :, None], meta["groups"])[:, :, 0].T
    close(table, saved["problem_grid"])
    close(table.mean(axis=0), result["problem_means"])
    problems = sorted(set(meta["groups"]))
    assert problems == design["problems"]
    folds = np.array([design["fold_assignment"][p] for p in problems])
    choices, oof = choose_folds(table, folds)
    close(oof, saved["oof"])
    close(oof.mean(), result["oof_map8"])
    for i, selected in enumerate(choices):
        item = result["folds"][i]
        assert item["fold"] == i and item["alpha"] == grid[selected]
        close(table[folds != i, selected].mean(), item["train_map8"])
        close(oof[folds == i].mean(), item["held_map8"])
    assert grid[int(np.argmax(table.mean(axis=0)))] == result["alpha"] == .10
    for key, alpha in (("dense_map8", 0.), ("fixed_025_map8", .25), ("in_sample_selected_map8", .10)):
        close(table[:, grid.index(alpha)].mean(), result[key])
    seal = read("data/development/weight_seal.json")
    assert seal["alpha"] == .10 and not seal["cpret_used"] and not seal["fresh_contents_read"]
    for key, name in (("result_sha256", "result.json"), ("design_sha256", "design_receipt.json"), ("verification_sha256", "verification.json")):
        assert sha(folder/name) == seal[key]
    print("PASS development: all grid weights, five folds, selected alpha 0.10", flush=True)
    return result["alpha"]


def apps(alpha):
    folder, meta = ROOT/"data/APPS", cohort("APPS")
    result, saved = read("data/APPS/result.json"), arrays("data/APPS/arrays.npz")
    for key, name in (("design_sha256", "design_receipt.json"), ("selection_sha256", "selection_seal.json"),
                      ("features_sha256", "features.json"), ("arrays_sha256", "arrays.npz")):
        assert sha(folder/name) == result[key]
    assert sha(ROOT/"data/development/weight_seal.json") == result["weight_seal_sha256"]
    assert alpha == result["alpha"] and not result["parameters_fit_on_new_cohort"] and not result["java_test_accessed"]
    embeddings("APPS", meta, saved["dense"])
    qi, gi = (meta["positions"][r] for r in ("query", "gallery"))
    counts, features = load_npz(folder/"counts.npz"), load_npz(folder/"ret_root_features.npz")
    difference = (transform(counts, gi, .5, False)-features).tocsr()
    close(difference.data, np.zeros(len(difference.data)))
    close((features[qi]@features[gi].T).toarray(), saved["structure"])
    ties = np.argsort([s.encode() for s in meta["ids"]["gallery"]], kind="stable")
    quality = np.empty_like(saved["quality"])
    for i, positive in enumerate(meta["positive_gallery_indices"]):
        assert len(positive) == 8
        for j, a in enumerate((0., .25, alpha)):
            ranking = limited(saved["dense"][i].astype(float), saved["structure"][i], a, ties)
            np.testing.assert_array_equal(ranking, saved["rankings"][j, i])
            common, native = metrics(ranking, positive)
            quality[j, i] = [native[0], *common]
    close(quality, saved["quality"])
    means = grouped(quality, meta["groups"])
    close(means, saved["problem_means"])
    for j, method in enumerate(("dense", "fixed025", "selected")):
        close(means[j].mean(axis=0), [result["methods"][method][k] for k in ("MAP@8", *KEYS)])
    rng = np.random.Generator(np.random.PCG64(20261014))
    indices = rng.integers(len(means[0]), size=(20000, len(means[0])))
    contrasts = {}
    for j, name in ((0, "selected_minus_dense"), (1, "selected_minus_fixed025")):
        delta = means[2, :, 0] - means[j, :, 0]
        effect = {"mean": float(delta.mean()), "percentile95": np.quantile(delta[indices].mean(axis=1), [.025, .975]).tolist(),
                  "higher": int((delta > 1e-12).sum()), "lower": int((delta < -1e-12).sum()), "equal": int((np.abs(delta) <= 1e-12).sum())}
        for key, value in effect.items():
            close(value, result["contrasts"][name][key])
        contrasts[name] = effect
    print("PASS APPS: 4,800 full rankings, all metrics and both contrasts", flush=True)
    return {"methods": result["methods"], "contrasts": contrasts}


def main():
    manifest()
    output = ROOT/"reproduced"
    output.mkdir(exist_ok=True)
    cross = crossed()
    alpha = development()
    ap = apps(alpha)
    report = {"status": "passed", "manifest_sha256": sha(ROOT/"MANIFEST.json"),
              "crossed_rankings": 41712, "apps_rankings": 4800, "development_queries": 2600,
              "alpha": alpha, "crossed": cross, "APPS": ap,
              "numpy": np.__version__, "scipy": scipy.__version__, "metric_atol": 1e-12,
              "encoding_replayed": False, "raw_corpus_parsing_replayed": False}
    (output/"verification.json").write_text(json.dumps(report, indent=2, sort_keys=True)+"\n")
    print("PASS all external numerical results", flush=True)


if __name__ == "__main__":
    main()
