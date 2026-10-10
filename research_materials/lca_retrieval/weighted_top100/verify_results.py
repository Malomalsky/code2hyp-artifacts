"""Offline replay of the weighted, BM25 and top-budget evaluations."""

import hashlib
import json
import os
from pathlib import Path
import sys

for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[name] = "1"

import numpy as np
import scipy
from scipy.sparse import load_npz
from kernels import HYBRIDS, GRID, mixed, hybrid, weighted_gallery, transform, group_counts, contrast, choose_folds

ROOT = Path(__file__).resolve().parent
KEYS = ("MAP_at_8", "MRR", "Success_at_10")
OLD_KEYS = ("problem_macro_map_at_r", "mrr", "recall_at_10")


def read(name):
    return json.loads((ROOT / name).read_text())


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def arrays(path):
    # Materialize once: NpzFile indexing otherwise decompresses on every query.
    with np.load(path, allow_pickle=False) as saved:
        return {name: saved[name] for name in saved.files}


def close(actual, expected):
    np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=0)


def manifest():
    if not __debug__:
        raise RuntimeError("Run without -O: validation assertions must be enabled")
    entries = read("MANIFEST.json")
    for name, entry in entries.items():
        relative = Path(name)
        assert not relative.is_absolute() and ".." not in relative.parts
        path = ROOT / relative
        assert path.is_file() and not path.is_symlink(), name
        assert path.stat().st_size == entry["bytes"] and sha(path) == entry["sha256"], name
    print(f"PASS: {len(entries)} manifested files", flush=True)


def bindings():
    for name in ("weighted", "top100", "two_stage", "bm25"):
        folder = ROOT / "data" / name
        result = read(f"data/{name}/result.json")
        assert not result["java_test_accessed"]
        for key, filename in (("design_sha256", "design_receipt.json"), ("selection_sha256", "selection_seal.json"),
                              ("encoding_receipt_sha256", "encoding_receipt.json"), ("quality_sha256", "quality.npz"),
                              ("evaluation_code_receipt_sha256", "kaggle_evaluation_code_receipt.json"),
                              ("calibration_sha256", "calibration.json")):
            if key in result:
                assert sha(folder / filename) == result[key], (name, key)
        for filename, expected in result.get("files_sha256", {}).items():
            assert sha(folder / filename) == expected
    for folder in (ROOT / "data/weighted", ROOT / "data/top100", ROOT / "data/development", ROOT / "data/bm25/old", ROOT / "data/bm25/later"):
        for filename in ("features.json", "bm25_features.json", "selection_seal.json"):
            if (folder / filename).exists():
                for name, expected in json.loads((folder / filename).read_text())["sha256"].items():
                    if name != "selected_programs.jsonl":
                        assert sha(folder / name) == expected, (folder, name)
    provenance = read("PROVENANCE.json")["files"]
    for name, entry in provenance.items():
        if name != "extracted_functions" and entry["export"] == "original_bytes":
            assert sha(ROOT / name) == entry["original_sha256"], name


def ranked(scores, tie_order):
    return tie_order[np.argsort(-scores[tie_order], kind="stable")]


def limited(dense_order, scores, tie_order, k):
    ids = tie_order[np.isin(tie_order, dense_order[:k])]
    return np.r_[ids[np.argsort(-scores[ids], kind="stable")], dense_order[k:]]


def metrics(ranking, labels, problem):
    positions = np.flatnonzero(labels[ranking] == problem) + 1
    assert len(positions) == 8
    hits = positions[positions <= 8]
    return [sum((j + 1) / p for j, p in enumerate(hits)) / 8,
            1 / positions[0], float(positions[0] <= 10)], positions[0]


def cohort(name):
    folder = ROOT / "data" / name
    rows = [json.loads(line) for line in (folder / "metadata.jsonl").read_text().splitlines()]
    assert len({r["source_relpath"] for r in rows}) == len(rows)
    qi = np.array([i for i, r in enumerate(rows) if r["role"] == "query"])
    gi = np.array([i for i, r in enumerate(rows) if r["role"] == "gallery"])
    assert len(rows) == len(qi) + len(gi) and len(qi) == len(gi)
    queries, gallery = [rows[i] for i in qi], [rows[i] for i in gi]
    tie = np.array([r["source_relpath"].encode() for r in gallery])
    tie_order = np.argsort(tie)
    labels = np.array([r["cluster_id"] for r in gallery])
    problems, groups = np.unique([r["cluster_id"] for r in queries], return_inverse=True)
    np.testing.assert_array_equal(np.bincount(groups), np.full(len(problems), 8))
    assert set(labels) == set(problems) and all(np.count_nonzero(labels == p) == 8 for p in problems)
    dense = np.load(folder / "dense.npy", mmap_mode="r", allow_pickle=False)
    assert dense.shape == (len(qi), len(gi)) and np.isfinite(dense).all()
    embeddings = np.load(folder / "embeddings.npy", allow_pickle=False)
    assert embeddings.shape == (len(rows), 768) and embeddings.dtype == np.float32
    np.testing.assert_allclose(np.linalg.norm(embeddings, axis=1), 1., atol=1e-5)
    # shortcut: score replay starts from saved embeddings; model/source regeneration is separate.
    np.testing.assert_allclose(embeddings[qi] @ embeddings[gi].T, -dense, atol=1e-6, rtol=0)
    receipt = read(f"data/{name}/encoding_receipt.json")
    assert sha(folder / "embeddings.npy") == receipt["embeddings_sha256"]
    assert sha(folder / "token_lengths.npy") == receipt["token_lengths_sha256"]
    assert receipt["programs"] == len(rows) and not receipt["training_performed"] and not receipt["java_test_accessed"]
    assert receipt["settings"]["revision"] == "3c4b60807d71f79b43f3c4363786d9493691f8b1"
    features = {}
    if name != "development":
        for key in ("ret_root_idf", "ret_raw_idf", "mask_root_idf", "ret_root", "nodes", "lexical"):
            features[key] = load_npz(folder / (key + "_features.npz"))
            assert features[key].shape[0] == len(rows) and np.isfinite(features[key].data).all()
        counts = load_npz(folder / "counts.npz")
        masked = group_counts(counts, np.load(folder / "group.npy", allow_pickle=False))
        for key, matrix, beta, idf in (("ret_root_idf", counts, .5, True),
                                     ("ret_raw_idf", counts, 1., True),
                                     ("mask_root_idf", masked, .5, True),
                                     ("ret_root", counts, .5, False)):
            difference = (transform(matrix, gi, beta, idf) - features[key]).tocsr()
            close(difference.data, np.zeros(len(difference.data)))
    return folder, rows, qi, gi, queries, tie, tie_order, labels, problems, groups, dense, features


def problem_means(values, groups, problems):
    counts = np.bincount(groups, minlength=len(problems))
    present = counts > 0
    means = np.bincount(groups, weights=values, minlength=len(problems))[present] / counts[present]
    return dict(zip(problems[present], means.tolist()))


def summary(quality, groups, problems, expected, old=False):
    close(quality.mean(axis=0), [expected[k] for k in (OLD_KEYS if old else KEYS)])
    means = problem_means(quality[:, 0], groups, problems)
    reference = expected["task_scores" if old else "problem_ap8"]
    assert set(means) == set(reference)
    close(list(means.values()), [reference[p] for p in means])
    return np.array([means[p] for p in sorted(means)])


def replay_weighted(data):
    folder, _, qi, gi, queries, tie, ties, labels, problems, groups, dense, features = data
    result = read("data/weighted/result.json")
    names = list(result["methods"])
    quality = np.empty((len(names), len(queries), 3))
    headers = read("data/weighted/score_headers.json")
    digests = {n: hashlib.sha256(bytes.fromhex(headers[n])) for n in names}
    transposed = {n: f[gi].T.tocsr() for n, f in features.items()}
    for i, query in enumerate(queries):
        auxiliary = {n: (f.getrow(qi[i]) @ transposed[n]).toarray().ravel() for n, f in features.items()}
        scores = mixed(-dense[i], auxiliary, tie)
        for j, name in enumerate(names):
            digests[name].update(np.asarray(-scores[name], dtype="<f8").tobytes())
            quality[j, i], _ = metrics(ranked(scores[name], ties), labels, query["cluster_id"])
            close(quality[j, i, 0], result["methods"][name]["metrics"]["query_scores"][query["source_relpath"]])
        if (i + 1) % 1300 == 0:
            print(f"Weighted: {i + 1}/{len(queries)}", flush=True)
    means = {}
    for j, name in enumerate(names):
        assert digests[name].hexdigest() == result["methods"][name]["negative_score_sha256"], name
        means[name] = summary(quality[j], groups, problems, result["methods"][name]["metrics"], old=True)
    indices = np.random.Generator(np.random.PCG64(20261010)).integers(650, size=(20000, 650))
    for name, expected in result["contrasts"].items():
        delta = means["candidate"] - means[name]
        close(contrast(delta, indices)["mean"], expected["mean"])
        close(contrast(delta, indices)["percentile_95_interval"], expected["percentile_95_interval"])
        close(np.quantile(delta[indices].mean(axis=1), [.05 / 6, 1 - .05 / 6]), expected["bonferroni_three_interval"])
        close(delta, [expected["problem_differences"][p] for p in problems])
    for name, expected in result["all_comparator_differences"].items():
        close((means["candidate"] - means[name]).mean(), expected)
    assert result["primary_supported"] == (result["contrasts"]["previous_tfidf"]["bonferroni_three_interval"][0] > 0)
    assert result["ancestor_supported"] == (result["contrasts"]["matched_mask"]["bonferroni_three_interval"][0] > 0)
    print("PASS: 14 weighted methods, all score SHA-256 hashes and three paired intervals", flush=True)


def replay_bm25(data, which):
    folder, _, qi, gi, queries, _, ties, labels, problems, groups, dense, features = data
    source = ROOT / "data/bm25" / which
    result = read("data/bm25/result.json")
    entry = result["cohorts"]["old_validation" if which == "old" else "later_reused"]
    stored = arrays(source / "quality.npz")
    selected = stored["query_indices"]
    q, g = (load_npz(source / (n + "_counts.npz")) for n in ("query", "gallery"))
    lexical = (q @ weighted_gallery(g).T).tocsr()
    alpha = result["alpha"]
    transposed = {n: features[HYBRIDS[n][0]][gi].T.tocsr() for n in entry["method_order"] if n in HYBRIDS}
    for ii, i in enumerate(selected):
        b, d = lexical.getrow(i).toarray().ravel(), -dense[i]
        scores = {"D": d, "BM25": b, "D_BM25": hybrid(d, b, alpha)}
        for name, transpose in transposed.items():
            feature, weight = HYBRIDS[name]
            scores[name] = (1 - weight) * d + weight * (features[feature].getrow(qi[i]) @ transpose).toarray().ravel()
        for j, name in enumerate(entry["method_order"]):
            ranking = ranked(scores[name], ties)
            close(metrics(ranking, labels, queries[i]["cluster_id"])[0], stored["metrics"][j, ii])
            np.testing.assert_array_equal(ranking[:10], stored["top10"][j, ii])
    for j, name in enumerate(entry["method_order"]):
        summary(stored["metrics"][j], groups[selected], problems, entry["methods"][name])
        if name in HYBRIDS:
            original = read("data/weighted/result.json")["methods"][name]["metrics"]
            close(stored["metrics"][j, :, 0], [original["query_scores"][queries[i]["source_relpath"]] for i in selected])
    assert sha(source / "quality.npz") == entry["quality_sha256"]
    assert sha(source / "features.json") == entry["features_sha256"]
    for name, expected in entry["differences"].items():
        a, b = name.split("_minus_")
        delta = np.array([entry["methods"][a]["problem_ap8"][p] - entry["methods"][b]["problem_ap8"][p]
                          for p in sorted(entry["methods"][a]["problem_ap8"])])
        close(delta.mean(), expected["mean_MAP_difference"])
        assert [int((delta > 1e-12).sum()), int((delta < -1e-12).sum()), int((np.abs(delta) <= 1e-12).sum())] == [
            expected[k] for k in ("better_problems", "worse_problems", "tied_problems")]
    if which == "old":
        calibration = read("data/bm25/calibration.json")
        design = read("data/bm25/design_receipt.json")
        values = arrays(ROOT / "data/bm25/calibration_ap8.npz")
        assert sha(ROOT / "data/bm25/calibration_ap8.npz") == calibration["query_values_sha256"]
        assert sha(ROOT / "data/bm25/design_receipt.json") == calibration["design_sha256"]
        cal = values["query_indices"]
        assert not set(design["calibration_problems"]) & set(design["validation_problems"])
        assert set(problems[groups[cal]]) == set(design["calibration_problems"])
        assert set(problems[groups[selected]]) == set(design["validation_problems"])
        for ci, i in enumerate(cal):
            b = lexical.getrow(i).toarray().ravel()
            for j, weight in enumerate(GRID):
                actual, _ = metrics(ranked(hybrid(-dense[i], b, weight), ties), labels, queries[i]["cluster_id"])
                close(actual[0], values["values"][j, ci])
        means = [np.mean(list(problem_means(v, groups[cal], problems).values())) for v in values["values"]]
        close(means, calibration["MAP_at_8_by_weight"])
        assert alpha == calibration["alpha"] == GRID[int(np.argmax(means))] == .10
    print(f"PASS: BM25 {which}, including development fit" if which == "old" else "PASS: BM25 later reused cohort", flush=True)


def development_grids(data):
    folder, _, _, _, queries, _, _, _, _, _, _, _ = data
    design = read("data/development/factorial_design_receipt.json")
    result = read("data/development/factorial_result.json")
    assert sha(folder / "factorial_design_receipt.json") == result["design_sha256"]
    assert design["grid"] == GRID and design["problems"] == result["problems"]
    problems = result["problems"]
    assert len(problems) == 325 and set(problems) <= {q["cluster_id"] for q in queries}
    folds = np.array([design["fold_assignment"][p] for p in problems])
    np.testing.assert_array_equal(np.bincount(folds), np.full(5, 65))
    for name in design["configs"]:
        values = np.load(folder / (name + "_problem_grid.npy"), allow_pickle=False)
        assert values.shape == (325, len(GRID)) and np.isfinite(values).all()
        choices, held = choose_folds(values, folds)
        expected = result["methods"][name]
        close(held, expected["oof_problem_ap8"])
        close(values[:, GRID.index(.15)], expected["fixed_problem_ap8"])
        close(values[:, -1], expected["standalone_problem_ap8"])
        for key in ("oof", "fixed", "standalone"):
            close(np.mean(expected[key + "_problem_ap8"]), expected[key + "_map8"])
        for f, choice in enumerate(choices):
            assert expected["folds"][f]["alpha"] == GRID[choice]
            close(values[folds != f, choice].mean(), expected["folds"][f]["training_map8"])
            close(held[folds == f].mean(), expected["folds"][f]["held_map8"])
    for name, expected in result["contrasts"].items():
        a, b = name.split("_minus_")
        for mode in ("oof", "fixed"):
            delta = np.array(result["methods"][a][mode + "_problem_ap8"]) - result["methods"][b][mode + "_problem_ap8"]
            close(delta.mean(), expected[mode]["mean"])
            assert [int((delta > 1e-12).sum()), int((delta < -1e-12).sum()), int((np.abs(delta) <= 1e-12).sum())] == [
                expected[mode][k] for k in ("higher", "lower", "equal")]
    print("PASS: eight development grids, all fold selections and 12 descriptive contrasts", flush=True)


def replay_budgets(data, fresh):
    _, _, qi, gi, queries, _, ties, labels, problems, groups, dense, features = data
    name = "top100" if fresh else "two_stage"
    result = read(f"data/{name}/result.json")
    stored = arrays(ROOT / f"data/{name}/quality.npz")
    np.testing.assert_array_equal(groups, stored["groups"])
    methods = list(result["methods"])
    transposed = {n: f[gi].T.tocsr() for n, f in features.items()}
    if fresh:
        q = load_npz(ROOT / "data/top100/query_lexical_counts.npz")
        bm = (q @ weighted_gallery(load_npz(ROOT / "data/top100/gallery_lexical_counts.npz")).T).tocsr()
    for i, query in enumerate(queries):
        d = -dense[i].astype(np.float64)
        dorder = ranked(d, ties)
        full = {}
        for f, key, alpha in (("nodes", "nodes", .50), ("root", "ret_root", .25), ("idf", "ret_root_idf", .20)):
            if fresh or f != "nodes":
                full[f] = (1 - alpha) * d + alpha * (features[key].getrow(qi[i]) @ transposed[key]).toarray().ravel()
        for j, method in enumerate(methods):
            if method == "D":
                ranking = dorder
            elif method in ("BM25", "D_BM25"):
                b = bm.getrow(i).toarray().ravel()
                ranking = ranked(b if method == "BM25" else hybrid(d, b, .10), ties)
            else:
                f, k = method.rsplit("_", 1)
                f = {"candidate": "idf", "simpler_root": "root"}.get(f, f)
                ranking = ranked(full[f], ties) if k == "full" else limited(dorder, full[f], ties, int(k))
            actual, first = metrics(ranking, labels, query["cluster_id"])
            close(actual, stored["metrics"][j, i])
            assert first == stored["first"][j, i]
            np.testing.assert_array_equal(ranking[:10], stored["top10"][j, i])
        for ki, k in enumerate((100,) if fresh else (50, 100, 500)):
            recall = np.count_nonzero(labels[dorder[:k]] == query["cluster_id"]) / 8
            assert recall == (stored["recall"][i] if fresh else stored["recall"][ki, i])
            if not fresh:
                for fi, f in enumerate(("idf", "root")):
                    assert np.isin(ranked(full[f], ties)[:8], dorder[:k]).mean() == stored["coverage"][fi, ki, i]
    for j, method in enumerate(methods):
        expected = result["methods"][method]
        reference = {**expected, "problem_ap8": expected["problem_ap8"] if fresh else expected["problem_MAP"]}
        summary(stored["metrics"][j], groups, problems, reference)
        if not fresh:
            base = result["methods"]["D"]["MAP_at_8"]
            full_method = method.rsplit("_", 1)[0] + "_full" if method != "D" else "D"
            full_map = result["methods"][full_method]["MAP_at_8"]
            close(expected["MAP_at_8"] - base, expected["delta_MAP_vs_dense"])
            close(expected["MAP_at_8"] - full_map, expected["delta_MAP_vs_full"])
            if full_map > base:
                close((expected["MAP_at_8"] - base) / (full_map - base), expected["fraction_full_MAP_gain_retained"])
    if fresh:
        means = {m: np.array([result["methods"][m]["problem_ap8"][p] for p in problems]) for m in methods}
        delta = means["root_100"] - means["D_BM25"]
        indices = np.random.Generator(np.random.PCG64(20261010)).integers(400, size=(20000, 400))
        assert contrast(delta, indices) == result["primary_root100_minus_denseBM25"]
        assert result["primary_supported"] == (result["primary_root100_minus_denseBM25"]["percentile_95_interval"][0] > 0)
        assert result["primary_problem_counts"] == {"better": int((delta > 0).sum()), "worse": int((delta < 0).sum()), "tied": int((delta == 0).sum())}
        assert result["dense_candidate_recall_100"] == float(stored["recall"].mean())
        assert result["queries_without_relevant_candidate_fraction"] == float((stored["recall"] == 0).mean())
        for base, comparisons in result["descriptive_MAP_deltas"].items():
            for method, value in comparisons.items():
                close(result["methods"][method]["MAP_at_8"] - result["methods"][base]["MAP_at_8"], value)
        for f, value in result["top100_minus_full_MAP"].items():
            close(result["methods"][f + "_100"]["MAP_at_8"] - result["methods"][f + "_full"]["MAP_at_8"], value)
    else:
        timing = arrays(ROOT / "data/two_stage/timing.npz")
        values = timing["milliseconds"]
        assert values.shape == (5, 512, 9, 5) and np.isfinite(values).all() and (values >= 0).all()
        np.testing.assert_allclose(values[..., :4].sum(axis=-1), values[..., 4], atol=1e-10, rtol=0)
        selected = sorted(range(len(queries)), key=lambda i: hashlib.sha256(queries[i]["source_relpath"].encode()).hexdigest())[:512]
        np.testing.assert_array_equal(selected, timing["query_indices"])
        for j, method in enumerate(methods):
            v = values[:, :, j]
            means = v.mean(axis=1)
            t = result["methods"][method]["timing_ms"]
            close(means[:, 4], t["per_pass_mean_total"])
            close(np.median(means[:, 4]), t["median_pass_mean_total"])
            close(np.median(v[:, :, 4]), t["individual_median_total"])
            close(np.quantile(v[:, :, 4], .95), t["individual_p95_total"])
            close(means.mean(axis=0), t["mean_components_dense_select_structure_sort_total"])
        for ki, k in enumerate((50, 100, 500)):
            entry = result["candidate_sets"][str(k)]
            close(stored["recall"][ki].mean(), entry["recall_at_K"])
            close((stored["recall"][ki] > 0).mean(), entry["any_relevant_fraction"])
            for fi, f in enumerate(("candidate", "simpler_root")):
                close(stored["coverage"][fi, ki].mean(), entry["full_hybrid_top8_coverage"][f])
    print(f"PASS: {name}, all {len(methods) * len(queries)} query/method outcomes", flush=True)


def main():
    manifest()
    bindings()
    from test_reproduction import self_check
    self_check()
    data = {n: cohort(n) for n in ("development", "weighted", "top100")}
    membership = [{r["cluster_id"] for r in d[4]} for d in data.values()]
    assert all(not a & b for i, a in enumerate(membership) for b in membership[i + 1:])
    assert len({json.dumps(read(f"data/{n}/encoding_receipt.json")["model_files"], sort_keys=True) for n in data}) == 1
    development_grids(data["development"])
    replay_weighted(data["weighted"])
    replay_bm25(data["development"], "old")
    replay_bm25(data["weighted"], "later")
    replay_budgets(data["weighted"], False)
    replay_budgets(data["top100"], True)
    from reproduce_figures import generate
    generate()
    output = ROOT / "reproduced"
    receipt = {"status": "passed", "numpy": np.__version__, "scipy": scipy.__version__,
               "manifest_sha256": sha(ROOT / "MANIFEST.json"), "neural_inference_performed": False,
               "training_performed": False, "java_test_accessed": False,
               "timing_scope": "All statistics replayed from original measured samples; no new hardware latency claim",
               "figure_source_sha256": {p.name: sha(p) for p in sorted(output.glob("figure-*.tex"))}}
    (output / "verification.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print("PASS: all packaged evaluations and three figure sources", flush=True)


if __name__ == "__main__":
    assert not sys.argv[1:], "Usage: python verify_results.py"
    main()
