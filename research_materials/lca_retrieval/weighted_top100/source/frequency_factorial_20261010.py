"""Bounded, count-only factorial study on the old calibration problems."""

import conditional_structure_pilot_20261010 as previous
from followup_baselines import normalized
from rank_fusion_followup import ROOT, sha, write
from datetime import datetime, timezone
import hashlib
import itertools
import json
from pathlib import Path
import resource
import shutil
import sys
import time

import numpy as np
from scipy.sparse import csr_matrix, load_npz

OUT = ROOT / "followup_results/frequency_factorial_20261010"
PROTOCOL = ROOT / "frequency_factorial_protocol_20261010.md"
GRID = previous.GRID
CONFIGS = {f"{representation}_{tf}{'_idf' if idf else ''}": (representation, tf, idf)
           for representation, tf, idf in itertools.product(("ret", "mask"), ("raw", "root"), (False, True))}


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


def ap8(scores, tie_order, relevant):
    order = tie_order[np.argsort(-scores[tie_order], kind="stable")[:8]]
    hits = relevant[order]
    return float(np.sum(np.cumsum(hits) * hits / np.arange(1, len(order) + 1)) / 8)


def choose_folds(values, folds):
    choices, held = [], np.empty(len(values))
    for fold in range(int(folds.max()) + 1):
        test = folds == fold
        assert test.any() and (~test).any()
        choice = int(np.argmax(values[~test].mean(axis=0)))
        choices.append(choice)
        held[test] = values[test, choice]
    return choices, held


def prepare():
    rows, queries, gallery = previous.prepare()
    assert sha(previous.OUT / "result.json") == "40fbe3fccf8299fe8444ccc56dfce5c859066f1035a402548a65e19911c9056e"
    seal = json.loads((previous.OUT / "verification.json").read_text())
    assert seal["status"] == "passed" and seal["result_sha256"] == sha(previous.OUT / "result.json")
    features = json.loads((previous.OUT / "features.json").read_text())
    previous.verify_files(features)
    old_design = json.loads((previous.OUT / "design_receipt.json").read_text())
    problems = sorted(old_design["calibration_problems"])
    ordered = sorted(problems, key=lambda p: (
        hashlib.sha256(("frequency-factorial-20261010:" + p).encode()).hexdigest(), p))
    assignment = {p: i % 5 for i, p in enumerate(ordered)}
    assert len(problems) == 325 and set(np.bincount(list(assignment.values()))) == {65}
    indices = [i for i, q in enumerate(queries) if q["cluster_id"] in assignment]
    assert len(indices) == 2600
    assert not {queries[i]["cluster_id"] for i in indices} & set(old_design["validation_problems"])
    paths = [Path(__file__), PROTOCOL, ROOT / "test_frequency_factorial_20261010.py",
             *(previous.OUT / n for n in ("design_receipt.json", "features.json", "result.json", "verification.json"))]
    fixed = {"bindings": {str(p): sha(p) for p in paths}, "grid": GRID, "configs": CONFIGS,
             "problems": problems, "fold_assignment": assignment,
             "query_ids": [queries[i]["source_relpath"] for i in indices],
             "gallery_ids": [g["source_relpath"] for g in gallery], "fixed_alpha": .15,
             "status": "exploratory_old_calibration_only", "threads": 1}
    OUT.mkdir(exist_ok=True)
    path = OUT / "design_receipt.json"
    if path.exists():
        old = json.loads(path.read_text())
        for key, value in fixed.items():
            assert json.loads(json.dumps(value)) == old[key], key
    else:
        assert not list(OUT.iterdir()) and shutil.disk_usage(OUT).free > 3 * 1024**3
        write(path, {**fixed, "fixed_before_new_scores_utc": datetime.now(timezone.utc).isoformat(),
                     "resources": {"model": "Mac15,12", "ram_bytes": 25769803776,
                                   "free_disk_bytes": shutil.disk_usage(OUT).free}})
    return rows, queries, gallery, indices, problems, np.array([assignment[p] for p in problems])


def problem_values(values, queries, indices, problems):
    mapping = {p: [] for p in problems}
    for i, value in zip(indices, values):
        mapping[queries[i]["cluster_id"]].append(value)
    assert {len(v) for v in mapping.values()} == {8}
    return np.array([np.mean(mapping[p], axis=0) for p in problems])


def build(rows, queries, gallery, indices, problems):
    counts = load_npz(previous.OUT / "counts.npz")
    group = np.load(previous.OUT / "group.npy", allow_pickle=False)
    gi = np.array([i for i, row in enumerate(rows) if row["role"] == "gallery"])
    qi = np.array([i for i, row in enumerate(rows) if row["role"] == "query"])[indices]
    assert counts.shape == (len(rows), len(group))
    bags = {"ret": counts, "mask": group_counts(counts, group)}
    np.testing.assert_array_equal(bags["ret"].sum(axis=1), bags["mask"].sum(axis=1))
    dense = np.load(previous.DENSE / "D.npy", mmap_mode="r", allow_pickle=False)
    tie = np.argsort(np.array([g["source_relpath"].encode() for g in gallery]))
    labels = np.array([g["cluster_id"] for g in gallery])
    relevant = {p: labels == p for p in problems}
    assert all(mask.sum() == 8 for mask in relevant.values())
    d_result = json.loads((previous.DENSE / "result.json").read_text())["methods"]["D"]["metrics"]["query_scores"]
    d_values = np.array([d_result[queries[i]["source_relpath"]] for i in indices])
    old_dense = problem_values(d_values, queries, indices, problems)
    for name, (representation, tf, idf) in CONFIGS.items():
        receipt = OUT / (name + ".json")
        if receipt.exists():
            saved = json.loads(receipt.read_text())
            assert saved["design_sha256"] == sha(OUT / "design_receipt.json")
            for filename, expected in saved["output_sha256"].items():
                assert sha(OUT / filename) == expected, filename
            continue
        path = OUT / (name + ".npy")
        assert not path.exists(), "Inspect an incomplete stage; do not overwrite it"
        started = time.perf_counter()
        vectors = transform(bags[representation], gi, 1. if tf == "raw" else .5, idf)
        feature_seconds = time.perf_counter() - started
        score = np.lib.format.open_memmap(path, mode="w+", dtype=np.float64,
                                         shape=(len(indices), len(gallery)))
        old = None
        if name == "ret_raw":
            old = np.load(previous.COHORT / "fragment_retained.npy", mmap_mode="r", allow_pickle=False)
        elif name == "ret_raw_idf":
            old = np.load(previous.OUT / "I.npy", mmap_mode="r", allow_pickle=False)
        maximum_error = 0.
        started = time.perf_counter()
        for start in range(0, len(qi), 128):
            block = (vectors[qi[start:start + 128]] @ vectors[gi].T).toarray()
            assert np.isfinite(block).all() and block.min() >= 0 and block.max() <= 1 + 1e-12
            score[start:start + len(block)] = block
            if old is not None:
                reference = old[np.array(indices[start:start + len(block)])]
                if name == "ret_raw":
                    reference = 1 - reference
                np.testing.assert_allclose(block, reference, rtol=0, atol=2e-12)
                maximum_error = max(maximum_error, float(np.max(np.abs(block - reference))))
        score.flush()
        scoring_seconds = time.perf_counter() - started
        started = time.perf_counter()
        values = np.empty((len(indices), len(GRID)))
        for j, i in enumerate(indices):
            d = -np.asarray(dense[i], dtype=np.float64)
            positive = relevant[queries[i]["cluster_id"]]
            for a, alpha in enumerate(GRID):
                values[j, a] = ap8((1 - alpha) * d + alpha * score[j], tie, positive)
        np.testing.assert_allclose(values[:, 0], d_values, rtol=0, atol=1e-12)
        by_problem = problem_values(values, queries, indices, problems)
        np.save(OUT / (name + "_problem_grid.npy"), by_problem, allow_pickle=False)
        write(receipt, {"design_sha256": sha(OUT / "design_receipt.json"),
            "completed_utc": datetime.now(timezone.utc).isoformat(),
            "dimensions": vectors.shape[1], "zero_vectors": int((vectors.getnnz(axis=1) == 0).sum()),
            "reconstructed_previous_max_error": maximum_error if old is not None else None,
            "timing": {"feature_seconds": feature_seconds, "scoring_seconds": scoring_seconds,
                       "grid_seconds": time.perf_counter() - started},
            "output_sha256": {n: sha(OUT / n) for n in (path.name, name + "_problem_grid.npy")}})
        print("Completed development grid:", name, flush=True)
    return old_dense


def summarize(problems, folds, dense):
    fixed_index = GRID.index(.15)
    methods = {"D": {"oof_problem_ap8": dense.tolist(), "fixed_problem_ap8": dense.tolist(),
                      "oof_map8": float(dense.mean()), "fixed_map8": float(dense.mean())}}
    for name in CONFIGS:
        values = np.load(OUT / (name + "_problem_grid.npy"), allow_pickle=False)
        assert values.shape == (325, len(GRID)) and np.isfinite(values).all()
        choices, oof = choose_folds(values, folds)
        fixed, alone = values[:, fixed_index], values[:, -1]
        methods[name] = {"oof_problem_ap8": oof.tolist(), "fixed_problem_ap8": fixed.tolist(),
            "standalone_problem_ap8": alone.tolist(), "oof_map8": float(oof.mean()),
            "fixed_map8": float(fixed.mean()), "standalone_map8": float(alone.mean()),
            "folds": [{"fold": f, "alpha": GRID[a],
                       "training_map8": float(values[folds != f, a].mean()),
                       "held_map8": float(oof[folds == f].mean())} for f, a in enumerate(choices)]}
        print(name, "OOF", float(oof.mean()), "fixed", float(fixed.mean()),
              "alphas", [GRID[a] for a in choices], flush=True)
    contrasts = {}
    for low, a in CONFIGS.items():
        for high, b in CONFIGS.items():
            changed = [i for i in range(3) if a[i] != b[i]]
            if len(changed) != 1:
                continue
            factor = changed[0]
            if b[factor] != ("ret", "root", True)[factor]:
                continue
            comparison = {}
            for mode in ("oof", "fixed"):
                delta = np.array(methods[high][mode + "_problem_ap8"]) - methods[low][mode + "_problem_ap8"]
                comparison[mode] = {"mean": float(delta.mean()), "higher": int((delta > 1e-12).sum()),
                    "lower": int((delta < -1e-12).sum()), "equal": int((np.abs(delta) <= 1e-12).sum())}
            contrasts[high + "_minus_" + low] = {"factor": ("category", "root_tf", "idf")[factor], **comparison}
    assert len(contrasts) == 12
    write(OUT / "result.json", {"status": "completed_exploratory_old_calibration_only",
        "completed_utc": datetime.now(timezone.utc).isoformat(), "problems": problems,
        "methods": methods, "contrasts": contrasts, "design_sha256": sha(OUT / "design_receipt.json"),
        "method_receipt_sha256": {name: sha(OUT / (name + ".json")) for name in CONFIGS},
        "peak_rss_bytes_before_verification": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "prior_validation_queries_scored": 0, "neural_training_or_inference": False,
        "java_test_accessed": False, "cpret_test_accessed": False,
        "limitations": "All problems previously analysed; CV does not remove adaptive design history. No independent confirmation or significance claim."})


def verify(queries, gallery, indices, problems, folds):
    result = json.loads((OUT / "result.json").read_text())
    assert result["design_sha256"] == sha(OUT / "design_receipt.json")
    dense = np.load(previous.DENSE / "D.npy", mmap_mode="r", allow_pickle=False)
    tie = np.array([g["source_relpath"].encode() for g in gallery])
    labels = np.array([g["cluster_id"] for g in gallery])
    positive = {p: np.flatnonzero(labels == p) for p in problems}
    fold_by_problem = dict(zip(problems, folds))
    for name in CONFIGS:
        receipt = json.loads((OUT / (name + ".json")).read_text())
        assert sha(OUT / (name + ".json")) == result["method_receipt_sha256"][name]
        for filename, expected in receipt["output_sha256"].items():
            assert sha(OUT / filename) == expected, filename
        table = np.load(OUT / (name + "_problem_grid.npy"), allow_pickle=False)
        entry = result["methods"][name]
        for f, saved in enumerate(entry["folds"]):
            assert saved["alpha"] == GRID[int(np.argmax(table[folds != f].mean(axis=0)))]
        matrix = np.load(OUT / (name + ".npy"), mmap_mode="r", allow_pickle=False)
        for mode in ("oof", "fixed", "standalone"):
            values = []
            for j, i in enumerate(indices):
                p = queries[i]["cluster_id"]
                alpha = entry["folds"][fold_by_problem[p]]["alpha"] if mode == "oof" else .15 if mode == "fixed" else 1.
                scores = -(1 - alpha) * np.asarray(dense[i], dtype=np.float64) + alpha * matrix[j]
                values.append(previous.ap8(scores, tie, positive[p]))
            actual = problem_values(values, queries, indices, problems)
            np.testing.assert_allclose(actual, entry[mode + "_problem_ap8"], rtol=0, atol=1e-12)
        print("Independent ranks verified:", name, flush=True)
    seal = {"status": "passed", "result_sha256": sha(OUT / "result.json"),
            "verified_rankings": 24, "queries_per_ranking": len(indices), "problems": len(problems),
            "method": "existing independent lexsort rank-position AP8",
            "previous_validation_queries_scored": 0}
    path = OUT / "verification.json"
    if path.exists():
        assert json.loads(path.read_text()) == seal
    else:
        write(path, seal)


if __name__ == "__main__":
    if sys.argv[1:] not in (["run"], ["verify"]):
        raise SystemExit("Usage: frequency_factorial_20261010.py run|verify")
    rows, queries, gallery, indices, problems, folds = prepare()
    if sys.argv[1] == "run":
        assert not (OUT / "result.json").exists(), "Use verify for a completed study"
        dense = build(rows, queries, gallery, indices, problems)
        summarize(problems, folds, dense)
    verify(queries, gallery, indices, problems, folds)
