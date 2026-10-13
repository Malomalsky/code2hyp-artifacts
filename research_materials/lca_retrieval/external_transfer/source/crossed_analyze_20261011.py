"""Common-metric 2x2 analysis of saved embeddings; no fitting or neural inference."""
import os
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[key] = "1"
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.sparse import load_npz
from crossed_prepare_20261011 import OUT, EXECUTION, CC, CP, OLD_CC, OLD_CP, ROOT, check, sha, write
from crossed_encode_20261011 import coderank_settings, QWEN
from cpret_external_20261011 import rankings, METHODS
from cpret_evaluate import inputs as cpret_inputs, query_metrics, KEYS
from bm25_20261010 import weighted_gallery


def freeze():
    check()
    value = {"design_sha256": sha(OUT/"crossed_design.json"),
        "bindings": {str(p): sha(p) for p in (Path(__file__), ROOT/"test_crossed_analysis_20261011.py")}}
    path = OUT/"analysis_receipt.json"
    if path.exists():
        old = json.loads(path.read_text())
        assert all(old[k] == v for k, v in value.items())
    else:
        assert not list(OUT.glob("*_result.json"))
        write(path, {**value, "fixed_before_missing_cell_scores_utc": datetime.now(timezone.utc).isoformat()})


def cohort(name):
    check()
    if name == "CPRet":
        manifest, rows, _ = cpret_inputs()
        ids, rel = manifest["python_ids"], manifest["python_qrels"]
        group_names = [manifest["problem_urls"][q] for q in ids["query"]]
        features = {n: load_npz(CP/(n+".npz")) for n in ("root", "nodes", "lexical")}
    else:
        assert name == "CodeContests"
        rows = [json.loads(line) for line in (CC/"metadata.jsonl").read_text().splitlines()]
        ids = {r: [x["source_relpath"] for x in rows if x["role"] == r] for r in ("query", "gallery")}
        gallery = [x for x in rows if x["role"] == "gallery"]
        queries = [x for x in rows if x["role"] == "query"]
        rel = {q["source_relpath"]: {g["source_relpath"]: 1 for g in gallery if g["cluster_id"] == q["cluster_id"]} for q in queries}
        assert all(len(v) == 8 for v in rel.values())
        group_names = [q["cluster_id"] for q in queries]
        features = {n: load_npz(CC/(f+"_features.npz")) for n, f in (("root", "ret_root"), ("nodes", "nodes"))}
    qi = np.array([i for i, r in enumerate(rows) if r["role"] == "query"])
    gi = np.array([i for i, r in enumerate(rows) if r["role"] == "gallery"])
    auxiliary = {n: (features[n][qi] @ features[n][gi].T).toarray() for n in ("root", "nodes")}
    if name == "CPRet":
        q, g = features["lexical"][qi], features["lexical"][gi]
    else:
        q, g = (load_npz(CC/(r+"_lexical_counts.npz")) for r in ("query", "gallery"))
    auxiliary["bm"] = (q @ weighted_gallery(g).T).toarray()
    problems, groups = np.unique(group_names, return_inverse=True)
    assert len(problems) == (276 if name == "CPRet" else 400)
    assert set(np.bincount(groups)) == ({1} if name == "CPRet" else {8})
    return ids, rel, qi, gi, auxiliary, problems, groups


def embeddings(corpus, model, downloads, count):
    diagonal = (corpus, model) in (("CodeContests", "CodeRank"), ("CPRet", "Qwen"))
    if diagonal:
        folder = OLD_CC if model == "CodeRank" else OLD_CP
    else:
        name = "coderank_cpret" if model == "CodeRank" else "qwen_codecontests"
        base = downloads/"crossed_retrievers"
        folder, inputs = base/name/"encoding", base/name/"input"
        execution = json.loads((OUT/"crossed_execution_receipt.json").read_text())
        digest = sha(OUT/"crossed_execution_receipt.json")
        assert sha(base/"crossed_execution_receipt.json") == digest
        assert sha(inputs/"execution_receipt.json") == digest
        assert sha(base/"submitted_launcher.py") == execution["launcher_sha256"]
        for file, expected in execution["payload_sha256"].items():
            if file.endswith((".py", ".md")) or file == "crossed_design.json":
                assert sha(inputs/file) == expected
        prefix = "coderank" if model == "CodeRank" else "qwen"
        source = "cpret" if corpus == "CPRet" else "codecontests"
        assert sha(inputs/"selected_programs.jsonl") == execution["payload_sha256"][source+"_programs.jsonl"]
        assert sha(inputs/"expected_model_files.json") == execution["payload_sha256"][prefix+"_expected_model_files.json"]
    receipt = json.loads((folder/"encoding_receipt.json").read_text())
    expected_files = json.loads(((OLD_CC if model == "CodeRank" else OLD_CP)/"encoding_receipt.json").read_text())["model_files"]
    assert receipt["programs"] == count and receipt["gpu"] == "Tesla T4"
    assert receipt["model_files"] == expected_files and not receipt["training_performed"] and not receipt["java_test_accessed"]
    assert receipt["status"].startswith("completed_")
    if not diagonal:
        if model == "CodeRank":
            assert receipt["settings"] == coderank_settings(execution)
            assert receipt["settings_sha256"] == hashlib.sha256(json.dumps(receipt["settings"], sort_keys=True).encode()).hexdigest()
            assert receipt["script_sha256"] == sha(ROOT/"coderank_kaggle_20261010.py")
        else:
            assert receipt["settings"] == QWEN and receipt["binding"] == digest
            assert receipt["script_sha256"] == sha(ROOT/"crossed_encode_20261011.py")
            assert receipt["source_sha256"] == sha(inputs/"selected_programs.jsonl")
    for name in ("embeddings", "token_lengths"):
        assert sha(folder/(name+".npy")) == receipt[name+"_sha256"]
    vectors = np.load(folder/"embeddings.npy", allow_pickle=False)
    assert vectors.shape == (count, 768 if model == "CodeRank" else 1024) and vectors.dtype == np.float32
    assert np.isfinite(vectors).all()
    np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), 1., atol=1e-5)
    lengths = np.load(folder/"token_lengths.npy", allow_pickle=False)
    assert lengths.shape == (count,) and lengths.dtype.kind in "iu" and (lengths > 0).all()
    assert int((lengths > receipt["settings"]["max_length"]).sum()) == receipt["truncated_programs"]
    return vectors, {"receipt_sha256": sha(folder/"encoding_receipt.json"), "reused_diagonal": diagonal,
        "embedding_sha256": receipt["embeddings_sha256"], "truncated_programs": receipt["truncated_programs"],
        "programs": count, "encoding_seconds": receipt["encoding_seconds"]}


def mean_problems(quality, groups):
    return np.stack([quality[:, groups == i, :].mean(axis=1) for i in range(groups.max()+1)], axis=1)


def native_metrics(ranking, ids, relevant):
    positions = np.array([i+1 for i, g in enumerate(ranking) if ids[g] in relevant])
    assert len(positions) == len(relevant)
    top = positions[positions <= 8]
    return [float(np.sum(np.arange(1, len(top)+1)/top)/8), float(1/positions[0]), float(positions[0] <= 10)]


def compute_cell(corpus, model, data, downloads):
    ids, rel, qi, gi, auxiliary, problems, groups = data
    vectors, receipt = embeddings(corpus, model, downloads, len(qi)+len(gi))
    dense = vectors[qi] @ vectors[gi].T
    del vectors
    tie = np.array([g.encode() for g in ids["gallery"]])
    nq, ng = len(qi), len(gi)
    quality = np.empty((6, nq, 3))
    native = np.empty((6, nq, 3)) if corpus == "CodeContests" else np.empty((0,))
    saved = np.empty((6, nq, ng), dtype=np.int32)
    for i, q in enumerate(ids["query"]):
        ranked = rankings(dense[i].astype(np.float64), auxiliary["root"][i], auxiliary["nodes"][i], auxiliary["bm"][i], tie)
        for j, name in enumerate(METHODS):
            saved[j, i] = ranked[name]
            metrics = query_metrics(ranked[name], ids["gallery"], rel[q])
            quality[j, i] = [metrics[k] for k in KEYS]
            if corpus == "CodeContests":
                native[j, i] = native_metrics(ranked[name], ids["gallery"], rel[q])
    for name in ("root_100", "nodes_100"):
        np.testing.assert_array_equal(quality[METHODS.index(name), :, 2], quality[0, :, 2])
    by_problem = mean_problems(quality, groups)
    return saved, quality, native, by_problem, receipt, dense


def check_cell(corpus, model, data, dense, saved, quality, native, by_problem):
    ids, rel, _, _, auxiliary, problems, groups = data
    tie = np.array([g.encode() for g in ids["gallery"]])
    tie_order = np.argsort(tie, kind="stable")
    actual = np.empty_like(quality)
    for i, q in enumerate(ids["query"]):
        d, b = dense[i].astype(float), auxiliary["bm"][i]
        scale = lambda v: (v-v.min())/(v.max()-v.min()) if v.max() > v.min() else np.zeros_like(v)
        scores = {"D": d, "BM25": b, "D_BM25": .9*scale(d)+.1*scale(b),
            "nodes_100": .5*d+.5*auxiliary["nodes"][i], "root_full": .75*d+.25*auxiliary["root"][i]}
        scores["root_100"] = scores["root_full"]
        dense_order = tie_order[np.argsort(-d[tie_order], kind="stable")]
        positive = np.array([g in rel[q] for g in ids["gallery"]])
        ideal = np.sum(1/np.log2(np.arange(2, min(10, len(rel[q]))+2)))
        for j, name in enumerate(METHODS):
            candidates = np.sort(dense_order[:100]) if name.endswith("_100") else np.arange(len(tie))
            candidates = candidates[np.argsort(tie[candidates], kind="stable")]
            ranking = candidates[np.argsort(-scores[name][candidates], kind="stable")]
            if name.endswith("_100"):
                ranking = np.concatenate((ranking, dense_order[100:]))
            np.testing.assert_array_equal(ranking, saved[j, i])
            hits = positive[ranking]
            first = int(np.flatnonzero(hits)[0])+1
            actual[j, i] = [np.sum(hits[:10]/np.log2(np.arange(2, 12)))/ideal,
                1/first if first <= 10 else 0., hits[:100].sum()/positive.sum()]
            if corpus == "CodeContests":
                expected = [np.sum(np.cumsum(hits[:8])*hits[:8]/np.arange(1, 9))/8,
                            1/first, float(first <= 10)]
                np.testing.assert_allclose(native[j, i], expected, rtol=0, atol=1e-12)
    np.testing.assert_allclose(actual, quality, rtol=0, atol=1e-12)
    np.testing.assert_allclose(mean_problems(actual, groups), by_problem, rtol=0, atol=1e-12)
    if (corpus, model) == ("CodeContests", "CodeRank"):
        old = json.loads((CC/"result.json").read_text())["methods"]
        for j, name in enumerate(METHODS):
            np.testing.assert_allclose(native[j].mean(axis=0), [old[name][k] for k in ("MAP_at_8", "MRR", "Success_at_10")], rtol=0, atol=1e-12)
    if (corpus, model) == ("CPRet", "Qwen"):
        old = json.loads((CP/"result.json").read_text())["methods"]
        for j, name in enumerate(METHODS):
            np.testing.assert_allclose(quality[j], [[old[name]["per_query"][q][k] for k in KEYS] for q in ids["query"]], rtol=0, atol=1e-12)


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


def run(downloads, verify_only):
    freeze()
    cells, problem_scores = {}, {}
    for corpus in ("CodeContests", "CPRet"):
        data = cohort(corpus)
        for model in ("CodeRank", "Qwen"):
            name = corpus+"_"+model
            path = OUT/(name+"_result.json")
            if path.exists():
                result = json.loads(path.read_text())
                assert result["analysis_sha256"] == sha(OUT/"analysis_receipt.json")
                assert result["arrays_sha256"] == sha(OUT/(name+"_arrays.npz"))
                with np.load(OUT/(name+"_arrays.npz"), allow_pickle=False) as arrays:
                    saved, quality, native, by_problem = (arrays[k] for k in ("rankings", "quality", "native", "by_problem"))
                vectors, receipt = embeddings(corpus, model, downloads, len(data[2])+len(data[3]))
                dense = vectors[data[2]] @ vectors[data[3]].T
                assert receipt == result["encoding"]
                del vectors
            else:
                assert not verify_only
                saved, quality, native, by_problem, receipt, dense = compute_cell(corpus, model, data, downloads)
                np.savez_compressed(OUT/(name+"_arrays.npz"), rankings=saved, quality=quality, native=native, by_problem=by_problem)
                result = {"corpus": corpus, "model": model, "encoding": receipt,
                    "analysis_sha256": sha(OUT/"analysis_receipt.json"), "arrays_sha256": sha(OUT/(name+"_arrays.npz")),
                    "problems": data[5].tolist(), "methods": {n: dict(zip(KEYS, by_problem[j].mean(axis=0).tolist())) for j, n in enumerate(METHODS)}}
                if corpus == "CodeContests":
                    result["native_metrics"] = {n: dict(zip(("MAP_at_8", "MRR", "Success_at_10"), native[j].mean(axis=0).tolist())) for j, n in enumerate(METHODS)}
                write(path, result)
            check_cell(corpus, model, data, dense, saved, quality, native, by_problem)
            for j, method in enumerate(METHODS):
                np.testing.assert_allclose(by_problem[j].mean(axis=0), [result["methods"][method][k] for k in KEYS], rtol=0, atol=1e-12)
            cells[name], problem_scores[name] = result, by_problem
            print("PASS", name, json.dumps(result["methods"]), flush=True)
            del saved, dense, quality, native
    contrasts = crossed_contrasts(problem_scores)
    summary = {"status": "verified_crossed_diagnostic", "analysis_sha256": sha(OUT/"analysis_receipt.json"),
        "cell_result_sha256": {n: sha(OUT/(n+"_result.json")) for n in cells},
        "cells": {n: c["methods"] for n, c in cells.items()}, "contrasts": contrasts,
        "method_changed": False, "hyperparameters_selected": False, "java_test_accessed": False}
    path = OUT/"summary.json"
    if path.exists():
        assert json.loads(path.read_text()) == summary
    else:
        write(path, summary)
    receipt = {"status": "passed", "summary_sha256": sha(path), "verified_rankings": 41712,
        "original_diagonal_metrics_reproduced": True, "analysis_sha256": sha(OUT/"analysis_receipt.json")}
    if (OUT/"verification.json").exists():
        assert json.loads((OUT/"verification.json").read_text()) == receipt
    else:
        write(OUT/"verification.json", receipt)
    print(json.dumps(contrasts, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("freeze", "run", "check"))
    parser.add_argument("--downloads", type=Path, default=EXECUTION/"output_v1")
    args = parser.parse_args()
    if args.mode == "freeze":
        freeze()
    else:
        run(args.downloads, args.mode == "check")
