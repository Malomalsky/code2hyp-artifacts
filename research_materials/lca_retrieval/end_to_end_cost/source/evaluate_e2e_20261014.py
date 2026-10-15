"""Require complete paired same-host timings and independently replay quality."""

import argparse
import json
from pathlib import Path
import numpy as np
from scipy.sparse import load_npz
from e2e_common_20261014 import METHODS, now, sha, write, summary, metrics


def require_complete(out, root):
    receipt = json.loads((out / "complete_receipt.json").read_text())
    assert receipt["status"] == "complete" and receipt["methods"] == list(METHODS)
    assert receipt["binding"] == sha(root / "e2e_manifest.json")
    host_id = receipt["host"]["host_id"]
    for method in METHODS:
        worker = out / method
        assert sha(worker / "complete.json") == receipt["worker_receipts"][method]
        saved = json.loads((worker / "complete.json").read_text())
        assert saved["host_id"] == host_id and saved["binding"] == receipt["binding"]
        assert saved["requests"] == 1200 and saved["quality_queries"] == 3200
        assert not saved["training_performed"] and not saved["java_test_accessed"]
        for name, digest in saved["files"].items():
            assert Path(name).name == name and sha(worker / name) == digest
    return receipt


def timing_records(folder, method, host_id, design, programs):
    times = np.full((3, 400), np.nan)
    live_quality = np.full((3, 400, 3), np.nan)
    gallery_labels = np.array(design["gallery_problems"])
    seen = set()
    for line in (folder / "requests.jsonl").read_text().splitlines():
        row = json.loads(line)
        key = row["repeat"], row["slot"]
        assert key not in seen and row["host_id"] == host_id and row["method"] == method
        assert 0 <= key[0] < 3 and 0 <= key[1] < 400
        seen.add(key)
        i = design["selected_indices"][row["slot"]]
        assert row["query_index"] == i and row["query_id"] == programs["query"][i]["id"]
        ranking = np.array(row["ranking"])
        np.testing.assert_array_equal(np.sort(ranking), np.arange(3200))
        components = np.array(row["seconds"])
        assert components.shape == (8,) and np.isfinite(components).all() and (components >= 0).all()
        assert components[-1] >= components[:-1].sum() - 1e-7
        times[key] = components[-1]
        live_quality[key] = metrics(ranking, gallery_labels == design["query_problems"][i])
    assert seen == {(r, q) for r in range(3) for q in range(400)}
    return times, live_quality


def evaluate_methods(out, root, receipt, methods, strict_rankings=True):
    design = json.loads((root / "e2e_design.json").read_text())
    programs = json.loads((root / "programs.json").read_text())
    tie_order = np.argsort(np.array([p["id"].encode() for p in programs["gallery"]]))
    tie_position = tie_order.argsort()
    gallery_labels = np.array(design["gallery_problems"])
    with np.load(root / "reference_vectors.npz", allow_pickle=False) as vectors:
        dense = vectors["query"] @ vectors["gallery"].T
    structural = load_npz(root / "reference_ast_features.npz")
    roles = json.loads((root / "source_roles.json").read_text())
    lexical_reference = None
    import sys
    sys.path.insert(0, str(root))
    from frozen_helpers import hybrid, vectorize, lexical_token_stream
    from collections import Counter
    result = {"status": "complete_same_host_cost_study", "completed_utc": now(),
              "receipt_sha256": sha(out / "complete_receipt.json") if (out / "complete_receipt.json").exists() else None,
              "host": receipt["host"], "methods": {}}
    for method in methods:
        folder = out / method
        report = json.loads((folder / "complete.json").read_text())
        times, live_quality = timing_records(folder, method, receipt["host"]["host_id"], design, programs)
        with np.load(folder / "full_quality.npz", allow_pickle=False) as saved:
            expected_quality, expected_top = saved["metrics"], saved["top100"]
        assert expected_quality.shape == (3200, 3) and expected_top.shape == (3200, 100)
        actual = np.empty_like(expected_quality)
        top_changes, examples = 0, []
        if method == "QwenEmbedding":
            q = np.load(folder / "quality_query_vectors.npy", allow_pickle=False)
            g = np.load(folder / "gallery_vectors.npy", allow_pickle=False)
            np.testing.assert_allclose(np.linalg.norm(q, axis=1), 1., atol=1e-5)
            np.testing.assert_allclose(np.linalg.norm(g, axis=1), 1., atol=1e-5)
            values = q @ g.T
        else:
            values = dense
        if method in ("BM25", "D_BM25"):
            vocab = {k: i for i, k in enumerate(json.loads((root / "lexical_vocabulary.json").read_text()))}
            index = load_npz(folder / "lexical_index.npz")
            counts = vectorize([Counter(lexical_token_stream(p["source"])) for p in programs["query"]], vocab)
            counts.sort_indices()
            lexical_reference = (counts @ index).tocsr()
        if method == "D_QwenReranker":
            pair = np.load(root / "reference_reranker_scores.npy", allow_pickle=False)
            candidates = np.load(root / "reference_candidates.npy", allow_pickle=False)
        for i in range(3200):
            scores = values[i].astype(np.float64)
            ranked = tie_order[np.argsort(-scores[tie_order], kind="stable")]
            if method in ("BM25", "D_BM25"):
                b = lexical_reference.getrow(i).toarray().ravel()
                scores = b if method == "BM25" else hybrid(scores, b, .10)
                ranked = tie_order[np.argsort(-scores[tie_order], kind="stable")]
            elif method == "D_AST":
                prefix = ranked[:100]
                f = (structural[roles["query"][i]] @ structural[roles["gallery"]][prefix].T).toarray().ravel()
                score = .75*scores[prefix]+.25*f
                ranked = np.concatenate((prefix[np.lexsort((tie_position[prefix], -score))], ranked[100:]))
            elif method == "D_QwenReranker":
                prefix = ranked[:100]
                np.testing.assert_array_equal(prefix, candidates[i])
                ranked = np.concatenate((prefix[np.lexsort((tie_position[prefix], -pair[i]))], ranked[100:]))
            actual[i] = metrics(ranked, gallery_labels == design["query_problems"][i])
            if not np.array_equal(ranked[:100], expected_top[i]):
                top_changes += 1
                if len(examples) < 3:
                    positions = np.flatnonzero(ranked[:100] != expected_top[i])
                    examples.append({"query_index": i, "positions": positions.tolist(),
                                     "local_ids": ranked[positions].tolist(),
                                     "remote_ids": expected_top[i, positions].tolist()})
                if strict_rankings:
                    np.testing.assert_array_equal(ranked[:100], expected_top[i])
        if strict_rankings:
            np.testing.assert_allclose(actual, expected_quality, rtol=0, atol=1e-12)
        latency = summary(times)
        startup = report["index"]["ready_startup_excluding_network_seconds"]
        delta = live_quality - actual[np.array(design["selected_indices"])][None, :, :]
        result["methods"][method] = {"quality": dict(zip(("MAP_at_8", "MRR", "Success_at_10"), actual.mean(axis=0).tolist())),
              "latency": latency, "index": report["index"], "memory": report["memory"],
              "live_400_query_quality_by_pass": live_quality.mean(axis=1).tolist(),
              "live_vs_reference_maximum_metric_difference": float(np.max(np.abs(delta))),
              "live_vs_reference_metric_changed_queries_by_pass": np.any(np.abs(delta) > 1e-12, axis=2).sum(axis=1).tolist(),
              "live_vs_reference_mean_metric_difference_by_pass": delta.mean(axis=1).tolist(),
              "local_vs_remote_top100_changed_queries": top_changes,
              "local_vs_remote_top100_examples": examples,
              "local_vs_remote_metric_changed_queries": int(np.any(np.abs(actual-expected_quality) > 1e-12, axis=1).sum()),
              "local_vs_remote_maximum_metric_difference": float(np.max(np.abs(actual-expected_quality))),
              "remote_saved_quality": dict(zip(("MAP_at_8", "MRR", "Success_at_10"), expected_quality.mean(axis=0).tolist())),
              "projected_workload_seconds_excluding_network": {str(n): startup+n*latency["mean_seconds"] for n in (1, 100, 1000, 10000)}}
    result["interpretation_limits"] = ["Inspected cohort; no new confirmation of novelty", "Latency not concurrent-service throughput",
           "Quality of previous five methods replays complete neural scores; live serving checked on 400 queries",
           "One host and gallery size; no energy, financial saving or industrial utility measurement"]
    return result


def run(out, root):
    receipt = require_complete(out, root)
    result = evaluate_methods(out, root, receipt, METHODS)
    write(out / "evaluated_result.json", result)
    print(json.dumps({m: {"quality": v["quality"], "latency": v["latency"]} for m, v in result["methods"].items()}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    args = parser.parse_args()
    run(args.output, args.input)
