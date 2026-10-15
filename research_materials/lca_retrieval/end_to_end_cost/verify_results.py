"""Recompute fixed retrieval quality and separate-session serving statistics."""

import json
from pathlib import Path
import sys
import numpy as np
from scipy.sparse import load_npz

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "source"))
from e2e_common_20261014 import sha, write, order, rerank, metrics, summary
from evaluate_e2e_20261014 import timing_records
from audit_e2e_partial_20261015 import verify_worker
from continue_e2e_20261015 import cached_ranking, MANIFEST_SHA
from frozen_helpers import hybrid, normalized, weighted_gallery


def read(path):
    return json.loads((ROOT / path).read_text())


def verify_manifest():
    manifest = read("MANIFEST.json")
    for name, item in manifest.items():
        path = Path(name)
        assert not path.is_absolute() and ".." not in path.parts
        target = ROOT / path
        assert not target.is_symlink() and target.stat().st_size == item["bytes"]
        assert sha(target) == item["sha256"], name
    return sha(ROOT / "MANIFEST.json")


def run():
    binding = verify_manifest()
    data = ROOT / "data"
    assert sha(data / "input/e2e_manifest.json") == MANIFEST_SHA
    original = read("data/input/e2e_manifest.json")
    for name, digest in original["files"].items():
        if name == "programs.json":
            continue
        path = data / "input" / name
        if not path.exists():
            path = ROOT / "source" / name
        assert sha(path) == digest, name
    design = read("data/input/e2e_design.json")
    programs = read("data/input/program_metadata.json")
    assert all("source" not in p for role in programs.values() for p in role)
    tie = np.array([p["id"].encode() for p in programs["gallery"]])
    labels = np.array(design["gallery_problems"])
    with np.load(data / "input/reference_vectors.npz", allow_pickle=False) as saved:
        dense = saved["query"] @ saved["gallery"].T
    counts = load_npz(data / "input/reference_ast_counts.npz")
    counts.data **= .5
    features = normalized(counts)
    roles = read("data/input/source_roles.json")
    qlex = load_npz(data / "input/lexical_query_counts.npz")
    glex = load_npz(data / "input/lexical_gallery_counts.npz")
    lex_index = weighted_gallery(glex).T.tocsr()
    np.testing.assert_allclose((lex_index-load_npz(data / "session_A/BM25/lexical_index.npz")).data, 0, rtol=0, atol=1e-12)
    lexical = (qlex @ lex_index).toarray()
    candidates = np.load(data / "input/reference_candidates.npy", allow_pickle=False)
    pair = np.load(data / "input/reference_reranker_scores.npy", allow_pickle=False)
    result = {"status": "passed", "scope": "approved_two_session_cost_study",
              "manifest_sha256": binding, "original_single_session_complete": False,
              "sessions": {}, "neural_inference_repeated": False}
    new = read("data/session_B/continuation_receipt.json")
    assert new["status"] == "complete_approved_two_session_continuation"
    assert new["methods"] == ["D_AST", "D"] and new["binding"] == MANIFEST_SHA
    assert not new["original_six_method_single_session_complete"]
    assert sha(data / "session_B/submitted_launcher.py") == new["launcher_sha256"]
    for method in new["methods"]:
        assert sha(data / f"session_B/{method}/complete.json") == new["worker_receipts"][method]
    assert sha(data / "session_B/reranker_quality_replay/receipt.json") == new["reranker_replay_receipt_sha256"]
    expected_models = read("data/input/expected_model_files.json")
    keys = {"coderank": "coderank_settings", "qwen_embedding": "qwen_embedding_settings",
            "qwen_reranker": "qwen_reranker_settings"}
    total_timed, total_quality = 0, 0
    for session, methods in (("A", ("BM25", "D_BM25", "D_AST", "QwenEmbedding", "D_QwenReranker")),
                             ("B", ("D_AST", "D"))):
        folder = data / f"session_{session}"
        host = read(f"data/session_{session}/host.json")
        assert sha(folder / "e2e_manifest.json") == MANIFEST_SHA
        assert sha(folder / "e2e_design.json") == sha(data / "input/e2e_design.json")
        result["sessions"][session] = {"host": host, "methods": {}}
        for method in methods:
            worker = folder / method
            if method != "D_QwenReranker":
                verify_worker(worker, method, host["host_id"], MANIFEST_SHA)
            started = json.loads((worker / "started.json").read_text())
            assert started["binding"] == MANIFEST_SHA and started["host_id"] == host["host_id"]
            assert started["method"] == method and started["threads"] == 1
            index = json.loads((worker / "index_receipt.json").read_text())
            memory = json.loads((worker / "memory.json").read_text())
            for name, model in index["models"].items():
                assert model["model_files"] == expected_models[name]
                assert model["settings"] == design[keys[name]]
            times, live = timing_records(worker, method, host["host_id"], design, programs)
            values, top = np.empty((3200, 3)), np.empty((3200, 100), dtype=np.int32)
            similarities = dense
            if method == "QwenEmbedding":
                q = np.load(worker / "quality_query_vectors.npy", allow_pickle=False)
                g = np.load(worker / "gallery_vectors.npy", allow_pickle=False)
                similarities = q @ g.T
            for i in range(3200):
                d = similarities[i].astype(np.float64)
                ranked = order(d, tie)
                if method in ("BM25", "D_BM25"):
                    ranked = order(lexical[i] if method == "BM25" else hybrid(d, lexical[i], .10), tie)
                elif method == "D_AST":
                    prefix = ranked[:100]
                    f = (features[roles["query"][i]] @ features[np.array(roles["gallery"])[prefix]].T).toarray().ravel()
                    ranked = rerank(ranked, .75*d[prefix]+.25*f, tie)
                elif method == "D_QwenReranker":
                    ranked = cached_ranking(d, candidates[i], pair[i], tie)
                values[i] = metrics(ranked, labels == design["query_problems"][i])
                top[i] = ranked[:100]
            quality_file = worker / "full_quality.npz"
            if method == "D_QwenReranker":
                quality_file = data / "session_B/reranker_quality_replay/full_quality.npz"
                receipt = read("data/session_B/reranker_quality_replay/receipt.json")
                assert receipt["queries"] == 3200 and not receipt["neural_inference_performed"]
                assert receipt["binding"] == MANIFEST_SHA and receipt["host_id"] == new["host"]["host_id"]
                assert sha(quality_file) == receipt["files"]["full_quality.npz"]
            with np.load(quality_file, allow_pickle=False) as saved:
                np.testing.assert_allclose(values, saved["metrics"], rtol=0, atol=1e-12)
                changed = int(np.any(top != saved["top100"], axis=1).sum())
            delta = live - values[np.array(design["selected_indices"])][None, :, :]
            latency = summary(times)
            startup = index["ready_startup_excluding_network_seconds"]
            result["sessions"][session]["methods"][method] = {
                "quality": dict(zip(("MAP_at_8", "MRR", "Success_at_10"), values.mean(axis=0).tolist())),
                "latency": latency, "index": index, "memory": memory,
                "local_vs_saved_top100_changed_queries": changed,
                "live_400_query_quality_by_pass": live.mean(axis=1).tolist(),
                "live_vs_reference_changed_queries_by_pass": np.any(np.abs(delta)>1e-12, axis=2).sum(axis=1).tolist(),
                "live_vs_reference_mean_difference_by_pass": delta.mean(axis=1).tolist(),
                "projected_workload_seconds": {str(n): startup+n*latency["mean_seconds"] for n in (1,100,1000,10000)}}
            total_timed += times.size
            total_quality += len(values)
    assert result["sessions"]["A"]["host"]["host_id"] != result["sessions"]["B"]["host"]["host_id"]
    result.update(timing_records_checked=total_timed, quality_query_method_checks=total_quality,
                  cross_session_timings_pooled=False)
    assert total_timed == 8400 and total_quality == 22400
    (ROOT / "reproduced").mkdir(exist_ok=True)
    write(ROOT / "reproduced/result.json", result)
    print("PASS: 22400 quality evaluations, 8400 raw timing records; sessions kept separate")
    return result


if __name__ == "__main__":
    run()
