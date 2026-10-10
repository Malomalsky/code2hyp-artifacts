"""Fixed top-100 comparisons from checked remote embeddings; no calibration."""

from top100_fresh_20261010 import OUT, TARGET, PROTOCOL, prepare, previous
import coderank_kaggle_20261010 as encoder
from rank_fusion_followup import metadata, sha, write, reviewer
from bm25_20261010 import weighted_gallery, hybrid
from two_stage_20261010 import order, rerank, query_metrics
from pathlib import Path
import argparse
import json
import time
import numpy as np
from scipy.sparse import load_npz

METHODS = ("D", "BM25", "D_BM25", "nodes_full", "nodes_100",
           "root_full", "root_100", "idf_full", "idf_100")


def rankings(dense, structural, bm, tie):
    dense_order = order(dense, tie)
    result = {"D": dense_order, "BM25": order(bm, tie), "D_BM25": order(hybrid(dense, bm, .10), tie)}
    prefix = dense_order[:100]
    for name, alpha in (("nodes", .50), ("root", .25), ("idf", .20)):
        fused = (1-alpha)*dense + alpha*structural[name]
        result[name+"_full"] = order(fused, tie)
        result[name+"_100"] = rerank(dense_order, fused[prefix], tie)
    return result


def inputs(encoding):
    prepare()
    for name in ("selection_seal.json", "features.json", "bm25_features.json"):
        for filename, digest in json.loads((OUT/name).read_text())["sha256"].items():
            assert sha(OUT/filename) == digest
    execution = json.loads((encoding/"top100_execution_receipt.json").read_text())
    assert sha(encoding/"top100_execution_receipt.json") == sha(OUT/"top100_execution_receipt.json")
    assert execution["design_sha256"] == sha(OUT/"design_receipt.json")
    assert execution["protocol_sha256"] == sha(PROTOCOL)
    assert execution["machine_shape"] == "NvidiaTeslaT4"
    assert sha(encoding/"submitted_launcher.py") == execution["launcher_sha256"]
    for name in ("top100_fresh_encode_20261010.py", "coderank_kaggle_20261010.py"):
        assert execution["payload_sha256"][name] == sha(OUT.parent.parent/name)
    receipt = json.loads((encoding/"encoding_receipt.json").read_text())
    assert sha(OUT/"selected_programs.jsonl") == execution["source_sha256"]
    expected = {**encoder.SETTINGS, "source_sha256": execution["source_sha256"],
        "external_design_sha256": execution["design_sha256"],
        "execution_addendum_sha256": execution["protocol_sha256"], "study": "top100_fresh_20261010"}
    assert receipt["settings"] == expected
    assert receipt["settings_sha256"] == encoder.hashlib.sha256(json.dumps(expected, sort_keys=True).encode()).hexdigest()
    assert receipt["script_sha256"] == sha(OUT.parent.parent/"coderank_kaggle_20261010.py")
    assert receipt["programs"] == TARGET*16 and receipt["gpu"] == "Tesla T4"
    assert not receipt["training_performed"] and not receipt["java_test_accessed"]
    old = json.loads((previous.ROOT/"followup_results/coderank_colab_20261007/coderank_output/encoding_receipt.json").read_text())
    assert receipt["model_files"] == old["model_files"]
    for name in ("embeddings", "token_lengths"):
        assert sha(encoding/(name+".npy")) == receipt[name+"_sha256"]
    vectors = np.load(encoding/"embeddings.npy", allow_pickle=False)
    assert vectors.shape == (16*TARGET, 768) and vectors.dtype == np.float32 and np.isfinite(vectors).all()
    np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), 1., atol=1e-5)
    lengths = np.load(encoding/"token_lengths.npy", allow_pickle=False)
    assert lengths.shape == (16*TARGET,) and lengths.dtype.kind in "iu" and (lengths > 0).all()
    rows, queries, gallery = metadata(OUT)
    qi = np.array([i for i, r in enumerate(rows) if r["role"] == "query"])
    gi = np.array([i for i, r in enumerate(rows) if r["role"] == "gallery"])
    assert len(qi) == len(gi) == TARGET*8
    structure = {n: load_npz(OUT/(source+"_features.npz")) for n, source in
                 (("nodes", "nodes"), ("root", "ret_root"), ("idf", "ret_root_idf"))}
    bm = load_npz(OUT/"query_lexical_counts.npz") @ weighted_gallery(load_npz(OUT/"gallery_lexical_counts.npz")).T
    dense = vectors[qi] @ vectors[gi].T
    auxiliary = {n: (v[qi] @ v[gi].T).toarray() for n, v in structure.items()}
    return queries, gallery, dense, auxiliary, bm.tocsr()


def run(encoding):
    assert not (OUT/"result.json").exists(), "Completed study: use --check"
    started = time.perf_counter()
    queries, gallery, dense, auxiliary, bm = inputs(encoding)
    tie = np.array([r["source_relpath"].encode() for r in gallery])
    labels = np.array([r["cluster_id"] for r in gallery])
    problems, groups = np.unique([r["cluster_id"] for r in queries], return_inverse=True)
    quality = np.empty((len(METHODS), len(queries), 3))
    top10 = np.empty((len(METHODS), len(queries), 10), dtype=np.int32)
    first = np.empty((len(METHODS), len(queries)), dtype=np.int32)
    recall = np.empty(len(queries))
    for i, query in enumerate(queries):
        relevant = labels == query["cluster_id"]
        assert relevant.sum() == 8
        ranked = rankings(dense[i].astype(np.float64), {n: v[i] for n, v in auxiliary.items()}, bm.getrow(i).toarray().ravel(), tie)
        for mi, name in enumerate(METHODS):
            quality[mi, i], first[mi, i] = query_metrics(ranked[name], relevant)
            top10[mi, i] = ranked[name][:10]
        recall[i] = relevant[ranked["D"][:100]].sum()/8
    np.savez_compressed(OUT/"quality.npz", metrics=quality, top10=top10, first=first, groups=groups, recall=recall)
    means = np.stack([np.bincount(groups, weights=v[:, 0])/8 for v in quality])
    methods = {name: {**dict(zip(("MAP_at_8", "MRR", "Success_at_10"), quality[mi].mean(axis=0).tolist())),
        "problem_ap8": dict(zip(problems, means[mi].tolist()))} for mi, name in enumerate(METHODS)}
    delta = means[METHODS.index("root_100")] - means[METHODS.index("D_BM25")]
    indices = np.random.Generator(np.random.PCG64(20261010)).integers(TARGET, size=(20000, TARGET))
    primary = reviewer.contrast(delta, indices)
    write(OUT/"result.json", {"completed_utc": previous.now(), "status": "completed_prospective_top100_validation",
        "design_sha256": sha(OUT/"design_receipt.json"), "selection_sha256": sha(OUT/"selection_seal.json"),
        "encoding_receipt_sha256": sha(encoding/"encoding_receipt.json"), "quality_sha256": sha(OUT/"quality.npz"),
        "methods": methods, "primary_root100_minus_denseBM25": primary,
        "primary_supported": bool(primary["percentile_95_interval"][0] > 0),
        "primary_problem_counts": {"better": int((delta > 0).sum()), "worse": int((delta < 0).sum()), "tied": int((delta == 0).sum())},
        "dense_candidate_recall_100": float(recall.mean()), "queries_without_relevant_candidate_fraction": float((recall == 0).mean()),
        "descriptive_MAP_deltas": {base: {n: methods[n]["MAP_at_8"]-methods[base]["MAP_at_8"] for n in METHODS}
            for base in ("D", "nodes_100")},
        "top100_minus_full_MAP": {n: methods[n+"_100"]["MAP_at_8"]-methods[n+"_full"]["MAP_at_8"] for n in ("nodes", "root", "idf")},
        "evaluation_seconds_including_checks": time.perf_counter()-started,
        "java_test_accessed": False, "parameter_selection_performed": False})
    print(json.dumps({"methods": {n: {k: v for k, v in m.items() if k != "problem_ap8"} for n, m in methods.items()},
        "primary": primary}, indent=2), flush=True)


def verify(encoding):
    queries, gallery, dense, auxiliary, bm = inputs(encoding)
    result = json.loads((OUT/"result.json").read_text())
    assert result["design_sha256"] == sha(OUT/"design_receipt.json")
    assert result["selection_sha256"] == sha(OUT/"selection_seal.json")
    assert result["encoding_receipt_sha256"] == sha(encoding/"encoding_receipt.json")
    assert result["quality_sha256"] == sha(OUT/"quality.npz")
    with np.load(OUT/"quality.npz", allow_pickle=False) as stored:
        quality, tops, first, groups, recall = (stored[n] for n in ("metrics", "top10", "first", "groups", "recall"))
    tie_order = np.argsort(np.array([r["source_relpath"].encode() for r in gallery]))
    labels = np.array([r["cluster_id"] for r in gallery])
    problems, expected_groups = np.unique([r["cluster_id"] for r in queries], return_inverse=True)
    np.testing.assert_array_equal(groups, expected_groups)
    np.testing.assert_array_equal(np.bincount(groups), np.full(TARGET, 8))
    for i, query in enumerate(queries):
        d = dense[i].astype(np.float64)
        b = bm.getrow(i).toarray().ravel()
        scores = {"D": d, "BM25": b, "D_BM25": hybrid(d, b, .10)}
        dense_order = tie_order[np.argsort(-d[tie_order], kind="stable")]
        for mi, name in enumerate(METHODS):
            if name in scores:
                ranking = tie_order[np.argsort(-scores[name][tie_order], kind="stable")]
            else:
                feature, mode = name.split("_")
                alpha = {"nodes": .50, "root": .25, "idf": .20}[feature]
                s = (1-alpha)*d + alpha*auxiliary[feature][i]
                if mode == "full":
                    ranking = tie_order[np.argsort(-s[tie_order], kind="stable")]
                else:
                    ids = dense_order[:100]
                    ids = ids[np.argsort(np.array([gallery[j]["source_relpath"].encode() for j in ids]))]
                    ranking = np.concatenate((ids[np.argsort(-s[ids], kind="stable")], dense_order[100:]))
            positions = np.flatnonzero(labels[ranking] == query["cluster_id"])+1
            at8 = positions[positions <= 8]
            actual = [sum((j+1)/p for j,p in enumerate(at8))/8, 1/positions[0], float(positions[0] <= 10)]
            np.testing.assert_array_equal(ranking[:10], tops[mi, i])
            np.testing.assert_allclose(actual, quality[mi, i], atol=1e-12, rtol=0)
            assert positions[0] == first[mi, i]
        assert recall[i] == (labels[dense_order[:100]] == query["cluster_id"]).sum()/8
    for mi, name in enumerate(METHODS):
        np.testing.assert_allclose(quality[mi].mean(axis=0), [result["methods"][name][k] for k in
            ("MAP_at_8", "MRR", "Success_at_10")], rtol=0, atol=1e-12)
        means = np.bincount(groups, weights=quality[mi, :, 0])/8
        assert dict(zip(problems, means.tolist())) == result["methods"][name]["problem_ap8"]
    p = np.bincount(groups, weights=quality[METHODS.index("root_100"), :, 0])/8
    b = np.bincount(groups, weights=quality[METHODS.index("D_BM25"), :, 0])/8
    indices = np.random.Generator(np.random.PCG64(20261010)).integers(TARGET, size=(20000, TARGET))
    assert reviewer.contrast(p-b, indices) == result["primary_root100_minus_denseBM25"]
    assert result["primary_supported"] == (result["primary_root100_minus_denseBM25"]["percentile_95_interval"][0] > 0)
    assert result["primary_problem_counts"] == {"better": int((p > b).sum()), "worse": int((p < b).sum()), "tied": int((p == b).sum())}
    assert result["dense_candidate_recall_100"] == float(recall.mean())
    assert result["queries_without_relevant_candidate_fraction"] == float((recall == 0).mean())
    receipt = {"status": "passed", "result_sha256": sha(OUT/"result.json"), "query_method_rankings": len(queries)*len(METHODS),
        "primary_bootstrap_reproduced": True, "java_test_accessed": False}
    if (OUT/"verification.json").exists():
        assert json.loads((OUT/"verification.json").read_text()) == receipt
    else:
        write(OUT/"verification.json", receipt)
    print("PASS: all rankings, metrics and primary contrast", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoding", type=Path, required=True)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if not args.check:
        run(args.encoding)
    verify(args.encoding)
