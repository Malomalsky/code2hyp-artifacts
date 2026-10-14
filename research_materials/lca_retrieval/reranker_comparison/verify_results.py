"""Offline replay of fixed ancestor masking and learned top-100 reranking."""
import os
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[key] = "1"
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.sparse import load_npz
from kernels import transform, group_counts, weighted_gallery, hybrid, order, rerank, query_metrics, summarize

ROOT = Path(__file__).resolve().parent
KEYS = ("MAP_at_8", "MRR", "Success_at_10")


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(name):
    return json.loads((ROOT / name).read_text())


def arrays(name):
    with np.load(ROOT / name, allow_pickle=False) as saved:
        return {key: saved[key] for key in saved.files}


def close(actual, expected):
    np.testing.assert_allclose(actual, expected, rtol=0, atol=2e-12)


def effect(a, b, indices):
    delta = a - b
    signs = np.rint(delta * 53760).astype(np.int64)
    return {"mean": float(delta.mean()),
            "percentile_95_interval": np.quantile(delta[indices].mean(axis=1), [.025, .975]).tolist(),
            "problem_counts": {"better": int((signs > 0).sum()), "worse": int((signs < 0).sum()),
                               "tied": int((signs == 0).sum())}}


def main():
    assert __debug__, "Run without -O"
    manifest = read("MANIFEST.json")
    for name, row in manifest.items():
        relative = Path(name)
        assert not relative.is_absolute() and ".." not in relative.parts
        path = ROOT / relative
        assert path.is_file() and not path.is_symlink()
        assert path.stat().st_size == row["bytes"] and sha(path) == row["sha256"], name
    expected = read("data/reranker_result.json")
    original = read("data/primary_result.json")
    ablation = read("data/ablation_result.json")
    final, previous = arrays("data/session2/reranker_checkpoint.npz"), arrays("data/session1/reranker_checkpoint.npz")
    receipt, start = read("data/session2/reranker_receipt.json"), read("data/session1/runtime_start.json")
    assert receipt["status"] == "completed" and receipt["completed_queries"] == 3200
    assert receipt["initial_completed_queries"] == len(previous["scores"]) == 2091
    assert not receipt["training_performed"] and not receipt["java_test_accessed"]
    assert not receipt["relevance_labels_available"]
    binding = sha(ROOT / "data/reranker_manifest.json")
    assert binding == receipt["binding"] == start["binding"] == str(final["binding"]) == str(previous["binding"])
    assert sha(ROOT / "data/session2/reranker_checkpoint.npz") == receipt["checkpoint_sha256"]
    for key in ("settings", "model_files", "gpu", "script_sha256", "packages"):
        assert receipt[key] == start[key]
    for key in ("scores", "lengths", "times"):
        assert final[key][:2091].dtype == previous[key].dtype
        assert final[key][:2091].shape == previous[key].shape
        assert final[key][:2091].tobytes() == previous[key].tobytes()
    assert final["scores"].shape == final["lengths"].shape == (3200, 100)
    assert final["times"].shape == (3200, 3) and np.isfinite(final["scores"]).all()
    # Large time sums differ by a few floating-point units across NumPy builds.
    np.testing.assert_allclose(final["times"].sum(axis=0), [receipt[k] for k in
          ("preprocessing_seconds", "inference_seconds", "query_processing_seconds")], rtol=0, atol=1e-7)
    assert int((final["lengths"] > 8192).sum()) == receipt["truncated_pairs"]

    rows = [json.loads(line) for line in (ROOT / "data/metadata.jsonl").read_text().splitlines()]
    assert len(rows) == 6400 and all("source" not in row for row in rows)
    qi = np.array([i for i, row in enumerate(rows) if row["role"] == "query"])
    gi = np.array([i for i, row in enumerate(rows) if row["role"] == "gallery"])
    assert len(qi) == len(gi) == 3200
    queries, gallery = [rows[i] for i in qi], [rows[i] for i in gi]
    vectors = np.load(ROOT / "data/embeddings.npy", allow_pickle=False)
    enc = read("data/encoding_receipt.json")
    assert sha(ROOT / "data/embeddings.npy") == enc["embeddings_sha256"]
    assert vectors.shape == (6400, 768) and vectors.dtype == np.float32
    assert np.isfinite(vectors).all()
    np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), 1., rtol=0, atol=1e-5)
    dense = vectors[qi] @ vectors[gi].T
    counts = load_npz(ROOT / "data/counts.npz")
    groups = np.load(ROOT / "data/group.npy", allow_pickle=False)
    masked = group_counts(counts, groups)
    close(counts.sum(axis=1), masked.sum(axis=1))
    features = [transform(counts, gi, .5, False), transform(masked, gi, .5, False)]
    lexical = (load_npz(ROOT / "data/query_lexical_counts.npz") @
               weighted_gallery(load_npz(ROOT / "data/gallery_lexical_counts.npz")).T).tocsr()
    saved = arrays("data/ablation_arrays.npz")
    paths = np.array([row["source_relpath"].encode() for row in gallery])
    labels = np.array([row["cluster_id"] for row in gallery])
    quality = {name: np.empty((3200, 3)) for name in ("D", "AST", "masked", "D_BM25", "Qwen_reranker")}
    digest = hashlib.sha256()
    for i, query in enumerate(queries):
        ranked = order(dense[i].astype(np.float64), paths)
        prefix = ranked[:100]
        np.testing.assert_array_equal(prefix, saved["candidates"][i])
        rankings = {"D": ranked, "D_BM25": order(hybrid(dense[i], lexical.getrow(i).toarray().ravel(), .1), paths),
                    "Qwen_reranker": rerank(ranked, final["scores"][i].astype(np.float64), paths)}
        for j, name in enumerate(("AST", "masked")):
            structural = (features[j][qi[i]] @ features[j][gi[prefix]].T).toarray().ravel()
            close(structural, saved["structural"][j, i])
            rankings[name] = rerank(ranked, .75*dense[i, prefix].astype(np.float64) + .25*structural, paths)
        for name, ranking in rankings.items():
            quality[name][i] = query_metrics(ranking, labels == query["cluster_id"])[0]
            positions = np.flatnonzero(labels[ranking] == query["cluster_id"]) + 1
            assert len(positions) == 8
            close(quality[name][i], [sum((j+1)/rank for j, rank in enumerate(positions) if rank <= 8)/8,
                                    1/positions[0], float(positions[0] <= 10)])
        close(quality["D"][i], saved["D"][i])
        close(quality["AST"][i], saved["retained"][i])
        close(quality["masked"][i], saved["masked"][i])
        candidate_scores = dict(zip(prefix, final["scores"][i]))
        independent = sorted(prefix, key=lambda p: (-float(candidate_scores[p]), paths[p])) + ranked[100:].tolist()
        np.testing.assert_array_equal(rankings["Qwen_reranker"], independent)
        digest.update(rankings["Qwen_reranker"].astype("<i4").tobytes())
    assert digest.hexdigest() == expected["rankings_sha256"]
    methods, problems = summarize(quality, queries)
    means = {name: np.array([row["problem_ap8"][p] for p in problems]) for name, row in methods.items()}
    for name, row in expected["methods"].items():
        close([methods[name][k] for k in KEYS], [row[k] for k in KEYS])
        close(means[name], [row["problem_ap8"][p] for p in problems])
    indices = np.random.Generator(np.random.PCG64(20261013)).integers(400, size=(20000, 400))
    for name, row in expected["contrasts"].items():
        a, b = name.split("_minus_")
        actual = effect(means[a], means[b], indices)
        close(actual["mean"], row["mean"])
        close(actual["percentile_95_interval"], row["percentile_95_interval"])
        assert actual["problem_counts"] == row["problem_counts"]
    actual = effect(means["AST"], means["masked"], indices)
    close(actual["percentile_95_interval"], ablation["retained_minus_masked"]["percentile_95_interval"])
    assert actual["problem_counts"] == ablation["retained_minus_masked"]["problem_counts"]
    indices = np.random.Generator(np.random.PCG64(20261010)).integers(400, size=(20000, 400))
    actual = effect(means["AST"], means["D_BM25"], indices)
    close(actual["percentile_95_interval"], original["primary_root100_minus_denseBM25"]["percentile_95_interval"])
    out = ROOT / "reproduced"
    out.mkdir(exist_ok=True)
    (out / "verification.json").write_text(json.dumps({"status": "passed", "manifest_sha256": sha(ROOT/"MANIFEST.json"),
        "rankings_replayed": 16000, "prefix_queries_preserved": 2091,
        "methods": methods, "query_processing_seconds": float(final["times"][:, 2].sum()),
        "neural_inference_replayed": False}, indent=2, sort_keys=True) + "\n")
    print("PASS: 16000 rankings, both ancestor representations, six additional contrasts, primary interval and unchanged resume prefix")


if __name__ == "__main__":
    main()
