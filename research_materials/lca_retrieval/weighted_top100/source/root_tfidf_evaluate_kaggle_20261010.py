"""Evaluate frozen new-problem hybrids from remote embeddings, without tuning."""

from root_tfidf_external_20261010 import OUT, ROOT, now
from rank_fusion_followup import metadata, reviewer, ranks, sha, write
from followup_baselines import summarize_problem_macro_retrieval, torch
from dataclasses import asdict
from pathlib import Path
import argparse
import json
import time
import numpy as np
from scipy.sparse import load_npz

HYBRIDS = {"candidate": ("ret_root_idf", .20), "previous_tfidf": ("ret_raw_idf", .15),
           "matched_mask": ("mask_root_idf", .20), "simpler_root": ("ret_root", .25),
           "node_counts": ("nodes", .50), "fixed_weight_control": ("ret_root_idf", .15)}
REFERENCES = ("previous_tfidf", "D", "matched_mask")


def mixed(dense, auxiliary, tie):
    values = {"D": dense, **{n: (1 - a) * dense + a * auxiliary[c] for n, (c, a) in HYBRIDS.items()}}
    values["lexical_dense_rrf"] = 1 / (60 + ranks(-dense, tie)) + 1 / (60 + ranks(-auxiliary["lexical"], tie))
    values.update({"standalone_" + n: v for n, v in auxiliary.items()})
    return values


def load_inputs(encoding):
    design = json.loads((OUT / "design_receipt.json").read_text())
    for path, expected in design["bindings"].items():
        assert sha(Path(path)) == expected, path
    for receipt in ("selection_seal.json", "features.json"):
        for name, expected in json.loads((OUT / receipt).read_text())["sha256"].items():
            assert sha(OUT / name) == expected, name
    receipt = json.loads((encoding / "encoding_receipt.json").read_text())
    settings = receipt["settings"]
    import coderank_kaggle_20261010 as coderank_colab
    for key, expected in coderank_colab.SETTINGS.items():
        assert settings[key] == (sha(OUT / "selected_programs.jsonl") if key == "source_sha256" else expected), key
    assert settings["external_design_sha256"] == sha(OUT / "design_receipt.json")
    assert settings["execution_addendum_sha256"] == sha(ROOT / "root_tfidf_kaggle_addendum_20261010.md")
    assert receipt["script_sha256"] == sha(ROOT / "coderank_kaggle_20261010.py")
    assert receipt["programs"] == 10400 and not receipt["training_performed"] and not receipt["java_test_accessed"]
    old_receipt = json.loads((ROOT / "followup_results/coderank_colab_20261007/coderank_output/encoding_receipt.json").read_text())
    assert receipt["model_files"] == old_receipt["model_files"], "Model files changed"
    for name in ("embeddings", "token_lengths"):
        assert sha(encoding / (name + ".npy")) == receipt[name + "_sha256"]
    vectors = np.load(encoding / "embeddings.npy", allow_pickle=False)
    assert vectors.shape == (10400, 768) and vectors.dtype == np.float32 and np.isfinite(vectors).all()
    np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), 1., atol=1e-5)
    rows, queries, gallery = metadata(OUT)
    assert len(rows) == 10400 and len(queries) == len(gallery) == 5200
    auxiliary = {name: load_npz(OUT / (name + "_features.npz")) for name in
                 ("ret_root_idf", "ret_raw_idf", "mask_root_idf", "ret_root", "nodes", "lexical")}
    for matrix in auxiliary.values():
        assert matrix.shape[0] == len(rows) and np.isfinite(matrix.data).all()
    return rows, queries, gallery, vectors, auxiliary


def run(encoding, check):
    rows, queries, gallery, vectors, features = load_inputs(encoding)
    folder = OUT / "evaluation_kaggle"
    receipt = OUT / "kaggle_evaluation_code_receipt.json"
    binding = {str(p): sha(p) for p in (Path(__file__), ROOT / "test_root_tfidf_kaggle_20261010.py",
               OUT / "design_receipt.json", ROOT / "coderank_evaluate.py")}
    assert json.loads(receipt.read_text())["bindings"] == binding, "Freeze evaluator before remote encoding"
    if check:
        result = json.loads((folder / "result.json").read_text())
        for name, method in result["methods"].items():
            assert sha(folder / (name + ".npy")) == method["negative_score_sha256"]
            reviewer.metrics(folder / (name + ".npy"), rows, method["metrics"])
        assert result["evaluation_code_receipt_sha256"] == sha(receipt)
        print("PASS: every saved ranking and metric independently verified")
        return
    assert not folder.exists(), "Do not overwrite an evaluation; use --check"
    folder.mkdir()
    qi = np.array([i for i, r in enumerate(rows) if r["role"] == "query"])
    gi = np.array([i for i, r in enumerate(rows) if r["role"] == "gallery"])
    tie = np.array([r["source_relpath"].encode() for r in gallery])
    names = ["D", *HYBRIDS, "lexical_dense_rrf", *("standalone_" + n for n in features)]
    matrices = {n: np.lib.format.open_memmap(folder / (n + ".npy"), mode="w+", dtype=np.float64,
                shape=(len(qi), len(gi))) for n in names}
    started = time.perf_counter()
    dense = vectors[qi] @ vectors[gi].T
    dense_seconds = time.perf_counter() - started
    started = time.perf_counter()
    auxiliary = {name: (v[qi] @ v[gi].T).toarray() for name, v in features.items()}
    auxiliary_seconds = time.perf_counter() - started
    started = time.perf_counter()
    for i in range(len(qi)):
        scores = mixed(dense[i].astype(np.float64), {n: v[i] for n, v in auxiliary.items()}, tie)
        for name, values in scores.items():
            assert np.isfinite(values).all()
            matrices[name][i] = -values
    fusion_seconds = time.perf_counter() - started
    del dense, auxiliary
    methods = {}
    for name, matrix in matrices.items():
        matrix.flush()
        metrics = asdict(summarize_problem_macro_retrieval(torch.from_numpy(matrix),
            query_ids=[r["source_relpath"] for r in queries], query_cluster_ids=[r["cluster_id"] for r in queries],
            gallery_ids=[r["source_relpath"] for r in gallery], gallery_cluster_ids=[r["cluster_id"] for r in gallery]))
        reviewer.metrics(folder / (name + ".npy"), rows, metrics)
        methods[name] = {"metrics": metrics, "negative_score_sha256": sha(folder / (name + ".npy"))}
    indices = np.random.Generator(np.random.PCG64(20261010)).integers(650, size=(20000, 650))
    problems = sorted(methods["candidate"]["metrics"]["task_scores"])
    values = {n: np.array([m["metrics"]["task_scores"][p] for p in problems]) for n, m in methods.items()}
    contrasts = {}
    for name in REFERENCES:
        delta = values["candidate"] - values[name]
        contrasts[name] = {**reviewer.contrast(delta, indices), "bonferroni_three_interval":
            np.quantile(delta[indices].mean(axis=1), [.05/6, 1-.05/6]).tolist(),
            "problem_differences": dict(zip(problems, delta.tolist()))}
    write(folder / "result.json", {"completed_utc": now(), "status": "completed_new_problem_replication",
        "evaluation_code_receipt_sha256": sha(receipt), "encoding_receipt_sha256": sha(encoding / "encoding_receipt.json"),
        "methods": methods, "contrasts": contrasts, "all_comparator_differences": {n: float((values["candidate"] - v).mean())
            for n, v in values.items() if n != "candidate"},
        "timing": {"dense_seconds": dense_seconds, "auxiliary_seconds": auxiliary_seconds, "fusion_seconds": fusion_seconds},
        "primary_supported": bool(contrasts["previous_tfidf"]["bonferroni_three_interval"][0] > 0),
        "ancestor_supported": bool(contrasts["matched_mask"]["bonferroni_three_interval"][0] > 0),
        "independent_metric_check": "passed_all_rankings", "java_test_accessed": False,
        "no_new_parameter_selection": True})
    print(json.dumps({"MAP_at_8": {n: float(v.mean()) for n, v in values.items()},
                      "contrasts": {n: {k: v for k, v in c.items() if k != "problem_differences"} for n, c in contrasts.items()}}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoding", type=Path, default=OUT / "encoding")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--freeze", action="store_true")
    args = parser.parse_args()
    if args.freeze:
        assert not (OUT / "evaluation_kaggle").exists() and not (args.encoding / "encoding_receipt.json").exists()
        bindings = {str(p): sha(p) for p in (Path(__file__), ROOT / "test_root_tfidf_kaggle_20261010.py",
                    OUT / "design_receipt.json", ROOT / "coderank_evaluate.py")}
        write(OUT / "kaggle_evaluation_code_receipt.json", {"fixed_before_remote_encoding_utc": now(), "bindings": bindings})
        print("Evaluation code frozen before remote encoding")
    else:
        run(args.encoding, args.check)
