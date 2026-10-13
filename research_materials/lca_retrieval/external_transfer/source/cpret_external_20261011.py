"""Prepare and evaluate the frozen CPRet/Qwen transfer test without local neural inference."""
import os
for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[variable] = "1"

from collections import Counter
from datetime import datetime, timezone
import argparse
import json
from pathlib import Path
import re
import shutil
import subprocess
import time
from urllib.request import urlopen
import numpy as np
from scipy.sparse import save_npz, load_npz
from cpret_evaluate import inputs, query_metrics, summarize, KEYS
from cpret_prepare import OUT as COHORT, CACHE, FILES
from rank_fusion_followup import ROOT, sha, write
from followup_baselines import vectorize, lexical_token_stream, ast_node_histogram
from frequency_factorial_20261010 import transform
from bm25_20261010 import weighted_gallery, hybrid
from two_stage_20261010 import order, rerank
from qwen_external_encode_20261011 import SETTINGS, validate

OUT = ROOT / "followup_results/cpret_external_20261011"
EXECUTION = ROOT.parents[2] / "code2hyp_informatics_artifacts/local_execution/kaggle_cpret_external_20261011"
PROTOCOL = ROOT / "cpret_external_protocol_20261011.md"
METHODS = ("D", "BM25", "D_BM25", "nodes_100", "root_full", "root_100")


def now():
    return datetime.now(timezone.utc).isoformat()


def check_bindings():
    design = json.loads((OUT / "design_receipt.json").read_text())
    for path, digest in design["bindings"].items():
        assert sha(Path(path)) == digest, path
    assert sha(OUT / "selected_programs.jsonl") == design["source_sha256"]
    return design


def prepare():
    if (OUT / "design_receipt.json").exists():
        return check_bindings()
    manifest, rows, _ = inputs()
    assert not manifest["retrieval_scores_inspected"]
    for name, digest in FILES.items():
        assert sha(CACHE / name) == digest
    assert len({r["problem_url"] for r in rows if r["role"] == "query"}) == 276
    assert Counter(s for rel in manifest["python_qrels"].values() for s in rel.values()) == {1: 927}
    fingerprints = [{r[k] for r in rows} for k in ("d0", "d1", "d2")]
    urls = {r["problem_url"] for r in rows}
    cf_ids = set()
    for url in urls:
        match = re.fullmatch(r"https://codeforces\.com/(?:contest|problemset/problem)/(\d+)/(?:problem/)?([A-Za-z0-9]+)", url)
        if match:
            cf_ids.add(f"{match[1]}:{match[2]}")
    late = [ROOT / f"followup_results/{n}/metadata.jsonl" for n in
            ("root_tfidf_external_20261010", "top100_fresh_20261010")]
    for path in late:
        records = [json.loads(line) for line in path.read_text().splitlines()]
        assert not cf_ids & {r["cluster_id"] for r in records}, path
        for key, values in zip(("d0", "d1", "d2"), fingerprints):
            assert not values & {r[key] for r in records}, (path, key)
    from path_context_followup import VALIDATION, CHECKPOINT_HASH
    from validation_diagnostic import REPO
    checkpoint = VALIDATION / "seed_20260711_encoder.pt"
    assert sha(checkpoint) == CHECKPOINT_HASH
    commit = "c419f6418056b618ce373ebd6fafe6601ff51566"
    assert subprocess.check_output(["git", "-C", str(REPO), "rev-parse", commit], text=True).strip() == commit
    names = ("cpret_external_20261011.py", "test_cpret_external_20261011.py", "qwen_external_encode_20261011.py",
        "coderank_kaggle_20261010.py", "cpret_evaluate.py", "cpret_prepare.py", "followup_baselines.py",
        "path_context_followup.py", "path_fragment_followup.py", "frequency_factorial_20261010.py",
        "bm25_20261010.py", "two_stage_20261010.py", "rank_fusion_followup.py")
    paths = [PROTOCOL, checkpoint, COHORT / "selection_manifest.json", COHORT / "selected_programs.jsonl",
        *late, *(ROOT / n for n in names), EXECUTION / "kernel/run.py", EXECUTION / "kernel/kernel-metadata.json",
        *sorted((REPO / "geometry_profile_research").glob("*.py"))]
    OUT.mkdir(exist_ok=True)
    assert not list(OUT.iterdir()) and shutil.disk_usage(OUT).free > 2 * 1024**3
    shutil.copy2(COHORT / "selected_programs.jsonl", OUT / "selected_programs.jsonl")
    shutil.copy2("/tmp/cpret-external-resources-20261011.json", OUT / "resources.json")
    design = {"fixed_before_scores_utc": now(), "bindings": {str(p): sha(p) for p in paths},
        "source_sha256": sha(OUT / "selected_programs.jsonl"), "settings": SETTINGS,
        "cohort_manifest_sha256": sha(COHORT / "selection_manifest.json"), "frozen_commit": commit,
        "queries": 276, "gallery": 3060, "problems": 276, "positive_qrels": 927,
        "methods": METHODS, "primary": "root_100_minus_D_BM25", "primary_metric": "nDCG@10",
        "bootstrap_resamples": 20000, "bootstrap_seed": 20261011,
        "late_cohort_source_token_ast_overlap": 0, "late_codeforces_problem_overlap": 0,
        "retrieval_scores_inspected": False, "java_test_accessed": False}
    write(OUT / "design_receipt.json", design)
    print("Protocol and unchanged 3336-program cohort bound before scores", flush=True)
    return design


def features():
    check_bindings()
    if (OUT / "features.json").exists():
        for name, digest in json.loads((OUT / "features.json").read_text())["sha256"].items():
            assert sha(OUT / name) == digest
        return
    from path_context_followup import load_model, path_keys, parse_python_ast_tree
    from path_fragment_followup import fragments
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    _, rows, _ = inputs()
    gi = np.array([i for i, r in enumerate(rows) if r["role"] == "gallery"])
    started = time.perf_counter()
    model = load_model()
    bags = {k: [] for k in ("root", "nodes", "lexical")}
    unknown, selected_paths = 0, 0
    for row in rows:
        source = row["source"]
        tree = parse_python_ast_tree(source)
        counts = Counter()
        paths = model._resolve_paths(tree, None)
        assert len(paths) <= 64
        for path in paths:
            retained, _, missing = path_keys(tree, path, model.token_to_id)
            counts.update(fragments(retained))
            unknown += missing
            selected_paths += 1
        bags["root"].append(counts)
        bags["nodes"].append(ast_node_histogram(source))
        bags["lexical"].append(Counter(lexical_token_stream(source)))
    files, dimensions = [], {}
    for name, values in bags.items():
        keys = sorted(set().union(*(set(values[i]) for i in gi)))
        counts = vectorize(values, {key: j for j, key in enumerate(keys)})
        counts.sort_indices()
        matrix = counts if name == "lexical" else transform(counts, gi, .5 if name == "root" else 1., False)
        assert matrix.shape[0] == 3336 and np.isfinite(matrix.data).all()
        save_npz(OUT / (name + ".npz"), matrix)
        write(OUT / (name + "_vocabulary.json"), keys)
        files.extend((name + ".npz", name + "_vocabulary.json"))
        dimensions[name] = len(keys)
    write(OUT / "features.json", {"sha256": {n: sha(OUT/n) for n in files},
        "design_sha256": sha(OUT / "design_receipt.json"), "dimensions": dimensions,
        "selected_paths": selected_paths, "unknown_path_tokens": unknown,
        "seconds_including_checkpoint_loading": time.perf_counter() - started,
        "retrieval_scores_inspected": False, "neural_inference_performed": False})
    print("Frozen structural and lexical features prepared; no retrieval scores evaluated", flush=True)


def payload():
    check_bindings()
    path = EXECUTION / "payload"
    assert not path.exists(), "Do not overwrite an uploaded payload"
    path.mkdir()
    files = ("config.json", "tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt", "model.safetensors")
    api = f'https://huggingface.co/api/models/{SETTINGS["model"]}/revision/{SETTINGS["revision"]}?blobs=true'
    with urlopen(api, timeout=120) as response:
        metadata = json.load(response)
    assert metadata["sha"] == SETTINGS["revision"]
    expected = {}
    for item in metadata["siblings"]:
        name = item["rfilename"]
        if name not in files:
            continue
        if item.get("lfs"):
            expected[name] = item["lfs"]["sha256"]
        else:
            url = f'https://huggingface.co/{SETTINGS["model"]}/resolve/{SETTINGS["revision"]}/{name}'
            with urlopen(url, timeout=120) as response:
                data = response.read()
            import hashlib
            assert hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest() == item["blobId"]
            expected[name] = hashlib.sha256(data).hexdigest()
    assert set(expected) == set(files)
    write(path / "expected_model_files.json", expected)
    for source in (OUT / "selected_programs.jsonl", OUT / "design_receipt.json", PROTOCOL,
                   ROOT / "qwen_external_encode_20261011.py", ROOT / "coderank_kaggle_20261010.py"):
        shutil.copy2(source, path / source.name)
    receipt = {"created_utc": now(), "machine_shape": "NvidiaTeslaT4",
        "launcher_sha256": sha(EXECUTION / "kernel/run.py"),
        "payload_sha256": {p.name: sha(p) for p in sorted(path.iterdir())}}
    write(path / "execution_receipt.json", receipt)
    write(OUT / "execution_receipt.json", receipt)
    write(path / "dataset-metadata.json", {"title": "Code2Hyp CPRet external inputs 20261011",
        "id": "ivankosyanenko/code2hyp-cpret-external-inputs-20261011",
        "licenses": [{"name": "CC-BY-NC-4.0"}],
        "description": "Private frozen CPRet Python C2C subset for fixed-method external validation. Sources: coldchair16/CPRet-data; model: Qwen/Qwen3-Embedding-0.6B. No manuscript."})
    print("Private payload sealed; no upload or GPU launch performed by this script", flush=True)


def rankings(d, root, nodes, bm, tie):
    dense = order(d, tie)
    prefix = dense[:100]
    fused = .75 * d + .25 * root
    return {"D": dense, "BM25": order(bm, tie), "D_BM25": order(hybrid(d, bm, .10), tie),
        "nodes_100": rerank(dense, .50 * d[prefix] + .50 * nodes[prefix], tie),
        "root_full": order(fused, tie), "root_100": rerank(dense, fused[prefix], tie)}


def evaluation_inputs(encoding):
    design = check_bindings()
    manifest, rows, _ = inputs()
    execution = json.loads((OUT / "execution_receipt.json").read_text())
    assert sha(encoding / "execution_receipt.json") == sha(OUT / "execution_receipt.json")
    for name, digest in execution["payload_sha256"].items():
        assert sha(encoding / name) == digest, name
    assert sha(encoding / "submitted_launcher.py") == execution["launcher_sha256"]
    receipt = json.loads((encoding / "encoding_receipt.json").read_text())
    assert receipt["status"] == "completed_external_encoding" and receipt["programs"] == 3336
    assert receipt["settings"] == SETTINGS and receipt["gpu"] == "Tesla T4"
    assert receipt["source_sha256"] == design["source_sha256"]
    assert receipt["script_sha256"] == sha(ROOT / "qwen_external_encode_20261011.py")
    assert receipt["binding"] == sha(OUT / "execution_receipt.json")
    assert receipt["model_files"] == json.loads((encoding / "expected_model_files.json").read_text())
    assert not receipt["training_performed"] and not receipt["java_test_accessed"]
    for name in ("embeddings", "token_lengths"):
        assert sha(encoding / (name + ".npy")) == receipt[name + "_sha256"]
    vectors = np.load(encoding / "embeddings.npy", allow_pickle=False)
    validate(vectors, 3336)
    lengths = np.load(encoding / "token_lengths.npy", allow_pickle=False)
    assert lengths.shape == (3336,) and lengths.dtype == np.int32 and (lengths > 0).all()
    assert int((lengths > 2048).sum()) == receipt["truncated_programs"]
    saved = json.loads((OUT / "features.json").read_text())
    assert saved["design_sha256"] == sha(OUT / "design_receipt.json")
    for name, digest in saved["sha256"].items():
        assert sha(OUT / name) == digest
    qi = np.array([i for i, r in enumerate(rows) if r["role"] == "query"])
    gi = np.array([i for i, r in enumerate(rows) if r["role"] == "gallery"])
    scores = {"D": vectors[qi] @ vectors[gi].T}
    for name in ("root", "nodes", "lexical"):
        feature = load_npz(OUT / (name + ".npz"))
        gallery = weighted_gallery(feature[gi]) if name == "lexical" else feature[gi]
        scores[name] = (feature[qi] @ gallery.T).toarray()
    assert all(v.shape == (276, 3060) and np.isfinite(v).all() for v in scores.values())
    return manifest, scores


def evaluate(encoding):
    assert not (OUT / "result.json").exists(), "Use --check for completed results"
    manifest, scores = evaluation_inputs(encoding)
    ids = manifest["python_ids"]
    tie = np.array([g.encode() for g in ids["gallery"]])
    saved = np.empty((len(METHODS), 276, 3060), dtype=np.int32)
    methods = {name: {} for name in METHODS}
    for i, q in enumerate(ids["query"]):
        ranked = rankings(scores["D"][i].astype(np.float64), scores["root"][i], scores["nodes"][i], scores["lexical"][i], tie)
        for j, name in enumerate(METHODS):
            saved[j, i] = ranked[name]
            methods[name][q] = query_metrics(ranked[name], ids["gallery"], manifest["python_qrels"][q])
        assert all(methods[n][q]["Recall@100"] == methods["D"][q]["Recall@100"] for n in ("root_100", "nodes_100"))
    np.savez_compressed(OUT / "rankings.npz", rankings=saved)
    methods = {n: summarize(v, manifest["problem_urls"]) for n, v in methods.items()}
    problems = sorted(methods["D"]["per_problem"])
    delta = np.array([methods["root_100"]["per_problem"][p]["nDCG@10"] -
                      methods["D_BM25"]["per_problem"][p]["nDCG@10"] for p in problems])
    samples = np.random.Generator(np.random.PCG64(20261011)).integers(276, size=(20000, 276))
    interval = np.quantile(delta[samples].mean(axis=1), [.025, .975]).tolist()
    primary = {"mean": float(delta.mean()), "percentile_95_interval": interval,
        "better": int((delta > 1e-12).sum()), "worse": int((delta < -1e-12).sum()),
        "tied": int((np.abs(delta) <= 1e-12).sum()),
        "interpretation": "advantage_supported" if interval[0] > 0 else "opposite_supported" if interval[1] < 0 else "inconclusive"}
    write(OUT / "result.json", {"completed_utc": now(), "status": "completed_external_fixed_method_validation",
        "design_sha256": sha(OUT / "design_receipt.json"), "rankings_sha256": sha(OUT / "rankings.npz"),
        "encoding_receipt_sha256": sha(encoding / "encoding_receipt.json"), "methods": methods, "primary": primary,
        "parameter_selection_performed": False, "java_test_accessed": False})
    print(json.dumps({"methods": {n: m["problem_macro"] for n, m in methods.items()}, "primary": primary}, indent=2))


def verify(encoding):
    manifest, scores = evaluation_inputs(encoding)
    result = json.loads((OUT / "result.json").read_text())
    assert result["design_sha256"] == sha(OUT / "design_receipt.json")
    assert result["rankings_sha256"] == sha(OUT / "rankings.npz")
    assert result["encoding_receipt_sha256"] == sha(encoding / "encoding_receipt.json")
    ids = manifest["python_ids"]
    saved = np.load(OUT / "rankings.npz", allow_pickle=False)["rankings"]
    actual = np.empty((len(METHODS), 276, 3))
    for i, q in enumerate(ids["query"]):
        d, b = scores["D"][i].astype(float), scores["lexical"][i]
        scaled = lambda v: (v-v.min())/(v.max()-v.min()) if v.max() > v.min() else np.zeros_like(v)
        fused = {"D": d, "BM25": b, "D_BM25": .9*scaled(d)+.1*scaled(b),
                 "nodes_100": .5*d+.5*scores["nodes"][i], "root_full": .75*d+.25*scores["root"][i]}
        fused["root_100"] = fused["root_full"]
        dense = sorted(range(3060), key=lambda g: (-d[g], ids["gallery"][g].encode()))
        rel = manifest["python_qrels"][q]
        assert set(rel.values()) == {1}
        ideal = sum(1/np.log2(k+1) for k in range(1, min(10, len(rel))+1))
        for j, name in enumerate(METHODS):
            pool = dense[:100] if name.endswith("_100") else range(3060)
            ranked = sorted(pool, key=lambda g: (-fused[name][g], ids["gallery"][g].encode()))
            if name.endswith("_100"):
                ranked += dense[100:]
            np.testing.assert_array_equal(ranked, saved[j, i])
            hits = [rank+1 for rank, g in enumerate(ranked[:100]) if ids["gallery"][g] in rel]
            top = [h for h in hits if h <= 10]
            actual[j, i] = [sum(1/np.log2(h+1) for h in top)/ideal,
                            1/top[0] if top else 0., len(hits)/len(rel)]
            np.testing.assert_allclose(actual[j, i], [result["methods"][name]["per_query"][q][k] for k in KEYS], rtol=0, atol=1e-12)
    for j, name in enumerate(METHODS):
        np.testing.assert_allclose(actual[j].mean(axis=0), [result["methods"][name]["problem_macro"][k] for k in KEYS], atol=1e-12, rtol=0)
    ordered = sorted(range(276), key=lambda i: manifest["problem_urls"][ids["query"][i]])
    delta = (actual[METHODS.index("root_100"), :, 0] - actual[METHODS.index("D_BM25"), :, 0])[ordered]
    samples = np.random.Generator(np.random.PCG64(20261011)).integers(276, size=(20000, 276))
    np.testing.assert_allclose([delta.mean(), *np.quantile(delta[samples].mean(axis=1), [.025, .975])],
        [result["primary"]["mean"], *result["primary"]["percentile_95_interval"]], atol=1e-12, rtol=0)
    receipt = {"status": "passed", "result_sha256": sha(OUT / "result.json"),
        "independently_verified_rankings": 1656, "independently_verified_primary_bootstrap": True,
        "java_test_accessed": False}
    path = OUT / "verification.json"
    if path.exists():
        assert json.loads(path.read_text()) == receipt
    else:
        write(path, receipt)
    print("PASS: independent ranking/metric/primary-interval verification", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("prepare", "features", "payload", "evaluate", "check"))
    parser.add_argument("--encoding", type=Path)
    args = parser.parse_args()
    if args.mode in ("evaluate", "check"):
        assert args.encoding is not None
        if args.mode == "evaluate":
            evaluate(args.encoding)
        verify(args.encoding)
    else:
        {"prepare": prepare, "features": features, "payload": payload}[args.mode]()
