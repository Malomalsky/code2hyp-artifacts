"""Same-VM fixed retrieval pipelines; raw-query clocks and isolated memory."""

import argparse
from collections import Counter
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time
import uuid

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["USE_TF"] = "0"

import numpy as np
from scipy.sparse import load_npz, save_npz
from e2e_common_20261014 import METHODS, feature_bag, metrics, now, order, rerank, sha, write

COLUMNS = ("query_encoding", "query_features", "dense_search", "candidate_gather",
           "pair_tokenization", "pair_inference", "final_sort", "total")


def load(root):
    manifest = json.loads((root / "e2e_manifest.json").read_text())
    for name, expected in manifest["files"].items():
        relative = Path(name)
        assert not relative.is_absolute() and ".." not in relative.parts
        assert sha(root / relative) == expected, name
    return json.loads((root / "e2e_design.json").read_text())


def host():
    cpu = next(line.split(":", 1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines()
               if line.startswith("model name"))
    gpu = subprocess.check_output(["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
                                   "--format=csv,noheader"], text=True).strip()
    return {"host_id": str(uuid.uuid4()), "hostname": platform.node(), "cpu": cpu,
            "logical_cpus": os.cpu_count(), "platform": platform.platform(),
            "python": platform.python_version(), "gpu": gpu,
            "ram_bytes": os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"),
            "cpu_threads": 1, "created_utc": now()}


def neural(root, name):
    import torch
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
    from qwen_external_encode_20261011 import SETTINGS as QE, last_token_pool
    from qwen_reranker_20261013 import SETTINGS as QR, PREFIX, SUFFIX, format_pair
    assert torch.cuda.is_available() and torch.cuda.get_device_name(0) == "Tesla T4"
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    design = json.loads((root / "e2e_design.json").read_text())
    key = {"coderank": "coderank_settings", "qwen_embedding": "qwen_embedding_settings",
           "qwen_reranker": "qwen_reranker_settings"}[name]
    settings = design[key]
    if name != "coderank":
        assert settings == (QE if name == "qwen_embedding" else QR)
    expected = json.loads((root / "expected_model_files.json").read_text())[name]
    started = time.perf_counter()
    snapshot = Path(snapshot_download(settings["model"], revision=settings["revision"], allow_patterns=list(expected)))
    download = time.perf_counter() - started
    assert {n: sha(snapshot / n) for n in expected} == expected, "Frozen model files changed"
    started = time.perf_counter()
    if name == "coderank":
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(str(snapshot), trust_remote_code=True, device="cuda:0").float().eval()
        model.max_seq_length = 512
        assert model[1].pooling_mode_cls_token and not model[1].pooling_mode_mean_tokens

        def encode(sources, query=False):
            del query
            with torch.inference_mode():
                return model.encode(sources, batch_size=16, normalize_embeddings=True,
                                    convert_to_numpy=True, show_progress_bar=False)
    else:
        tokenizer = AutoTokenizer.from_pretrained(snapshot, padding_side="left", trust_remote_code=False)
        tokenizer.truncation_side = "right"
        cls = AutoModel if name == "qwen_embedding" else AutoModelForCausalLM
        model = cls.from_pretrained(snapshot, torch_dtype=torch.float16,
                                    attn_implementation="sdpa", trust_remote_code=False).to("cuda:0").eval()
        if name == "qwen_embedding":
            def encode(sources, query=False):
                texts = [f'Instruct: {QE["instruction"]}\nQuery:{s}' for s in sources] if query else sources
                with torch.inference_mode():
                    tokens = tokenizer(texts, padding=True, truncation=True, max_length=2048,
                                       return_tensors="pt").to("cuda:0")
                    pooled = last_token_pool(model(**tokens).last_hidden_state, tokens["attention_mask"]).float()
                    return torch.nn.functional.normalize(pooled, p=2, dim=1).cpu().numpy()
        else:
            prefix = tokenizer.encode(PREFIX, add_special_tokens=False)
            suffix = tokenizer.encode(SUFFIX, add_special_tokens=False)
            cap = QR["max_length"] - len(prefix) - len(suffix)
            yes, no = (tokenizer.convert_tokens_to_ids(word) for word in ("yes", "no"))

            def encode(query, documents):
                t = time.perf_counter()
                ids = tokenizer([format_pair(query, s) for s in documents], add_special_tokens=False,
                                truncation=False)["input_ids"]
                ids = [prefix + row[:cap] + suffix for row in ids]
                tokenize_seconds = time.perf_counter() - t
                scores = []
                t = time.perf_counter()
                with torch.inference_mode():
                    for start in range(0, len(ids), 8):
                        batch = tokenizer.pad({"input_ids": ids[start:start+8]}, padding=True,
                                              return_tensors="pt").to("cuda:0")
                        logits = model(**batch, use_cache=False, logits_to_keep=1).logits[:, -1, :].float()
                        scores.append((logits[:, yes] - logits[:, no]).cpu().numpy())
                torch.cuda.synchronize()
                return np.concatenate(scores), tokenize_seconds, time.perf_counter() - t
    torch.cuda.synchronize()
    report = {"settings": settings, "model_files": expected, "model_download_seconds": download,
              "model_load_seconds": time.perf_counter() - started,
              "weight_files_bytes": sum((snapshot / n).stat().st_size for n in expected if n.endswith("safetensors"))}
    # Keep the resident model alive in the closure, including both models for reranking.
    return encode, report


def index(root, out, method, programs):
    from frozen_helpers import vectorize, normalized, lexical_token_stream, weighted_gallery
    index_start = time.perf_counter()
    preparation, reports = {}, {}
    gallery = [p["source"] for p in programs["gallery"]]
    state = {}
    if method != "BM25":
        kind = "qwen_embedding" if method == "QwenEmbedding" else "coderank"
        state["encode"], reports[kind] = neural(root, kind)
        batch = 8 if kind == "qwen_embedding" else 16
        state["encode"](["pass\n"] * batch)
        t = time.perf_counter()
        blocks = []
        for start in range(0, len(gallery), batch):
            blocks.append(state["encode"](gallery[start:start+batch]))
        state["gallery_vectors"] = np.concatenate(blocks).astype(np.float32)
        preparation["gallery_encoding_seconds"] = time.perf_counter() - t
        np.save(out / "gallery_vectors.npy", state["gallery_vectors"], allow_pickle=False)
        if kind == "coderank":
            with np.load(root / "reference_vectors.npz", allow_pickle=False) as saved:
                error = float(np.max(np.abs(state["gallery_vectors"] - saved["gallery"])))
            reports[kind]["maximum_reference_gallery_error"] = error
            assert error < 1e-4, "Encoder output changed materially; stop without tuning"
    if method in ("BM25", "D_BM25"):
        t = time.perf_counter()
        bags = [Counter(lexical_token_stream(s)) for s in gallery]
        keys = sorted(set().union(*(set(b) for b in bags)))
        assert keys == json.loads((root / "lexical_vocabulary.json").read_text())
        state["lex_vocab"] = {key: i for i, key in enumerate(keys)}
        counts = vectorize(bags, state["lex_vocab"])
        counts.sort_indices()
        state["lex_index"] = weighted_gallery(counts).T.tocsr()
        preparation["gallery_lexical_index_seconds"] = time.perf_counter() - t
        save_npz(out / "lexical_index.npz", state["lex_index"])
        write(out / "lexical_vocabulary.json", keys)
    if method == "D_AST":
        t = time.perf_counter()
        state["token_vocab"] = json.loads((root / "node_token_vocabulary.json").read_text())
        bags = [feature_bag(s, state["token_vocab"]) for s in gallery]
        keys = sorted(set().union(*(set(b) for b in bags)))
        expected = [tuple(map(tuple, key)) for key in json.loads((root / "ast_vocabulary.json").read_text())]
        assert keys == expected, "AST parser/vocabulary changed"
        state["ast_vocab"] = {key: i for i, key in enumerate(keys)}
        counts = vectorize(bags, state["ast_vocab"])
        gi = json.loads((root / "source_roles.json").read_text())["gallery"]
        assert (counts != load_npz(root / "reference_ast_counts.npz")[gi]).nnz == 0
        counts.data **= .5
        state["ast_index"] = normalized(counts)
        preparation["gallery_ast_index_seconds"] = time.perf_counter() - t
        save_npz(out / "ast_index.npz", state["ast_index"])
        write(out / "ast_vocabulary.json", keys)
    if method == "D_QwenReranker":
        state["pair"], reports["qwen_reranker"] = neural(root, "qwen_reranker")
        state["pair"]("pass\n", ["pass\n"] * 8)
    files = [out / name for name in ("gallery_vectors.npy", "lexical_index.npz", "lexical_vocabulary.json",
                                    "ast_index.npz", "ast_vocabulary.json") if (out / name).exists()]
    # Serialized index bytes and resident arrays are different measures.
    memory = sum(v.nbytes if isinstance(v, np.ndarray) else v.data.nbytes + v.indices.nbytes + v.indptr.nbytes
                 for v in state.values() if isinstance(v, np.ndarray) or hasattr(v, "indptr"))
    raw_bytes = sum(len(s.encode()) for s in gallery) if method == "D_QwenReranker" else 0
    return state, {"preparation": preparation, "models": reports,
                   "index_array_bytes": int(memory), "reranker_gallery_source_utf8_bytes": raw_bytes,
                   "serialized_index_bytes": sum(p.stat().st_size for p in files),
                   "serialization_boundary_note": "Index saves are outside component preparation clocks",
                   "startup_and_index_seconds": time.perf_counter() - index_start}


def serve(source, method, state, gallery, tie):
    from frozen_helpers import vectorize, normalized, lexical_token_stream, hybrid
    clocks = np.zeros(len(COLUMNS))
    if method != "BM25":
        import torch
        torch.cuda.synchronize()
    start = time.perf_counter()
    if method != "BM25":
        t = time.perf_counter()
        query = state["encode"]([source], query=True)[0]
        clocks[0] = time.perf_counter() - t
        t = time.perf_counter()
        dense = (query @ state["gallery_vectors"].T).astype(np.float64)
        clocks[2] = time.perf_counter() - t
    if method in ("BM25", "D_BM25"):
        t = time.perf_counter()
        counts = vectorize([Counter(lexical_token_stream(source))], state["lex_vocab"])
        counts.sort_indices()
        lexical = (counts @ state["lex_index"]).toarray().ravel()
        clocks[1] = time.perf_counter() - t
    if method == "D_AST":
        t = time.perf_counter()
        counts = vectorize([feature_bag(source, state["token_vocab"])], state["ast_vocab"])
        counts.data **= .5
        features = normalized(counts)
        clocks[1] = time.perf_counter() - t
    t = time.perf_counter()
    if method == "BM25":
        ranking = order(lexical, tie)
    elif method == "D_BM25":
        ranking = order(hybrid(dense, lexical, .10), tie)
    elif method in ("D", "QwenEmbedding"):
        ranking = order(dense, tie)
    else:
        dense_order = order(dense, tie)
        prefix = dense_order[:100]
        documents = [gallery[int(i)]["source"] for i in prefix] if method == "D_QwenReranker" else None
        clocks[3] = time.perf_counter() - t
        if method == "D_AST":
            t = time.perf_counter()
            structural = (features @ state["ast_index"][prefix].T).toarray().ravel()
            clocks[1] += time.perf_counter() - t
            scores = .75 * dense[prefix] + .25 * structural
        else:
            scores, clocks[4], clocks[5] = state["pair"](source, documents)
        t = time.perf_counter()
        ranking = rerank(dense_order, scores, tie)
    clocks[6] = time.perf_counter() - t
    if method != "BM25":
        torch.cuda.synchronize()
    clocks[-1] = time.perf_counter() - start
    return ranking, clocks


def full_quality(root, out, method, state, programs, tie, design, deadline):
    from frozen_helpers import vectorize, normalized, lexical_token_stream, hybrid
    labels = np.array(design["gallery_problems"])
    if method == "QwenEmbedding":
        vectors = []
        for start in range(0, 3200, 8):
            if time.time() > deadline:
                return False
            vectors.append(state["encode"]([p["source"] for p in programs["query"][start:start+8]], query=True))
        q = np.concatenate(vectors).astype(np.float32)
        np.save(out / "quality_query_vectors.npy", q, allow_pickle=False)
        dense = q @ state["gallery_vectors"].T
    else:
        with np.load(root / "reference_vectors.npz", allow_pickle=False) as saved:
            dense = saved["query"] @ saved["gallery"].T
    scores = np.load(root / "reference_reranker_scores.npy", allow_pickle=False) if method == "D_QwenReranker" else None
    candidates = np.load(root / "reference_candidates.npy", allow_pickle=False) if scores is not None else None
    expected_ast = load_npz(root / "reference_ast_features.npz") if method == "D_AST" else None
    roles = json.loads((root / "source_roles.json").read_text())
    values, top100 = np.empty((3200, 3)), np.empty((3200, 100), dtype=np.int32)
    for i, program in enumerate(programs["query"]):
        d = dense[i].astype(np.float64)
        ranked = order(d, tie)
        if method in ("BM25", "D_BM25"):
            q = vectorize([Counter(lexical_token_stream(program["source"]))], state["lex_vocab"])
            q.sort_indices()
            lexical = (q @ state["lex_index"]).toarray().ravel()
            ranked = order(lexical if method == "BM25" else hybrid(d, lexical, .10), tie)
        elif method == "D_AST":
            q = vectorize([feature_bag(program["source"], state["token_vocab"])], state["ast_vocab"])
            q.data **= .5
            q = normalized(q)
            difference = q - expected_ast[roles["query"][i]]
            assert not difference.nnz or np.max(np.abs(difference.data)) < 1e-12
            prefix = ranked[:100]
            structural = (q @ state["ast_index"][prefix].T).toarray().ravel()
            ranked = rerank(ranked, .75*d[prefix]+.25*structural, tie)
        elif method == "D_QwenReranker":
            np.testing.assert_array_equal(ranked[:100], candidates[i])
            ranked = rerank(ranked, scores[i], tie)
        values[i] = metrics(ranked, labels == design["query_problems"][i])
        top100[i] = ranked[:100]
    np.savez_compressed(out / "full_quality.npz", metrics=values, top100=top100)
    write(out / "quality_scope.json", {"queries": 3200,
          "scope": "new frozen Qwen embedding inference" if method == "QwenEmbedding" else
                   "numerical replay of complete old neural scores, freshly rebuilt lexical/AST features",
          "mean_metrics": values.mean(axis=0).tolist()})
    return True


def worker(root, out, method, deadline):
    assert Path("/kaggle/working").is_dir(), "Neural benchmark is Kaggle-only"
    design = load(root)
    programs = json.loads((root / "programs.json").read_text())
    assert set(programs) == {"query", "gallery"} and all(len(p) == 3200 for p in programs.values())
    out.mkdir(parents=True, exist_ok=True)
    assert not (out / "complete.json").exists(), "Completed worker must not be rerun"
    identity = json.loads((out.parent / "host.json").read_text())
    write(out / "started.json", {"method": method, "host_id": identity["host_id"], "started_utc": now(),
          "binding": sha(root / "e2e_manifest.json"), "threads": 1})
    if method != "BM25":
        import torch
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    state, report = index(root, out, method, programs)
    tie = np.array([p["id"].encode() for p in programs["gallery"]])
    selected = design["selected_indices"]
    warmup_start = time.perf_counter()
    for i in selected[:4]:
        serve(programs["query"][i]["source"], method, state, programs["gallery"], tie)
    report["request_warmup_seconds"] = time.perf_counter() - warmup_start
    report["ready_startup_excluding_network_seconds"] = (
        report["startup_and_index_seconds"] + report["request_warmup_seconds"]
        - sum(m["model_download_seconds"] for m in report["models"].values()))
    report["startup_scope"] = "Includes frozen-model integrity checks, loading, index build/serialization and warmup; excludes dependency install and Python imports"
    if method != "BM25":
        import torch
        report["preparation_peak_cuda_allocated_bytes"] = int(torch.cuda.max_memory_allocated())
        report["preparation_peak_cuda_reserved_bytes"] = int(torch.cuda.max_memory_reserved())
        torch.cuda.reset_peak_memory_stats()
    print(json.dumps({"phase": "timing_started", "method": method, "host_id": identity["host_id"],
                      "index": report}), flush=True)
    write(out / "index_receipt.json", report)
    record_path = out / "requests.jsonl"
    completed = set()
    if record_path.exists():
        for line in record_path.read_text().splitlines():
            row = json.loads(line)
            assert row["method"] == method and row["host_id"] == identity["host_id"]
            key = (row["repeat"], row["slot"])
            assert key not in completed
            completed.add(key)
    with record_path.open("a") as stream:
        for repeat, indices in enumerate(design["query_orders"]):
            for slot in indices:
                if (repeat, slot) in completed:
                    continue
                if time.time() >= deadline:
                    write(out / "partial.json", {"status": "budget_stop", "requests": len(completed), "host_id": identity["host_id"]})
                    return
                i = selected[slot]
                ranked, times = serve(programs["query"][i]["source"], method, state, programs["gallery"], tie)
                record = {"method": method, "host_id": identity["host_id"], "repeat": repeat, "slot": slot,
                          "query_index": i, "query_id": programs["query"][i]["id"],
                          "seconds": times.tolist(), "ranking": ranked.tolist()}
                stream.write(json.dumps(record, separators=(",", ":")) + "\n")
                stream.flush()
                completed.add((repeat, slot))
                if len(completed) % 50 == 0:
                    print(json.dumps({"method": method, "timed_requests": len(completed), "target": 1200}), flush=True)
    memory = {"peak_host_rss_bytes_before_quality": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)}
    if method != "BM25":
        memory.update(peak_cuda_allocated_bytes=max(int(torch.cuda.max_memory_allocated()), report["preparation_peak_cuda_allocated_bytes"]),
                      peak_cuda_reserved_bytes=max(int(torch.cuda.max_memory_reserved()), report["preparation_peak_cuda_reserved_bytes"]))
    write(out / "memory.json", memory)
    if not full_quality(root, out, method, state, programs, tie, design, deadline):
        write(out / "partial.json", {"status": "budget_stop_during_quality", "requests": 1200})
        return
    write(out / "complete.json", {"status": "complete", "method": method, "host_id": identity["host_id"],
          "binding": sha(root / "e2e_manifest.json"), "requests": 1200, "quality_queries": 3200,
          "columns": COLUMNS, "memory": memory, "index": report,
          "packages": {n: importlib.metadata.version(n) for n in ("numpy", "scipy", "torch", "transformers", "sentence-transformers")},
          "training_performed": False, "java_test_accessed": False, "completed_utc": now(),
          "files": {p.name: sha(p) for p in out.iterdir() if p.is_file()}})


def controller(root, out):
    assert Path("/kaggle/working").is_dir()
    design = load(root)
    out.mkdir(parents=True, exist_ok=True)
    identity = host()
    assert "Tesla T4" in identity["gpu"]
    assert not (out / "host.json").exists(), "Do not overwrite an earlier session"
    write(out / "host.json", identity)
    shutil_manifest = out / "e2e_manifest.json"
    shutil_manifest.write_bytes((root / "e2e_manifest.json").read_bytes())
    (out / "e2e_design.json").write_bytes((root / "e2e_design.json").read_bytes())
    deadline = time.time() + design["controller_budget_seconds"]
    for method in design["method_order"]:
        if time.time() >= deadline:
            print("PARTIAL: controller budget exhausted; preserve outputs", flush=True)
            return
        command = [sys.executable, str(Path(__file__)), "--input", str(root), "--out", str(out / method),
                   "--method", method, "--deadline", str(deadline)]
        subprocess.run(command, check=True, timeout=max(1, deadline-time.time()+300))
        if not (out / method / "complete.json").exists():
            print("PARTIAL: worker did not complete; preserve outputs", flush=True)
            return
    write(out / "complete_receipt.json", {"status": "complete", "host": identity,
          "binding": sha(root / "e2e_manifest.json"), "methods": METHODS,
          "worker_receipts": {m: sha(out / m / "complete.json") for m in METHODS}, "completed_utc": now(),
          "scope": "same-VM warmed raw-query latency and separate gallery preparation, not network/concurrent service"})
    print("COMPLETE: all six methods, 7200 timed requests, full-cohort quality", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--method", choices=METHODS)
    parser.add_argument("--deadline", type=float)
    args = parser.parse_args()
    if args.method:
        worker(args.input, args.out, args.method, args.deadline)
    else:
        controller(args.input, args.out)
