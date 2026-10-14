"""Fixed Kaggle T4 reranking; labels stay local, every query is checkpointed."""

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import resource
import time

SETTINGS = {
    "model": "Qwen/Qwen3-Reranker-0.6B",
    "revision": "e61197ed45024b0ed8a2d74b80b4d909f1255473",
    "instruction": "Given a solution to a programming problem, retrieve other implementations that solve the same problem.",
    "max_length": 8192, "batch_size": 8, "precision": "float16",
    "attention": "sdpa", "padding_side": "left", "truncation_side": "right",
    "score": "float32_yes_minus_no_final_token_logits", "top_k": 100,
}
PREFIX = ('<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. '
          'Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n')
SUFFIX = '<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def format_pair(query, document):
    return f'<Instruct>: {SETTINGS["instruction"]}\n<Query>: {query}\n<Document>: {document}'


def validate_checkpoint(scores, total):
    import numpy as np
    assert scores.ndim == 2 and scores.shape[1] == 100 and scores.shape[0] <= total
    assert scores.dtype.kind == "f" and np.isfinite(scores).all()


def run(root, out, max_seconds):
    assert Path("/kaggle/working").is_dir(), "Remote inference only"
    import numpy as np
    import torch
    from huggingface_hub import snapshot_download
    from transformers import AutoModelForCausalLM, AutoTokenizer

    assert torch.cuda.is_available() and torch.cuda.get_device_name(0) == "Tesla T4"
    assert 60 <= max_seconds <= 30000
    torch.manual_seed(20261013)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    manifest_path = root / "reranker_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    assert manifest["settings"] == SETTINGS
    for name, expected in manifest["files"].items():
        assert Path(name).name == name and sha(root / name) == expected, name
    assert sha(Path(__file__)) == manifest["files"][Path(__file__).name]
    binding = sha(manifest_path)
    programs = json.loads((root / "programs.json").read_text())
    assert set(programs) == {"query", "gallery"}
    assert all(len(programs[k]) == 3200 for k in programs)
    assert all(set(p) == {"id", "source"} for rows in programs.values() for p in rows)
    candidates = np.load(root / "candidates.npy", allow_pickle=False)
    assert candidates.shape == (3200, 100) and candidates.dtype.kind in "iu"
    assert candidates.min() >= 0 and candidates.max() < 3200
    assert all(len(set(row)) == 100 for row in candidates)
    out.mkdir(parents=True, exist_ok=True)
    assert not (out / "reranker_receipt.json").exists(), "Already complete"
    checkpoint = out / "reranker_checkpoint.npz"
    scores = np.empty((3200, 100), dtype=np.float32)
    lengths = np.empty((3200, 100), dtype=np.int32)
    times = np.empty((3200, 3), dtype=np.float64)
    done, resume = 0, checkpoint if checkpoint.exists() else root / checkpoint.name
    if resume.exists():
        with np.load(resume, allow_pickle=False) as saved:
            assert str(saved["binding"]) == binding
            validate_checkpoint(saved["scores"], 3200)
            done = len(saved["scores"])
            assert saved["lengths"].shape == (done, 100) and saved["times"].shape == (done, 3)
            assert (saved["lengths"] > 0).all() and np.isfinite(saved["times"]).all()
            scores[:done], lengths[:done], times[:done] = saved["scores"], saved["lengths"], saved["times"]
    initial_done = done

    def save():
        temporary = checkpoint.with_suffix(".tmp")
        with temporary.open("wb") as stream:
            np.savez(stream, scores=scores[:done], lengths=lengths[:done], times=times[:done], binding=binding)
        temporary.replace(checkpoint)

    start = time.perf_counter()
    snapshot = Path(snapshot_download(SETTINGS["model"], revision=SETTINGS["revision"],
        allow_patterns=["config.json", "generation_config.json", "model.safetensors", "tokenizer.json",
                        "tokenizer_config.json", "vocab.json", "merges.txt", "chat_template.jinja"]))
    model_files = {p.name: sha(p) for p in snapshot.iterdir() if p.is_file()}
    download_seconds = time.perf_counter() - start
    start = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(snapshot, padding_side="left", trust_remote_code=False)
    tokenizer.truncation_side = "right"
    model = AutoModelForCausalLM.from_pretrained(snapshot, torch_dtype=torch.float16,
        attn_implementation="sdpa", trust_remote_code=False).to("cuda:0").eval()
    prefix = tokenizer.encode(PREFIX, add_special_tokens=False)
    suffix = tokenizer.encode(SUFFIX, add_special_tokens=False)
    cap = SETTINGS["max_length"] - len(prefix) - len(suffix)
    yes, no = (tokenizer.convert_tokens_to_ids(word) for word in ("yes", "no"))
    assert yes != no and cap > 0
    torch.cuda.synchronize()
    load_seconds = time.perf_counter() - start

    def infer(token_ids):
        batch = tokenizer.pad({"input_ids": token_ids}, padding=True, return_tensors="pt").to("cuda:0")
        logits = model(**batch, use_cache=False, logits_to_keep=1).logits[:, -1, :].float()
        result = (logits[:, yes] - logits[:, no]).cpu().numpy()
        assert np.isfinite(result).all()
        return result

    report = {"binding": binding, "settings": SETTINGS, "model_files": model_files,
              "gpu": torch.cuda.get_device_name(0), "script_sha256": sha(Path(__file__)),
              "download_seconds": download_seconds, "model_load_seconds": load_seconds,
              "initial_completed_queries": initial_done, "max_session_seconds": max_seconds,
              "packages": {n: importlib.metadata.version(n) for n in
                           ("torch", "transformers", "numpy", "huggingface-hub")},
              "training_performed": False, "relevance_labels_available": False,
              "java_test_accessed": False, "started_utc": datetime.now(timezone.utc).isoformat()}
    (out / "runtime_start.json").write_text(json.dumps(report, indent=2) + "\n")
    with torch.inference_mode():
        infer([prefix + tokenizer.encode(format_pair("pass", "pass"), add_special_tokens=False) + suffix])
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        session_start = time.perf_counter()
        while done < 3200:
            if time.perf_counter() - session_start > max_seconds:
                save()
                print(json.dumps({"status": "checkpointed_time_limit", "completed_queries": done}), flush=True)
                return
            query_start = time.perf_counter()
            query = programs["query"][done]["source"]
            texts = [format_pair(query, programs["gallery"][int(j)]["source"]) for j in candidates[done]]
            ids = tokenizer(texts, add_special_tokens=False, truncation=False)["input_ids"]
            lengths[done] = [len(row) + len(prefix) + len(suffix) for row in ids]
            ids = [prefix + row[:cap] + suffix for row in ids]
            preparation_seconds = time.perf_counter() - query_start
            torch.cuda.synchronize()
            inference_start = time.perf_counter()
            for start in range(0, 100, SETTINGS["batch_size"]):
                batch_ids = ids[start:start + SETTINGS["batch_size"]]
                scores[done, start:start + len(batch_ids)] = infer(batch_ids)
            torch.cuda.synchronize()
            inference_seconds = time.perf_counter() - inference_start
            times[done] = [preparation_seconds, inference_seconds, time.perf_counter() - query_start]
            done += 1
            save()
            if done == initial_done + 1 or done % 25 == 0:
                elapsed = time.perf_counter() - session_start
                print(json.dumps({"completed_queries": done, "total": 3200,
                    "session_seconds": elapsed, "projected_remaining_seconds": elapsed*(3200-done)/(done-initial_done)}), flush=True)
    validate_checkpoint(scores, 3200)
    report.update(status="completed", completed_queries=done,
        checkpoint_sha256=sha(checkpoint),
        preprocessing_seconds=float(times[:, 0].sum()), inference_seconds=float(times[:, 1].sum()),
        query_processing_seconds=float(times[:, 2].sum()),
        query_seconds_quantiles=np.quantile(times[:, 2], [0, .5, .95, 1]).tolist(),
        truncated_pairs=int((lengths > SETTINGS["max_length"]).sum()),
        token_length_quantiles=np.quantile(lengths, [0, .5, .95, .99, 1]).tolist(),
        peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
        peak_host_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        completed_utc=datetime.now(timezone.utc).isoformat())
    (out / "reranker_receipt.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-seconds", type=int, required=True)
    args = parser.parse_args()
    run(args.input, args.out, args.max_seconds)
