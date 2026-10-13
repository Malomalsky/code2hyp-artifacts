"""Missing-cell adapters; unchanged retrievers on the other sealed cohort."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time
import coderank_kaggle_20261010 as coderank
from qwen_external_encode_20261011 import SETTINGS as QWEN, texts, last_token_pool, validate

SOURCE_SHA = {"coderank": "375b1335f45ee50f84ac944341c490a2528ddf7063213da70c29be4593be3971",
              "qwen": "ae91ef4ac8b7d7703deb3bb1e90738362b04d9d0bd3ee1cc77de95fdda10a0ea"}


def source_rows(path, model):
    assert coderank.sha(path) == SOURCE_SHA[model]
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    key, n, q = ("id", 3336, 276) if model == "coderank" else ("source_relpath", 6400, 3200)
    assert len(rows) == n and len({r[key] for r in rows}) == n
    assert Counter(r["role"] for r in rows) == {"query": q, "gallery": n-q}
    assert all(hashlib.sha256(r["source"].encode()).hexdigest() == r["d0"] for r in rows)
    if model == "qwen":
        for role in ("query", "gallery"):
            counts = Counter(r["cluster_id"] for r in rows if r["role"] == role)
            assert len(counts) == 400 and set(counts.values()) == {8}
    return rows


def coderank_settings(receipt):
    return {**coderank.SETTINGS, "source_sha256": SOURCE_SHA["coderank"],
        "external_design_sha256": receipt["design_sha256"],
        "execution_addendum_sha256": receipt["protocol_sha256"], "study": "crossed_coderank_cpret_20261011"}


def run_qwen(source, out, receipt):
    # The encoding operations match qwen_external_encode_20261011; only the cohort checks differ.
    import numpy as np
    import torch
    from huggingface_hub import snapshot_download
    from transformers import AutoModel, AutoTokenizer
    rows = source_rows(source, "qwen")
    assert torch.cuda.is_available() and torch.cuda.get_device_name(0) == "Tesla T4"
    out.mkdir(parents=True, exist_ok=True)
    assert not (out / "encoding_receipt.json").exists()
    binding = coderank.sha(source.parent / "execution_receipt.json")
    checkpoint = out / "embeddings.partial.npz"
    vectors = np.empty((len(rows), 1024), dtype=np.float32)
    done, seconds = 0, 0.
    if checkpoint.exists():
        with np.load(checkpoint, allow_pickle=False) as saved:
            assert str(saved["binding"]) == binding
            done = len(saved["vectors"])
            assert done <= len(rows)
            validate(saved["vectors"], done)
            vectors[:done] = saved["vectors"]
            seconds = float(saved["encoding_seconds"])

    def save():
        temporary = checkpoint.with_suffix(".tmp")
        with temporary.open("wb") as stream:
            np.savez(stream, vectors=vectors[:done], binding=binding, encoding_seconds=seconds)
        temporary.replace(checkpoint)

    expected = json.loads((source.parent / "expected_model_files.json").read_text())
    t = time.perf_counter()
    snapshot = Path(snapshot_download(QWEN["model"], revision=QWEN["revision"], allow_patterns=list(expected)))
    actual = {name: coderank.sha(snapshot/name) for name in expected}
    assert actual == expected
    tokenizer = AutoTokenizer.from_pretrained(snapshot, padding_side="left", trust_remote_code=False)
    tokenizer.truncation_side = "right"
    model = AutoModel.from_pretrained(snapshot, torch_dtype=torch.float16,
        attn_implementation="sdpa", trust_remote_code=False).to("cuda:0").eval()
    assert model.config.hidden_size == 1024
    load_seconds = time.perf_counter() - t
    inputs = texts(rows)
    lengths = np.array([len(tokenizer.encode(s, truncation=False)) for s in inputs], dtype=np.int32)
    np.save(out / "token_lengths.npy", lengths, allow_pickle=False)

    def encode(batch):
        tokens = tokenizer(batch, padding=True, truncation=True, max_length=2048,
                           return_tensors="pt").to("cuda:0")
        hidden = model(**tokens).last_hidden_state
        pooled = last_token_pool(hidden, tokens["attention_mask"]).float()
        result = torch.nn.functional.normalize(pooled, p=2, dim=1).cpu().numpy()
        validate(result, len(batch))
        return result

    with torch.inference_mode():
        encode(["pass\n"] * 8)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        pilot_ids = np.argsort(lengths, kind="stable")[np.linspace(0, len(rows)-1, 32, dtype=int)]
        t = time.perf_counter()
        for start in range(0, 32, 8):
            encode([inputs[i] for i in pilot_ids[start:start+8]])
        torch.cuda.synchronize()
        pilot_seconds = time.perf_counter() - t
        report = {"settings": QWEN, "binding": binding, "source_sha256": coderank.sha(source),
            "script_sha256": coderank.sha(Path(__file__)), "model_files": actual,
            "created_utc": datetime.now(timezone.utc).isoformat(), "gpu": torch.cuda.get_device_name(0),
            "packages": {n: importlib.metadata.version(n) for n in ("torch", "transformers", "numpy", "huggingface-hub")},
            "model_load_seconds": load_seconds, "pilot_seconds": pilot_seconds,
            "projected_encoding_seconds": pilot_seconds * len(rows)/32,
            "token_length_quantiles": np.quantile(lengths, [0, .5, .9, .99, 1]).tolist(),
            "truncated_programs": int((lengths > 2048).sum()),
            "training_performed": False, "java_test_accessed": False}
        coderank.write_json(out / "resource_pilot.json", report)
        print(json.dumps(report), flush=True)
        assert report["projected_encoding_seconds"] < 3600
        started = time.perf_counter()
        while done < len(rows):
            if time.perf_counter() - started >= 3600:
                save()
                print("Time bound reached; resume checkpoint", flush=True)
                return
            end = min(done + 128, len(rows))
            t = time.perf_counter()
            for start in range(done, end, 8):
                stop = min(start+8, end)
                vectors[start:stop] = encode(inputs[start:stop])
            torch.cuda.synchronize()
            seconds += time.perf_counter() - t
            done = end
            save()
            print(f"Qwen encoded {done}/{len(rows)}", flush=True)
    validate(vectors, len(rows))
    np.save(out / "embeddings.npy", vectors, allow_pickle=False)
    report.update(status="completed_external_encoding", programs=done, encoding_seconds=seconds,
        peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
        embeddings_sha256=coderank.sha(out / "embeddings.npy"), token_lengths_sha256=coderank.sha(out / "token_lengths.npy"),
        completed_utc=datetime.now(timezone.utc).isoformat())
    coderank.write_json(out / "encoding_receipt.json", report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=("coderank", "qwen"), required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    assert Path("/kaggle/input").is_dir() and Path("/kaggle/working").is_dir()
    receipt = json.loads((args.source.parent / "execution_receipt.json").read_text())
    assert coderank.sha(Path(__file__)) == receipt["payload_sha256"][Path(__file__).name]
    if args.model == "coderank":
        coderank.SETTINGS = coderank_settings(receipt)
        coderank.SOURCE_SHA = SOURCE_SHA["coderank"]
        coderank.load_sources = lambda path: source_rows(path, "coderank")
        coderank.run(args.source, args.out, "full")
    else:
        run_qwen(args.source, args.out, receipt)
