"""Fixed, checkpointed Qwen inference; Kaggle T4 only, no retrieval evaluation."""

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time
from coderank_kaggle_20261010 import sha, write_json

SETTINGS = {"model": "Qwen/Qwen3-Embedding-0.6B",
    "revision": "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3",
    "max_length": 2048, "batch_size": 8, "precision": "float16",
    "pooling": "last_nonpadding_token", "padding_side": "left",
    "truncation_side": "right", "attention": "sdpa", "dimensions": 1024,
    "instruction": "Given a solution to a programming problem, retrieve other implementations that solve the same problem.",
    "output_precision": "float32", "normalized": True}


def texts(rows):
    assert all(r["role"] in ("query", "gallery") for r in rows)
    return [f'Instruct: {SETTINGS["instruction"]}\nQuery:{r["source"]}'
            if r["role"] == "query" else r["source"] for r in rows]


def last_token_pool(hidden, mask):
    import torch
    assert mask.ndim == 2 and (mask.sum(dim=1) > 0).all()
    positions = torch.arange(mask.shape[1], device=mask.device).expand_as(mask)
    last = positions.masked_fill(mask == 0, -1).max(dim=1).values
    return hidden[torch.arange(len(hidden), device=hidden.device), last]


def validate(vectors, count):
    import numpy as np
    assert vectors.shape == (count, 1024) and vectors.dtype == np.float32
    assert np.isfinite(vectors).all()
    np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), 1., atol=1e-5)


def run(source, out):
    assert Path("/kaggle/input").is_dir() and Path("/kaggle/working").is_dir(), "Kaggle only"
    import numpy as np
    import torch
    from huggingface_hub import snapshot_download
    from transformers import AutoModel, AutoTokenizer

    assert torch.cuda.is_available() and torch.cuda.get_device_name(0) == "Tesla T4"
    design = json.loads((source.parent / "design_receipt.json").read_text())
    assert design["settings"] == SETTINGS and sha(source) == design["source_sha256"]
    rows = [json.loads(line) for line in source.read_text().splitlines()]
    assert len(rows) == 3336 and len({r["id"] for r in rows}) == len(rows)
    assert all(hashlib.sha256(r["source"].encode()).hexdigest() == r["d0"] for r in rows)
    assert sum(r["role"] == "query" for r in rows) == 276
    binding = sha(source.parent / "execution_receipt.json")
    out.mkdir(parents=True, exist_ok=True)
    assert not (out / "encoding_receipt.json").exists(), "Already completed"
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
    snapshot = Path(snapshot_download(SETTINGS["model"], revision=SETTINGS["revision"],
                                     allow_patterns=list(expected)))
    actual = {name: sha(snapshot / name) for name in expected}
    assert actual == expected, "Model/tokenizer hashes differ"
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
        report = {"binding": binding, "settings": SETTINGS, "model_files": actual,
            "source_sha256": sha(source), "script_sha256": sha(Path(__file__)),
            "created_utc": datetime.now(timezone.utc).isoformat(), "gpu": torch.cuda.get_device_name(0),
            "packages": {n: importlib.metadata.version(n) for n in ("torch", "transformers", "numpy", "huggingface-hub")},
            "model_load_seconds": load_seconds, "pilot_seconds": pilot_seconds,
            "projected_encoding_seconds": pilot_seconds * len(rows) / 32,
            "token_length_quantiles": np.quantile(lengths, [0, .5, .9, .99, 1]).tolist(),
            "truncated_programs": int((lengths > 2048).sum()),
            "training_performed": False, "java_test_accessed": False}
        write_json(out / "resource_pilot.json", report)
        print(json.dumps(report), flush=True)
        assert report["projected_encoding_seconds"] < 3600, "One-hour inference budget exceeded"
        started = time.perf_counter()
        while done < len(rows):
            if time.perf_counter() - started >= 3600:
                save()
                print("Time bound reached; resume saved checkpoint, do not restart", flush=True)
                return
            end = min(done + 128, len(rows))
            t = time.perf_counter()
            for start in range(done, end, 8):
                stop = min(start + 8, end)
                vectors[start:stop] = encode(inputs[start:stop])
            torch.cuda.synchronize()
            seconds += time.perf_counter() - t
            done = end
            save()
            print(f"Encoded {done}/{len(rows)}", flush=True)
    validate(vectors, len(rows))
    np.save(out / "embeddings.npy", vectors, allow_pickle=False)
    report.update(status="completed_external_encoding", programs=done, encoding_seconds=seconds,
        peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
        embeddings_sha256=sha(out / "embeddings.npy"), token_lengths_sha256=sha(out / "token_lengths.npy"),
        completed_utc=datetime.now(timezone.utc).isoformat())
    write_json(out / "encoding_receipt.json", report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    run(args.source, args.out)
