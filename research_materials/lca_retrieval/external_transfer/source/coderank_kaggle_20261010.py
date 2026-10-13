"""Bounded, resumable Kaggle encoding of the frozen new-problem cohort."""

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

MODEL = "nomic-ai/CodeRankEmbed"
REVISION = "3c4b60807d71f79b43f3c4363786d9493691f8b1"
SOURCE_SHA = "fb94d1a3f179d76d5b1a97e3e70a8db9ee7da08957c3b5158662285ac6622ac4"
SETTINGS = {"model": MODEL, "revision": REVISION, "max_length": 512,
            "batch_size": 16, "precision": "float32", "query_prefix": "",
            "pooling": "cls", "normalized": True, "source_sha256": SOURCE_SHA,
            "external_design_sha256": "a6cc83dc1798888eb42ac96436823ba9b3050564c0b33b227030abaac02c0685",
            "execution_platform": "Kaggle", "execution_addendum_sha256": "c9b112fefd020371245f509d0c659f9dbcb9e7b6e8ebb56648d5ecd68ae12004"}


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def load_sources(path):
    if sha(path) != SOURCE_SHA:
        raise ValueError("Wrong source file")
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(rows) == 10400 and len({r["source_relpath"] for r in rows}) == 10400
    for role in ("query", "gallery"):
        counts = Counter(r["cluster_id"] for r in rows if r["role"] == role)
        assert len(counts) == 650 and set(counts.values()) == {8}
    assert all(hashlib.sha256(r["source"].encode()).hexdigest() == r["d0"] for r in rows)
    return rows


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def run(source, out, mode):
    # No fallback to local or CPU neural inference.
    assert Path("/kaggle/input").is_dir() and Path("/kaggle/working").is_dir(), "Kaggle only"
    import importlib.metadata
    import numpy as np
    import torch
    from huggingface_hub import snapshot_download
    from sentence_transformers import SentenceTransformer

    assert torch.cuda.is_available(), "A Kaggle GPU is required"
    assert torch.cuda.get_device_name() == "Tesla T4", "Only the registered T4 is allowed"
    rows = load_sources(source)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "encoding_receipt.json").exists():
        raise FileExistsError("Completed encoding already exists")
    checkpoint = out / "embeddings.partial.npz"
    binding = hashlib.sha256(json.dumps(SETTINGS, sort_keys=True).encode()).hexdigest()
    vectors = np.empty((len(rows), 768), dtype=np.float32)
    done, encoding_seconds = 0, 0.
    if checkpoint.exists():
        with np.load(checkpoint, allow_pickle=False) as state:
            assert str(state["binding"]) == binding
            old = state["vectors"]
            assert old.ndim == 2 and old.shape[1] == 768 and len(old) <= len(rows)
            assert np.isfinite(old).all()
            np.testing.assert_allclose(np.linalg.norm(old, axis=1), 1., atol=1e-5)
            done = len(old)
            vectors[:done] = old
            encoding_seconds = float(state["encoding_seconds"])

    def save():
        temporary = checkpoint.with_suffix(".tmp")
        with temporary.open("wb") as stream:
            np.savez(stream, vectors=vectors[:done], binding=binding,
                     encoding_seconds=encoding_seconds)
        temporary.replace(checkpoint)

    t = time.perf_counter()
    snapshot = Path(snapshot_download(MODEL, revision=REVISION))
    files = {str(p.relative_to(snapshot)): sha(p) for p in snapshot.rglob("*") if p.is_file()}
    expected = json.loads((source.parent / "expected_model_files.json").read_text())
    assert files == expected, "Frozen model-file hashes differ; no inference permitted"
    model = SentenceTransformer(str(snapshot), trust_remote_code=True, device="cuda").float().eval()
    model.max_seq_length = SETTINGS["max_length"]
    assert model[1].pooling_mode_cls_token and not model[1].pooling_mode_mean_tokens
    load_seconds = time.perf_counter() - t
    sources = [r["source"] for r in rows]
    lengths = np.array([len(model.tokenizer.encode(s, truncation=False)) for s in sources])
    files = {str(p.relative_to(snapshot)): sha(p) for p in snapshot.rglob("*") if p.is_file()}
    torch.cuda.reset_peak_memory_stats()

    def encode(texts):
        return model.encode(texts, batch_size=SETTINGS["batch_size"],
                            normalize_embeddings=True, convert_to_numpy=True,
                            show_progress_bar=False)

    with torch.inference_mode():
        encode(["pass\n"] * SETTINGS["batch_size"])
        torch.cuda.synchronize()
        pilot_ids = np.argsort(lengths, kind="stable")[np.linspace(0, len(rows)-1, 64, dtype=int)]
        t = time.perf_counter()
        pilot = encode([sources[i] for i in pilot_ids])
        torch.cuda.synchronize()
        pilot_seconds = time.perf_counter() - t
        assert pilot.shape == (64, 768) and np.isfinite(pilot).all()
        np.testing.assert_allclose(np.linalg.norm(pilot, axis=1), 1., atol=1e-5)
        report = {"settings": SETTINGS, "settings_sha256": binding,
            "script_sha256": sha(Path(__file__)), "created_utc": datetime.now(timezone.utc).isoformat(),
            "gpu": torch.cuda.get_device_name(), "gpu_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
            "packages": {n: importlib.metadata.version(n) for n in
                         ("torch", "transformers", "sentence-transformers", "numpy")},
            "model_files": files, "model_load_seconds": load_seconds,
            "pilot_seconds": pilot_seconds, "pilot_programs": 64,
            "projected_encoding_seconds": pilot_seconds * len(rows) / 64,
            "token_length_quantiles": np.quantile(lengths, [0, .5, .9, .99, 1]).tolist(),
            "truncated_programs": int((lengths > 512).sum()),
            "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
            "training_performed": False, "java_test_accessed": False}
        write_json(out / "resource_pilot.json", report)
        print(json.dumps(report), flush=True)
        if mode == "pilot":
            return
        assert report["projected_encoding_seconds"] < 3600, "Pilot exceeds one-hour inference budget"
        started = time.perf_counter()
        while done < len(rows):
            if time.perf_counter() - started >= 3600:
                save()
                print("One-hour bound reached; checkpoint saved", flush=True)
                return
            end = min(done + 128, len(rows))
            t = time.perf_counter()
            block = encode(sources[done:end])
            torch.cuda.synchronize()
            encoding_seconds += time.perf_counter() - t
            assert block.shape == (end-done, 768) and np.isfinite(block).all()
            np.testing.assert_allclose(np.linalg.norm(block, axis=1), 1., atol=1e-5)
            vectors[done:end] = block
            done = end
            save()
            print(f"Encoded {done}/{len(rows)}", flush=True)
    np.save(out / "embeddings.npy", vectors, allow_pickle=False)
    np.save(out / "token_lengths.npy", lengths, allow_pickle=False)
    report.update({"status": "completed_exploratory_encoding", "programs": done,
        "encoding_seconds": encoding_seconds, "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
        "embeddings_sha256": sha(out / "embeddings.npy"),
        "token_lengths_sha256": sha(out / "token_lengths.npy"),
        "completed_utc": datetime.now(timezone.utc).isoformat()})
    write_json(out / "encoding_receipt.json", report)
    print("Completed encoding; retrieve embeddings and receipt before releasing the VM", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("check", "pilot", "full"), default="pilot")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("/kaggle/working/root_tfidf_encoding"))
    args = parser.parse_args()
    if args.mode == "check":
        assert sha(Path(__file__)) and len(load_sources(args.source)) == 10400
        print("PASS: sealed cohort membership and all exact source hashes")
    else:
        run(args.source, args.out, args.mode)
