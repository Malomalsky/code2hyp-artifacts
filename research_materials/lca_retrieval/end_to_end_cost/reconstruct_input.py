"""Restore the exact private benchmark payload from locally supplied upstream shards."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
from verify_results import ROOT, read, sha, verify_manifest
from source.source_normalization import normalize_python_source


def run(parquet_dir, destination):
    import pyarrow.parquet as pq
    verify_manifest()
    assert not destination.exists(), "Use a new directory; existing inputs are never overwritten"
    design = read("data/input/original_selection_design.json")
    metadata = read("data/input/program_metadata.json")
    needed = {p["source_sha256"] for rows in metadata.values() for p in rows}
    found = {}
    for name, size, digest in design["shards"]:
        path = parquet_dir / name
        assert path.stat().st_size == size and sha(path) == digest, name
        for batch in pq.ParquetFile(path).iter_batches(batch_size=128, columns=["solutions"], use_threads=False):
            for row in batch.to_pylist():
                for language, source in zip(row["solutions"]["language"], row["solutions"]["solution"]):
                    if language != 3:
                        continue
                    canonical = normalize_python_source(source.encode("utf-8"))
                    if not canonical.decode_ok:
                        continue
                    fingerprint = hashlib.sha256(canonical.text.encode()).hexdigest()
                    if fingerprint in needed:
                        found[fingerprint] = canonical.text
    assert set(found) == needed, f"Missing {len(needed-set(found))} selected programs"
    programs = {role:[{"id":p["id"], "source":found[p["source_sha256"]]} for p in rows]
                for role, rows in metadata.items()}
    text = json.dumps(programs, indent=2, ensure_ascii=False)+"\n"
    manifest = read("data/input/e2e_manifest.json")
    assert hashlib.sha256(text.encode()).hexdigest() == manifest["files"]["programs.json"]
    destination.mkdir(parents=True)
    (destination / "programs.json").write_text(text)
    for name, digest in manifest["files"].items():
        if name == "programs.json":
            continue
        source = ROOT / "data/input" / name
        if not source.exists():
            source = ROOT / "source" / name
        assert sha(source) == digest, name
        shutil.copy2(source, destination / name)
    shutil.copy2(ROOT / "data/input/e2e_manifest.json", destination / "e2e_manifest.json")
    print("PASS: all 6400 program identities and original payload SHA-256 bindings restored")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.parquet_dir, args.output)
