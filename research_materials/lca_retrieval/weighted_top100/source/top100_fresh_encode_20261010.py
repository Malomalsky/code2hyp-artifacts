"""Cohort adapter around the unchanged, bounded Kaggle T4 encoder."""

import coderank_kaggle_20261010 as encoder
from collections import Counter
from pathlib import Path
import argparse
import hashlib
import json


def load_sources(path):
    assert encoder.sha(path) == encoder.SOURCE_SHA
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(rows) == 6400 and len({r["source_relpath"] for r in rows}) == 6400
    for role in ("query", "gallery"):
        counts = Counter(r["cluster_id"] for r in rows if r["role"] == role)
        assert len(counts) == 400 and set(counts.values()) == {8}
    assert Counter(r["cluster_id"] for r in rows if r["role"] == "query") == Counter(
        r["cluster_id"] for r in rows if r["role"] == "gallery")
    assert all(hashlib.sha256(r["source"].encode()).hexdigest() == r["d0"] for r in rows)
    return rows


def configure(source):
    receipt = json.loads((source.parent / "top100_execution_receipt.json").read_text())
    for name, digest in receipt["payload_sha256"].items():
        assert Path(name).name == name and encoder.sha(source.parent/name) == digest, name
    encoder.SOURCE_SHA = receipt["source_sha256"]
    encoder.SETTINGS = {**encoder.SETTINGS, "source_sha256": encoder.SOURCE_SHA,
        "external_design_sha256": receipt["design_sha256"],
        "execution_addendum_sha256": receipt["protocol_sha256"], "study": "top100_fresh_20261010"}
    encoder.load_sources = load_sources
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    configure(args.source)
    encoder.run(args.source, args.out, "full")
