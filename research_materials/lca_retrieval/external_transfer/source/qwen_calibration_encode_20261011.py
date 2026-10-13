"""Source adapter only; reuse the frozen Qwen inference operations."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import crossed_encode_20261011 as frozen
from coderank_kaggle_20261010 import sha


def source_rows(path, model):
    assert model == 'qwen'
    receipt = json.loads((path.parent/'execution_receipt.json').read_text())
    expected = receipt['cohort']
    assert sha(path) == expected['source_sha256']
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(rows) == expected['programs']
    assert len({r['source_relpath'] for r in rows}) == len(rows)
    assert Counter(r['role'] for r in rows) == expected['roles']
    assert all(hashlib.sha256(r['source'].encode()).hexdigest() == r['d0'] for r in rows)
    return rows


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    assert Path('/kaggle/input').is_dir() and Path('/kaggle/working').is_dir()
    receipt = json.loads((args.source.parent/'execution_receipt.json').read_text())
    for name, digest in receipt['payload_sha256'].items():
        assert Path(name).name == name and sha(args.source.parent/name) == digest, name
    frozen.source_rows = source_rows
    frozen.run_qwen(args.source, args.out, receipt)
