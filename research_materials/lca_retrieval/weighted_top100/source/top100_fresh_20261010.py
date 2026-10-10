"""Fixed unused-problem selection and AST/count preparation; no neural inference."""

import root_tfidf_external_20261010 as previous
from rank_fusion_followup import ROOT, sha, write
from followup_baselines import vectorize, lexical_token_stream
from collections import Counter
from pathlib import Path
import json
import subprocess
import sys
import time
import numpy as np
from scipy.sparse import save_npz

OUT = ROOT / "followup_results/top100_fresh_20261010"
TARGET = 400
PROTOCOL = ROOT / "top100_fresh_protocol_20261010.md"
SHARDS = (
    ("train-00033-of-00039-53b0f749aacff9c1.parquet", 203741870, "65366660f211cfa6e18c81086348713805dd874b708b0bf2e3d0b6de2556cebf"),
    ("train-00034-of-00039-a36ff0bff7d2a76f.parquet", 187767858, "5ea8c872ec944013b661f23203f8d4b82e46dc8dec204c32af89ba7153035171"),
    ("train-00035-of-00039-d28f9be60314601f.parquet", 151356957, "4a7ed07e1f67ca32ed03a874b96dfc9f4f43be0563734b33085bf0b653c65c47"),
    ("train-00036-of-00039-146e1a11c054aeab.parquet", 204394454, "cb14ebff9529e89c5fd5659b7de0352858f6a514fc63e2111daed392511eafa8"),
    ("train-00037-of-00039-995207c374a4e6f2.parquet", 231402023, "bc6e40d8b2de70a52056f0d25d70cac57dd77b8671fa5d1e467e330156addca0"),
    ("train-00038-of-00039-96a59dd6a98cd075.parquet", 204141641, "62a3f1e1a56485b7b8514d969e803a844fae89e29b985995109f860f50d042ed"),
)


def exclusions():
    ids, hashes, bindings = previous.exclusions()
    late = previous.OUT
    seal = json.loads((late / "selection_seal.json").read_text())
    assert sha(late / "selection_seal.json") == "ee964544f65bd1c7cac3ef40a97c653e093ac7448a7edd220bad44c59c67927d"
    for name in ("metadata.jsonl", "selection_receipt.json"):
        assert sha(late / name) == seal["sha256"][name]
        bindings[str(late / name)] = seal["sha256"][name]
    ids.update(json.loads((late / "selection_receipt.json").read_text())["excluded_problem_ids"])
    for row in map(json.loads, (late / "metadata.jsonl").read_text().splitlines()):
        ids.add(row["cluster_id"])
        for values, key in zip(hashes, ("d0", "d1", "d2")):
            values.add(row[key])
    for name, size, digest in previous.SHARDS:
        path = late / "input" / name
        assert path.stat().st_size == size and sha(path) == digest
        bindings[str(path)] = digest
        for batch in previous.cpret_prepare.pq.ParquetFile(path).iter_batches(
                columns=["source", "cf_contest_id", "cf_index"], use_threads=False):
            ids.update(f"{r['cf_contest_id']}:{r['cf_index']}" for r in batch.to_pylist() if r["source"] == 2)
    return ids, hashes, bindings


def prepare():
    _, _, bindings = exclusions()
    old_design = json.loads((previous.OUT / "design_receipt.json").read_text())
    for path, digest in old_design["bindings"].items():
        assert sha(Path(path)) == digest, path
    bindings.update(old_design["bindings"])
    bm = ROOT / "followup_results/bm25_20261010"
    assert sha(bm / "result.json") == "056c91f91942ccff203b73e999ad443e847503b5328800a71818b731a50f0f8e"
    calibration = json.loads((bm / "calibration.json").read_text())
    assert calibration["alpha"] == .10
    assert sha(bm / "calibration.json") == "94c82b2e5a261fe9b1f5a9b2ed8ccfbc0ad9568dcd4596abae219785cf25c244"
    paths = [Path(__file__), PROTOCOL, ROOT / "top100_fresh_evaluate_20261010.py",
        ROOT / "top100_fresh_encode_20261010.py", ROOT / "test_top100_fresh_20261010.py",
        ROOT / "coderank_kaggle_20261010.py", ROOT / "root_tfidf_external_20261010.py",
        ROOT / "bm25_20261010.py", ROOT / "two_stage_20261010.py",
        ROOT / "followup_results/top100_fresh_resources_20261010.json", bm / "calibration.json"]
    bindings.update({str(p): sha(p) for p in paths})
    commit = "c419f6418056b618ce373ebd6fafe6601ff51566"
    assert subprocess.check_output(["git", "-C", str(previous.old.previous.REPO), "rev-parse", commit], text=True).strip() == commit
    fixed = {"bindings": bindings, "target": TARGET, "shards": SHARDS,
        "revision": previous.old.previous.REVISION, "frozen_commit": commit,
        "primary": "root_100_minus_D_BM25", "cutoff": 100, "bootstrap_resamples": 20000,
        "bootstrap_seed": 20261010, "bm25_alpha": .10,
        "old_full_python_inventory_available": False, "java_test_accessed": False}
    OUT.mkdir(exist_ok=True)
    path = OUT / "design_receipt.json"
    if path.exists():
        saved = json.loads(path.read_text())
        assert all(saved[k] == json.loads(json.dumps(v)) for k, v in fixed.items())
    else:
        assert not list(OUT.iterdir()), "Inspect unbound outputs before proceeding"
        write(path, {**fixed, "fixed_before_new_contents_utc": previous.now()})
    return path


def select():
    seal = OUT / "selection_seal.json"
    if seal.exists():
        for name, digest in json.loads(seal.read_text())["sha256"].items():
            assert sha(OUT / name) == digest
        return [json.loads(line) for line in (OUT / "selected_programs.jsonl").open()]
    assert not (OUT / "selection_receipt.json").exists()
    ids, hashes, _ = exclusions()
    previous.download(SHARDS, OUT / "input")
    candidates, attrition = [], Counter()
    for name, _, _ in SHARDS:
        for record in previous.old.previous.records(OUT / "input" / name):
            attrition["input_problems"] += 1
            if record["source"] != 2:
                attrition["non_codeforces"] += 1
                continue
            problem = f"{record['cf_contest_id']}:{record['cf_index']}"
            if problem in ids:
                attrition["previous_problem"] += 1
                continue
            for language, source in zip(record["solutions"]["language"], record["solutions"]["solution"]):
                if language != 3:
                    continue
                row, reason = previous.old.previous.candidate(source)
                attrition["python3_candidates"] += 1
                if reason:
                    attrition[reason] += 1
                elif any(row[k] in h for k, h in zip(("d0", "d1", "d2"), hashes)):
                    attrition["previous_fingerprint"] += 1
                else:
                    candidates.append({**row, "cluster_id": problem, "name": record["name"]})
        print("Eligibility processed", name, flush=True)
    rows, eligible, removed = previous.choose(candidates, TARGET)
    attrition.update(removed)
    write(OUT / "selection_receipt.json", {"completed_utc": previous.now(), "eligible_problems": eligible,
        "target": TARGET, "attrition": dict(attrition), "excluded_problem_ids": sorted(ids),
        "exclusion_fingerprint_counts": list(map(len, hashes)), "retrieval_scores_inspected": False})
    if not rows:
        write(OUT / "status.json", {"status": "insufficient_eligible_problems_no_scores"})
        return []
    assert len(rows) == 16*TARGET and not {r["cluster_id"] for r in rows} & ids
    for key, previous_hashes in zip(("d0", "d1", "d2"), hashes):
        assert len({r[key] for r in rows}) == len(rows)
        assert not {r[key] for r in rows} & previous_hashes
    for name, source in (("selected_programs.jsonl", True), ("metadata.jsonl", False)):
        with (OUT / name).open("x") as stream:
            for row in rows:
                assert previous.old.previous.digest(row["source"]) == row["d0"]
                stream.write(json.dumps({k: v for k, v in row.items() if source or k != "source"}, sort_keys=True)+"\n")
    write(seal, {"fixed_before_scores_utc": previous.now(), "sha256": {n: sha(OUT/n) for n in
        ("selected_programs.jsonl", "metadata.jsonl", "selection_receipt.json", "design_receipt.json")}})
    return rows


def features(rows):
    # Reuse the frozen AST implementation with only its output directory rebound.
    original = previous.OUT
    try:
        previous.OUT = OUT
        previous.features(rows)
    finally:
        previous.OUT = original
    path = OUT / "bm25_features.json"
    if path.exists():
        for name, digest in json.loads(path.read_text())["sha256"].items():
            assert sha(OUT/name) == digest
        return
    started = time.perf_counter()
    bags = [Counter(lexical_token_stream(row["source"])) for row in rows]
    qi = [i for i, r in enumerate(rows) if r["role"] == "query"]
    gi = [i for i, r in enumerate(rows) if r["role"] == "gallery"]
    keys = sorted(set().union(*(set(bags[i]) for i in gi)))
    vocab = {k: i for i, k in enumerate(keys)}
    for name, indices in (("query", qi), ("gallery", gi)):
        counts = vectorize([bags[i] for i in indices], vocab)
        counts.sort_indices()
        save_npz(OUT / (name+"_lexical_counts.npz"), counts)
    write(OUT / "lexical_vocabulary.json", keys)
    names = ("query_lexical_counts.npz", "gallery_lexical_counts.npz", "lexical_vocabulary.json")
    write(path, {"sha256": {n: sha(OUT/n) for n in names}, "seconds": time.perf_counter()-started,
        "query_oov_occurrences": sum(v for i in qi for k, v in bags[i].items() if k not in vocab)})


if __name__ == "__main__":
    prepare()
    if sys.argv[1:] == ["--prepare"]:
        print("Design frozen before new contents", flush=True)
    elif sys.argv[1:] == ["--run"]:
        rows = select()
        if rows:
            features(rows)
            print(f"Prepared {TARGET} unseen problems / {len(rows)} programs", flush=True)
    else:
        raise SystemExit("Usage: top100_fresh_20261010.py --prepare|--run")
