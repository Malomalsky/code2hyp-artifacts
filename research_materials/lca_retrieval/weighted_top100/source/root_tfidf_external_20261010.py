"""Quality-blind new-problem selection and non-neural feature preparation."""

import fresh_fusion_followup as old
import cpret_prepare
from conditional_structure_pilot_20261010 import masked_key
from frequency_factorial_20261010 import group_counts, transform
from followup_baselines import vectorize, lexical_token_stream
from rank_fusion_followup import ROOT, sha, write
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
import json
import re
import shutil
import subprocess
import sys
import time
import numpy as np
from scipy.sparse import save_npz

OUT = ROOT / "followup_results/root_tfidf_external_20261010"
PROTOCOL = ROOT / "root_tfidf_external_protocol_20261010.md"
TARGET = 650
SHARDS = (
    ("train-00024-of-00039-f4e3da0e5661e6d1.parquet", 180979071, "c958b0997f636ea83fbf21c08adda22897dd35ca8cd1e5a97e5be16f832af28d"),
    ("train-00025-of-00039-3f6ebfbaba5f4c70.parquet", 205823196, "5ae9793f8d97c4448d8053475c10007215a9d819168e3f68a9af4cc2e335da84"),
    ("train-00026-of-00039-7d4898300894cbbe.parquet", 188681982, "cb60563bba14228dba3addd8a5be6a998654113bd734b3c54271dfe6e30e63c7"),
    ("train-00027-of-00039-f8196766547533a2.parquet", 217005966, "8b1464860f954fff785255b6f744aeaaba224c0e6b9ac2c74c6b832eedbbccbb"),
    ("train-00028-of-00039-79a302af3c924863.parquet", 178924886, "f32b70263520ba04d01efda103d8ddbbf2f158b60931caae83d67f5d49b6dc62"),
    ("train-00029-of-00039-2b6615897d038115.parquet", 197596694, "a2d76b2c01471ec07d4a724391f76123d54f66dc11fb24b531c3a056cbbfe7b0"),
    ("train-00030-of-00039-4135cc54050afc22.parquet", 223399896, "fe9d77f481b639e21867942ad1061f8a3b7bfb82b3357e6c7ec96da55c6373cd"),
    ("train-00031-of-00039-40309dd907c042b7.parquet", 181140154, "2daeff73dc92b289ec9aeaa6f52af9e7665aae45ade8a172e5dad9a130a48fb2"),
    ("train-00032-of-00039-7b7d2068a3d9c359.parquet", 186381035, "fe24e9818595a3fa467a8986f4d341cf1cbcdb6f1d2c0900b4e995b25597f9ad"),
)


def now():
    return datetime.now(timezone.utc).isoformat()


def cf_id(url):
    match = re.fullmatch(r"https?://codeforces\.com/(?:contest/(\d+)/problem|problemset/problem/(\d+))/([A-Za-z0-9]+)(?:/)?", url)
    return f"{match[1] or match[2]}:{match[3]}" if match else None


def exclusions():
    ids, hashes, bindings = cpret_prepare.previous()
    receipt = old.OUT / "selection_receipt.json"
    assert sha(receipt) == "7d3a5d37c11e2247490d12a56337f382b01491a7da286f409c86b502aa56593a"
    ids.update(json.loads(receipt.read_text())["excluded_previous_problem_ids"])
    source = cpret_prepare.OUT / "selected_programs.jsonl"
    manifest_path = cpret_prepare.OUT / "selection_manifest.json"
    assert sha(manifest_path) == "8bf4f6e420dd631c2acd8370e10714bec1b295130b2a191ed6ff21e409802aa3"
    manifest = json.loads(manifest_path.read_text())
    assert sha(source) == manifest["selection_sha256"]
    for line in source.open():
        row = json.loads(line)
        problem = cf_id(row["problem_url"])
        if problem:
            ids.add(problem)
        for values, key in zip(hashes, ("d0", "d1", "d2")):
            values.add(row[key])
    for path in (receipt, source, manifest_path):
        bindings[str(path)] = sha(path)
    return ids, hashes, bindings


def prepare():
    _, _, bindings = exclusions()
    old_bindings = json.loads((old.OUT / "design_receipt.json").read_text())["bindings"]
    helper_names = ("path_context_followup.py", "path_fragment_followup.py", "codecontests_replication.py",
                    "followup_baselines.py", "validation_diagnostic.py", "rank_fusion_followup.py",
                    "reviewer_reproduction/verify_results.py")
    helpers = [ROOT / n for n in helper_names]
    helpers += sorted((old.previous.REPO / "geometry_profile_research").glob("*.py"))
    for path in helpers:
        assert sha(path) == old_bindings[str(path)], path
    checkpoint = old.VALIDATION / "seed_20260711_encoder.pt"
    assert sha(checkpoint) == old.previous.CHECKPOINT_HASH
    assert sha(ROOT / "frequency_factorial_20261010.py") == "c0e223e006bf60e747db2921f1c8c9beaa8e12f1f4926acfa670cd22d150e83e"
    assert sha(ROOT / "followup_results/frequency_factorial_20261010/result.json") == "f28c4ed4445dd32ecf58712d19ea767714f7dd2b8f4edb974c2bda28bf1824c9"
    paths = [Path(__file__), PROTOCOL, checkpoint, *helpers,
             *(ROOT / n for n in ("test_root_tfidf_external_20261010.py", "root_tfidf_colab_20261010.py",
                 "coderank_colab.py", "cpret_prepare.py", "fresh_fusion_followup.py",
                 "independent_fragment_followup.py", "frequency_factorial_20261010.py",
                 "conditional_structure_pilot_20261010.py"))]
    bindings.update({str(p): sha(p) for p in paths})
    commit = "c419f6418056b618ce373ebd6fafe6601ff51566"
    assert subprocess.check_output(["git", "-C", str(old.previous.REPO), "rev-parse", commit], text=True).strip() == commit
    fixed = {"bindings": bindings, "target": TARGET, "shards": SHARDS,
             "old_shards_for_problem_exclusion": old.SHARDS, "revision": old.previous.REVISION,
             "old_full_python_inventory_available": False,
             "frozen_commit": commit, "training_performed": False, "java_test_accessed": False}
    OUT.mkdir(exist_ok=True)
    path = OUT / "design_receipt.json"
    if path.exists():
        saved = json.loads(path.read_text())
        for key, value in fixed.items():
            assert saved[key] == json.loads(json.dumps(value)), key
    else:
        assert not list(OUT.iterdir()) and shutil.disk_usage(OUT).free > 8 * 1024**3
        write(path, {**fixed, "fixed_before_new_contents_utc": now()})
    return path


def download(shards, folder):
    folder.mkdir(exist_ok=True)
    for name, size, digest in shards:
        path = folder / name
        if not path.exists():
            partial = path.with_suffix(".part")
            url = f"https://huggingface.co/datasets/deepmind/code_contests/resolve/{old.previous.REVISION}/data/{name}?download=true"
            subprocess.run(["curl", "-fLsS", "--retry", "3", "--max-time", "600", "-o", str(partial), url], check=True)
            assert partial.stat().st_size == size and sha(partial) == digest, name
            partial.rename(path)
        assert path.stat().st_size == size and sha(path) == digest, name
        print("Verified", name, flush=True)


def choose(candidates, target):
    shared = [defaultdict(set) for _ in range(3)]
    for row in candidates:
        for mapping, key in zip(shared, ("d0", "d1", "d2")):
            mapping[row[key]].add(row["cluster_id"])
    groups, removed = defaultdict(dict), Counter()
    for row in sorted(candidates, key=lambda r: r["d0"]):
        if any(len(m[row[k]]) > 1 for m, k in zip(shared, ("d0", "d1", "d2"))):
            removed["cross_problem_duplicate"] += 1
        elif row["d2"] in groups[row["cluster_id"]]:
            removed["within_problem_duplicate"] += 1
        else:
            groups[row["cluster_id"]][row["d2"]] = row
    eligible = [p for p, rows in groups.items() if len(rows) >= 16]
    if len(eligible) < target:
        return [], len(eligible), removed
    salt = "root-tfidf-external-20261010:"
    rows = []
    for p in sorted(eligible, key=lambda p: (old.previous.digest(salt + p), p))[:target]:
        ordered = sorted(groups[p].values(), key=lambda r: (old.previous.digest(salt + p + ":" + r["d0"]), r["d0"]))[:16]
        rows.extend({**r, "role": "query" if i < 8 else "gallery",
                     "source_relpath": f"{p}/{r['d0']}.py"} for i, r in enumerate(ordered))
    return rows, len(eligible), removed


def select():
    seal = OUT / "selection_seal.json"
    if seal.exists():
        for name, expected in json.loads(seal.read_text())["sha256"].items():
            assert sha(OUT / name) == expected, name
        return [json.loads(line) for line in (OUT / "selected_programs.jsonl").open()]
    assert not (OUT / "selection_receipt.json").exists(), "Do not replace an earlier selection"
    ids, hashes, _ = exclusions()
    download(old.SHARDS, OUT / "old_input")
    for name, _, _ in old.SHARDS:
        for batch in cpret_prepare.pq.ParquetFile(OUT / "old_input" / name).iter_batches(
                columns=["source", "cf_contest_id", "cf_index"], use_threads=False):
            ids.update(f"{r['cf_contest_id']}:{r['cf_index']}" for r in batch.to_pylist() if r["source"] == 2)
    download(SHARDS, OUT / "input")
    candidates, attrition = [], Counter()
    for name, _, _ in SHARDS:
        for record in old.previous.records(OUT / "input" / name):
            attrition["input_problems"] += 1
            if record["source"] != 2:
                attrition["non_codeforces_problem"] += 1
                continue
            problem = f"{record['cf_contest_id']}:{record['cf_index']}"
            if problem in ids:
                attrition["previous_problem"] += 1
                continue
            for language, source in zip(record["solutions"]["language"], record["solutions"]["solution"]):
                if language != 3:
                    continue
                row, reason = old.previous.candidate(source)
                attrition["python3_candidates"] += 1
                if reason:
                    attrition[reason] += 1
                elif any(row[k] in h for k, h in zip(("d0", "d1", "d2"), hashes)):
                    attrition["previous_fingerprint"] += 1
                else:
                    candidates.append({**row, "cluster_id": problem, "name": record["name"]})
        print("Eligibility processed", name, flush=True)
    rows, eligible, removed = choose(candidates, TARGET)
    attrition.update(removed)
    write(OUT / "selection_receipt.json", {"completed_utc": now(), "eligible_problems": eligible,
        "target": TARGET, "attrition": dict(attrition), "excluded_problem_ids": sorted(ids),
        "exclusion_fingerprint_counts": list(map(len, hashes)), "retrieval_scores_inspected": False})
    if not rows:
        write(OUT / "status.json", {"status": "insufficient_eligible_problems_no_scores"})
        return []
    assert len(rows) == TARGET * 16 and not {r["cluster_id"] for r in rows} & ids
    for key, previous in zip(("d0", "d1", "d2"), hashes):
        assert len({r[key] for r in rows}) == len(rows)
        assert not {r[key] for r in rows} & previous
    for name, source in (("selected_programs.jsonl", True), ("metadata.jsonl", False)):
        with (OUT / name).open("x") as stream:
            for row in rows:
                assert old.previous.digest(row["source"]) == row["d0"]
                stream.write(json.dumps({k: v for k, v in row.items() if source or k != "source"}, sort_keys=True) + "\n")
    write(seal, {"fixed_before_scores_utc": now(), "sha256": {name: sha(OUT / name) for name in
        ("selected_programs.jsonl", "metadata.jsonl", "selection_receipt.json", "design_receipt.json")}})
    print("Sealed", len(rows), "programs from", TARGET, "new problems", flush=True)
    return rows


def features(rows):
    path = OUT / "features.json"
    if path.exists():
        for name, expected in json.loads(path.read_text())["sha256"].items():
            assert sha(OUT / name) == expected, name
        return
    assert not (OUT / "counts.npz").exists(), "Inspect incomplete features instead of overwriting"
    model = old.previous.load_model()
    started = time.perf_counter()
    bags, nodes, lexical, unknown = [], [], [], 0
    for i, row in enumerate(rows):
        tree = old.previous.parse_python_ast_tree(row["source"])
        counts = Counter()
        for p in model._resolve_paths(tree, None):
            retained, _, missing = old.previous.path_keys(tree, p, model.token_to_id)
            counts.update(old.previous.fragments(retained))
            unknown += missing
        bags.append(counts)
        nodes.append(old.previous.ast_node_histogram(row["source"]))
        lexical.append(Counter(lexical_token_stream(row["source"])))
        if (i + 1) % 1000 == 0:
            print("AST counts", i + 1, "/", len(rows), flush=True)
    extraction_seconds = time.perf_counter() - started
    gi = np.array([i for i, row in enumerate(rows) if row["role"] == "gallery"])
    qi = np.array([i for i, row in enumerate(rows) if row["role"] == "query"])
    keys = sorted(set().union(*(set(bags[i]) for i in gi)))
    vocabulary = {key: i for i, key in enumerate(keys)}
    groups = {key: i for i, key in enumerate(sorted({masked_key(k) for k in keys}))}
    group = np.array([groups[masked_key(k)] for k in keys])
    counts = vectorize(bags, vocabulary)
    masked = group_counts(counts, group)
    np.testing.assert_array_equal(masked.sum(axis=1), counts.sum(axis=1))
    matrices = {"ret_root_idf": transform(counts, gi, .5, True),
                "ret_raw_idf": transform(counts, gi, 1., True),
                "mask_root_idf": transform(masked, gi, .5, True),
                "ret_root": transform(counts, gi, .5, False)}
    for name, values, idf in (("nodes", nodes, False), ("lexical", lexical, True)):
        vocab = {key: i for i, key in enumerate(sorted(set().union(*(set(values[i]) for i in gi))))}
        matrices[name] = transform(vectorize(values, vocab), gi, 1., idf)
    save_npz(OUT / "counts.npz", counts)
    np.save(OUT / "group.npy", group, allow_pickle=False)
    write(OUT / "vocabulary.json", keys)
    for name, values in matrices.items():
        save_npz(OUT / (name + "_features.npz"), values)
    names = ["counts.npz", "group.npy", "vocabulary.json", *(n + "_features.npz" for n in matrices)]
    write(path, {"completed_utc": now(), "selection_seal_sha256": sha(OUT / "selection_seal.json"),
        "sha256": {n: sha(OUT / n) for n in names}, "dimensions": len(keys), "masked_groups": len(groups),
        "unknown_path_tokens": unknown, "unseen_query_count_fraction": sum(v for i in qi for k, v in bags[i].items()
            if k not in vocabulary) / sum(sum(bags[i].values()) for i in qi),
        "extraction_seconds": extraction_seconds, "total_seconds": time.perf_counter() - started,
        "retrieval_scores_inspected": False, "neural_inference_performed": False})
    print("Non-neural features complete; waiting for remote dense encoding", flush=True)


if __name__ == "__main__":
    prepare()
    if len(sys.argv) > 1 and sys.argv[1] == "prepare":
        print("Design bound before new contents", flush=True)
    else:
        rows = select()
        if rows:
            features(rows)
