"""One prospective replication of the frozen RRF, without training or tuning."""

from rank_fusion_followup import (ROOT, BASE, FUSED, REFERENCES, ranks,
    self_check, estimates, reviewer, sha, write, np)
import independent_fragment_followup as previous
from validation_diagnostic import VALIDATION
from collections import Counter, defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
import json
import platform
import resource
import shutil
import subprocess
import sys
import time

OUT = ROOT / "followup_results/fresh_fusion"
PROTOCOL = ROOT / "fresh_fusion_protocol.md"
TARGET = 650
SHARDS = (
    ("train-00015-of-00039-19ccd7331d695677.parquet", 178946180, "130ef71583565d5b2887a186cb0c64578486f1c7dfdff4bcb9c1107f807b0768"),
    ("train-00016-of-00039-bf38b0908b322307.parquet", 202144829, "7528a2ee39b08a78c6f9ffe16d9a18fb3e4de24e34fe2e899c795fc31acf2c76"),
    ("train-00017-of-00039-ae5533a2f822e6ef.parquet", 169304362, "895fe5a77c17f0ab4a364e01745a882b977e6da2e3580e8b171d59fcf816f971"),
    ("train-00018-of-00039-8c793837880f5507.parquet", 184831619, "8a214241c33628e6a01952088046b0561bd6e2941a39ce492bb59597bc93e8f6"),
    ("train-00019-of-00039-d688fad5ee604390.parquet", 190554330, "bd4b85d9ec0f7b16d38c9a296c6333f15720f0a019582d223480be5c90b2a864"),
    ("train-00020-of-00039-5d59387098675b73.parquet", 211262604, "b2a03b3940b6de703c18a0b55111e7aa979e2399b29f3f96c4abca706d5830db"),
    ("train-00021-of-00039-b257bf03d6876780.parquet", 181119876, "444b82044438c76d6e2cf79000615349b1dd7ba165da9118924ebcb427e6febc"),
    ("train-00022-of-00039-1cfd39fa43c1917c.parquet", 193915223, "63ff8266ec628a64417d7dda7754ed0366522c4cef5217e990487f83af2b1e53"),
    ("train-00023-of-00039-d078bcb55e45cbf0.parquet", 176115120, "2aa78452a6a3306536b2f000a4431c048282426a47e7197210155bedbad82c49"),
)


def old_inputs():
    paths = [(previous.OLD_INPUT, previous.OLD_HASH), (previous.INPUT, previous.INPUT_HASH)]
    for directory, shards in ((previous.OLD_OUT, previous.OLD_SHARDS),
            (previous.ORDER_OUT, previous.ORDER_SHARDS), (previous.OUT, previous.SHARDS)):
        for name, size, expected in shards:
            path = directory / "input" / name
            assert path.stat().st_size == size
            paths.append((path, expected))
    return paths


def prepare():
    paths = [PROTOCOL, Path(__file__), ROOT / "rank_fusion_followup.py",
        ROOT / "independent_fragment_followup.py", previous.INVENTORY,
        VALIDATION / "seed_20260711_encoder.pt",
        *(ROOT / name for name in ("path_fragment_followup.py", "path_order_followup.py",
            "path_context_followup.py", "codecontests_replication.py", "followup_baselines.py",
            "validation_diagnostic.py", "reviewer_reproduction/verify_results.py")),
        *sorted((previous.REPO / "geometry_profile_research").glob("*.py"))]
    bindings = {str(p): sha(p) for p in paths}
    assert bindings[str(previous.INVENTORY)] == previous.INVENTORY_HASH
    assert bindings[str(VALIDATION / "seed_20260711_encoder.pt")] == previous.CHECKPOINT_HASH
    for path, expected in old_inputs():
        assert sha(path) == expected, path
        bindings[str(path)] = expected
    assert subprocess.check_output(["git", "-C", str(previous.REPO), "rev-parse",
        "c419f6418056b618ce373ebd6fafe6601ff51566"], text=True).strip() == "c419f6418056b618ce373ebd6fafe6601ff51566"
    OUT.mkdir(parents=True, exist_ok=True)
    receipt = OUT / "design_receipt.json"
    if receipt.exists():
        assert json.loads(receipt.read_text())["bindings"] == bindings
    else:
        assert not list(OUT.iterdir()), "Unbound existing output"
        assert shutil.disk_usage(OUT).free > 8 * 1024**3
        write(receipt, {"fixed_before_new_contents_utc": datetime.now(timezone.utc).isoformat(),
            "bindings": bindings, "shards": SHARDS, "revision": previous.REVISION,
            "target": TARGET, "k": 60, "threads": 1, "training_performed": False,
            "machine": platform.machine(), "disk_free_bytes": shutil.disk_usage(OUT).free})
    return receipt


def download():
    folder = OUT / "input"
    folder.mkdir(exist_ok=True)
    for name, size, expected in SHARDS:
        path = folder / name
        if not path.exists():
            partial = path.with_suffix(".parquet.part")
            url = f"https://huggingface.co/datasets/deepmind/code_contests/resolve/{previous.REVISION}/data/{name}?download=true"
            subprocess.run(["curl", "--fail", "--location", "--silent", "--show-error",
                "--retry", "3", "--max-time", "600", "--output", str(partial), url], check=True)
            assert partial.stat().st_size == size and sha(partial) == expected, name
            partial.rename(path)
        assert path.stat().st_size == size and sha(path) == expected, name
        print("New shard verified: " + name, flush=True)


def select():
    old = [set(), set(), set()]
    count = 0
    with previous.INVENTORY.open() as stream:
        for line in stream:
            row = json.loads(line)
            count += 1
            for values, key in zip(old, ("d0_sha256", "d1_sha256", "d2_sha256")):
                if row[key]:
                    values.add(row[key])
    assert count == 240000
    old_ids = set()
    for path, _ in old_inputs():
        for record in previous.records(path):
            if record["source"] != 2:
                continue
            old_ids.add(f"{record['cf_contest_id']}:{record['cf_index']}")
            for lang, source in zip(record["solutions"]["language"], record["solutions"]["solution"]):
                if lang == 3:
                    row, _ = previous.candidate(source)
                    if row:
                        for values, key in zip(old, ("d0", "d1", "d2")):
                            values.add(row[key])
        print("Excluded previous input: " + path.name, flush=True)
    candidates, attrition, input_problems = [], Counter(), 0
    for name, _, _ in SHARDS:
        for record in previous.records(OUT / "input" / name):
            input_problems += 1
            if record["source"] != 2:
                attrition["non_codeforces_problem"] += 1
                continue
            problem = f"{record['cf_contest_id']}:{record['cf_index']}"
            if problem in old_ids:
                attrition["previous_problem"] += 1
                continue
            for lang, source in zip(record["solutions"]["language"], record["solutions"]["solution"]):
                if lang != 3:
                    continue
                attrition["python3_candidates"] += 1
                row, reason = previous.candidate(source)
                if reason:
                    attrition[reason] += 1
                elif any(row[k] in v for v, k in zip(old, ("d0", "d1", "d2"))):
                    attrition["matches_previous_inventory"] += 1
                else:
                    candidates.append({**row, "cluster_id": problem, "name": record["name"]})
        print("New eligibility processed: " + name, flush=True)
    shared = [defaultdict(set) for _ in range(3)]
    for row in candidates:
        for mapping, key in zip(shared, ("d0", "d1", "d2")):
            mapping[row[key]].add(row["cluster_id"])
    groups = defaultdict(dict)
    for row in sorted(candidates, key=lambda r: r["d0"]):
        if any(len(m[row[k]]) > 1 for m, k in zip(shared, ("d0", "d1", "d2"))):
            attrition["cross_problem_duplicate"] += 1
        elif row["d2"] in groups[row["cluster_id"]]:
            attrition["within_problem_duplicate"] += 1
        else:
            groups[row["cluster_id"]][row["d2"]] = row
    eligible = [p for p, g in groups.items() if len(g) >= 16]
    write(OUT / "selection_receipt.json", {"completed_utc": datetime.now(timezone.utc).isoformat(),
        "target": TARGET, "eligible_problems": len(eligible), "input_problems": input_problems,
        "attrition": dict(attrition), "excluded_previous_problem_ids": sorted(old_ids),
        "previous_inputs": {str(p): h for p, h in old_inputs()},
        "inventory_sha256": previous.INVENTORY_HASH})
    print(f"Fresh eligibility: {len(eligible)}; required {TARGET}", flush=True)
    if len(eligible) < TARGET:
        write(OUT / "result.json", {"status": "insufficient_eligible_problems_no_scores",
            "design_receipt_sha256": sha(OUT / "design_receipt.json"),
            "selection_receipt_sha256": sha(OUT / "selection_receipt.json")})
        return []
    rows = []
    for p in sorted(eligible, key=lambda p: previous.digest(f"fresh-fusion-problems-20261002:{p}"))[:TARGET]:
        ordered = sorted(groups[p].values(), key=lambda r: previous.digest(f"fresh-fusion-sources-20261002:{p}:{r['d0']}"))[:16]
        rows.extend({**r, "role": "query" if i < 8 else "gallery",
            "source_relpath": f"{p}/{r['d0']}.py"} for i, r in enumerate(ordered))
    for name, include_source in (("selected_programs.jsonl", True), ("metadata.jsonl", False)):
        with (OUT / name).open("x") as stream:
            for row in rows:
                stream.write(json.dumps({k: v for k, v in row.items() if include_source or k != "source"}, sort_keys=True) + "\n")
    write(OUT / "selection_seal.json", {"fixed_before_scores_utc": datetime.now(timezone.utc).isoformat(),
        "sha256": {name: sha(OUT / name) for name in ("selected_programs.jsonl", "metadata.jsonl",
            "selection_receipt.json", "design_receipt.json")}})
    return rows


def run():
    design = prepare()
    assert not (OUT / "result.json").exists(), "Verify completed outputs; do not overwrite"
    download()
    started = time.perf_counter()
    if (OUT / "selected_programs.jsonl").exists():
        for name, expected in json.loads((OUT / "selection_seal.json").read_text())["sha256"].items():
            assert sha(OUT / name) == expected
        rows = [json.loads(line) for line in (OUT / "selected_programs.jsonl").read_text().splitlines()]
    else:
        rows = select()
    if not rows:
        return
    assert len(rows) == TARGET * 16 and len({r["d2"] for r in rows}) == len(rows)
    assert all(previous.digest(r["source"]) == r["d0"] for r in rows)
    _, queries, gallery = __import__("rank_fusion_followup").metadata(OUT)
    assert len({r["cluster_id"] for r in rows}) == TARGET
    selection_seconds = time.perf_counter() - started
    model = previous.load_model()
    bags, unknown = {name: [] for name in BASE}, 0
    feature_started = time.perf_counter()
    for i, row in enumerate(rows):
        tree = previous.parse_python_ast_tree(row["source"])
        retained, masked = Counter(), Counter()
        for path in model._resolve_paths(tree, None):
            a, b, missing = previous.path_keys(tree, path, model.token_to_id)
            retained.update(previous.fragments(a))
            masked.update(previous.fragments(b))
            unknown += missing
        assert sum(retained.values()) == sum(masked.values())
        for name, bag in zip(BASE, (previous.ast_node_histogram(row["source"]), retained, masked)):
            bags[name].append(bag)
        if (i + 1) % 1000 == 0:
            print(f"Unchanged features: {i+1}/{len(rows)}", flush=True)
    feature_seconds = time.perf_counter() - feature_started
    methods = previous.evaluate(bags, rows, OUT)
    matrices = {n: np.load(OUT / (n + ".npy"), mmap_mode="r", allow_pickle=False) for n in BASE}
    tie = np.array([r["source_relpath"].encode("utf-8") for r in gallery])
    from followup_baselines import torch, summarize_problem_macro_retrieval
    for name, source in zip(FUSED, BASE[1:]):
        path = OUT / (name + ".npy")
        matrix = np.lib.format.open_memmap(path, mode="w+", dtype=np.float64, shape=matrices["ast_nodes"].shape)
        for i in range(len(queries)):
            matrix[i] = -(1. / (60 + ranks(matrices["ast_nodes"][i], tie))
                        + 1. / (60 + ranks(matrices[source][i], tie)))
        matrix.flush()
        metrics = asdict(summarize_problem_macro_retrieval(torch.from_numpy(matrix),
            query_ids=[r["source_relpath"] for r in queries], query_cluster_ids=[r["cluster_id"] for r in queries],
            gallery_ids=[r["source_relpath"] for r in gallery], gallery_cluster_ids=[r["cluster_id"] for r in gallery]))
        methods[name] = {"metrics": metrics, "negative_score_sha256": sha(path)}
    indices = np.random.Generator(np.random.PCG64(20261002)).integers(TARGET, size=(20000, TARGET))
    np.save(OUT / "bootstrap_indices.npy", indices, allow_pickle=False)
    result = {"status": "completed_prospective_fixed_fusion", "completed_utc": datetime.now(timezone.utc).isoformat(),
        "design_receipt_sha256": sha(design), "selection_seal_sha256": sha(OUT / "selection_seal.json"),
        "problems": TARGET, "queries": len(queries), "gallery": len(gallery), "k": 60,
        "methods": methods, "contrasts": estimates(methods, indices),
        "bootstrap_indices_sha256": sha(OUT / "bootstrap_indices.npy"),
        "unknown_path_tokens": unknown, "selection_seconds": selection_seconds, "feature_seconds": feature_seconds,
        "elapsed_seconds_excluding_download": time.perf_counter() - started,
        "python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__,
        "max_rss_bytes_macos": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "threads": 1, "training_performed": False, "java_test_accessed": False}
    write(OUT / "result.json", result)
    print(json.dumps({"MAP@8": {n: v["metrics"]["problem_macro_map_at_r"] for n, v in methods.items()},
        "contrasts": result["contrasts"]}), flush=True)


def verify():
    result = json.loads((OUT / "result.json").read_text())
    assert result["status"] == "completed_prospective_fixed_fusion"
    design = OUT / "design_receipt.json"
    assert sha(design) == result["design_receipt_sha256"]
    for name, expected in json.loads(design.read_text())["bindings"].items():
        assert sha(Path(name)) == expected, name
    seal = OUT / "selection_seal.json"
    assert sha(seal) == result["selection_seal_sha256"]
    for name, expected in json.loads(seal.read_text())["sha256"].items():
        assert sha(OUT / name) == expected
    rows, queries, gallery = __import__("rank_fusion_followup").metadata(OUT)
    excluded = set(json.loads((OUT / "selection_receipt.json").read_text())["excluded_previous_problem_ids"])
    assert not excluded.intersection(r["cluster_id"] for r in rows)
    assert len(rows) == TARGET * 16 and len({r["d2"] for r in rows}) == len(rows)
    checked = {}
    matrices = {n: np.load(OUT / (n + ".npy"), mmap_mode="r", allow_pickle=False) for n in (*BASE, *FUSED)}
    tie_order = np.argsort([r["source_relpath"].encode("utf-8") for r in gallery])
    for i in range(len(queries)):
        weights = {}
        for name in BASE:
            order = tie_order[np.argsort(matrices[name][i, tie_order], kind="stable")]
            weights[name] = np.reciprocal(60. + np.argsort(order) + 1)
        for name, source in zip(FUSED, BASE[1:]):
            np.testing.assert_array_equal(matrices[name][i], -weights["ast_nodes"] - weights[source])
    for name in (*BASE, *FUSED):
        path = OUT / (name + ".npy")
        key = "distance_sha256" if name in BASE else "negative_score_sha256"
        assert sha(path) == result["methods"][name][key]
        reconstructed = reviewer.metrics(path, rows, result["methods"][name]["metrics"])
        checked[name] = {"metrics": {"task_scores": reconstructed["task_scores"]}}
        print("Independent ranking passed: " + name, flush=True)
    indices = np.load(OUT / "bootstrap_indices.npy", allow_pickle=False)
    assert sha(OUT / "bootstrap_indices.npy") == result["bootstrap_indices_sha256"]
    np.testing.assert_array_equal(indices, np.random.Generator(np.random.PCG64(20261002)).integers(TARGET, size=(20000, TARGET)))
    assert estimates(checked, indices) == result["contrasts"]
    write(OUT / "independent_verification.json", {"status": "passed",
        "verified_utc": datetime.now(timezone.utc).isoformat(), "result_sha256": sha(OUT / "result.json"),
        "rankings": 5, "contrasts": 3, "independent_fusion_reconstruction": True})
    print("All five rankings and three fixed contrasts verified", flush=True)


if __name__ == "__main__":
    self_check()
    previous.self_check()
    previous.torch.set_num_threads(1)
    previous.torch.set_num_interop_threads(1)
    if sys.argv[1:] == ["--prepare"]:
        print("Prospective design SHA-256: " + sha(prepare()), flush=True)
    elif sys.argv[1:] == ["--verify"]:
        verify()
    elif sys.argv[1:] == ["--self-check"]:
        print("Existing fragment/masking and RRF/tie self-checks passed")
    elif not sys.argv[1:]:
        run()
    else:
        raise ValueError("Use --prepare, --verify, --self-check, or no arguments")
