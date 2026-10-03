"""Offline numerical reproduction from saved scores; only NumPy is required."""

import hashlib
import json
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
KEYS = ("problem_macro_map_at_r", "mrr", "recall_at_10")
CELLS = ("EEE_true_LCA", "EEE_zero_anchor")


def read(name):
    return json.loads((ROOT / name).read_text(encoding="utf-8"))


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rows(name):
    return [json.loads(line) for line in (ROOT / name).read_text(encoding="utf-8").splitlines()]


def metrics(matrix, metadata, expected):
    queries = [r for r in metadata if r["role"] == "query"]
    gallery = [r for r in metadata if r["role"] == "gallery"]
    distances = np.load(ROOT / matrix, mmap_mode="r", allow_pickle=False)
    assert distances.shape == (len(queries), len(gallery))
    assert len({r["source_relpath"] for r in metadata}) == len(metadata)
    tie = np.array([r["source_relpath"].encode("utf-8") for r in gallery])
    labels = np.array([r["cluster_id"] for r in gallery])
    task, reciprocal, success = {}, [], []
    for values, query in zip(distances, queries):
        assert np.isfinite(values).all()
        relevant = labels[np.lexsort((tie, values))] == query["cluster_id"]
        assert relevant.sum() == 8
        ap = float(np.sum(np.cumsum(relevant[:8]) * relevant[:8] / np.arange(1, 9)) / 8)
        np.testing.assert_allclose(ap, expected["query_scores"][query["source_relpath"]], atol=1e-12, rtol=0)
        task.setdefault(query["cluster_id"], []).append(ap)
        first = int(np.flatnonzero(relevant)[0]) + 1
        reciprocal.append(1 / first)
        success.append(first <= 10)
    task = {p: float(np.mean(v)) for p, v in sorted(task.items())}
    assert set(task) == set(expected["task_scores"])
    np.testing.assert_allclose(list(task.values()), [expected["task_scores"][p] for p in task], atol=1e-12, rtol=0)
    aggregate = [float(np.mean(list(task.values()))), float(np.mean(reciprocal)), float(np.mean(success))]
    np.testing.assert_allclose(aggregate, [expected[k] for k in KEYS], atol=1e-12, rtol=0)
    return {"MAP_at_8": aggregate[0], "MRR": aggregate[1], "Success_at_10": aggregate[2], "task_scores": task}


def contrast(values, indices):
    return {"mean": float(values.mean()), "percentile_95_interval": np.quantile(
        values[indices].mean(axis=1), [.025, .975]).tolist()}


def verify():
    manifest = read("MANIFEST.json")
    for name, entry in manifest.items():
        relative = Path(name)
        assert not relative.is_absolute() and ".." not in relative.parts
        path = ROOT / relative
        assert not path.is_symlink() and path.is_file()
        assert path.stat().st_size == entry["bytes"] and sha(path) == entry["sha256"], name
    print(f"Verified SHA-256 for {len(manifest)} supplied files", flush=True)
    metadata = rows("data/test_programs.jsonl")
    h1 = {cell: [] for cell in CELLS}
    for seed in range(20260711, 20260721):
        receipt = read(f"data/h1/seed_{seed}.json")
        assert receipt["seed"] == seed
        for cell in CELLS:
            h1[cell].append(metrics(f"data/h1/{seed}_{cell}.npy", metadata, receipt["cells"][cell]["metrics"]))
        print(f"H1 seed {seed}: both rankings verified", flush=True)
    report = read("data/h1/interim_h1_report.json")
    problems = sorted(h1[CELLS[0]][0]["task_scores"])
    means = {c: np.array([sum(r["task_scores"][p] for r in h1[c]) / 10 for p in problems]) for c in CELLS}
    indices = np.load(ROOT / "data/h1/bootstrap_indices.npy", allow_pickle=False)
    assert indices.shape == (20000, 386) and indices.min() >= 0 and indices.max() < 386
    digest = hashlib.sha256(indices.astype("<i8").tobytes(order="C")).hexdigest()
    assert digest == report["bootstrap"]["shared_resample_index_sha256"]
    values = means[CELLS[0]] - means[CELLS[1]]
    estimate = contrast(values, indices)
    expected = report["H1_EEE_true_LCA_minus_EEE_zero_anchor"]
    np.testing.assert_allclose([estimate["mean"], *estimate["percentile_95_interval"]],
        [expected["point_estimate_delta_problem_macro_MAP_at_8"], expected["percentile_interval"]["lower"],
         expected["percentile_interval"]["upper"]], atol=1e-12, rtol=0)
    assert [int((values > 0).sum()), int((values == 0).sum()), int((values < 0).sum())] == [
        expected[k] for k in ("positive_problem_count", "zero_problem_count", "negative_problem_count")]
    result = {"H1": estimate, "Table_2": {}, "Table_3": {}, "Table_4": {}, "Table_5": {}}
    for cell in CELLS:
        result["Table_2"][cell] = {k: float(np.mean([r[k] for r in h1[cell]])) for k in ("MAP_at_8", "MRR", "Success_at_10")}
        np.testing.assert_allclose(result["Table_2"][cell]["MAP_at_8"], report["primary_cell_means"][cell], atol=1e-12, rtol=0)
    for name in ("token_tfidf", "ast_nodes", "ast_markov", "unixcoder"):
        receipt = read(f"data/{name}/result.json")
        assert receipt["query_ids"] == [r["source_relpath"] for r in metadata if r["role"] == "query"]
        assert receipt["gallery_ids"] == [r["source_relpath"] for r in metadata if r["role"] == "gallery"]
        result["Table_3"][name] = metrics(f"data/{name}/distances.npy", metadata, receipt["metrics"])
    for folder, table in (("codecontests_replication", "Table_4"), ("path_context", "Table_5")):
        receipt = read(f"data/{folder}/result.json")
        external = rows(f"data/{folder}/metadata.jsonl")
        for name, method in receipt["methods"].items():
            result[table][name] = metrics(f"data/{folder}/{name}.npy", external, method["metrics"])
        scores = {n: r["task_scores"] for n, r in result[table].items()}
        problems = sorted(next(iter(scores.values())))
        assert all(set(s) == set(problems) for s in scores.values())
        indices = np.random.Generator(np.random.PCG64(20260930)).integers(len(problems), size=(20000, len(problems)))
        if folder == "codecontests_replication":
            values = np.array([scores["path_triples"][p] - scores["endpoint_pairs"][p] for p in problems])
            result["external_82_contrast"] = contrast(values, indices)
            expected = receipt["primary_contrast"]
            np.testing.assert_allclose([result["external_82_contrast"]["mean"], *result["external_82_contrast"]["percentile_95_interval"]],
                [expected["delta_MAP_at_8"], *expected["percentile_95_interval"]], atol=1e-12, rtol=0)
        else:
            compact = np.array([scores["compact_retained"][p] - scores["compact_removed"][p] for p in problems])
            full = np.array([scores["full_retained"][p] - scores["full_masked"][p] for p in problems])
            result["external_76_contrasts"] = {}
            for name, values in (("compact_gain", compact), ("full_gain", full), ("primary_interaction", compact - full)):
                actual, expected = contrast(values, indices), receipt["contrasts"][name]
                np.testing.assert_allclose([actual["mean"], *actual["percentile_95_interval"]],
                    [expected["mean"], *expected["percentile_95_interval"]], atol=1e-12, rtol=0)
                result["external_76_contrasts"][name] = actual
    diagnostic = read("data/validation_diagnostic/result.json")
    validation = diagnostic["queries"] + diagnostic["gallery"]
    result["validation_diagnostic"] = {}
    for name, method in diagnostic["methods"].items():
        result["validation_diagnostic"][name] = metrics(f"data/validation_diagnostic/{name}.npy", validation, method["metrics"])
    for reference in diagnostic["saved_transport"]:
        for cell, entry in reference["cells"].items():
            key = f"{reference['seed']}_{cell}"
            result["validation_diagnostic"][key] = metrics(f"data/validation_diagnostic/{key}.npy", validation, entry["metrics"])
    timing = read("data/lca_gallery_runtime.json")
    original = np.load(ROOT / "data/h1/20260711_EEE_true_LCA.npy", mmap_mode="r", allow_pickle=False)
    timed = np.load(ROOT / "data/lca_gallery_runtime.npy", allow_pickle=False)
    ids = [r["source_relpath"] for r in metadata if r["role"] == "query"]
    assert timing["gallery_ids"] == [r["source_relpath"] for r in metadata if r["role"] == "gallery"]
    assert timed.shape == (8, 3088) and np.isfinite(timed).all()
    selected = original[[ids.index(q) for q in timing["query_ids"]]]
    np.testing.assert_allclose(timed, selected, atol=1e-6, rtol=1e-6)
    error = float(np.max(np.abs(timed - selected)))
    np.testing.assert_allclose(error, timing["maximum_saved_matrix_absolute_error"], atol=1e-12, rtol=0)
    assert sum(timing["four_query_batch_seconds"]) == timing["cross_scoring_seconds"] > 0
    result["recorded_timing"] = {"cross_scoring_seconds": timing["cross_scoring_seconds"], "maximum_saved_matrix_absolute_error": error}
    for family, table, cohorts, seed in (
        ("path_fragments", "Table_6", ("path_context", "codecontests_replication", "fresh_train"), 20261001),
        ("path_order", "Table_7", ("exploratory120", "fresh_train"), 20261002)):
        result[table] = {}
        for cohort in cohorts:
            folder = f"data/{family}/{cohort}"
            receipt = read(f"{folder}/result.json")
            assert receipt["bootstrap_seed"] == seed and receipt["bootstrap_resamples"] == 20000
            assert receipt["training_performed"] is False and receipt["java_test_accessed"] is False
            metadata = rows(f"{folder}/metadata.jsonl")
            assert len(metadata) == receipt["queries"] + receipt["gallery"]
            methods = {name: metrics(f"{folder}/{name}.npy", metadata, method["metrics"])
                       for name, method in receipt["methods"].items()}
            assert len(methods) == 7
            problems = sorted(next(iter(methods.values()))["task_scores"])
            assert len(problems) == receipt["problems"]
            scores = {n: np.array([m["task_scores"][p] for p in problems]) for n, m in methods.items()}
            indices = np.random.Generator(np.random.PCG64(seed)).integers(len(problems), size=(20000, len(problems)))
            fragment = scores["fragment_retained"] - scores["fragment_masked"]
            if family == "path_fragments":
                full = scores["full_retained"] - scores["full_masked"]
                differences = {"primary_fragment_gain": fragment, "full_gain": full,
                    "compact_gain": scores["compact_retained"] - scores["compact_removed"],
                    "fragment_minus_full_gain": fragment - full,
                    "fragment_retained_minus_full_retained": scores["fragment_retained"] - scores["full_retained"],
                    "fragment_retained_minus_ast_nodes": scores["fragment_retained"] - scores["ast_nodes"]}
            else:
                differences = {"ancestor_gain": scores["ordered_retained"] - scores["ordered_masked"],
                    "adjacency_minus_frequency": scores["ordered_retained"] - scores["path_frequencies"],
                    "adjacency_minus_shuffle": scores["ordered_retained"] - scores["shuffled_retained"],
                    "raw_ancestor_gain": fragment,
                    "equal_weight_minus_raw": scores["ordered_retained"] - scores["fragment_retained"],
                    "equal_weight_minus_ast": scores["ordered_retained"] - scores["ast_nodes"]}
            assert set(differences) == set(receipt["contrasts"])
            verified = {}
            for name, values in differences.items():
                actual, expected = contrast(values, indices), receipt["contrasts"][name]
                np.testing.assert_allclose([actual["mean"], *actual["percentile_95_interval"]],
                    [expected["mean"], *expected["percentile_95_interval"]], atol=1e-12, rtol=0)
                if "problem_values" in expected:
                    assert set(expected["problem_values"]) == set(problems)
                    np.testing.assert_allclose(values, [expected["problem_values"][p] for p in problems], atol=1e-12, rtol=0)
                if "positive_problems" in expected:
                    assert [int((values > 0).sum()), int((values == 0).sum()), int((values < 0).sum())] == [
                        expected[k] for k in ("positive_problems", "zero_problems", "negative_problems")]
                if "bonferroni_98_333_interval" in expected:
                    actual["bonferroni_98_333_interval"] = np.quantile(values[indices].mean(axis=1), [1/120, 119/120]).tolist()
                    np.testing.assert_allclose(actual["bonferroni_98_333_interval"], expected["bonferroni_98_333_interval"], atol=1e-12, rtol=0)
                verified[name] = actual
            result[table][cohort] = {"methods": methods, "contrasts": verified}
            print(f"{family}/{cohort}: seven rankings and all contrasts verified", flush=True)
    # Match the rounded printed cells, not only the underlying receipts.
    names = ("compact_retained", "compact_removed", "full_retained", "full_masked", "fragment_retained", "fragment_masked", "ast_nodes")
    order_names = ("ordered_retained", "ordered_masked", "path_frequencies", "shuffled_retained", "fragment_retained", "fragment_masked", "ast_nodes")
    keys = ("MAP_at_8", "MRR", "Success_at_10")
    table_values = {
        2: [[result["Table_2"][c][k] for c in reversed(CELLS)] for k in keys],
        3: [[result["Table_2"][CELLS[0]][k] for k in keys]] + [[result["Table_3"][n][k] for k in keys]
            for n in ("token_tfidf", "ast_nodes", "ast_markov", "unixcoder")],
        4: [[result["Table_4"][n][k] for k in keys] for n in ("path_triples", "endpoint_pairs", "ast_nodes")],
        5: [[result["Table_5"][n][k] for k in keys] for n in names[:4] + ("ast_nodes",)],
        6: [[result["Table_6"][c]["methods"][n]["MAP_at_8"] for c in ("path_context", "codecontests_replication", "fresh_train")] for n in names],
        7: [[result["Table_7"][c]["methods"][n]["MAP_at_8"] for c in ("exploratory120", "fresh_train")] for n in order_names]}
    source = (ROOT / "manuscript.tex").read_text(encoding="utf-8")
    for number, values in table_values.items():
        body = re.search(rf"Table {number}\. .*?\\begin\{{longtable\}}(.*?)\\end\{{longtable\}}", source, re.S).group(1)
        printed = [re.findall(r"(?<!\d)0\.\d{5}(?!\d)", line) for line in body.splitlines() if re.search(r"0\.\d{5}", line)]
        assert printed == [[f"{v:.5f}" for v in row] for row in values], f"Printed Table {number} mismatch"
    result["printed_tables_checked"] = list(table_values)
    (ROOT / "verified_results.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print("PASS: Tables 2--7 (including printed cells), all bootstrap intervals, validation diagnostic and timing-score agreement", flush=True)


if __name__ == "__main__":
    verify()
