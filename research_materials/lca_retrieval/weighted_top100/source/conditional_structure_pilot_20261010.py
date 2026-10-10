"""Isolated, exploratory CPU score calibration; no neural forward or training."""

from rank_fusion_followup import ROOT, metadata, ranks, sha, write
from followup_baselines import normalized, vectorize
from path_context_followup import load_model, path_keys, parse_python_ast_tree, VALIDATION, CHECKPOINT_HASH
from path_fragment_followup import fragments
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import resource
import shutil
import subprocess
import sys
import time

import numpy as np
from scipy.sparse import csr_matrix, save_npz, load_npz

COHORT = ROOT / "followup_results/fresh_fusion"
DENSE = ROOT / "followup_results/coderank_colab_20261007/evaluation"
OUT = ROOT / "followup_results/conditional_structure_20261010"
PROTOCOL = ROOT / "conditional_structure_pilot_protocol_20261010.md"
GRID = [0., .005, .01, .02, .05, .10, .15, .20, .25, .30, .35, .40,
        .45, .50, .55, .60, .65, .70, .75, .80, .85, .90, .95, 1.]
NAMES = ("N", "F", "I", "W", "R", "T")


def masked_key(fragment):
    masked = tuple((m, -1 if m else c) for m, c in fragment)
    return min(masked, masked[::-1])


def channels(counts, gallery_indices, group):
    assert counts.shape[1] == len(group) and np.isfinite(counts.data).all()
    assert (counts.data >= 0).all() and (group >= 0).all()
    groups = int(group.max()) + 1
    membership = csr_matrix((np.ones(len(group)), (np.arange(len(group)), group)),
                            shape=(len(group), groups))
    context = (counts @ membership).tocsr()
    size = np.bincount(group, minlength=groups)
    assert (size > 0).all()
    gallery = counts[gallery_indices]
    frequency = gallery.sum(axis=0).A1
    assert (frequency > 0).all(), "Vocabulary must be fitted only on the gallery"
    total = context[gallery_indices].sum(axis=0).A1
    probability = (frequency + 1 / size[group]) / (total[group] + 1)
    np.testing.assert_allclose(np.bincount(group, weights=probability), 1., atol=1e-12)
    scale = 1 / np.sqrt(probability)
    z = counts.multiply(scale).tocsr()
    norm = np.sqrt(z.multiply(z).sum(axis=1)).A1
    inverse = 1 / np.where(norm > 0, norm, 1)
    # Subtract counts before scaling; avoid cancellation of near-equal Gram matrices.
    expected = context @ membership.T.multiply(probability)
    residual = (counts - expected).multiply(scale).tocsr()
    residual.eliminate_zeros()
    idf = np.log((len(gallery_indices) + 1) / ((gallery > 0).sum(axis=0).A1 + 1)) + 1
    result = {"I": normalized(counts.multiply(idf)),
              "W": z.multiply(inverse[:, None]).tocsr(),
              "R": residual.multiply(inverse[:, None]).tocsr(),
              "C": context.multiply(inverse[:, None]).tocsr()}
    energy = result["R"].multiply(result["R"]).sum(axis=1).A1
    assert energy.min() >= 0 and energy.max() <= 1 + 1e-12
    return result, {"conditional_probability": probability.tolist(),
                    "contexts": groups, "singleton_contexts": int((size == 1).sum()),
                    "residual_energy_fraction": energy.tolist(),
                    "zero_count_vectors": int((norm == 0).sum())}


def ap8(scores, tie, relevant):
    positions = np.sort(ranks(-scores, tie)[relevant])
    return float(np.sum(np.arange(1, len(positions) + 1) / positions * (positions <= 8)) / 8)


def prepare():
    dense_result = DENSE / "result.json"
    assert sha(dense_result) == "d25fe73921b328604e9bf317af2a7e9c02c0b67b85ed22b9f7e7679da7f73524"
    dense = json.loads(dense_result.read_text())
    structural = json.loads((COHORT / "result.json").read_text())
    expected = dict(json.loads((DENSE / "analysis_receipt.json").read_text())["bindings"])
    assert sha(DENSE / "analysis_receipt.json") == dense["analysis_receipt_sha256"]
    expected.update({str(COHORT / k): v for k, v in json.loads((COHORT / "selection_seal.json").read_text())["sha256"].items()})
    expected.update({str(DENSE / "D.npy"): dense["methods"]["D"]["score_sha256"],
                     str(DENSE / "T.npy"): dense["lexical_score_sha256"]})
    for name in ("ast_nodes", "fragment_retained"):
        expected[str(COHORT / (name + ".npy"))] = structural["methods"][name]["distance_sha256"]
    source = COHORT / "selected_programs.jsonl"
    assert expected[str(source)] == "f5186a1945cbe65a9a38846ca5a305ddfb7a766f89ea10fc8ec98cdd228ab729"
    original = json.loads((COHORT / "design_receipt.json").read_text())["bindings"]
    helpers = [ROOT / n for n in ("path_context_followup.py", "path_fragment_followup.py",
        "codecontests_replication.py", "validation_diagnostic.py", "followup_baselines.py",
        "rank_fusion_followup.py", "reviewer_reproduction/verify_results.py")]
    geometry = ROOT.parents[2] / "code2hyp_informatics_artifacts/geometry_profile_research"
    helpers.extend(sorted(geometry.glob("*.py")))
    for path in helpers:
        assert sha(path) == original[str(path)], path
        expected[str(path)] = original[str(path)]
    checkpoint = VALIDATION / "seed_20260711_encoder.pt"
    expected[str(checkpoint)] = CHECKPOINT_HASH
    for path, digest in expected.items():
        assert sha(Path(path)) == digest, path
    rows, queries, gallery = metadata(COHORT)
    assert len(rows) == 10400 and len(queries) == len(gallery) == 5200
    problems = sorted({r["cluster_id"] for r in queries}, key=lambda p: (
        hashlib.sha256(("conditional-structure-development-20261010:" + p).encode()).hexdigest(), p))
    assert len(problems) == 650
    expected.update({str(path): sha(path) for path in (Path(__file__), PROTOCOL,
        ROOT / "test_conditional_structure_pilot_20261010.py",
        ROOT / "CONDITIONAL_STRUCTURE_CANDIDATE_20261010.md", dense_result,
        DENSE / "analysis_receipt.json", COHORT / "selection_seal.json")})
    frozen = {"bindings": expected, "weight_grid": GRID, "calibration_problems": problems[:325],
              "validation_problems": problems[325:], "methods": NAMES,
              "protocol_status": "exploratory_reused_cohort", "threads": 1}
    OUT.mkdir(exist_ok=True)
    path = OUT / "design_receipt.json"
    if path.exists():
        old = json.loads(path.read_text())
        for k, v in frozen.items():
            assert json.loads(json.dumps(v)) == old[k], k
    else:
        assert not list(OUT.iterdir())
        assert shutil.disk_usage(OUT).free > 3 * 1024**3
        write(path, {**frozen, "fixed_before_new_scores_utc": datetime.now(timezone.utc).isoformat(),
            "resources": {"platform": platform.platform(), "cpu_threads_used": 1,
                "ram_bytes": int(subprocess.check_output(["sysctl", "-n", "hw.memsize"])),
                "free_disk_bytes": shutil.disk_usage(OUT).free}})
    return rows, queries, gallery


def verify_files(receipt):
    for name, expected in receipt["output_sha256"].items():
        assert sha(OUT / name) == expected, name


def build_features(rows):
    receipt = OUT / "features.json"
    if receipt.exists():
        verify_files(json.loads(receipt.read_text()))
        return
    assert not (OUT / "counts.npz").exists(), "Incomplete feature stage requires inspection"
    source = COHORT / "selected_programs.jsonl"
    programs = [json.loads(line) for line in source.read_text().splitlines()]
    assert [{k: r[k] for k in ("source_relpath", "role", "cluster_id")} for r in programs] == [
        {k: r[k] for k in ("source_relpath", "role", "cluster_id")} for r in rows]
    assert all(hashlib.sha256(r["source"].encode()).hexdigest() == r["d0"] for r in programs)
    model = load_model()
    started = time.perf_counter()
    bags, unknown = [], 0
    for i, row in enumerate(programs):
        tree = parse_python_ast_tree(row["source"])
        counts = Counter()
        for path in model._resolve_paths(tree, None):
            a, _, missing = path_keys(tree, path, model.token_to_id)
            counts.update(fragments(a))
            unknown += missing
        bags.append(counts)
        if (i + 1) % 2000 == 0:
            print(f"Non-neural AST counts: {i+1}/{len(rows)}", flush=True)
    extraction_seconds = time.perf_counter() - started
    qi = np.array([i for i, r in enumerate(rows) if r["role"] == "query"])
    gi = np.array([i for i, r in enumerate(rows) if r["role"] == "gallery"])
    keys = sorted(set().union(*(set(bags[i]) for i in gi)))
    vocabulary = {key: i for i, key in enumerate(keys)}
    masked = sorted({masked_key(key) for key in keys})
    groups = {key: i for i, key in enumerate(masked)}
    group = np.array([groups[masked_key(key)] for key in keys])
    unseen = sum(v for i in qi for k, v in bags[i].items() if k not in vocabulary)
    mass = sum(sum(bags[i].values()) for i in qi)
    counts = vectorize(bags, vocabulary)
    del bags, programs, model
    previous = json.loads((COHORT / "result.json").read_text())
    assert len(keys) == previous["methods"]["fragment_retained"]["dimensions"]
    assert unknown == previous["unknown_path_tokens"]
    assert unseen / mass == previous["methods"]["fragment_retained"]["unseen_query_count_fraction"]
    started = time.perf_counter()
    transformed, info = channels(counts, gi, group)
    transform_seconds = time.perf_counter() - started
    save_npz(OUT / "counts.npz", counts)
    np.save(OUT / "group.npy", group, allow_pickle=False)
    write(OUT / "vocabulary.json", keys)
    for name, matrix in transformed.items():
        save_npz(OUT / (name + "_features.npz"), matrix)
    raw = normalized(counts)
    old = np.load(COHORT / "fragment_retained.npy", mmap_mode="r", allow_pickle=False)
    raw_error, identity_error = 0., 0.
    matrices = {}
    for name in ("I", "W", "R"):
        path = OUT / (name + ".npy")
        assert not path.exists()
        matrices[name] = np.lib.format.open_memmap(path, mode="w+", dtype=np.float64,
                                                   shape=(len(qi), len(gi)))
    started = time.perf_counter()
    for start in range(0, len(qi), 128):
        selection = qi[start:start + 128]
        raw_distance = 1 - (raw[selection] @ raw[gi].T).toarray()
        np.testing.assert_allclose(raw_distance, old[start:start + len(selection)], rtol=0, atol=2e-12)
        raw_error = max(raw_error, float(np.max(np.abs(raw_distance - old[start:start + len(selection)]))))
        for name, matrix in matrices.items():
            scores = (transformed[name][selection] @ transformed[name][gi].T).toarray()
            assert np.isfinite(scores).all() and np.abs(scores).max() <= 1 + 1e-10
            matrix[start:start + len(selection)] = scores
        context = (transformed["C"][selection] @ transformed["C"][gi].T).toarray()
        difference = matrices["W"][start:start + len(selection)] - context
        actual = matrices["R"][start:start + len(selection)]
        np.testing.assert_allclose(actual, difference, atol=2e-12, rtol=0)
        identity_error = max(identity_error, float(np.max(np.abs(actual - difference))))
    for matrix in matrices.values():
        matrix.flush()
    scoring_seconds = time.perf_counter() - started
    names = ["counts.npz", "group.npy", "vocabulary.json",
             *(n + "_features.npz" for n in transformed), *(n + ".npy" for n in matrices)]
    write(receipt, {"completed_utc": datetime.now(timezone.utc).isoformat(),
        "output_sha256": {n: sha(OUT / n) for n in names}, "dimensions": len(keys),
        "unknown_path_tokens": unknown, "unseen_query_count_fraction": unseen / mass,
        "raw_score_max_abs_error": raw_error, "kernel_identity_max_abs_error": identity_error,
        "background": info, "timing": {"ast_count_seconds": extraction_seconds,
            "transform_seconds": transform_seconds, "scoring_and_identity_check_seconds": scoring_seconds}})
    print("New representations saved; original fragment scores reproduced", flush=True)


def matrices():
    locations = {"D": DENSE / "D.npy", "T": DENSE / "T.npy",
                 "N": COHORT / "ast_nodes.npy", "F": COHORT / "fragment_retained.npy",
                 **{n: OUT / (n + ".npy") for n in ("I", "W", "R")}}
    result = {n: np.load(path, mmap_mode="r", allow_pickle=False) for n, path in locations.items()}
    assert all(m.shape == (5200, 5200) for m in result.values())
    return result


def score_row(matrix, name, i):
    values = np.asarray(matrix[name][i], dtype=np.float64)
    return -values if name == "D" else 1 - values if name in ("N", "F", "T") else values


def problem_means(values, queries, indices):
    grouped = {}
    for value, i in zip(values, indices):
        grouped.setdefault(queries[i]["cluster_id"], []).append(float(value))
    assert set(map(len, grouped.values())) == {8}
    return {p: float(np.mean(v)) for p, v in sorted(grouped.items())}


def evaluate(queries, gallery):
    assert not (OUT / "result.json").exists(), "Verify a completed run; do not overwrite"
    matrix = matrices()
    design = json.loads((OUT / "design_receipt.json").read_text())
    tie = np.array([r["source_relpath"].encode("utf-8") for r in gallery])
    labels = np.array([r["cluster_id"] for r in gallery])
    relevant = {p: np.flatnonzero(labels == p) for p in set(labels)}
    cal = [i for i, q in enumerate(queries) if q["cluster_id"] in set(design["calibration_problems"])]
    val = [i for i, q in enumerate(queries) if q["cluster_id"] in set(design["validation_problems"])]
    assert len(cal) == len(val) == 2600 and not set(cal) & set(val)
    previous = json.loads((DENSE / "result.json").read_text())["methods"]["D"]["metrics"]
    d_ap = []
    for i, query in enumerate(queries):
        value = ap8(score_row(matrix, "D", i), tie, relevant[query["cluster_id"]])
        np.testing.assert_allclose(value, previous["query_scores"][query["source_relpath"]], atol=1e-12, rtol=0)
        d_ap.append(value)
    path = OUT / "calibration.json"
    if path.exists():
        calibration = json.loads(path.read_text())
    else:
        started = time.perf_counter()
        summaries = {}
        for name in NAMES:
            means = []
            for alpha in GRID:
                values = []
                for i in cal:
                    score = (1 - alpha) * score_row(matrix, "D", i) + alpha * score_row(matrix, name, i)
                    values.append(ap8(score, tie, relevant[queries[i]["cluster_id"]]))
                means.append(float(np.mean(list(problem_means(values, queries, cal).values()))))
            selected = int(np.argmax(means))
            summaries[name] = {"calibration_map8_by_weight": means, "alpha": GRID[selected],
                               "selected_calibration_map8": means[selected]}
            print("Calibration", name, summaries[name], flush=True)
        calibration = {"fixed_before_validation_scoring_utc": datetime.now(timezone.utc).isoformat(),
            "weight_grid": GRID, "methods": summaries, "seconds": time.perf_counter() - started,
            "design_sha256": sha(OUT / "design_receipt.json"), "features_sha256": sha(OUT / "features.json")}
        write(path, calibration)
    assert calibration["design_sha256"] == sha(OUT / "design_receipt.json")
    assert calibration["features_sha256"] == sha(OUT / "features.json")
    started = time.perf_counter()
    output = {"D": {"validation_query_ap8": [d_ap[i] for i in val],
        "validation_problem_ap8": problem_means([d_ap[i] for i in val], queries, val)}}
    for name in NAMES:
        choice = calibration["methods"][name]
        assert choice["alpha"] == GRID[int(np.argmax(choice["calibration_map8_by_weight"]))]
        entry = {"alpha": choice["alpha"], "selected_calibration_map8": choice["selected_calibration_map8"]}
        for key, alpha in (("validation", choice["alpha"]), ("standalone_validation", 1.)):
            values = []
            for i in val:
                score = (1 - alpha) * score_row(matrix, "D", i) + alpha * score_row(matrix, name, i)
                values.append(ap8(score, tie, relevant[queries[i]["cluster_id"]]))
            entry[key + "_query_ap8"] = values
            entry[key + "_problem_ap8"] = problem_means(values, queries, val)
        output[name] = entry
        print("Validation", name, "alpha", choice["alpha"], "MAP8",
              np.mean(entry["validation_query_ap8"]), "standalone", np.mean(entry["standalone_validation_query_ap8"]), flush=True)
    contrasts = {}
    residual = output["R"]["validation_problem_ap8"]
    for name, entry in output.items():
        entry["validation_map8"] = float(np.mean(list(entry["validation_problem_ap8"].values())))
        if name != "R":
            differences = np.array([residual[p] - entry["validation_problem_ap8"][p] for p in residual])
            contrasts["R_versus_" + name] = {"mean": float(differences.mean()),
                "improved_problems": int((differences > 1e-12).sum()),
                "worsened_problems": int((differences < -1e-12).sum()),
                "unchanged_problems": int((np.abs(differences) <= 1e-12).sum())}
    write(OUT / "result.json", {"status": "completed_exploratory_internal_validation",
        "completed_utc": datetime.now(timezone.utc).isoformat(), "methods": output, "contrasts": contrasts,
        "validation_query_ids": [queries[i]["source_relpath"] for i in val],
        "calibration_sha256": sha(path), "features_sha256": sha(OUT / "features.json"),
        "design_sha256": sha(OUT / "design_receipt.json"), "validation_seconds": time.perf_counter() - started,
        "peak_rss_bytes_macos": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "score_weight_calibration_performed": True, "neural_training_performed": False,
        "neural_inference_performed": False, "java_test_accessed": False,
        "limitations": "All 650 problems were previously analysed. This is internal development validation, not independent confirmation; no new inferential tests or superiority claim."})


def verify(queries, gallery):
    result = json.loads((OUT / "result.json").read_text())
    for key, path in (("design", "design_receipt.json"), ("features", "features.json"), ("calibration", "calibration.json")):
        assert sha(OUT / path) == result[key + "_sha256"]
    verify_files(json.loads((OUT / "features.json").read_text()))
    calibration = json.loads((OUT / "calibration.json").read_text())
    matrix = matrices()
    tie_order = np.argsort(np.array([r["source_relpath"].encode("utf-8") for r in gallery]))
    labels = np.array([r["cluster_id"] for r in gallery])
    lookup = {q["source_relpath"]: i for i, q in enumerate(queries)}
    indices = [lookup[q] for q in result["validation_query_ids"]]
    for name, entry in result["methods"].items():
        if name != "D":
            means = calibration["methods"][name]["calibration_map8_by_weight"]
            assert entry["alpha"] == GRID[int(np.argmax(means))]
        for prefix, alpha in (("validation", entry.get("alpha", 0.)), ("standalone_validation", 1.)):
            if name == "D" and prefix == "standalone_validation":
                continue
            values = []
            for i in indices:
                scores = ((1 - alpha) * score_row(matrix, "D", i) + alpha * score_row(matrix, name, i))
                order = tie_order[np.argsort(-scores[tie_order], kind="stable")][:8]
                hits = labels[order] == queries[i]["cluster_id"]
                values.append(float(np.sum(np.cumsum(hits) * hits / np.arange(1, 9)) / 8))
            np.testing.assert_allclose(values, entry[prefix + "_query_ap8"], rtol=0, atol=1e-12)
            for problem, value in problem_means(values, queries, indices).items():
                np.testing.assert_allclose(value, entry[prefix + "_problem_ap8"][problem], rtol=0, atol=1e-12)
    path = OUT / "verification.json"
    assert not path.exists()
    write(path, {"status": "passed", "verified_utc": datetime.now(timezone.utc).isoformat(),
        "result_sha256": sha(OUT / "result.json"), "selected_rankings_and_standalone_checked": 13,
        "validation_queries_per_ranking": len(indices), "rank_algorithm": "independent stable sorting in UTF-8 ID order",
        "input_and_output_hashes_checked": True, "original_dense_AP8_reconstructed": True})
    print("PASS: all selected and standalone validation AP@8 independently reconstructed", flush=True)


if __name__ == "__main__":
    if sys.argv[1:] not in (["run"], ["verify"]):
        raise SystemExit("Usage: conditional_structure_pilot_20261010.py run|verify")
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    rows, queries, gallery = prepare()
    if sys.argv[1] == "run":
        build_features(rows)
        evaluate(queries, gallery)
    verify(queries, gallery)
