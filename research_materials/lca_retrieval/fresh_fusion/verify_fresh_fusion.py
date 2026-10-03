"""Reconstruct the prospective fusion results from saved scores using NumPy only."""

import os
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[key] = "1"

import importlib.util
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("numerical_helper", ROOT / "reviewer_reproduction/verify_results.py")
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)
BASE = ("ast_nodes", "fragment_retained", "fragment_masked")
FUSED = ("fusion_retained", "fusion_masked")


def verify(folder):
    manifest = ROOT / "MANIFEST.json"
    if manifest.exists():
        for name, entry in json.loads(manifest.read_text()).items():
            relative = Path(name)
            assert not relative.is_absolute() and ".." not in relative.parts
            path = ROOT / relative
            assert path.is_file() and not path.is_symlink()
            assert path.stat().st_size == entry["bytes"] and helper.sha(path) == entry["sha256"], name
    result = json.loads((folder / "result.json").read_text())
    assert result["status"] == "completed_prospective_fixed_fusion"
    assert (result["problems"], result["queries"], result["gallery"], result["k"]) == (650, 5200, 5200, 60)
    assert not result["training_performed"] and not result["java_test_accessed"]
    assert helper.sha(folder / "design_receipt.json") == result["design_receipt_sha256"]
    assert helper.sha(folder / "selection_seal.json") == result["selection_seal_sha256"]
    seal = json.loads((folder / "selection_seal.json").read_text())["sha256"]
    for name in ("metadata.jsonl", "selection_receipt.json", "design_receipt.json"):
        assert helper.sha(folder / name) == seal[name]
    rows = [json.loads(line) for line in (folder / "metadata.jsonl").read_text().splitlines()]
    assert len(rows) == 10400 and len({r["d2"] for r in rows}) == 10400
    excluded = set(json.loads((folder / "selection_receipt.json").read_text())["excluded_previous_problem_ids"])
    assert not excluded.intersection(r["cluster_id"] for r in rows)
    gallery = [r for r in rows if r["role"] == "gallery"]
    assert len(gallery) == 5200
    tie_order = np.argsort([r["source_relpath"].encode("utf-8") for r in gallery])
    matrices = {n: np.load(folder / (n + ".npy"), mmap_mode="r", allow_pickle=False) for n in (*BASE, *FUSED)}
    for i in range(5200):
        weights = {}
        for name in BASE:
            order = tie_order[np.argsort(matrices[name][i, tie_order], kind="stable")]
            weights[name] = 1. / (np.argsort(order) + 61.)
        for name, component in zip(FUSED, BASE[1:]):
            np.testing.assert_array_equal(matrices[name][i], -weights["ast_nodes"] - weights[component])
    metrics = {}
    for name in (*BASE, *FUSED):
        path = folder / (name + ".npy")
        key = "distance_sha256" if name in BASE else "negative_score_sha256"
        assert helper.sha(path) == result["methods"][name][key]
        metrics[name] = helper.metrics(path, rows, result["methods"][name]["metrics"])
        print("Verified ranking: " + name, flush=True)
    indices = np.load(folder / "bootstrap_indices.npy", allow_pickle=False)
    assert helper.sha(folder / "bootstrap_indices.npy") == result["bootstrap_indices_sha256"]
    np.testing.assert_array_equal(indices, np.random.Generator(np.random.PCG64(20261002)).integers(650, size=(20000, 650)))
    problems = sorted(metrics["fusion_retained"]["task_scores"])
    assert len(problems) == 650
    contrasts = {}
    for reference in ("ast_nodes", "fragment_retained", "fusion_masked"):
        delta = np.array([metrics["fusion_retained"]["task_scores"][p] - metrics[reference]["task_scores"][p] for p in problems])
        draws = delta[indices].mean(axis=1)
        contrasts[reference] = {"mean": float(delta.mean()),
            "percentile_95_interval": np.quantile(draws, [.025, .975]).tolist(),
            "bonferroni_three_interval": np.quantile(draws, [.05/6, 1-.05/6]).tolist()}
    assert contrasts.keys() == result["contrasts"].keys()
    for reference, values in contrasts.items():
        for key, value in values.items():
            np.testing.assert_allclose(value, result["contrasts"][reference][key], atol=1e-12, rtol=0)
    return {"status": "passed", "result_sha256": helper.sha(folder / "result.json"),
        "methods": {name: {k: v for k, v in value.items() if k != "task_scores"} for name, value in metrics.items()},
        "contrasts": contrasts}


if __name__ == "__main__":
    assert len(sys.argv) <= 2, "Usage: verify_fresh_fusion.py [data_directory]"
    folder = Path(sys.argv[1]).resolve() if len(sys.argv) == 2 else ROOT / "followup_results/fresh_fusion"
    report = verify(folder)
    (folder / "verified_results.json").write_text(json.dumps(report, indent=2) + "\n")
    print("PASS: five rankings, both fusions and all three prospective contrasts")
