"""Approved two-session continuation; original benchmark and weights unchanged."""

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

MANIFEST_SHA = "9d2834bf8f1a1ee1e552f769c3f4c0462a77077eceb70c7061c150612536e0e8"
METHODS = ("D_AST", "D")


def cached_ranking(dense, candidates, scores, tie):
    import numpy as np
    from e2e_common_20261014 import order
    if (dense.ndim != 1 or tie.shape != dense.shape or candidates.ndim != 1
            or scores.shape != candidates.shape or candidates.dtype.kind not in "iu"
            or len(np.unique(candidates)) != len(candidates)
            or np.any(candidates < 0) or np.any(candidates >= len(dense))):
        raise ValueError("Invalid candidate-to-score binding")
    ranked = order(dense, tie)
    # Cached scores belong to saved candidate IDs, not a new BLAS ordering.
    prefix = candidates[order(scores, tie[candidates])]
    return np.concatenate((prefix, ranked[~np.isin(ranked, candidates)]))


def replay_reranker(root, out, host_id):
    import numpy as np
    from e2e_common_20261014 import metrics, order, sha, write
    design = json.loads((root / "e2e_design.json").read_text())
    programs = json.loads((root / "programs.json").read_text())
    tie = np.array([p["id"].encode() for p in programs["gallery"]])
    labels = np.array(design["gallery_problems"])
    with np.load(root / "reference_vectors.npz", allow_pickle=False) as vectors:
        dense = vectors["query"] @ vectors["gallery"].T
    candidates = np.load(root / "reference_candidates.npy", allow_pickle=False)
    scores = np.load(root / "reference_reranker_scores.npy", allow_pickle=False)
    assert dense.shape == (3200, 3200) and candidates.shape == scores.shape == (3200, 100)
    values, top = np.empty((3200, 3)), np.empty((3200, 100), dtype=np.int32)
    changed_order, changed_membership = [], []
    for i in range(3200):
        current = order(dense[i].astype(np.float64), tie)[:100]
        if not np.array_equal(current, candidates[i]):
            changed_order.append(i)
        if not np.array_equal(np.sort(current), np.sort(candidates[i])):
            changed_membership.append(i)
        ranked = cached_ranking(dense[i].astype(np.float64), candidates[i], scores[i], tie)
        values[i] = metrics(ranked, labels == design["query_problems"][i])
        top[i] = ranked[:100]
    previous = json.loads((root / "reference_reranker_result.json").read_text())["methods"]["Qwen_reranker"]
    keys = ("MAP_at_8", "MRR", "Success_at_10")
    np.testing.assert_allclose(values.mean(axis=0), [previous[k] for k in keys], rtol=0, atol=1e-12)
    out.mkdir()
    np.savez_compressed(out / "full_quality.npz", metrics=values, top100=top)
    write(out / "receipt.json", {
        "status": "complete_cached_pair_score_replay", "host_id": host_id,
        "binding": sha(root / "e2e_manifest.json"), "queries": 3200,
        "quality": dict(zip(keys, values.mean(axis=0).tolist())),
        "recomputed_dense_prefix_order_changed_queries": changed_order,
        "recomputed_dense_prefix_membership_changed_queries": changed_membership,
        "scope": "Historical fixed candidates and pair scores, not new inference or session-A completion",
        "neural_inference_performed": False, "training_performed": False,
        "java_test_accessed": False, "files": {"full_quality.npz": sha(out / "full_quality.npz")}})


def main():
    assert Path("/kaggle/working").is_dir(), "GPU continuation is Kaggle-only"
    manifests = list(Path("/kaggle/input").rglob("e2e_manifest.json"))
    assert len(manifests) == 1
    root = manifests[0].parent
    assert hashlib.sha256(manifests[0].read_bytes()).hexdigest() == MANIFEST_SHA
    manifest = json.loads(manifests[0].read_text())
    assert manifest["private_only"] and manifest["machine_shape"] == "NvidiaTeslaT4"
    for name, digest in manifest["files"].items():
        assert Path(name).name == name
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == digest, name
    package_root = Path("/kaggle/working/frozen_geometry")
    package = package_root / "geometry_profile_research"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("")
    for name in ("raw_ast.py", "python_raw_ast.py"):
        shutil.copy2(root / name, package / name)
    environment = {"USE_TF": "0", "TOKENIZERS_PARALLELISM": "false",
                   "HF_HOME": "/kaggle/temp/e2e-hf", "PYTHONPATH": str(package_root),
                   "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
                   "MKL_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1"}
    os.environ.update(environment)
    subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", "--disable-pip-version-check",
                    "transformers==4.57.1", "sentence-transformers==3.1.1", "numpy==2.1.3", "scipy==1.14.1"],
                   check=True, timeout=1200)
    sys.path.insert(0, str(root))
    from e2e_benchmark_20261014 import host, load
    from e2e_common_20261014 import now, sha, write
    design = load(root)
    assert tuple(m for m in design["method_order"] if m in METHODS) == METHODS
    identity = host()
    assert "Tesla T4" in identity["gpu"]
    out = Path("/kaggle/working/end_to_end_cost_continuation")
    out.mkdir()
    write(out / "host.json", identity)
    for name in ("e2e_manifest.json", "e2e_design.json"):
        shutil.copy2(root / name, out / name)
    shutil.copy2(Path(__file__), out / "submitted_launcher.py")
    deadline = time.time() + 2400
    for method in METHODS:
        subprocess.run([sys.executable, str(root / "e2e_benchmark_20261014.py"),
                        "--input", str(root), "--out", str(out / method),
                        "--method", method, "--deadline", str(deadline)],
                       check=True, timeout=max(1, deadline-time.time()+60))
        if not (out / method / "complete.json").exists():
            print("PARTIAL: preserve continuation outputs", flush=True)
            return
    replay_reranker(root, out / "reranker_quality_replay", identity["host_id"])
    write(out / "continuation_receipt.json", {
        "status": "complete_approved_two_session_continuation", "host": identity,
        "binding": MANIFEST_SHA, "methods": METHODS, "completed_utc": now(),
        "launcher_sha256": sha(Path(__file__)),
        "worker_receipts": {m: sha(out / m / "complete.json") for m in METHODS},
        "reranker_replay_receipt_sha256": sha(out / "reranker_quality_replay/receipt.json"),
        "scope": "D and D_AST paired on session B; retain session A separately; no pooled timing",
        "original_six_method_single_session_complete": False})
    print("COMPLETE: approved paired continuation, not a six-method same-VM run", flush=True)


if __name__ == "__main__":
    main()
