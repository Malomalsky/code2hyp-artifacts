"""Audit saved workers without manufacturing a full-study completion receipt."""

import argparse
import json
from pathlib import Path
import numpy as np
from e2e_common_20261014 import METHODS, now, sha, summary, write
from evaluate_e2e_20261014 import evaluate_methods, timing_records


def verify_worker(folder, method, host_id, binding):
    report = json.loads((folder / "complete.json").read_text())
    assert report["status"] == "complete" and report["method"] == method
    assert report["host_id"] == host_id and report["binding"] == binding
    assert report["requests"] == 1200 and report["quality_queries"] == 3200
    assert not report["training_performed"] and not report["java_test_accessed"]
    for name, digest in report["files"].items():
        assert Path(name).name == name and sha(folder / name) == digest
    assert report["index"] == json.loads((folder / "index_receipt.json").read_text())
    assert report["memory"] == json.loads((folder / "memory.json").read_text())
    return report


def run(out, root):
    assert not (out / "complete_receipt.json").exists(), "Use the full evaluator for complete studies"
    binding = sha(root / "e2e_manifest.json")
    manifest = json.loads((root / "e2e_manifest.json").read_text())
    for name, digest in manifest["files"].items():
        assert Path(name).name == name and sha(root / name) == digest
    for name in ("e2e_manifest.json", "e2e_design.json"):
        assert sha(out / name) == sha(root / name)
    assert sha(out / "submitted_launcher.py") == manifest["launcher_sha256"]
    host = json.loads((out / "host.json").read_text())
    complete = [m for m in METHODS if (out / m / "complete.json").exists()]
    for method in complete:
        verify_worker(out / method, method, host["host_id"], binding)
    # Retain cross-BLAS rank differences in this incomplete-study audit, not in the full completion gate.
    result = evaluate_methods(out, root, {"host": host}, complete, strict_rankings=False)
    result.update(status="incomplete_study_audited_saved_workers", audited_utc=now(),
                  manifest_sha256=binding, complete_workers=complete,
                  missing_complete_workers=[m for m in METHODS if m not in complete])
    design = json.loads((root / "e2e_design.json").read_text())
    programs = json.loads((root / "programs.json").read_text())
    expected_models = json.loads((root / "expected_model_files.json").read_text())
    settings_keys = {"coderank": "coderank_settings", "qwen_embedding": "qwen_embedding_settings",
                     "qwen_reranker": "qwen_reranker_settings"}
    for method in METHODS:
        folder = out / method
        if not folder.exists():
            continue
        started = json.loads((folder / "started.json").read_text())
        assert started["host_id"] == host["host_id"] and started["binding"] == binding
        assert started["method"] == method and started["threads"] == 1
        index = json.loads((folder / "index_receipt.json").read_text())
        for name, model in index["models"].items():
            assert model["model_files"] == expected_models[name]
            assert model["settings"] == design[settings_keys[name]]
        if method in complete:
            continue
        times, live = timing_records(folder, method, host["host_id"], design, programs)
        latency = summary(times)
        result["methods"][method] = {
            "status": "all_timing_requests_saved_but_worker_not_complete",
            "quality": None, "latency": latency, "index": index,
            "memory": json.loads((folder / "memory.json").read_text()),
            "live_400_query_quality_by_pass": live.mean(axis=1).tolist(),
            "projected_workload_seconds_excluding_network": {
                str(n): index["ready_startup_excluding_network_seconds"]+n*latency["mean_seconds"]
                for n in (1, 100, 1000, 10000)}}
        if method == "D_QwenReranker":
            previous = json.loads((root / "reference_reranker_result.json").read_text())
            assert previous["status"] == "complete_exploratory_comparison"
            result["methods"][method]["previous_full_quality_not_new_host_receipt"] = {
                key: previous["methods"]["Qwen_reranker"][key]
                for key in ("MAP_at_8", "MRR", "Success_at_10")}
            result["methods"][method]["previous_full_quality_result_sha256"] = sha(root / "reference_reranker_result.json")
    result["downloaded_file_sha256"] = {
        str(p.relative_to(out)): sha(p) for p in sorted(out.rglob("*"))
        if p.is_file() and p.name != "audited_partial_result.json"}
    result["evaluator_sha256"] = sha(Path(__file__))
    result["shared_evaluator_sha256"] = sha(Path(__file__).with_name("evaluate_e2e_20261014.py"))
    result["interpretation_limits"] += [
        "Kernel ERROR after 6000 timed requests; D-only worker never started",
        "Reranker full-quality replay failed exact candidate-order assertion; its prior 3200-query quality is separate",
        "No completion receipt is created or inferred; no cross-VM timing is pooled"]
    write(out / "audited_partial_result.json", result)
    print(json.dumps({"status": result["status"], "methods": {
        m: {k: v[k] for k in ("quality", "latency", "live_400_query_quality_by_pass")}
        for m, v in result["methods"].items()}}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    args = parser.parse_args()
    run(args.output, args.input)
