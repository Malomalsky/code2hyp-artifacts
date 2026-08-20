from __future__ import annotations

import json
from pathlib import Path

from scripts.check_codenet_java_stage_b_readiness import evaluate_stage_b_readiness


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_current_stage_b_design_passes_scientific_and_implementation_freeze_checks() -> None:
    design = json.loads((PROJECT_ROOT / "configs/codenet_java_stage_b_draft_v1.json").read_text())
    d4_path = PROJECT_ROOT / "data/codenet_java_stage_b_eligibility_d4_train32_v2/d4_manifest.json"
    d5_path = PROJECT_ROOT / "data/codenet_java_stage_b_metadata_v2/d5_metadata_manifest.json"
    d5_index_path = PROJECT_ROOT / "data/codenet_java_stage_b_metadata_v2/d5_metadata_index.jsonl"
    sampling_path = PROJECT_ROOT / "reports/codenet_java_stage_b_sampling_design_v2.json"
    power_path = PROJECT_ROOT / "reports/codenet_java_stage_b_power_precheck_train32_v2.json"

    from geometry_profile_research.codenet_eligibility import stable_sha256

    artifact_paths = {
        "metadata_frame": PROJECT_ROOT / "reports/codenet_java_stage_b_frame_v1.json",
        "candidate_manifest": PROJECT_ROOT / "data/codenet_java_stage_b_candidates_v1/manifest.json",
        "d0_d2_manifest": PROJECT_ROOT / "data/codenet_java_stage_b_eligibility_d0_d2_v1/manifest.json",
        "statement_audit": PROJECT_ROOT / "reports/codenet_java_stage_b_statement_overlap_v1.json",
        "d3_primary_manifest": PROJECT_ROOT / "data/codenet_java_stage_b_eligibility_d0_d3_v1/d3_manifest.json",
        "d4_manifest": d4_path,
        "d3_sensitivity_0p80_manifest": PROJECT_ROOT / "data/codenet_java_stage_b_eligibility_d0_d3_jaccard0p80_v1/d3_manifest.json",
        "d4_sensitivity_0p80_manifest": PROJECT_ROOT / "data/codenet_java_stage_b_eligibility_d4_train32_jaccard0p80_v2/d4_manifest.json",
        "d3_sensitivity_0p95_manifest": PROJECT_ROOT / "data/codenet_java_stage_b_eligibility_d0_d3_jaccard0p95_v1/d3_manifest.json",
        "d4_sensitivity_0p95_manifest": PROJECT_ROOT / "data/codenet_java_stage_b_eligibility_d4_train32_jaccard0p95_v2/d4_manifest.json",
        "d5_manifest": d5_path,
        "d5_index": d5_index_path,
        "sampling_design": sampling_path,
        "power": power_path,
    }
    actual_hashes = {name: stable_sha256(path.read_bytes()) for name, path in artifact_paths.items()}

    implementation_commit = design["freeze"]["implementation_commit"]
    assert implementation_commit == "c419f6418056b618ce373ebd6fafe6601ff51566"
    assert design["freeze"]["container_digest"] == (
        "sha256:f58fec9b2a2a1f3d58f462596486f6dca1e1a29c5d303f083d145fb19bea4204"
    )

    report = evaluate_stage_b_readiness(
        design=design,
        d4=json.loads(d4_path.read_text()),
        d5=json.loads(d5_path.read_text()),
        sampling_design=json.loads(sampling_path.read_text()),
        power=json.loads(power_path.read_text()),
        actual_hashes=actual_hashes,
        repository_commit="a" * 40,
        tracked_worktree_clean=False,
        dependency_lock_sha256="b" * 64,
        frozen_commit_exists=True,
        runner_tag_commits={
            design["freeze"]["validation_runner_tag"]: implementation_commit,
            design["freeze"]["test_runner_tag"]: implementation_commit,
        },
    )

    assert report["blocking_failures"] == ["tracked_worktree_clean"]
    workload = next(check for check in report["checks"] if check["id"] == "registered_compute_workload")
    assert workload["passed"] is True

    design["freeze"]["implementation_commit"] = "c" * 40
    mismatch = evaluate_stage_b_readiness(
        design=design,
        d4=json.loads(d4_path.read_text()),
        d5=json.loads(d5_path.read_text()),
        sampling_design=json.loads(sampling_path.read_text()),
        power=json.loads(power_path.read_text()),
        actual_hashes=actual_hashes,
        repository_commit="a" * 40,
        tracked_worktree_clean=True,
        dependency_lock_sha256="b" * 64,
        frozen_commit_exists=True,
        runner_tag_commits={
            design["freeze"]["validation_runner_tag"]: "a" * 40,
            design["freeze"]["test_runner_tag"]: "b" * 40,
        },
    )
    assert "runner_tags_point_to_implementation_commit" in mismatch["blocking_failures"]
