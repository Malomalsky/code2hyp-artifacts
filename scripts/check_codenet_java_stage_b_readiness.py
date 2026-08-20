from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from geometry_profile_research.codenet_eligibility import (
    canonical_json_bytes,
    portable_manifest_path,
    stable_sha256,
)


EXPECTED_TEST_CELLS = (
    "prefix_EEE_true_LCA",
    "prefix_EEE_zero_anchor",
    "prefix_HEE_near_zero_true_LCA",
    "prefix_HEE_active_true_LCA",
    "label_only_EEE_true_LCA",
    "label_only_HEE_active_true_LCA",
    "prefix_HHH_active_true_LCA",
)


def evaluate_stage_b_readiness(
    *,
    design: Mapping[str, Any],
    d4: Mapping[str, Any],
    d5: Mapping[str, Any],
    sampling_design: Mapping[str, Any],
    power: Mapping[str, Any],
    actual_hashes: Mapping[str, str],
    repository_commit: str | None,
    tracked_worktree_clean: bool,
    dependency_lock_sha256: str | None,
    frozen_commit_exists: bool | None = None,
    runner_tag_commits: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []

    def add(identifier: str, passed: bool, detail: object, *, blocking: bool = True) -> None:
        checks.append(
            {
                "id": identifier,
                "passed": bool(passed),
                "blocking": blocking,
                "detail": str(detail),
            }
        )

    add(
        "design_schema",
        design.get("schema_version") == "code2hyp-codenet-java-stage-b-design-v1",
        design.get("schema_version"),
    )
    add(
        "pre_registration_state",
        design.get("status") == "draft_before_registration_implementation_freeze_or_split"
        and design.get("registration_doi") is None,
        f"status={design.get('status')}, doi={design.get('registration_doi')}",
    )
    add(
        "d4_pre_split_state",
        d4.get("protocol", {}).get("split_status") == "not_generated"
        and d4.get("protocol", {}).get("retrieval_metrics_opened") is False,
        d4.get("protocol"),
    )
    add(
        "d5_pre_split_state",
        d5.get("status") == "pre_registration_pre_split_without_retrieval_metrics"
        and d5.get("split_generated") is False
        and d5.get("retrieval_metrics_opened") is False,
        d5.get("status"),
    )

    counts = design["eligibility"]["primary_counts"]
    d4_summary = d4["summary"]
    d5_summary = d5["summary"]
    observed_counts = {
        "problem_clusters_after_d4_union": d4_summary["problem_clusters_after_d4_union"],
        "eligible_evaluation_clusters": d4_summary["eligible_evaluation_clusters_minimum_16_users_16"],
        "eligible_train_clusters": d4_summary["eligible_train_clusters"],
        "retained_programs_in_eligible_clusters": d5_summary["programs"],
    }
    add(
        "eligibility_counts_pinned",
        all(int(counts[key]) == int(value) for key, value in observed_counts.items()),
        observed_counts,
    )
    expected_quotas = tuple(design["split"]["weights_train_validation_test"])
    selected_quotas = tuple(sampling_design["selected_quotas_train_validation_test"])
    upper_bound = d4["role_specific_upper_bound"]
    add(
        "role_constrained_quotas",
        selected_quotas
        == (
            int(upper_bound["train_clusters"]),
            int(upper_bound["validation_clusters"]),
            int(upper_bound["test_clusters"]),
        )
        == (199, 66, 266)
        and expected_quotas == (3, 1, 4),
        selected_quotas,
    )
    add(
        "user_distinct_train_capacity",
        int(sampling_design["selected_train_programs_per_cluster"])
        == int(design["sampling"]["train_programs_per_cluster"])
        == int(design["eligibility"]["minimum_distinct_users_train"])
        == 32,
        sampling_design["selected_train_programs_per_cluster"],
    )

    expected_hashes = design["eligibility"]["artifacts"]
    hash_mapping = {
        "metadata_frame_sha256": "metadata_frame",
        "candidate_materialization_manifest_sha256": "candidate_manifest",
        "d0_d2_manifest_sha256": "d0_d2_manifest",
        "statement_overlap_audit_sha256": "statement_audit",
        "d3_primary_manifest_sha256": "d3_primary_manifest",
        "d4_primary_manifest_sha256": "d4_manifest",
        "d3_sensitivity_0p80_manifest_sha256": "d3_sensitivity_0p80_manifest",
        "d4_sensitivity_0p80_manifest_sha256": "d4_sensitivity_0p80_manifest",
        "d3_sensitivity_0p95_manifest_sha256": "d3_sensitivity_0p95_manifest",
        "d4_sensitivity_0p95_manifest_sha256": "d4_sensitivity_0p95_manifest",
        "d5_metadata_manifest_sha256": "d5_manifest",
        "d5_metadata_index_sha256": "d5_index",
        "sampling_design_report_sha256": "sampling_design",
    }
    for design_key, actual_key in hash_mapping.items():
        add(
            f"{actual_key}_hash_pinned",
            str(expected_hashes[design_key]) == str(actual_hashes[actual_key]),
            f"actual={actual_hashes[actual_key]}, expected={expected_hashes[design_key]}",
        )
    add(
        "power_report_hash_pinned",
        str(design["power"]["report_sha256"]) == str(actual_hashes["power"]),
        actual_hashes["power"],
    )
    add(
        "power_gate",
        int(power["planned_test_clusters"]) == int(design["power"]["planned_test_clusters"])
        and float(design["power"]["gate_marginal_power"]) >= float(design["power"]["required_power"])
        and float(design["power"]["gate_two_contrast_joint_power_lower_bound"])
        >= float(design["power"]["required_power"]),
        {
            "clusters": power["planned_test_clusters"],
            "marginal": design["power"]["gate_marginal_power"],
            "joint_lower": design["power"]["gate_two_contrast_joint_power_lower_bound"],
        },
    )
    add(
        "seven_test_cells_frozen",
        tuple(design["geometry"]["test_cells"]) == EXPECTED_TEST_CELLS,
        design["geometry"]["test_cells"],
    )
    encoder = design["encoder_training"]
    add(
        "deterministic_cuda_execution",
        encoder.get("compute_device") == "cuda"
        and encoder.get("torch_deterministic_algorithms") is True
        and int(encoder.get("torch_num_threads_per_process", 0)) == 1,
        {
            "compute_device": encoder.get("compute_device"),
            "deterministic": encoder.get("torch_deterministic_algorithms"),
            "torch_num_threads": encoder.get("torch_num_threads_per_process"),
        },
    )
    transport = design["transport"]
    seed_count = len(encoder["model_seeds"])
    curvature_count = len(design["geometry"]["active_curvature_candidates"])
    validation_matrices = seed_count * (6 + 3 * curvature_count)
    test_matrices = seed_count * len(design["geometry"]["test_cells"])
    validation_shape = tuple(int(value) for value in transport["validation_matrix_shape_per_seed_and_cell"])
    test_shape = tuple(int(value) for value in transport["test_matrix_shape_per_seed_and_cell"])
    validation_pairs = validation_matrices * validation_shape[0] * validation_shape[1]
    test_pairs = test_matrices * test_shape[0] * test_shape[1]
    expected_workload = {
        "registered_validation_distance_matrices": validation_matrices,
        "registered_test_distance_matrices": test_matrices,
        "registered_validation_pair_evaluations": validation_pairs,
        "registered_test_pair_evaluations": test_pairs,
        "minimum_final_distance_matrix_storage_bytes": (validation_pairs + test_pairs) * 8,
    }
    add(
        "registered_compute_workload",
        all(int(transport.get(key, -1)) == value for key, value in expected_workload.items()),
        expected_workload,
    )
    inference = design["inference"]
    add(
        "cluster_bootstrap_contract",
        int(inference["bootstrap_resamples"]) == 20_000
        and inference["bootstrap_unit"] == "duplicate_closed_problem_cluster"
        and tuple(inference["two_sided_interval_quantiles"]) == (0.025, 0.975)
        and bool(inference["bootstrap_domain"]),
        {
            "resamples": inference["bootstrap_resamples"],
            "unit": inference["bootstrap_unit"],
            "interval": inference["two_sided_interval_quantiles"],
            "domain": inference["bootstrap_domain"],
        },
    )
    freeze = design["freeze"]
    add(
        "runner_tags_frozen",
        freeze.get("validation_runner_tag") == "codenet-java-stage-b-validation-runner-v1"
        and freeze.get("test_runner_tag") == "codenet-java-stage-b-test-runner-v1",
        {
            "validation": freeze.get("validation_runner_tag"),
            "test": freeze.get("test_runner_tag"),
        },
    )
    add(
        "downstream_data_unopened",
        design["split"]["generated"] is False
        and freeze["registration_complete"] is False
        and freeze["java_validation_metrics_opened"] is False
        and freeze["java_test_program_ids_materialized"] is False
        and freeze["java_test_relevance_labels_opened"] is False
        and freeze["java_test_retrieval_metrics_computed"] is False,
        freeze,
    )

    add("immutable_implementation_commit", bool(freeze.get("implementation_commit")), freeze.get("implementation_commit"))
    if freeze.get("implementation_commit"):
        frozen_commit = str(freeze["implementation_commit"])
        add(
            "implementation_commit_exists",
            frozen_commit_exists is True,
            f"commit={frozen_commit}, current_protocol_commit={repository_commit}",
        )
        tag_commits = dict(runner_tag_commits or {})
        add(
            "runner_tags_point_to_implementation_commit",
            all(
                tag_commits.get(str(freeze[key])) == frozen_commit
                for key in ("validation_runner_tag", "test_runner_tag")
            ),
            tag_commits,
        )
    add("container_digest", bool(freeze.get("container_digest")), freeze.get("container_digest"))
    add("tracked_worktree_clean", tracked_worktree_clean, tracked_worktree_clean)
    add("dependency_lockfile", bool(dependency_lock_sha256), dependency_lock_sha256)

    blocking_failures = [check["id"] for check in checks if check["blocking"] and not check["passed"]]
    return {
        "schema_version": "code2hyp-codenet-java-stage-b-readiness-v1",
        "ready_for_public_registration": not blocking_failures,
        "blocking_failures": blocking_failures,
        "checks": checks,
    }


def build_stage_b_readiness_report(*, project_root: Path, design_path: Path, output_path: Path) -> dict[str, Any]:
    paths = {
        "metadata_frame": project_root / "reports/codenet_java_stage_b_frame_v1.json",
        "candidate_manifest": project_root / "data/codenet_java_stage_b_candidates_v1/manifest.json",
        "d0_d2_manifest": project_root / "data/codenet_java_stage_b_eligibility_d0_d2_v1/manifest.json",
        "statement_audit": project_root / "reports/codenet_java_stage_b_statement_overlap_v1.json",
        "d3_primary_manifest": project_root / "data/codenet_java_stage_b_eligibility_d0_d3_v1/d3_manifest.json",
        "d4_manifest": project_root / "data/codenet_java_stage_b_eligibility_d4_train32_v2/d4_manifest.json",
        "d3_sensitivity_0p80_manifest": project_root / "data/codenet_java_stage_b_eligibility_d0_d3_jaccard0p80_v1/d3_manifest.json",
        "d4_sensitivity_0p80_manifest": project_root / "data/codenet_java_stage_b_eligibility_d4_train32_jaccard0p80_v2/d4_manifest.json",
        "d3_sensitivity_0p95_manifest": project_root / "data/codenet_java_stage_b_eligibility_d0_d3_jaccard0p95_v1/d3_manifest.json",
        "d4_sensitivity_0p95_manifest": project_root / "data/codenet_java_stage_b_eligibility_d4_train32_jaccard0p95_v2/d4_manifest.json",
        "d5_manifest": project_root / "data/codenet_java_stage_b_metadata_v2/d5_metadata_manifest.json",
        "d5_index": project_root / "data/codenet_java_stage_b_metadata_v2/d5_metadata_index.jsonl",
        "sampling_design": project_root / "reports/codenet_java_stage_b_sampling_design_v2.json",
        "power": project_root / "reports/codenet_java_stage_b_power_precheck_train32_v2.json",
    }
    design = json.loads(design_path.read_text(encoding="utf-8"))
    git_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=project_root, capture_output=True, text=True, check=False
    ).stdout.strip() or None
    tracked_status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=project_root,
        capture_output=True,
        text=True,
        check=False,
    )
    frozen_commit = design["freeze"].get("implementation_commit")
    frozen_commit_exists = None
    runner_tag_commits: dict[str, str] = {}
    if frozen_commit:
        frozen_commit_exists = subprocess.run(
            ["git", "cat-file", "-e", f"{frozen_commit}^{{commit}}"],
            cwd=project_root,
            capture_output=True,
            text=True,
            check=False,
        ).returncode == 0
        for key in ("validation_runner_tag", "test_runner_tag"):
            tag = str(design["freeze"][key])
            runner_tag_commits[tag] = subprocess.run(
                ["git", "rev-list", "-n", "1", tag],
                cwd=project_root,
                capture_output=True,
                text=True,
                check=False,
            ).stdout.strip()
    lock_path = project_root / "uv.lock"
    report = evaluate_stage_b_readiness(
        design=design,
        d4=json.loads(paths["d4_manifest"].read_text(encoding="utf-8")),
        d5=json.loads(paths["d5_manifest"].read_text(encoding="utf-8")),
        sampling_design=json.loads(paths["sampling_design"].read_text(encoding="utf-8")),
        power=json.loads(paths["power"].read_text(encoding="utf-8")),
        actual_hashes={name: stable_sha256(path.read_bytes()) for name, path in paths.items()},
        repository_commit=git_commit,
        tracked_worktree_clean=tracked_status.returncode == 0 and not tracked_status.stdout.strip(),
        dependency_lock_sha256=stable_sha256(lock_path.read_bytes()) if lock_path.is_file() else None,
        frozen_commit_exists=frozen_commit_exists,
        runner_tag_commits=runner_tag_commits,
    )
    report["inputs"] = {
        "design": {
            "path": portable_manifest_path(design_path, project_root=project_root),
            "sha256": stable_sha256(design_path.read_bytes()),
        },
        **{
            name: {
                "path": portable_manifest_path(path, project_root=project_root),
                "sha256": stable_sha256(path.read_bytes()),
            }
            for name, path in paths.items()
        },
    }
    report["repository_commit"] = git_commit
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(canonical_json_bytes(report))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Fail-closed pre-registration audit for CodeNet Java Stage B.")
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--design", type=Path, default=PROJECT_ROOT / "configs/codenet_java_stage_b_draft_v1.json")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "reports/codenet_java_stage_b_readiness_v1.json")
    args = parser.parse_args()
    report = build_stage_b_readiness_report(
        project_root=args.project_root,
        design_path=args.design,
        output_path=args.output,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    if not report["ready_for_public_registration"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
