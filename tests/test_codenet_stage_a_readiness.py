from __future__ import annotations

from pathlib import Path

from scripts.check_codenet_stage_a_readiness import evaluate_readiness


def test_readiness_fails_closed_on_pending_official_map_and_registration(tmp_path: Path) -> None:
    report = evaluate_readiness(
        project_root=tmp_path,
        design={
            "registration_doi": None,
            "status": "draft_not_registered",
            "eligibility": {"required_parse_rate": 0.95, "minimum_clusters_for_practical_claim": 764},
            "split": {"generated": False},
            "test_policy": {"test_labels_opened": False},
        },
        d0_d2={"summary": {"source_files": 240000, "parse_rate": 0.98}},
        d3={"experiment_role": "full_pre_split_D3_eligibility_without_retrieval_metrics"},
        d4={
            "summary": {"eligible_problem_clusters_minimum_64": 773, "descriptions_missing": 0},
            "protocol": {"official_identical_problem_map": "pending_full_CodeNet_derived_metadata"},
        },
        d5={"summary": {"minimum_distinct_users_per_problem_cluster": 92}},
        attrition={"summary": {"decision": "global author removal is not estimand-preserving"}},
    )
    assert report["ready_for_stage_a_registration"] is False
    assert "official_identical_problem_map" in report["blocking_failures"]
    assert "registration_doi_pending" not in report["blocking_failures"]
    assert "independent_git_repository" in report["blocking_failures"]


def test_readiness_fails_closed_on_unpinned_official_map_checksum(tmp_path: Path) -> None:
    report = evaluate_readiness(
        project_root=tmp_path,
        design={
            "registration_doi": None,
            "status": "draft_not_registered",
            "dataset": {
                "official_identical_problem_map": {"sha256": "a" * 64},
                "official_full_archive": {"sha256": "b" * 64},
            },
            "eligibility": {"required_parse_rate": 0.95, "minimum_clusters_for_practical_claim": 764},
            "split": {"generated": False},
            "test_policy": {"test_labels_opened": False},
        },
        d0_d2={"summary": {"source_files": 240000, "parse_rate": 0.98}},
        d3={"experiment_role": "full_pre_split_D3_eligibility_without_retrieval_metrics"},
        d4={
            "input": {
                "official_identical_problem_map_sha256": "c" * 64,
                "official_full_archive_sha256": "b" * 64,
            },
            "summary": {"eligible_problem_clusters_minimum_64": 773, "descriptions_missing": 0},
            "protocol": {"official_identical_problem_map": "applied_and_verified"},
        },
        d5={"summary": {"minimum_distinct_users_per_problem_cluster": 92}},
        attrition={"summary": {"decision": "global author removal is not estimand-preserving"}},
    )

    assert "official_identical_problem_map_checksum" in report["blocking_failures"]
    assert "official_full_archive_checksum" not in report["blocking_failures"]
