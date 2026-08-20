import pytest

from scripts.build_codenet_java_stage_b_metadata import build_metadata_rows


def test_java_metadata_join_matches_duplicate_closed_d4_counts() -> None:
    d3 = [
        {
            "problem_id": "component-a",
            "original_problem_id": "p1",
            "retained_after_d0_d3": True,
            "source_relpath": "p1/s1.java",
            "submission_id": "s1",
            "user_id": "u1",
        },
        {
            "problem_id": "component-b",
            "original_problem_id": "p2",
            "retained_after_d0_d3": True,
            "source_relpath": "p2/s2.java",
            "submission_id": "s2",
            "user_id": "u2",
        },
    ]
    d4 = [
        {
            "cluster_id": "cluster-ab",
            "official_component_ids": ["component-a", "component-b"],
            "eligible_evaluation_minimum_16": True,
            "retained_programs_after_d0_d4": 2,
            "distinct_users_after_d0_d4": 2,
        }
    ]

    rows, summary = build_metadata_rows(d3, d4)

    assert summary["programs"] == 2
    assert {row["problem_cluster_id"] for row in rows} == {"cluster-ab"}
    assert all(len(row["user_id_sha256"]) == 64 for row in rows)

    d4[0]["distinct_users_after_d0_d4"] = 3
    with pytest.raises(ValueError, match="user count differs"):
        build_metadata_rows(d3, d4)
