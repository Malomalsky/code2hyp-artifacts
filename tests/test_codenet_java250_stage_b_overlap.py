from __future__ import annotations

import json
from pathlib import Path

from scripts.audit_codenet_java250_stage_b_overlap import audit_java250_overlap


def test_overlap_audit_rejects_java_benchmark_as_independent_when_tasks_were_opened(tmp_path: Path) -> None:
    java_root = tmp_path / "java"
    for problem_id in ("p00001", "p00002", "p00003"):
        problem = java_root / problem_id
        problem.mkdir(parents=True)
        (problem / "s1.java").write_text("class A {}", encoding="utf-8")

    clusters = tmp_path / "clusters.jsonl"
    clusters.write_text(
        "\n".join(
            json.dumps(row)
            for row in (
                {"cluster_id": "c1", "problem_ids": ["p00001", "p00002"]},
                {"cluster_id": "c2", "problem_ids": ["p00004"]},
            )
        )
        + "\n",
        encoding="utf-8",
    )
    assignments = tmp_path / "assignments.jsonl"
    assignments.write_text(json.dumps({"cluster_id": "c1"}) + "\n", encoding="utf-8")
    archive = tmp_path / "java.tar.gz"
    archive.write_bytes(b"archive")

    result = audit_java250_overlap(
        java_root=java_root,
        python_clusters_path=clusters,
        stage_a_assignments_path=assignments,
        archive_path=archive,
        output_path=tmp_path / "report.json",
    )

    assert result["counts"]["java250_problem_overlap_with_opened_stage_a"] == 2
    assert result["counts"]["java250_overlapping_stage_a_duplicate_components"] == 1
    assert result["java250_problem_ids_not_in_opened_stage_a"] == ["p00003"]
