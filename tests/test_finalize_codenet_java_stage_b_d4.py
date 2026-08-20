from __future__ import annotations

import json
from pathlib import Path

from scripts.finalize_codenet_java_stage_b_d4 import finalize_java_stage_b_d4


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_d4_merges_exact_statements_and_excludes_missing_clusters(tmp_path: Path) -> None:
    frame_path = tmp_path / "frame.json"
    frame_path.write_text(
        json.dumps(
            {
                "components": [
                    {"component_id": "c1", "java_problem_ids": ["p1"]},
                    {"component_id": "c2", "java_problem_ids": ["p2"]},
                    {"component_id": "c3", "java_problem_ids": ["p3"]},
                ]
            }
        ),
        encoding="utf-8",
    )
    d3_dir = tmp_path / "d3"
    d3_dir.mkdir()
    (d3_dir / "d3_manifest.json").write_text("{}\n", encoding="utf-8")
    _write_jsonl(
        d3_dir / "post_d3_problem_clusters.jsonl",
        [
            {"cluster_id": "d1", "problem_ids": ["c1"]},
            {"cluster_id": "d2", "problem_ids": ["c2"]},
            {"cluster_id": "d3", "problem_ids": ["c3"]},
        ],
    )
    _write_jsonl(
        d3_dir / "d3_index.jsonl",
        [
            {"problem_id": "c1", "user_id": "u1", "retained_after_d0_d3": True},
            {"problem_id": "c2", "user_id": "u1", "retained_after_d0_d3": True},
            {"problem_id": "c3", "user_id": "u3", "retained_after_d0_d3": True},
        ],
    )
    statement_path = tmp_path / "statements.json"
    statement_path.write_text(
        json.dumps(
            {
                "java_descriptions_missing": ["p3"],
                "cross_frame_exact_statement_collisions": [],
                "cross_component_exact_statement_groups": [
                    {
                        "normalized_text_sha256": "abc",
                        "official_component_ids": ["c1", "c2"],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    result = finalize_java_stage_b_d4(
        frame_report_path=frame_path,
        d3_dir=d3_dir,
        statement_audit_path=statement_path,
        output_dir=tmp_path / "d4",
        train_programs_per_cluster=16,
    )

    assert result["summary"]["problem_clusters_after_d4_union"] == 2
    assert result["summary"]["problem_clusters_excluded_for_missing_statements"] == 1
    clusters = [
        json.loads(line)
        for line in (tmp_path / "d4" / "post_d4_problem_clusters.jsonl").read_text().splitlines()
    ]
    merged = next(cluster for cluster in clusters if cluster["official_component_ids"] == ["c1", "c2"])
    assert merged["retained_programs_after_d0_d4"] == 2
    assert merged["distinct_users_after_d0_d4"] == 1
    assert merged["eligible_train"] is False
    assert result["summary"]["eligible_train_clusters"] == 0
