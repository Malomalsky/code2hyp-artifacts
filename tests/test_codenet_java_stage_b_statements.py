from __future__ import annotations

import json
from pathlib import Path

from scripts.audit_codenet_java_stage_b_statements import audit_statement_overlap


def test_statement_audit_detects_cross_frame_identity(tmp_path: Path) -> None:
    descriptions = tmp_path / "descriptions"
    descriptions.mkdir()
    (descriptions / "p00001.html").write_text("<h1>Same task</h1>", encoding="utf-8")
    (descriptions / "p00002.html").write_text("<h1> same  TASK </h1>", encoding="utf-8")
    frame = tmp_path / "frame.json"
    frame.write_text(
        json.dumps(
            {
                "components": [
                    {
                        "component_id": "c1",
                        "problem_ids": ["p00001"],
                        "java_problem_ids": ["p00001"],
                        "metadata_eligible_for_evaluation_minimum_16": True,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    d0_d2 = tmp_path / "d0.json"
    d0_d2.write_text(json.dumps({"preliminary_d4_edges": []}), encoding="utf-8")
    stage_a = tmp_path / "stage_a.jsonl"
    stage_a.write_text(json.dumps({"problem_ids": ["p00002"]}) + "\n", encoding="utf-8")

    result = audit_statement_overlap(
        frame_report_path=frame,
        d0_d2_manifest_path=d0_d2,
        stage_a_clusters_path=stage_a,
        descriptions_root=descriptions,
        output_path=tmp_path / "result.json",
    )

    assert result["counts"]["exact_normalized_statement_collisions_java_vs_opened_stage_a"] == 1
    assert not result["interpretation"]["no_exact_statement_leak_detected_among_available_descriptions"]
