from __future__ import annotations

import csv
import io
import json
import tarfile
from pathlib import Path

from scripts.audit_codenet_java_stage_b_frame import audit_java_stage_b_frame


def test_stage_b_frame_excludes_whole_official_component_touching_stage_a(tmp_path: Path) -> None:
    metadata = tmp_path / "metadata.tar.gz"
    rows = [
        ["s1", "p00001", "u1", "0", "Java", "Java", "java", "Accepted", "1", "1", "1", ""],
        ["s2", "p00002", "u2", "0", "Java", "Java", "java", "Accepted", "1", "1", "1", ""],
        ["s3", "p00003", "u3", "0", "Java", "Java", "java", "Accepted", "1", "1", "1", ""],
    ]
    header = [
        "submission_id", "problem_id", "user_id", "date", "language", "original_language",
        "filename_ext", "status", "cpu_time", "memory", "code_size", "accuracy",
    ]
    with tarfile.open(metadata, "w:gz") as archive:
        for problem_id in ("p00001", "p00002", "p00003"):
            text = io.StringIO()
            writer = csv.writer(text, lineterminator="\n")
            writer.writerow(header)
            writer.writerows(row for row in rows if row[1] == problem_id)
            payload = text.getvalue().encode("utf-8")
            member = tarfile.TarInfo(f"Project_CodeNet/metadata/{problem_id}.csv")
            member.size = len(payload)
            archive.addfile(member, io.BytesIO(payload))

    source = tmp_path / "source.tar.gz"
    source.write_bytes(b"source")
    clusters = tmp_path / "clusters.jsonl"
    clusters.write_text(json.dumps({"problem_ids": ["p00001"]}) + "\n", encoding="utf-8")
    duplicates = tmp_path / "duplicates"
    duplicates.write_text("p00001,p00002\n", encoding="utf-8")

    result = audit_java_stage_b_frame(
        metadata_archive=metadata,
        source_archive=source,
        stage_a_clusters_path=clusters,
        official_duplicates_path=duplicates,
        output_path=tmp_path / "report.json",
    )

    assert result["counts"]["java_components_overlapping_opened_stage_a"] == 1
    assert result["counts"]["independent_java_components"] == 1
    independent = [row for row in result["components"] if row["independent_of_opened_stage_a"]]
    assert independent[0]["problem_ids"] == ["p00003"]
