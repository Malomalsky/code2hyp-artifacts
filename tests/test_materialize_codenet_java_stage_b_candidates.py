from __future__ import annotations

import csv
import io
import json
import tarfile
from pathlib import Path

from scripts.materialize_codenet_java_stage_b_candidates import materialize_candidates


def test_materializer_copies_only_accepted_java_from_eligible_components(tmp_path: Path) -> None:
    header = [
        "submission_id", "problem_id", "user_id", "date", "language", "original_language",
        "filename_ext", "status", "cpu_time", "memory", "code_size", "accuracy",
    ]
    rows = [
        ["s000000001", "p00001", "u1", "0", "Java", "Java", "java", "Accepted", "1", "1", "1", ""],
        ["s000000002", "p00001", "u2", "0", "Java", "Java", "java", "Wrong Answer", "1", "1", "1", ""],
        ["s000000003", "p00002", "u3", "0", "Java", "Java", "java", "Accepted", "1", "1", "1", ""],
    ]
    metadata = tmp_path / "metadata.tar.gz"
    with tarfile.open(metadata, "w:gz") as archive:
        for problem_id in ("p00001", "p00002"):
            text = io.StringIO()
            writer = csv.writer(text, lineterminator="\n")
            writer.writerow(header)
            writer.writerows(row for row in rows if row[1] == problem_id)
            payload = text.getvalue().encode()
            member = tarfile.TarInfo(f"Project_CodeNet/metadata/{problem_id}.csv")
            member.size = len(payload)
            archive.addfile(member, io.BytesIO(payload))

    source = tmp_path / "source.tar.gz"
    with tarfile.open(source, "w:gz") as archive:
        for row in rows:
            payload = f"class {row[0]} {{}}".encode()
            member = tarfile.TarInfo(f"Project_CodeNet/data/{row[1]}/Java/{row[0]}.java")
            member.size = len(payload)
            archive.addfile(member, io.BytesIO(payload))

    frame = tmp_path / "frame.json"
    frame.write_text(
        json.dumps(
            {
                "schema_version": "code2hyp-codenet-java-stage-b-frame-v1",
                "inputs": {"metadata_archive": {"sha256": "m"}, "source_archive": {"sha256": "s"}},
                "components": [
                    {
                        "component_id": "c1",
                        "java_problem_ids": ["p00001"],
                        "metadata_eligible_for_evaluation_minimum_16": True,
                    },
                    {
                        "component_id": "c2",
                        "java_problem_ids": ["p00002"],
                        "metadata_eligible_for_evaluation_minimum_16": False,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    output_archive = tmp_path / "candidates.tar"
    output_manifest = tmp_path / "manifest.json"

    result = materialize_candidates(
        frame_report_path=frame,
        metadata_archive=metadata,
        source_archive=source,
        output_archive=output_archive,
        output_manifest=output_manifest,
    )

    assert result["counts"]["accepted_java_sources"] == 1
    with tarfile.open(output_archive) as archive:
        assert archive.getnames() == ["p00001/s000000001.java"]
