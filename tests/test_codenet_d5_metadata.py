from __future__ import annotations

import csv
import json
from pathlib import Path

from scripts.build_codenet_python800_d5_metadata import build_d5_metadata_artifacts


def test_d5_metadata_join_hashes_users_and_does_not_change_primary(tmp_path: Path) -> None:
    metadata = tmp_path / "metadata"
    metadata.mkdir()
    fieldnames = [
        "submission_id",
        "problem_id",
        "user_id",
        "date",
        "language",
        "original_language",
        "filename_ext",
        "status",
        "cpu_time",
        "memory",
        "code_size",
        "accuracy",
    ]
    examples = {
        "p1": [("s1", "u1"), ("s2", "u2")],
        "p2": [("s3", "u1")],
    }
    for problem, submissions in examples.items():
        with (metadata / f"{problem}.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for submission, user in submissions:
                writer.writerow(
                    {
                        "submission_id": submission,
                        "problem_id": problem,
                        "user_id": user,
                        "date": "1600000000",
                        "language": "Python",
                        "original_language": "Python (3.8.2)",
                        "filename_ext": "py",
                        "status": "Accepted",
                        "cpu_time": "1",
                        "memory": "1",
                        "code_size": "1",
                        "accuracy": "",
                    }
                )
    d3 = tmp_path / "d3"
    d3.mkdir()
    (d3 / "d3_manifest.json").write_text("{}\n", encoding="utf-8")
    d3_rows = [
        {
            "problem_id": problem,
            "submission_id": submission,
            "source_relpath": f"{problem}/{submission}.py",
            "retained_after_d0_d3": True,
        }
        for problem, submissions in examples.items()
        for submission, _ in submissions
    ]
    (d3 / "d3_index.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in d3_rows),
        encoding="utf-8",
    )
    d4 = tmp_path / "d4"
    d4.mkdir()
    (d4 / "statement_d4_manifest.json").write_text("{}\n", encoding="utf-8")
    clusters = [
        {"cluster_id": "c1", "problem_ids": ["p1"]},
        {"cluster_id": "c2", "problem_ids": ["p2"]},
    ]
    (d4 / "post_statement_d4_problem_clusters.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in clusters),
        encoding="utf-8",
    )
    output = tmp_path / "d5"
    manifest = build_d5_metadata_artifacts(
        metadata_root=metadata,
        d3_dir=d3,
        statement_d4_dir=d4,
        output_dir=output,
        metadata_archive_sha256="a" * 64,
    )
    assert manifest["protocol"]["primary_D0_D4_changed"] is False
    assert manifest["protocol"]["retrieval_metrics_opened"] is False
    assert "one retained program per user" in manifest["protocol"]["within_cluster_selection"]
    assert manifest["summary"]["distinct_users"] == 2
    assert manifest["summary"]["users_spanning_multiple_problem_clusters"] == 1
    assert manifest["summary"]["minimum_distinct_users_per_problem_cluster"] == 1
    rows = [json.loads(line) for line in (output / "d5_metadata_index.jsonl").read_text().splitlines()]
    assert all(row["user_id_sha256"] not in {"u1", "u2"} for row in rows)
    assert rows[0]["user_id_sha256"] == rows[2]["user_id_sha256"]
