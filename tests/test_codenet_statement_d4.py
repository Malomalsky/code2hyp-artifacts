from __future__ import annotations

import json
from pathlib import Path

from scripts.build_codenet_python800_statement_d4 import (
    build_statement_d4_artifacts,
    normalized_problem_statement,
    read_official_identical_problem_clusters,
)


def test_statement_normalization_ignores_markup_case_and_whitespace() -> None:
    left = "<h1>Sum</h1><p>A &lt; B</p><script>secret()</script>"
    right = "<H1>  sum </H1>\n<p>a &lt; b</p>"
    assert normalized_problem_statement(left) == normalized_problem_statement(right) == "sum a < b"


def test_statement_d4_merges_identical_descriptions_without_opening_metrics(tmp_path: Path) -> None:
    descriptions = tmp_path / "descriptions"
    descriptions.mkdir()
    (descriptions / "p1.html").write_text("<h1>Task</h1><p>Add A and B.</p>", encoding="utf-8")
    (descriptions / "p2.html").write_text("<h1>task</h1> <p>Add A and B.</p>", encoding="utf-8")
    (descriptions / "p3.html").write_text("<h1>Other</h1>", encoding="utf-8")
    d3 = tmp_path / "d3"
    d3.mkdir()
    (d3 / "d3_manifest.json").write_text("{}\n", encoding="utf-8")
    rows = [
        {
            "cluster_id": problem,
            "problem_ids": [problem],
            "retained_programs_after_d0_d3": 10,
        }
        for problem in ("p1", "p2", "p3")
    ]
    (d3 / "post_d3_problem_clusters.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    output = tmp_path / "d4"
    manifest = build_statement_d4_artifacts(
        descriptions_root=descriptions,
        d3_dir=d3,
        output_dir=output,
        minimum_cluster_programs=1,
    )
    assert manifest["summary"]["identical_statement_groups"] == 1
    assert manifest["summary"]["problem_cluster_count_after_statement_d4"] == 2
    assert manifest["protocol"]["retrieval_metrics_opened"] is False
    clusters = [json.loads(line) for line in (output / "post_statement_d4_problem_clusters.jsonl").read_text().splitlines()]
    merged = next(cluster for cluster in clusters if cluster["problem_count"] == 2)
    assert merged["retained_programs_after_statement_d4"] == 20


def test_official_d4_map_is_applied_before_eligibility_count(tmp_path: Path) -> None:
    descriptions = tmp_path / "descriptions"
    descriptions.mkdir()
    for problem in ("p00001", "p00002", "p00003"):
        (descriptions / f"{problem}.html").write_text(f"<p>{problem}</p>", encoding="utf-8")
    d3 = tmp_path / "d3"
    d3.mkdir()
    (d3 / "d3_manifest.json").write_text("{}\n", encoding="utf-8")
    rows = [
        {
            "cluster_id": problem,
            "problem_ids": [problem],
            "retained_programs_after_d0_d3": 10,
        }
        for problem in ("p00001", "p00002", "p00003")
    ]
    (d3 / "post_d3_problem_clusters.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    official_map = tmp_path / "identical_problem_clusters"
    official_map.write_text("p00001,p00002\np00003,p99999\n", encoding="utf-8")
    official_readme = tmp_path / "README"
    official_readme.write_text("official provenance\n", encoding="utf-8")

    output = tmp_path / "d4"
    manifest = build_statement_d4_artifacts(
        descriptions_root=descriptions,
        d3_dir=d3,
        output_dir=output,
        minimum_cluster_programs=1,
        official_identical_problem_map=official_map,
        official_source_readme=official_readme,
        official_archive_sha256="a" * 64,
    )

    assert manifest["protocol"]["official_identical_problem_map"] == "applied_and_verified"
    assert manifest["summary"]["official_identical_problem_clusters_total"] == 2
    assert manifest["summary"]["official_identical_problem_clusters_touching_sampling_frame"] == 2
    assert manifest["summary"]["official_identical_problem_clusters_with_multiple_sampling_frame_members"] == 1
    assert manifest["summary"]["official_identical_problem_edges_within_sampling_frame"] == 1
    assert manifest["summary"]["problem_cluster_count_after_statement_d4"] == 2
    assert manifest["gate_precheck"]["final_eligibility"] == "failed"
    clusters = [
        json.loads(line)
        for line in (output / "post_statement_d4_problem_clusters.jsonl").read_text().splitlines()
    ]
    merged = next(cluster for cluster in clusters if cluster["problem_count"] == 2)
    assert merged["problem_ids"] == ["p00001", "p00002"]
    assert merged["retained_programs_after_statement_d4"] == 20


def test_official_cluster_parser_rejects_overlapping_rows(tmp_path: Path) -> None:
    path = tmp_path / "identical_problem_clusters"
    path.write_text("p00001,p00002\np00002,p00003\n", encoding="utf-8")

    try:
        read_official_identical_problem_clusters(path)
    except ValueError as error:
        assert "occurs in clusters" in str(error)
    else:
        raise AssertionError("Overlapping official clusters must fail closed")


def test_official_cluster_parser_rejects_empty_map(tmp_path: Path) -> None:
    path = tmp_path / "identical_problem_clusters"
    path.write_text("\n", encoding="utf-8")

    try:
        read_official_identical_problem_clusters(path)
    except ValueError as error:
        assert "empty" in str(error)
    else:
        raise AssertionError("An empty official map must fail closed")
