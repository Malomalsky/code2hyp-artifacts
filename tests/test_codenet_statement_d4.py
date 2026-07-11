from __future__ import annotations

import json
from pathlib import Path

from scripts.build_codenet_python800_statement_d4 import (
    build_statement_d4_artifacts,
    normalized_problem_statement,
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
