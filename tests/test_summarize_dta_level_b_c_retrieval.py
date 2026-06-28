from __future__ import annotations

import json
from pathlib import Path

from scripts.summarize_dta_level_b_c_retrieval import format_markdown, summarize_level_b_c_retrieval


def test_summarizes_task_level_curvature_contrasts(tmp_path: Path) -> None:
    payload = {
        "benchmark_level": "B_independent_solution",
        "config": {"seed": 1},
        "runs": [
            _run(0.0, {"task-a": (2, 4), "task-b": (4, 4)}),
            _run(1e-4, {"task-a": (1, 4), "task-b": (2, 4)}),
            _run(1.0, {"task-a": (1, 4), "task-b": (1, 4)}),
        ],
    }
    input_path = tmp_path / "level_b.json"
    input_path.write_text(json.dumps(payload), encoding="utf-8")

    summary = summarize_level_b_c_retrieval((input_path,), bootstrap_samples=100, seed=7)
    markdown = format_markdown(summary)

    assert len(summary["task_metric_rows"]) == 6
    param = next(row for row in summary["contrast_rows"] if row["contrast"] == "param_H1e-4_minus_E")
    curvature = next(row for row in summary["contrast_rows"] if row["contrast"] == "curvature_H1_minus_H1e-4")
    assert param["positive_tasks"] == 2
    assert param["n_tasks"] == 2
    assert param["mean_task_delta"] > 0.0
    assert curvature["positive_tasks"] == 1
    assert "Primary curvature contrasts" in markdown


def _run(curvature: float, task_to_ranks: dict[str, tuple[int, int]]) -> dict[str, object]:
    records = []
    for task, ranks in task_to_ranks.items():
        for index, rank in enumerate(ranks):
            records.append({"query_task": task, "rank": rank, "query_id": f"{task}-{index}"})
    return {
        "benchmark_level": "B_independent_solution",
        "seed": 1,
        "curvature": curvature,
        "query_records": records,
    }
