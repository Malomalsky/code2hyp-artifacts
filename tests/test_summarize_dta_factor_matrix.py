from __future__ import annotations

import json
from pathlib import Path

from scripts.summarize_dta_factor_matrix import format_markdown, summarize_dta_factor_matrix


def test_summarizes_primary_factor_contrasts_with_holm(tmp_path: Path) -> None:
    input_path = tmp_path / "factor.json"
    input_path.write_text(json.dumps({"experiment": "dta_code2hyp_factor_matrix", "benchmark_level": "B", "runs": _runs()}), encoding="utf-8")

    summary = summarize_dta_factor_matrix(input_path, bootstrap_samples=100, seed=11)
    markdown = format_markdown(summary)

    assert len(summary["contrast_rows"]) == 4
    by_name = {row["contrast"]: row for row in summary["contrast_rows"]}
    assert by_name["C_path_LCA_product_minus_single_point"]["positive_tasks"] == 2
    assert by_name["C_measure_measure_minus_centroid"]["positive_tasks"] == 2
    assert by_name["C_curvature_H1_minus_H1e-4"]["positive_tasks"] == 2
    assert all("holm_p" in row for row in summary["contrast_rows"])
    assert "Primary contrasts" in markdown


def _runs() -> list[dict[str, object]]:
    cells = [
        ("E", 0.0, "single_point", "centroid", {"task-a": (5, 5), "task-b": (5, 5)}),
        ("E", 0.0, "single_point", "measure", {"task-a": (4, 4), "task-b": (4, 4)}),
        ("E", 0.0, "lca_product", "centroid", {"task-a": (3, 3), "task-b": (3, 3)}),
        ("E", 0.0, "lca_product", "measure", {"task-a": (3, 3), "task-b": (3, 3)}),
        ("H_1e-4", 1e-4, "single_point", "centroid", {"task-a": (5, 5), "task-b": (5, 5)}),
        ("H_1e-4", 1e-4, "single_point", "measure", {"task-a": (4, 4), "task-b": (4, 4)}),
        ("H_1e-4", 1e-4, "lca_product", "centroid", {"task-a": (3, 3), "task-b": (3, 3)}),
        ("H_1e-4", 1e-4, "lca_product", "measure", {"task-a": (2, 2), "task-b": (2, 2)}),
        ("H_1", 1.0, "single_point", "centroid", {"task-a": (5, 5), "task-b": (5, 5)}),
        ("H_1", 1.0, "single_point", "measure", {"task-a": (4, 4), "task-b": (4, 4)}),
        ("H_1", 1.0, "lca_product", "centroid", {"task-a": (3, 3), "task-b": (3, 3)}),
        ("H_1", 1.0, "lca_product", "measure", {"task-a": (1, 1), "task-b": (1, 1)}),
    ]
    return [
        {
            "cell_id": f"{geometry}__{path_object_mode}__{method_aggregation}",
            "geometry": geometry,
            "curvature": curvature,
            "path_object_mode": path_object_mode,
            "method_aggregation": method_aggregation,
            "query_records": [
                {"query_task": task, "rank": rank, "query_id": f"{task}-{index}"}
                for task, ranks in task_ranks.items()
                for index, rank in enumerate(ranks)
            ],
        }
        for geometry, curvature, path_object_mode, method_aggregation, task_ranks in cells
    ]
