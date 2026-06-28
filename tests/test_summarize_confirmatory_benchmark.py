from __future__ import annotations

import json
from pathlib import Path

from scripts.summarize_confirmatory_benchmark import (
    ConfirmatoryInput,
    format_markdown,
    summarize_confirmatory_benchmark,
)


def test_summarizes_cells_and_paired_query_contrasts(tmp_path: Path) -> None:
    path = tmp_path / "synthetic.json"
    path.write_text(
        json.dumps(
            {
                "experiment": "synthetic_confirmatory",
                "config": {"seed": 101},
                "runs": [
                    _run("single_point", "measure", [2, 4]),
                    _run("lca_product", "measure", [1, 2]),
                    _run("single_point", "centroid", [3, 6]),
                    _run("lca_product", "centroid", [1, 3]),
                ],
            }
        ),
        encoding="utf-8",
    )

    summary = summarize_confirmatory_benchmark(
        [ConfirmatoryInput(dataset="synthetic", path=path)],
        bootstrap_samples=100,
        seed=7,
    )
    markdown = format_markdown(summary)

    cell_by_id = {row["cell_id"]: row for row in summary["cell_summaries"]}
    assert cell_by_id["synthetic::E::train_weighted_combined_p1p00::lca_product::measure"]["mrr"] == 0.75
    assert cell_by_id["synthetic::E::train_weighted_combined_p1p00::single_point::measure"]["mrr"] == 0.375
    assert "95% bootstrap CI" in markdown

    contrasts = {row["contrast"]: row for row in summary["paired_contrasts"]}
    contrast = contrasts["lca_product_measure_minus_single_point_measure"]
    assert contrast["paired_query_count"] == 2
    assert contrast["delta_mrr"] == 0.375
    assert contrast["delta_recall_at_1"] == 0.5
    assert contrast["delta_mean_rank"] == -1.5
    assert contrast["bootstrap_ci"]["delta_mrr"][0] <= contrast["delta_mrr"] <= contrast["bootstrap_ci"]["delta_mrr"][1]
    assert "does not improve MRR by" not in markdown

    task_contrasts = {row["contrast"]: row for row in summary["task_level_contrasts"]}
    task_contrast = task_contrasts["lca_product_measure_minus_single_point_measure"]
    assert task_contrast["task_count"] == 1
    assert task_contrast["paired_query_count"] == 2
    assert task_contrast["delta_mrr"] == 0.375
    assert task_contrast["task_sign_test_p"]["delta_mrr"] == 1.0
    assert "Paired task-level contrasts" in markdown


def test_uses_payload_seed_when_run_seed_is_absent(tmp_path: Path) -> None:
    path = tmp_path / "seeded.json"
    path.write_text(
        json.dumps({"config": {"seed": 20260625}, "runs": [_run("single_point", "measure", [1])]}),
        encoding="utf-8",
    )

    summary = summarize_confirmatory_benchmark([ConfirmatoryInput(dataset="d", path=path)], bootstrap_samples=10, seed=1)

    assert summary["query_rows"][0]["seed"] == 20260625


def test_nonpositive_interpretation_uses_neutral_change_wording(tmp_path: Path) -> None:
    path = tmp_path / "negative.json"
    path.write_text(
        json.dumps(
            {
                "config": {"seed": 101},
                "runs": [
                    _run("single_point", "measure", [1, 1]),
                    _run("lca_product", "measure", [2, 2]),
                ],
            }
        ),
        encoding="utf-8",
    )

    summary = summarize_confirmatory_benchmark([ConfirmatoryInput(dataset="synthetic", path=path)], bootstrap_samples=10, seed=7)
    text = "\n".join(summary["interpretation"])

    assert "changes MRR by -0.5000" in text
    assert "does not improve MRR by" not in text


def _run(path_object_mode: str, method_aggregation: str, ranks: list[int]) -> dict[str, object]:
    return {
        "cell_id": f"E__{path_object_mode}__{method_aggregation}",
        "geometry": "E",
        "curvature": 0.0,
        "cost_mode": "train_weighted_combined_p1p00",
        "path_object_mode": path_object_mode,
        "method_aggregation": method_aggregation,
        "query_records": [
            {
                "query_id": f"q-{index}",
                "query_task": "task-a",
                "rank": rank,
            }
            for index, rank in enumerate(ranks)
        ],
    }
