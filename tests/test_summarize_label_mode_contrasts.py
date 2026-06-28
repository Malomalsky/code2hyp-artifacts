from __future__ import annotations

import json

from scripts.summarize_code2hyp_label_mode_contrasts import summarize


def test_label_mode_contrasts_pair_queries_by_task(tmp_path):
    rows = []
    for variant, mrr in [
        ("structural_categorical", 1.0),
        ("structural_scalar_hash", 0.5),
        ("structural_none", 0.25),
    ]:
        rows.append(
            {
                "dataset": "toy",
                "seed": 1,
                "variant": variant,
                "query_id": "q1",
                "query_task": "task-a",
                "rank": 1,
                "mrr": mrr,
                "recall_at_1": 1.0 if mrr == 1.0 else 0.0,
                "recall_at_5": 1.0,
                "mean_rank": 1.0 / mrr,
            }
        )
    path = tmp_path / "label_modes.json"
    path.write_text(
        json.dumps({"max_paths": 8, "distance_mode": "centroid_proxy", "query_rows": rows}),
        encoding="utf-8",
    )

    result = summarize(path, bootstrap_samples=10, seed=7)
    contrast = next(row for row in result["contrasts"] if row["contrast"] == "categorical_minus_scalar_hash")

    assert contrast["dataset"] == "toy"
    assert contrast["task_count"] == 1
    assert contrast["paired_query_count"] == 1
    assert contrast["delta_mrr"] == 0.5
