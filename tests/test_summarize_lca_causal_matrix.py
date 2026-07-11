from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.summarize_lca_causal_matrix import render_markdown, summarize_lca_causal_matrix


def test_summary_averages_seeds_within_task_before_task_cluster_bootstrap(tmp_path: Path) -> None:
    first = _write_payload(tmp_path / "seed-1.json", seed=1, true=(0.8, 0.4), zero=(0.2, 0.4), hee=(0.9, 0.5))
    second = _write_payload(tmp_path / "seed-2.json", seed=2, true=(0.6, 0.6), zero=(0.4, 0.2), hee=(0.8, 0.7))

    summary = summarize_lca_causal_matrix((first, second), bootstrap_resamples=500, bootstrap_seed=42)

    assert summary["status"] == "pilot"
    assert summary["study_stage"] == "pilot"
    assert summary["cells"]["EEE__true_lca__measure"]["task_scores"] == pytest.approx(
        {"task-a": 0.7, "task-b": 0.5}
    )
    h1 = next(row for row in summary["contrasts"] if row["name"] == "H1_true_lca_vs_zero_anchor")
    assert h1["task_differences"] == pytest.approx({"task-a": 0.4, "task-b": 0.2})
    assert h1["mean_task_difference"] == pytest.approx(0.3)
    assert h1["task_signs"] == {"positive": 2, "tie": 0, "negative": 0}
    assert h1["exact_sign_test_two_sided_p"] == pytest.approx(0.5)
    report = render_markdown(summary)
    assert "Planned contrasts" in report
    assert "Gate C pass" in report


def test_summary_rejects_duplicate_seed_files(tmp_path: Path) -> None:
    first = _write_payload(tmp_path / "seed-1a.json", seed=1, true=(0.8, 0.4), zero=(0.2, 0.4), hee=(0.9, 0.5))
    second = _write_payload(tmp_path / "seed-1b.json", seed=1, true=(0.6, 0.6), zero=(0.4, 0.2), hee=(0.8, 0.7))

    with pytest.raises(ValueError, match="distinct seeds"):
        summarize_lca_causal_matrix((first, second), bootstrap_resamples=10)


def _write_payload(
    path: Path,
    *,
    seed: int,
    true: tuple[float, float],
    zero: tuple[float, float],
    hee: tuple[float, float],
) -> Path:
    def row(cell_id: str, scores: tuple[float, float]) -> dict[str, object]:
        return {
            "cell_id": cell_id,
            "map_at_r": sum(scores) / 2.0,
            "task_scores": {"task-a": scores[0], "task-b": scores[1]},
        }

    payload = {
        "experiment": "code2hyp_lca_causal_and_role_geometry_matrix",
        "status": "complete",
        "config": {
            "seed": seed,
            "benchmark_level": "B_independent_solution",
            "language": "python",
            "dim": 4,
            "epochs": 2,
            "max_paths": 64,
            "path_selection_policy": "lca_depth_stratified",
            "item_scope": "callable",
            "ap_at_r": 8,
        },
        "tasks": [{"label": "task-a"}, {"label": "task-b"}],
        "runs": [
            row("EEE__true_lca__measure", true),
            row("EEE__zero_anchor__measure", zero),
            row("EEE__endpoint_only__measure", zero),
            row("HEE_near_zero__true_lca__measure", true),
            row("HEE__true_lca__measure", hee),
            row("HHH__true_lca__measure", true),
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path
