from __future__ import annotations

import json
from pathlib import Path

from scripts.analyze_dta_factor_matrix_diagnostics import (
    analyze_factor_matrix_diagnostics,
    format_markdown,
)


def test_analyzes_geometry_aware_diagnostics_and_flags_failure_modes(tmp_path: Path) -> None:
    input_path = tmp_path / "factor.json"
    input_path.write_text(
        json.dumps(
            {
                "experiment": "dta_code2hyp_factor_matrix",
                "benchmark_level": "B_independent_solution",
                "config": {"seed": 20260625, "encoder_policy": "geometry_aware"},
                "runs": [
                    _run("E", 0.0, "lca_product", "measure", 0.34, side_share=0.62),
                    _run("H_1e-4", 1e-4, "lca_product", "measure", 0.33, radius_fraction=0.01, side_share=0.70),
                    _run("H_1", 1.0, "lca_product", "measure", 0.29, radius_fraction=0.44, side_share=0.74),
                    _run("H_1", 1.0, "lca_product", "centroid", 0.36, radius_fraction=0.42, side_share=0.58),
                    _run("H_1", 1.0, "single_point", "measure", 0.31, radius_fraction=0.40, side_share=0.55),
                ],
            }
        ),
        encoding="utf-8",
    )

    summary = analyze_factor_matrix_diagnostics(input_path)
    markdown = format_markdown(summary)

    flags = summary["diagnostic_flags"]
    assert flags["active_curvature_underperforms_near_zero"] is True
    assert flags["active_curvature_underperforms_euclidean"] is True
    assert flags["measure_underperforms_centroid"] is True
    assert flags["side_cost_dominates"] is True
    assert flags["active_curvature_radius_active"] is True
    assert flags["near_zero_radius_near_center"] is True
    assert flags["geometry_confounded_aggregation_comparison"] is True
    assert "active-curvature cell underperforms" in markdown
    assert "encoder-confounded" in markdown


def test_treats_numerically_negligible_curvature_delta_as_tie(tmp_path: Path) -> None:
    input_path = tmp_path / "factor.json"
    input_path.write_text(
        json.dumps(
            {
                "experiment": "dta_code2hyp_factor_matrix",
                "benchmark_level": "B_independent_solution",
                "config": {"seed": 20260625, "encoder_policy": "shared_euclidean"},
                "runs": [
                    _run("E", 0.0, "lca_product", "measure", 0.3262000001, side_share=0.94),
                    _run("H_1e-4", 1e-4, "lca_product", "measure", 0.3262000000, radius_fraction=0.002, side_share=0.94),
                    _run("H_1", 1.0, "lca_product", "measure", 0.3261999999, radius_fraction=0.21, side_share=0.93),
                    _run("H_1", 1.0, "lca_product", "centroid", 0.3200, radius_fraction=0.21, side_share=0.93),
                    _run("H_1", 1.0, "single_point", "measure", 0.3250, radius_fraction=0.21, side_share=0.93),
                ],
            }
        ),
        encoding="utf-8",
    )

    summary = analyze_factor_matrix_diagnostics(input_path)
    markdown = format_markdown(summary)

    assert summary["diagnostic_flags"]["active_curvature_underperforms_near_zero"] is False
    assert summary["diagnostic_flags"]["active_curvature_underperforms_euclidean"] is False
    assert "negative active-curvature result" not in markdown
    assert "near-tie active-curvature result" in markdown


def test_keeps_cost_modes_as_separate_diagnostic_cells(tmp_path: Path) -> None:
    input_path = tmp_path / "factor.json"
    input_path.write_text(
        json.dumps(
            {
                "experiment": "dta_code2hyp_factor_matrix",
                "benchmark_level": "B_independent_solution",
                "config": {"seed": 20260625, "encoder_policy": "shared_euclidean"},
                "runs": [
                    _run("E", 0.0, "lca_product", "measure", 0.20, side_share=1.00, cost_mode="side_only"),
                    _run("H_1e-4", 1e-4, "lca_product", "measure", 0.20, radius_fraction=0.01, side_share=1.00, cost_mode="side_only"),
                    _run("H_1", 1.0, "lca_product", "measure", 0.20, radius_fraction=0.40, side_share=1.00, cost_mode="side_only"),
                    _run("E", 0.0, "lca_product", "measure", 0.31, side_share=0.25, cost_mode="train_normalized_combined"),
                    _run("H_1e-4", 1e-4, "lca_product", "measure", 0.32, radius_fraction=0.01, side_share=0.25, cost_mode="train_normalized_combined"),
                    _run("H_1", 1.0, "lca_product", "measure", 0.35, radius_fraction=0.40, side_share=0.25, cost_mode="train_normalized_combined"),
                ],
            }
        ),
        encoding="utf-8",
    )

    summary = analyze_factor_matrix_diagnostics(input_path)
    markdown = format_markdown(summary)

    cell_ids = {row["cell_id"] for row in summary["cell_summaries"]}
    assert "side_only__E__lca_product__measure" in cell_ids
    assert "train_normalized_combined__H_1__lca_product__measure" in cell_ids
    contrast_modes = {row["cost_mode"] for row in summary["diagnostic_contrasts"]}
    assert contrast_modes == {"side_only", "train_normalized_combined"}
    normalized_curvature = [
        row
        for row in summary["diagnostic_contrasts"]
        if row["cost_mode"] == "train_normalized_combined" and row["contrast"] == "H1_minus_H1e-4"
    ][0]
    assert normalized_curvature["mean_delta_mrr"] > 0.0
    normalized_cell = [
        row
        for row in summary["cell_summaries"]
        if row["cell_id"] == "train_normalized_combined__H_1__lca_product__measure"
    ][0]
    assert normalized_cell["mean_total_distance_side_expected_cost_spearman"] == 0.4
    assert normalized_cell["mean_transport_entropy"] == 1.25
    assert "train_normalized_combined" in markdown
    assert "rho(total, side)" in markdown


def _run(
    geometry: str,
    curvature: float,
    path_object_mode: str,
    method_aggregation: str,
    mrr: float,
    *,
    radius_fraction: float | None = None,
    side_share: float,
    cost_mode: str = "unnormalized_combined",
) -> dict[str, object]:
    diagnostics = {"curvature_radius_fractions": {}, "scaled_norm_median": 0.0}
    if radius_fraction is not None:
        diagnostics["curvature_radius_fractions"] = {
            str(curvature): {
                "scaled_radius_fraction_median": radius_fraction,
                "scaled_radius_fraction_mean": radius_fraction,
                "scaled_radius_fraction_max": radius_fraction + 0.05,
            }
        }
        diagnostics["scaled_norm_median"] = radius_fraction
    return {
        "cell_id": f"{geometry}__{path_object_mode}__{method_aggregation}",
        "geometry": geometry,
        "curvature": curvature,
        "path_object_mode": path_object_mode,
        "method_aggregation": method_aggregation,
        "cost_mode": cost_mode,
        "encoder_policy": "geometry_aware",
        "mrr": mrr,
        "recall_at_1": mrr / 2.0,
        "embedding_norm_diagnostics": diagnostics,
        "cost_component_diagnostics": {
            "point_cost_share": 1.0 - side_share,
            "side_cost_share": side_share,
            "total_cost_scale": 1.0,
        },
        "query_records": [
            {"query_task": "task-a", "rank": int(round(1.0 / mrr)), "query_id": "q0"}
        ],
        "retrieval_diagnostics": {
            "total_distance_side_expected_cost_spearman": 0.4,
            "total_distance_point_expected_cost_spearman": 0.8,
            "transport_entropy_mean": 1.25,
        },
    }
