from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib-cache"))
os.environ.setdefault("XDG_CACHE_HOME", str(PROJECT_ROOT / ".matplotlib-cache"))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from plotnine import (
    aes,
    element_text,
    facet_wrap,
    geom_col,
    geom_hline,
    geom_text,
    geom_tile,
    ggplot,
    labs,
    scale_fill_gradient2,
    scale_fill_manual,
    theme,
    theme_bw,
)

from scripts.summarize_raw_ast_factor_matrix import summarize_factor_matrix


CONTRAST_COLORS = {
    "curvature": "#3C6E71",
    "path_object": "#8C4A2F",
    "aggregation": "#6C5478",
    "orientation": "#5B6C9D",
    "full_model": "#375A7F",
}


def plot_factor_matrix(*, input_path: Path, output_prefix: Path) -> tuple[Path, Path, Path, Path]:
    summary = summarize_factor_matrix(input_path)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    heatmap_png = output_prefix.with_name(output_prefix.name + "_heatmap").with_suffix(".png")
    heatmap_pdf = output_prefix.with_name(output_prefix.name + "_heatmap").with_suffix(".pdf")
    deltas_png = output_prefix.with_name(output_prefix.name + "_deltas").with_suffix(".png")
    deltas_pdf = output_prefix.with_name(output_prefix.name + "_deltas").with_suffix(".pdf")

    _heatmap_plot(summary["means"]).save(heatmap_png, dpi=300, verbose=False)
    _heatmap_plot(summary["means"]).save(heatmap_pdf, verbose=False)
    _deltas_plot(_delta_rows(summary)).save(deltas_png, dpi=300, verbose=False)
    _deltas_plot(_delta_rows(summary)).save(deltas_pdf, verbose=False)
    return heatmap_png, heatmap_pdf, deltas_png, deltas_pdf


def _heatmap_plot(rows: Sequence[dict[str, Any]]) -> ggplot:
    if not rows:
        raise ValueError("factor-matrix heatmap requires at least one mean row")
    frame = pd.DataFrame(rows)
    frame["path_cost_orientation"] = frame.get("path_cost_orientation", "directed")
    include_orientation = len(set(frame["path_cost_orientation"])) > 1
    frame["cell"] = frame["path_object_mode"] + " / " + frame["method_aggregation"]
    if include_orientation:
        frame["cell"] = frame["cell"] + "\n" + frame["path_cost_orientation"]
    frame["geometry_label"] = frame.apply(_geometry_label, axis=1)
    frame["project_cell"] = frame["project"] + "\n" + frame["cell"]
    frame["label"] = frame.apply(lambda row: f"{row['mean_mrr']:.3f}\nn={int(row['n'])}", axis=1)
    return (
        ggplot(frame, aes(x="geometry_label", y="project_cell", fill="mean_mrr"))
        + geom_tile(color="#222222", size=0.25)
        + geom_text(aes(label="label"), size=7.0, color="#111111", lineheight=0.85)
        + scale_fill_gradient2(low="#B45A4A", mid="#F2EFE8", high="#315F72", midpoint=frame["mean_mrr"].mean())
        + labs(
            title="Raw-AST Code2Hyp factor matrix",
            subtitle="Mean MRR by project, path object, method aggregation, and geometry; labels show MRR and completed seeds",
            x="Node geometry",
            y="Project and representation cell",
            fill="Mean MRR",
        )
        + theme_bw(base_size=8.5)
        + theme(
            figure_size=(7.8, max(3.8, 0.36 * len(frame["project_cell"].unique()) + 1.4)),
            axis_text_x=element_text(rotation=20, ha="right"),
        )
    )


def _deltas_plot(rows: Sequence[dict[str, Any]]) -> ggplot:
    if not rows:
        raise ValueError("factor-matrix delta plot requires at least one contrast row")
    frame = pd.DataFrame(rows)
    frame["project_cell"] = frame.apply(_project_cell_label, axis=1)
    frame["label"] = frame.apply(lambda row: f"{row['delta_mrr']:+.3f}\nn={int(row['n'])}", axis=1)
    frame["label_y"] = frame["delta_mrr"] / 2.0
    frame["contrast"] = pd.Categorical(
        frame["contrast"],
        categories=("curvature", "path_object", "aggregation", "orientation", "full_model"),
        ordered=True,
    )
    return (
        ggplot(frame, aes(x="project_cell", y="delta_mrr", fill="contrast"))
        + geom_hline(yintercept=0.0, color="#222222", size=0.4)
        + geom_col(color="#222222", size=0.2, width=0.72)
        + geom_text(aes(y="label_y", label="label"), size=7.0, color="#F8F8F2", lineheight=0.85)
        + facet_wrap("~ contrast_label", scales="free_y", ncol=1)
        + scale_fill_manual(values=CONTRAST_COLORS)
        + labs(
            title="Factor effects in raw-AST Code2Hyp",
            subtitle="Mean paired MRR deltas; positive values favor the first term of each contrast",
            x="Project and path-cost orientation",
            y="Mean paired delta MRR",
            fill="Contrast",
        )
        + theme_bw(base_size=8.5)
        + theme(
            figure_size=(8.0, 7.2),
            legend_position="none",
            axis_text_x=element_text(rotation=20, ha="right"),
            strip_text=element_text(weight="bold"),
        )
    )


def _delta_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in summary.get("curvature_deltas", []):
        if row.get("path_object_mode") == "lca_product" and row.get("method_aggregation") == "measure":
            rows.append(_plot_delta_row(row, "curvature", f"curvature c={row['curvature']:g}"))
    for row in summary.get("path_object_deltas", []):
        if row.get("method_aggregation") == "measure" and row.get("geometry") == "poincare" and float(row.get("curvature", 0.0)) == 1.0:
            rows.append(_plot_delta_row(row, "path_object", "LCA-product - single point"))
    for row in summary.get("aggregation_deltas", []):
        if row.get("path_object_mode") == "lca_product" and row.get("geometry") == "poincare" and float(row.get("curvature", 0.0)) == 1.0:
            rows.append(_plot_delta_row(row, "aggregation", "measure - centroid"))
    for row in summary.get("orientation_deltas", []):
        if row.get("path_object_mode") == "lca_product" and row.get("method_aggregation") == "measure":
            rows.append(_plot_delta_row(row, "orientation", "unoriented - directed"))
    for row in summary.get("full_model_deltas", []):
        rows.append(_plot_delta_row(row, "full_model", "full Code2Hyp-v1 - simplest baseline"))
    return rows


def _plot_delta_row(row: dict[str, Any], contrast: str, contrast_label: str) -> dict[str, Any]:
    return {
        "project": row["project"],
        "contrast": contrast,
        "contrast_label": contrast_label,
        "delta_mrr": float(row["mean_delta_mrr"]),
        "n": int(row["n"]),
        "path_cost_orientation": row.get("path_cost_orientation"),
    }


def _project_cell_label(row: Any) -> str:
    orientation = row.get("path_cost_orientation")
    if orientation and orientation != "directed":
        return f"{row['project']}\n{orientation}"
    return str(row["project"])


def _geometry_label(row: Any) -> str:
    if row["geometry"] == "euclidean":
        return "Euclidean"
    curvature = float(row["curvature"])
    if abs(curvature - 1e-4) < 1e-12:
        return "Poincare c=1e-4"
    if abs(curvature - 1.0) < 1e-12:
        return "Poincare c=1"
    return f"Poincare c={curvature:g}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot raw-AST Code2Hyp factor-matrix heatmaps and deltas.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    for output in plot_factor_matrix(input_path=args.input, output_prefix=args.output_prefix):
        print(f"Wrote {output}")


if __name__ == "__main__":
    main()
