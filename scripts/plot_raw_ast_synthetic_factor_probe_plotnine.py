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
    geom_tile,
    ggplot,
    labs,
    scale_fill_gradient,
    scale_fill_manual,
    theme,
    theme_bw,
)

from scripts.summarize_raw_ast_synthetic_factor_probe import summarize_synthetic_factor_probe


CONTRAST_COLORS = {
    "lca_product": "#8C4A2F",
    "curvature": "#3C6E71",
    "full": "#375A7F",
}


def plot_synthetic_factor_probe(*, input_path: Path, output_prefix: Path) -> tuple[Path, Path, Path, Path]:
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    rows = payload.get("rows", [])
    if not rows:
        raise ValueError("synthetic factor probe plot requires at least one row")
    summary = summarize_synthetic_factor_probe(input_path)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    stress_png = output_prefix.with_name(output_prefix.name + "_stress_heatmap").with_suffix(".png")
    stress_pdf = output_prefix.with_name(output_prefix.name + "_stress_heatmap").with_suffix(".pdf")
    deltas_png = output_prefix.with_name(output_prefix.name + "_stress_deltas").with_suffix(".png")
    deltas_pdf = output_prefix.with_name(output_prefix.name + "_stress_deltas").with_suffix(".pdf")

    _stress_heatmap(rows).save(stress_png, dpi=300, verbose=False)
    _stress_heatmap(rows).save(stress_pdf, verbose=False)
    _delta_plot(_delta_rows(summary)).save(deltas_png, dpi=300, verbose=False)
    _delta_plot(_delta_rows(summary)).save(deltas_pdf, verbose=False)
    return stress_png, stress_pdf, deltas_png, deltas_pdf


def _stress_heatmap(rows: Sequence[dict[str, Any]]) -> ggplot:
    frame = pd.DataFrame(rows)
    frame["representation"] = frame.apply(_representation_label, axis=1)
    frame["case_dim"] = frame["case"] + "\ndim=" + frame["dim"].astype(str)
    return (
        ggplot(frame, aes(x="representation", y="case_dim", fill="path_stress"))
        + geom_tile(color="#222222", size=0.2)
        + scale_fill_gradient(low="#F2EFE8", high="#8A3030")
        + labs(
            title="Synthetic AST metric-distortion probe",
            subtitle="Lower path stress means better preservation of LCA-anchored raw path distances",
            x="Representation",
            y="Tree family and embedding dimension",
            fill="Path stress",
        )
        + theme_bw(base_size=8.5)
        + theme(
            figure_size=(8.2, max(4.6, 0.24 * len(frame["case_dim"].unique()) + 1.4)),
            axis_text_x=element_text(rotation=25, ha="right"),
        )
    )


def _delta_plot(rows: Sequence[dict[str, Any]]) -> ggplot:
    if not rows:
        raise ValueError("synthetic factor delta plot requires at least one row")
    frame = pd.DataFrame(rows)
    frame["contrast"] = pd.Categorical(frame["contrast"], categories=("lca_product", "curvature", "full"), ordered=True)
    return (
        ggplot(frame, aes(x="case", y="delta_path_stress", fill="contrast"))
        + geom_hline(yintercept=0.0, color="#222222", size=0.35)
        + geom_col(color="#222222", size=0.18, width=0.72)
        + facet_wrap("~ contrast_label", scales="free_y", ncol=1)
        + scale_fill_manual(values=CONTRAST_COLORS)
        + labs(
            title="Synthetic factor effects",
            subtitle="Positive delta means lower path stress after adding the factor",
            x="Synthetic tree family",
            y="Path stress reduction",
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
    for row in summary.get("lca_product_deltas", []):
        rows.append(_plot_delta_row(row, "lca_product", f"LCA-product - single point, {_factor_label(row)}"))
    for row in summary.get("curvature_deltas", []):
        if row.get("factor") == "lca_product":
            rows.append(_plot_delta_row(row, "curvature", f"Poincare - Euclidean, LCA-product, c={float(row['curvature']):g}"))
    for row in summary.get("full_model_deltas", []):
        rows.append(_plot_delta_row(row, "full", f"Poincare LCA-product - Euclidean single point, c={float(row['curvature']):g}"))
    return rows


def _plot_delta_row(row: dict[str, Any], contrast: str, contrast_label: str) -> dict[str, Any]:
    return {
        "case": row["case"],
        "dim": int(row["dim"]),
        "contrast": contrast,
        "contrast_label": f"{contrast_label}, dim={int(row['dim'])}",
        "delta_path_stress": float(row["delta_path_stress"]),
    }


def _representation_label(row: Any) -> str:
    geometry = "Euclidean" if row["geometry"] == "euclidean" else f"Poincare c={float(row['curvature']):g}"
    path = "single point" if row["path_object_mode"] == "single_point" else "LCA product"
    return f"{geometry}\n{path}"


def _factor_label(row: dict[str, Any]) -> str:
    if row["factor"] == "euclidean":
        return "Euclidean"
    return f"Poincare c={float(row['curvature']):g}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot synthetic raw-AST Code2Hyp factor probes.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    for output in plot_synthetic_factor_probe(input_path=args.input, output_prefix=args.output_prefix):
        print(f"Wrote {output}")


if __name__ == "__main__":
    main()
