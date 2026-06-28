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
    geom_errorbar,
    geom_hline,
    ggplot,
    labs,
    position_dodge,
    scale_fill_manual,
    theme,
    theme_bw,
)


METRICS = {
    "mrr": "MRR",
    "recall_at_1": "Recall@1",
    "margin_mean": "Mean margin",
}
METRIC_ORDER = ("MRR", "Recall@1", "Mean margin")
CURVATURE_ORDER = ("H(c=1e-4)", "H(c=1)")
PROJECT_COLORS = ("#375A7F", "#8C4A2F", "#4C6B3C", "#6C5478", "#8A3030", "#3E6F73")


def build_curvature_delta_rows(payloads: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for payload in payloads:
        grouped: dict[tuple[str, str, int, int], dict[tuple[str, float], dict[str, Any]]] = {}
        for run in payload.get("runs", []):
            key = (
                str(run["project"]),
                str(run["node_input_mode"]),
                int(run["dim"]),
                int(run["seed"]),
            )
            grouped.setdefault(key, {})[(str(run["geometry"]), float(run["curvature"]))] = run
        for (project, node_input_mode, dim, seed), variants in grouped.items():
            euclidean = variants.get(("euclidean", 1.0))
            if euclidean is None:
                continue
            for (geometry, curvature), poincare in variants.items():
                if geometry != "poincare":
                    continue
                for metric_key, metric_label in METRICS.items():
                    if metric_key not in euclidean or metric_key not in poincare:
                        continue
                    rows.append(
                        {
                            "project": project,
                            "node_input_mode": node_input_mode,
                            "dim": dim,
                            "seed": seed,
                            "curvature": curvature,
                            "curvature_label": _curvature_label(curvature),
                            "metric": metric_label,
                            "delta": float(poincare[metric_key]) - float(euclidean[metric_key]),
                            "poincare": float(poincare[metric_key]),
                            "euclidean": float(euclidean[metric_key]),
                        }
                    )
    return rows


def build_geometry_diagnostic_rows(payloads: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for payload in payloads:
        for run in payload.get("runs", []):
            if str(run.get("geometry")) != "poincare":
                continue
            diagnostics = run.get("geometry_diagnostics", {})
            rows.append(
                {
                    "project": str(run["project"]),
                    "node_input_mode": str(run["node_input_mode"]),
                    "dim": int(run["dim"]),
                    "seed": int(run["seed"]),
                    "curvature": float(run["curvature"]),
                    "curvature_label": _curvature_label(float(run["curvature"])),
                    "sqrt_curvature_norm_mean": float(diagnostics.get("sqrt_curvature_norm_mean", 0.0)),
                    "sqrt_curvature_norm_max": float(diagnostics.get("sqrt_curvature_norm_max", 0.0)),
                    "projection_active_fraction": float(diagnostics.get("projection_active_fraction", 0.0)),
                    "near_boundary_fraction": float(diagnostics.get("near_boundary_fraction", 0.0)),
                }
            )
    return rows


def plot_curvature_controls(
    *,
    inputs: Sequence[Path],
    output_prefix: Path,
) -> tuple[Path, Path, Path, Path]:
    payloads = tuple(_load_json(path) for path in inputs)
    delta_rows = build_curvature_delta_rows(payloads)
    diagnostic_rows = build_geometry_diagnostic_rows(payloads)
    if not delta_rows:
        raise ValueError("no matched Poincare/Euclidean curvature-control pairs found")
    if not diagnostic_rows:
        raise ValueError("no Poincare geometry diagnostics found")

    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    delta_png = output_prefix.with_name(output_prefix.name + "_deltas").with_suffix(".png")
    delta_pdf = output_prefix.with_name(output_prefix.name + "_deltas").with_suffix(".pdf")
    diagnostic_png = output_prefix.with_name(output_prefix.name + "_diagnostics").with_suffix(".png")
    diagnostic_pdf = output_prefix.with_name(output_prefix.name + "_diagnostics").with_suffix(".pdf")

    _delta_plot(delta_rows).save(delta_png, dpi=300, verbose=False)
    _delta_plot(delta_rows).save(delta_pdf, verbose=False)
    _diagnostic_plot(diagnostic_rows).save(diagnostic_png, dpi=300, verbose=False)
    _diagnostic_plot(diagnostic_rows).save(diagnostic_pdf, verbose=False)
    return delta_png, delta_pdf, diagnostic_png, diagnostic_pdf


def _delta_plot(rows: Sequence[dict[str, Any]]) -> ggplot:
    frame = pd.DataFrame(rows)
    summary = (
        frame.groupby(["project", "curvature_label", "metric"], as_index=False, observed=False)
        .agg(delta_mean=("delta", "mean"), delta_sd=("delta", "std"), n=("delta", "size"))
        .fillna({"delta_sd": 0.0})
    )
    summary["metric"] = pd.Categorical(summary["metric"], categories=METRIC_ORDER, ordered=True)
    summary["curvature_label"] = pd.Categorical(summary["curvature_label"], categories=CURVATURE_ORDER, ordered=True)
    colors = _project_colors(summary["project"])
    dodge = position_dodge(width=0.72)
    return (
        ggplot(summary, aes(x="curvature_label", y="delta_mean", fill="project"))
        + geom_hline(yintercept=0.0, color="#222222", size=0.45)
        + geom_col(position=dodge, width=0.64, color="#222222", size=0.25)
        + geom_errorbar(
            aes(ymin="delta_mean - delta_sd", ymax="delta_mean + delta_sd"),
            position=dodge,
            width=0.18,
            size=0.35,
        )
        + facet_wrap("~ metric", scales="free_y", ncol=1)
        + scale_fill_manual(values=colors)
        + labs(
            title="Raw-AST Code2Hyp curvature controls",
            subtitle="Paired delta against matched Euclidean baseline",
            x="Poincare curvature control",
            y="Poincare - Euclidean",
            fill="Project",
        )
        + theme_bw(base_size=9)
        + theme(
            figure_size=(7.2, 6.8),
            legend_position="bottom",
            axis_text_x=element_text(rotation=0, ha="center"),
            strip_text=element_text(weight="bold"),
        )
    )


def _diagnostic_plot(rows: Sequence[dict[str, Any]]) -> ggplot:
    frame = pd.DataFrame(rows)
    summary = (
        frame.groupby(["project", "curvature_label"], as_index=False, observed=False)
        .agg(
            mean_norm=("sqrt_curvature_norm_mean", "mean"),
            max_norm=("sqrt_curvature_norm_max", "max"),
            projection_fraction=("projection_active_fraction", "mean"),
        )
        .fillna(0.0)
    )
    summary["curvature_label"] = pd.Categorical(summary["curvature_label"], categories=CURVATURE_ORDER, ordered=True)
    colors = _project_colors(summary["project"])
    dodge = position_dodge(width=0.72)
    return (
        ggplot(summary, aes(x="curvature_label", y="mean_norm", fill="project"))
        + geom_col(position=dodge, width=0.64, color="#222222", size=0.25)
        + geom_errorbar(
            aes(ymin="mean_norm", ymax="max_norm"),
            position=dodge,
            width=0.18,
            size=0.35,
        )
        + scale_fill_manual(values=colors)
        + labs(
            title="Geometry diagnostics",
            subtitle="Mean sqrt(c)||x|| with maximum observed value as upper whisker",
            x="Poincare curvature control",
            y="sqrt(c)||x||",
            fill="Project",
        )
        + theme_bw(base_size=9)
        + theme(
            figure_size=(7.2, 3.8),
            legend_position="bottom",
            axis_text_x=element_text(rotation=0, ha="center"),
        )
    )


def _project_colors(projects: Sequence[Any]) -> dict[str, str]:
    unique_projects = list(dict.fromkeys(str(project) for project in projects))
    return {project: PROJECT_COLORS[index % len(PROJECT_COLORS)] for index, project in enumerate(unique_projects)}


def _curvature_label(curvature: float) -> str:
    if abs(curvature - 1e-4) < 1e-12:
        return "H(c=1e-4)"
    if abs(curvature - 1.0) < 1e-12:
        return "H(c=1)"
    return f"H(c={curvature:g})"


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot raw-AST Code2Hyp curvature-control deltas and diagnostics.")
    parser.add_argument("--input", action="append", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    outputs = plot_curvature_controls(inputs=tuple(args.input), output_prefix=args.output_prefix)
    for output in outputs:
        print(f"Wrote {output}")


if __name__ == "__main__":
    main()
