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
MODE_LABELS = {
    "label_only": "A: label only",
    "label_depth": "B: label + depth",
    "label_depth_prefix": "C: label + depth + prefix",
}
MODE_ORDER = tuple(MODE_LABELS.values())
PROJECT_COLORS = ("#4C78A8", "#F58518", "#54A24B", "#B279A2", "#E45756", "#72B7B2")


def build_paired_delta_rows(payloads: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for payload in payloads:
        grouped: dict[tuple[str, str, int, int], dict[str, dict[str, Any]]] = {}
        for run in payload.get("runs", []):
            key = (
                str(run["project"]),
                str(run["node_input_mode"]),
                int(run["dim"]),
                int(run["seed"]),
            )
            grouped.setdefault(key, {})[str(run["geometry"])] = run
        for (project, node_input_mode, dim, seed), pair in grouped.items():
            if "euclidean" not in pair or "poincare" not in pair:
                continue
            euclidean = pair["euclidean"]
            poincare = pair["poincare"]
            for metric_key, metric_label in METRICS.items():
                if metric_key not in euclidean or metric_key not in poincare:
                    continue
                rows.append(
                    {
                        "project": project,
                        "node_input_mode": node_input_mode,
                        "mode_label": MODE_LABELS.get(node_input_mode, node_input_mode),
                        "dim": dim,
                        "seed": seed,
                        "metric": metric_label,
                        "delta": float(poincare[metric_key]) - float(euclidean[metric_key]),
                        "poincare": float(poincare[metric_key]),
                        "euclidean": float(euclidean[metric_key]),
                    }
                )
    return rows


def plot_node_input_matrix_deltas(
    *,
    inputs: Sequence[Path],
    output_prefix: Path,
) -> tuple[Path, Path]:
    payloads = tuple(_load_json(path) for path in inputs)
    delta_rows = build_paired_delta_rows(payloads)
    if not delta_rows:
        raise ValueError("no matched Poincare/Euclidean run pairs found")

    frame = pd.DataFrame(delta_rows)
    summary = (
        frame.groupby(["project", "mode_label", "metric"], as_index=False, observed=False)
        .agg(delta_mean=("delta", "mean"), delta_sd=("delta", "std"), n=("delta", "size"))
        .fillna({"delta_sd": 0.0})
    )
    summary["metric"] = pd.Categorical(summary["metric"], categories=METRIC_ORDER, ordered=True)
    summary["mode_label"] = pd.Categorical(summary["mode_label"], categories=MODE_ORDER, ordered=True)
    projects = list(dict.fromkeys(str(project) for project in summary["project"]))
    colors = {project: PROJECT_COLORS[index % len(PROJECT_COLORS)] for index, project in enumerate(projects)}
    dodge = position_dodge(width=0.72)

    plot = (
        ggplot(summary, aes(x="mode_label", y="delta_mean", fill="project"))
        + geom_hline(yintercept=0.0, color="#2B2B2B", size=0.45)
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
            title="Conditional raw-AST Code2Hyp geometry effect",
            subtitle="Paired delta: Poincare minus matched Euclidean baseline",
            x="Node information available to the structural encoder",
            y="Poincare - Euclidean",
            fill="Project",
        )
        + theme_bw(base_size=9)
        + theme(
            figure_size=(7.6, 7.2),
            legend_position="bottom",
            axis_text_x=element_text(rotation=18, ha="right"),
            strip_text=element_text(weight="bold"),
        )
    )

    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    png_path = output_prefix.with_suffix(".png")
    pdf_path = output_prefix.with_suffix(".pdf")
    plot.save(png_path, dpi=300, verbose=False)
    plot.save(pdf_path, verbose=False)
    return png_path, pdf_path


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot paired Poincare-Euclidean deltas for the raw-AST node-input matrix.")
    parser.add_argument("--input", action="append", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    png_path, pdf_path = plot_node_input_matrix_deltas(inputs=tuple(args.input), output_prefix=args.output_prefix)
    print(f"Wrote {png_path}")
    print(f"Wrote {pdf_path}")


if __name__ == "__main__":
    main()
