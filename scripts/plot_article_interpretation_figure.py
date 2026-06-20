from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
from statistics import fmean, pstdev
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib-cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


METRIC_LABELS = {
    "node_count": "AST size",
    "ball_size_mean_r3": "Ball growth",
    "forman_mean": "Forman mean",
    "forman_negative_mass": "Forman neg.",
    "forman_positive_mass": "Forman pos.",
    "ollivier_mean": "OR mean",
    "ollivier_negative_mass": "OR neg.",
    "ollivier_near_zero_mass": "OR near-zero",
}

HEATMAP_METRICS = [
    "node_count",
    "ball_size_mean_r3",
    "forman_negative_mass",
    "forman_positive_mass",
    "ollivier_mean",
    "ollivier_negative_mass",
    "ollivier_near_zero_mass",
]

CURVATURE_METRICS = [
    "forman_mean",
    "forman_negative_mass",
    "forman_positive_mass",
    "ollivier_mean",
    "ollivier_negative_mass",
    "ollivier_near_zero_mass",
]

FAMILY_COLORS = {
    "forman": "#D55E00",
    "ollivier": "#0072B2",
}

ANNOTATION_OFFSETS = {
    "forman_mean": (-58, 12),
    "forman_negative_mass": (8, 6),
    "forman_positive_mass": (8, 3),
    "ollivier_mean": (8, 6),
    "ollivier_negative_mass": (8, 10),
    "ollivier_near_zero_mass": (8, -10),
}

DISPLAY_JITTER = {
    "ollivier_near_zero_mass": (0.008, -0.006),
}


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _float(row: dict[str, str], key: str) -> float:
    return float(row[key])


def _zscore(values: list[float]) -> list[float]:
    mean = fmean(values)
    std = pstdev(values) if len(values) > 1 else 0.0
    return [(value - mean) / std if std else 0.0 for value in values]


def _family(metric: str) -> str:
    return "ollivier" if metric.startswith("ollivier") else "forman"


def _build_task_heatmap(summary_rows: list[dict[str, str]]) -> tuple[list[str], list[list[float]]]:
    rows = sorted(summary_rows, key=lambda row: int(row["task_id"]))
    task_labels = [f"task-{int(row['task_id']):02d}" for row in rows]
    columns: list[list[float]] = []
    for metric in HEATMAP_METRICS:
        means = [_float(row, f"{metric}_mean") for row in rows]
        columns.append(_zscore(means))
    matrix = [[columns[column][row] for column in range(len(columns))] for row in range(len(rows))]
    return task_labels, matrix


def plot_article_interpretation_figure(
    *,
    summary_path: Path,
    effect_path: Path,
    residual_path: Path,
    permutation_path: Path,
    output_stem: Path,
) -> None:
    task_summary = _read_csv(summary_path)
    effects = {row["metric"]: row for row in _read_csv(effect_path)}
    residuals = {row["metric"]: row for row in _read_csv(residual_path)}
    permutation_rows = _read_csv(permutation_path)
    max_holm = max(float(row["p_value_holm"]) for row in permutation_rows)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7,
            "axes.labelsize": 8,
            "axes.titlesize": 9,
            "legend.fontsize": 7,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig = plt.figure(figsize=(7.4, 6.8))
    grid = fig.add_gridspec(
        2,
        2,
        height_ratios=[1.15, 1.0],
        width_ratios=[1.15, 1.0],
        hspace=0.62,
        wspace=0.35,
    )
    ax_heatmap = fig.add_subplot(grid[0, :])
    ax_bars = fig.add_subplot(grid[1, 0])
    ax_scatter = fig.add_subplot(grid[1, 1])

    _draw_heatmap(ax_heatmap, task_summary)
    _draw_raw_residual_bars(ax_bars, effects, residuals)
    _draw_control_scatter(ax_scatter, residuals)

    for label, axis in zip(["A", "B", "C"], [ax_heatmap, ax_bars, ax_scatter]):
        axis.text(
            -0.10,
            1.08,
            label,
            transform=axis.transAxes,
            fontsize=10,
            fontweight="bold",
            va="top",
            ha="right",
        )

    fig.suptitle(
        "Interpreting task-level geometry signal in AST curvature profiles",
        fontsize=10,
        y=0.99,
    )
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def _draw_heatmap(axis: Any, task_summary: list[dict[str, str]]) -> None:
    task_labels, matrix = _build_task_heatmap(task_summary)
    image = axis.imshow(matrix, cmap="RdBu_r", vmin=-2.5, vmax=2.5, aspect="auto")
    axis.set_title("Task-specific geometry profiles, standardized across tasks")
    axis.set_ylabel("DTA task")
    axis.set_xticks(range(len(HEATMAP_METRICS)))
    axis.set_xticklabels(
        [METRIC_LABELS[metric] for metric in HEATMAP_METRICS],
        rotation=25,
        ha="right",
    )
    axis.set_yticks(range(len(task_labels)))
    axis.set_yticklabels(task_labels)
    colorbar = axis.figure.colorbar(image, ax=axis, fraction=0.025, pad=0.015)
    colorbar.set_label("z-score")


def _draw_raw_residual_bars(
    axis: Any,
    effects: dict[str, dict[str, str]],
    residuals: dict[str, dict[str, str]],
) -> None:
    x_positions = list(range(len(CURVATURE_METRICS)))
    raw_values = [_float(effects[metric], "eta_squared_task") for metric in CURVATURE_METRICS]
    residual_values = [
        _float(residuals[metric], "eta_squared_task_residual")
        for metric in CURVATURE_METRICS
    ]
    width = 0.38
    axis.bar(
        [x - width / 2 for x in x_positions],
        raw_values,
        width=width,
        label="Raw task effect",
        color="#999999",
        edgecolor="#333333",
        linewidth=0.4,
    )
    axis.bar(
        [x + width / 2 for x in x_positions],
        residual_values,
        width=width,
        label="After size/growth control",
        color="#009E73",
        edgecolor="#333333",
        linewidth=0.4,
    )
    axis.set_title("Curvature signal before and after controls")
    axis.set_ylabel("Task-level eta-squared")
    axis.set_xticks(x_positions)
    axis.set_xticklabels([METRIC_LABELS[metric] for metric in CURVATURE_METRICS], rotation=40, ha="right")
    axis.set_ylim(0.0, 0.65)
    axis.legend(frameon=False, loc="upper right", bbox_to_anchor=(1.0, 1.0))
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.grid(axis="y", color="#E0E0E0", linewidth=0.5)


def _draw_control_scatter(
    axis: Any,
    residuals: dict[str, dict[str, str]],
) -> None:
    for metric in CURVATURE_METRICS:
        row = residuals[metric]
        family = _family(metric)
        x = _float(row, "covariate_r_squared")
        y = _float(row, "eta_squared_task_residual")
        jitter_x, jitter_y = DISPLAY_JITTER.get(metric, (0.0, 0.0))
        plot_x = x + jitter_x
        plot_y = y + jitter_y
        axis.scatter(
            plot_x,
            plot_y,
            s=42,
            color=FAMILY_COLORS[family],
            edgecolor="#222222",
            linewidth=0.4,
            zorder=3,
        )
        axis.annotate(
            METRIC_LABELS[metric],
            (plot_x, plot_y),
            xytext=ANNOTATION_OFFSETS.get(metric, (4, 3)),
            textcoords="offset points",
            fontsize=6.5,
        )
    axis.axvline(0.25, color="#777777", linewidth=0.8, linestyle="--")
    axis.axhline(0.25, color="#777777", linewidth=0.8, linestyle="--")
    axis.text(0.30, 0.08, "size-driven zone", fontsize=6.5, color="#555555")
    axis.text(0.01, 0.585, "task-specific curvature", fontsize=6.5, color="#555555")
    axis.set_title("What remains after size/growth controls")
    axis.set_xlabel("Variance explained by controls, R-squared")
    axis.set_ylabel("Residual task eta-squared")
    axis.set_xlim(0.0, 0.72)
    axis.set_ylim(0.0, 0.62)
    axis.grid(color="#E0E0E0", linewidth=0.5)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.legend(
        handles=[
            Patch(facecolor=FAMILY_COLORS["forman"], edgecolor="#222222", label="Forman-Ricci"),
            Patch(facecolor=FAMILY_COLORS["ollivier"], edgecolor="#222222", label="Ollivier-Ricci"),
        ],
        frameon=False,
        loc="upper right",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create an article-ready interpretation figure for AST geometry results."
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("reports/task_geometry_profile_summary_limit50.csv"),
    )
    parser.add_argument(
        "--effects",
        type=Path,
        default=Path("reports/task_geometry_effect_sizes_limit50.csv"),
    )
    parser.add_argument(
        "--residuals",
        type=Path,
        default=Path("reports/task_geometry_residual_effect_sizes_limit50.csv"),
    )
    parser.add_argument(
        "--permutation",
        type=Path,
        default=Path("reports/task_geometry_permutation_tests_limit50.csv"),
    )
    parser.add_argument(
        "--output-stem",
        type=Path,
        default=Path("figures/article_geometry_interpretation_limit50"),
    )
    args = parser.parse_args()
    plot_article_interpretation_figure(
        summary_path=args.summary,
        effect_path=args.effects,
        residual_path=args.residuals,
        permutation_path=args.permutation,
        output_stem=args.output_stem,
    )
    print(f"wrote {args.output_stem.with_suffix('.pdf')}")
    print(f"wrote {args.output_stem.with_suffix('.png')}")


if __name__ == "__main__":
    main()
