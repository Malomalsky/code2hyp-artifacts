from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib-cache"))
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from geometry_profile_research.task_geometry_analysis import (
    DEFAULT_TASK_GEOMETRY_METRICS,
    compute_metric_effect_sizes,
    compute_permutation_tests,
    compute_residual_effect_sizes,
    summarize_task_geometry,
    zscore_task_means,
)


METRIC_LABELS = {
    "node_count": "AST size",
    "ball_size_mean_r3": "Ball growth r=3",
    "forman_mean": "Forman mean",
    "forman_negative_mass": "Forman negative",
    "forman_positive_mass": "Forman positive",
    "ollivier_mean": "Ollivier mean",
    "ollivier_negative_mass": "Ollivier negative",
    "ollivier_near_zero_mass": "Ollivier near-zero",
}


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _plot_heatmap(
    zscores: list[dict[str, Any]],
    metrics: list[str],
    output_stem: Path,
) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 9,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    task_labels = [f"task-{int(row['task_id']):02d}" for row in zscores]
    matrix = [[float(row[metric]) for metric in metrics] for row in zscores]

    fig, ax = plt.subplots(figsize=(7.0, 3.5))
    image = ax.imshow(matrix, cmap="RdBu_r", vmin=-2.5, vmax=2.5, aspect="auto")
    ax.set_title("Task-level AST geometry profile")
    ax.set_xlabel("Geometry descriptor, z-score across tasks")
    ax.set_ylabel("DTA task")
    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels(
        [METRIC_LABELS.get(metric, metric) for metric in metrics],
        rotation=35,
        ha="right",
    )
    ax.set_yticks(range(len(task_labels)))
    ax.set_yticklabels(task_labels)
    colorbar = fig.colorbar(image, ax=ax, fraction=0.035, pad=0.02)
    colorbar.set_label("z-score")

    output_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_stem.with_suffix(".pdf"))
    fig.savefig(output_stem.with_suffix(".png"), dpi=300)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize and plot task-level AST geometry profiles."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("reports/code_geometry_atlas_ollivier.csv"),
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=Path("reports/task_geometry_profile_summary.csv"),
    )
    parser.add_argument(
        "--effect-output",
        type=Path,
        default=Path("reports/task_geometry_effect_sizes.csv"),
    )
    parser.add_argument(
        "--residual-effect-output",
        type=Path,
        default=Path("reports/task_geometry_residual_effect_sizes.csv"),
    )
    parser.add_argument(
        "--permutation-output",
        type=Path,
        default=Path("reports/task_geometry_permutation_tests.csv"),
    )
    parser.add_argument("--permutations", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument(
        "--figure-output-stem",
        type=Path,
        default=Path("figures/task_geometry_profile_heatmap"),
    )
    args = parser.parse_args()

    rows = _read_csv(args.input)
    metrics = list(DEFAULT_TASK_GEOMETRY_METRICS)
    task_summary = summarize_task_geometry(rows, metrics)
    effect_sizes = compute_metric_effect_sizes(rows, metrics)
    residual_effect_sizes = compute_residual_effect_sizes(rows)
    permutation_tests = compute_permutation_tests(
        rows,
        metrics,
        permutations=args.permutations,
        seed=args.seed,
    )
    zscores = zscore_task_means(task_summary, metrics)

    _write_csv(args.summary_output, task_summary)
    _write_csv(args.effect_output, effect_sizes)
    _write_csv(args.residual_effect_output, residual_effect_sizes)
    _write_csv(args.permutation_output, permutation_tests)
    _plot_heatmap(zscores, metrics, args.figure_output_stem)

    print(f"wrote {args.summary_output} ({len(task_summary)} rows)")
    print(f"wrote {args.effect_output} ({len(effect_sizes)} rows)")
    print(f"wrote {args.residual_effect_output} ({len(residual_effect_sizes)} rows)")
    print(f"wrote {args.permutation_output} ({len(permutation_tests)} rows)")
    print(f"wrote {args.figure_output_stem.with_suffix('.pdf')}")
    print(f"wrote {args.figure_output_stem.with_suffix('.png')}")


if __name__ == "__main__":
    main()
