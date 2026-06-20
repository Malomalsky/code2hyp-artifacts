from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib-cache"))
os.environ.setdefault("XDG_CACHE_HOME", str(PROJECT_ROOT / ".matplotlib-cache"))
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from geometry_profile_research.code2hyp_reporting import summarize_pilot_runs


DEFAULT_INPUTS = (
    (
        "512 / 3 epochs",
        "Original",
        PROJECT_ROOT / "outputs/code2hyp_schedule_sweep_original_512_3epochs_3seeds.json",
    ),
    (
        "512 / 3 epochs",
        "Structural only",
        PROJECT_ROOT / "outputs/code2hyp_schedule_sweep_structural_only_512_3epochs_3seeds.json",
    ),
    (
        "256 / 5 epochs",
        "Original",
        PROJECT_ROOT / "outputs/code2hyp_schedule_sweep_original_256_5epochs_3seeds.json",
    ),
    (
        "256 / 5 epochs",
        "Structural only",
        PROJECT_ROOT / "outputs/code2hyp_schedule_sweep_structural_only_256_5epochs_3seeds.json",
    ),
)
DEFAULT_OUTPUT_PREFIX = PROJECT_ROOT / "figures/code2hyp_schedule_sweep_f1_spearman"

VARIANT_ORDER = (
    "B4_hyperbolic_code2vec",
    "B8_hyperbolic_frechet_code2vec",
    "B17_hyperbolic_path_mp_code2vec",
    "B18_hyperbolic_path_mp_struct_rank",
    "B19_hyperbolic_path_mp_rank_annealed",
    "B20_hyperbolic_path_mp_rank_delayed",
    "B21_hyperbolic_path_mp_rank_cosine",
    "B22_hyperbolic_path_mp_rank_warmup_decay",
)

VARIANT_LABELS = {
    "B4_hyperbolic_code2vec": "B4",
    "B8_hyperbolic_frechet_code2vec": "B8",
    "B17_hyperbolic_path_mp_code2vec": "B17",
    "B18_hyperbolic_path_mp_struct_rank": "B18",
    "B19_hyperbolic_path_mp_rank_annealed": "B19",
    "B20_hyperbolic_path_mp_rank_delayed": "B20",
    "B21_hyperbolic_path_mp_rank_cosine": "B21",
    "B22_hyperbolic_path_mp_rank_warmup_decay": "B22",
}

VARIANT_FULL_LABELS = {
    "B4_hyperbolic_code2vec": "B4 full-context Poincare",
    "B8_hyperbolic_frechet_code2vec": "B8 Frechet aggregation",
    "B17_hyperbolic_path_mp_code2vec": "B17 no rank schedule",
    "B18_hyperbolic_path_mp_struct_rank": "B18 constant rank",
    "B19_hyperbolic_path_mp_rank_annealed": "B19 linear",
    "B20_hyperbolic_path_mp_rank_delayed": "B20 delayed linear",
    "B21_hyperbolic_path_mp_rank_cosine": "B21 cosine",
    "B22_hyperbolic_path_mp_rank_warmup_decay": "B22 warmup-decay",
}

VARIANT_COLORS = {
    "B4_hyperbolic_code2vec": "#D55E00",
    "B8_hyperbolic_frechet_code2vec": "#CC79A7",
    "B17_hyperbolic_path_mp_code2vec": "#0072B2",
    "B18_hyperbolic_path_mp_struct_rank": "#56B4E9",
    "B19_hyperbolic_path_mp_rank_annealed": "#009E73",
    "B20_hyperbolic_path_mp_rank_delayed": "#E69F00",
    "B21_hyperbolic_path_mp_rank_cosine": "#F0E442",
    "B22_hyperbolic_path_mp_rank_warmup_decay": "#000000",
}


def _load_summary(path: Path) -> dict[str, dict[str, float | int | str]]:
    result = json.loads(path.read_text(encoding="utf-8"))
    return {str(row["variant"]): row for row in summarize_pilot_runs(result)}


def _panel_offsets(row_label: str, column_label: str) -> dict[str, tuple[int, int]]:
    if row_label.startswith("512") and column_label == "Original":
        return {
            "B19_hyperbolic_path_mp_rank_annealed": (5, -12),
            "B20_hyperbolic_path_mp_rank_delayed": (5, 6),
            "B21_hyperbolic_path_mp_rank_cosine": (5, -2),
            "B22_hyperbolic_path_mp_rank_warmup_decay": (5, -12),
        }
    if row_label.startswith("512") and column_label == "Structural only":
        return {
            "B20_hyperbolic_path_mp_rank_delayed": (5, -12),
            "B21_hyperbolic_path_mp_rank_cosine": (5, 6),
            "B22_hyperbolic_path_mp_rank_warmup_decay": (5, -12),
        }
    return {}


def _plot_panel(
    ax: plt.Axes,
    row_label: str,
    column_label: str,
    summaries: dict[str, dict[str, float | int | str]],
) -> None:
    offsets = _panel_offsets(row_label, column_label)
    for variant in VARIANT_ORDER:
        if variant not in summaries:
            continue
        row = summaries[variant]
        x = float(row["validation_structural_spearman_mean"])
        y = float(row["validation_f1_mean"])
        ax.errorbar(
            x,
            y,
            xerr=float(row["validation_structural_spearman_std"]),
            yerr=float(row["validation_f1_std"]),
            fmt="o",
            color=VARIANT_COLORS[variant],
            ecolor=VARIANT_COLORS[variant],
            markersize=5.8,
            elinewidth=0.9,
            capsize=2.2,
            markeredgecolor="#222222",
            markeredgewidth=0.45,
            label=VARIANT_FULL_LABELS[variant],
        )
        ax.annotate(
            VARIANT_LABELS[variant],
            (x, y),
            xytext=offsets.get(variant, (4, 4)),
            textcoords="offset points",
            fontsize=7.4,
        )
    ax.axvline(0.0, color="#555555", linestyle="--", linewidth=0.75)
    ax.grid(alpha=0.20, linewidth=0.6)
    ax.set_title(f"{row_label}: {column_label}")
    ax.set_xlabel("AST-distance Spearman")
    ax.set_ylabel("Target-subtoken F1")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_schedule_sweep(inputs: tuple[tuple[str, str, Path], ...], output_prefix: Path) -> None:
    if len(inputs) != 4:
        raise ValueError("schedule sweep plot expects exactly four input files")
    rows = tuple(dict.fromkeys(row_label for row_label, _, _ in inputs))
    columns = tuple(dict.fromkeys(column_label for _, column_label, _ in inputs))
    if len(rows) != 2 or len(columns) != 2:
        raise ValueError("schedule sweep plot expects two row labels and two column labels")
    summaries = {
        (row_label, column_label): _load_summary(path)
        for row_label, column_label, path in inputs
    }

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 8.5,
            "axes.titlesize": 9.5,
            "axes.labelsize": 8.8,
            "legend.fontsize": 8.0,
            "figure.titlesize": 11.5,
            "xtick.labelsize": 7.8,
            "ytick.labelsize": 7.8,
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(10.8, 7.0), constrained_layout=False)
    fig.subplots_adjust(left=0.075, right=0.77, top=0.89, bottom=0.08, wspace=0.30, hspace=0.42)

    for row_index, row_label in enumerate(rows):
        for column_index, column_label in enumerate(columns):
            _plot_panel(axes[row_index][column_index], row_label, column_label, summaries[(row_label, column_label)])

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center left", bbox_to_anchor=(0.79, 0.50), frameon=False)
    fig.suptitle("Code2Hyp schedule sweep: task quality versus structural alignment")
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_prefix.with_suffix(".png"), dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(output_prefix.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot Code2Hyp schedule sweep F1/Spearman trade-offs.")
    parser.add_argument("--output-prefix", type=Path, default=DEFAULT_OUTPUT_PREFIX)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plot_schedule_sweep(DEFAULT_INPUTS, args.output_prefix)
    print(f"Wrote {args.output_prefix.with_suffix('.png')}")
    print(f"Wrote {args.output_prefix.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
