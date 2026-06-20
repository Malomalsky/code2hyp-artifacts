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
        PROJECT_ROOT / "outputs/code2hyp_path_attention_original_512_3epochs_3seeds.json",
    ),
    (
        "512 / 3 epochs",
        "Structural only",
        PROJECT_ROOT / "outputs/code2hyp_path_attention_structural_only_512_3epochs_3seeds.json",
    ),
)
DEFAULT_OUTPUT_PREFIX = PROJECT_ROOT / "figures/code2hyp_path_attention_f1_spearman"

VARIANT_ORDER = (
    "B17_hyperbolic_path_mp_code2vec",
    "B23_hyperbolic_path_attention_mp_code2vec",
    "B24_hyperbolic_path_attention_mp_rank_annealed",
    "B25_hyperbolic_path_depth_attention_mp_code2vec",
    "B26_hyperbolic_path_depth_attention_mp_rank_annealed",
    "B27_hyperbolic_path_attention_mp_monotone",
    "B28_hyperbolic_path_attention_mp_tree_distance",
    "B29_hyperbolic_path_dual_attention_mp_separated",
    "B30_hyperbolic_path_dual_attention_mp_rank_separated",
    "B31_hyperbolic_path_dual_attention_mp_soft_rank",
    "B32_lorentz_path_dual_attention_mp_soft_rank",
    "B34_hyperbolic_path_dual_attention_mp_adaptive_rank",
)

VARIANT_LABELS = {
    "B17_hyperbolic_path_mp_code2vec": "B17",
    "B23_hyperbolic_path_attention_mp_code2vec": "B23",
    "B24_hyperbolic_path_attention_mp_rank_annealed": "B24",
    "B25_hyperbolic_path_depth_attention_mp_code2vec": "B25",
    "B26_hyperbolic_path_depth_attention_mp_rank_annealed": "B26",
    "B27_hyperbolic_path_attention_mp_monotone": "B27",
    "B28_hyperbolic_path_attention_mp_tree_distance": "B28",
    "B29_hyperbolic_path_dual_attention_mp_separated": "B29",
    "B30_hyperbolic_path_dual_attention_mp_rank_separated": "B30",
    "B31_hyperbolic_path_dual_attention_mp_soft_rank": "B31",
    "B32_lorentz_path_dual_attention_mp_soft_rank": "B32",
    "B34_hyperbolic_path_dual_attention_mp_adaptive_rank": "B34",
}

VARIANT_FULL_LABELS = {
    "B17_hyperbolic_path_mp_code2vec": "B17 path message passing",
    "B23_hyperbolic_path_attention_mp_code2vec": "B23 path-node attention",
    "B24_hyperbolic_path_attention_mp_rank_annealed": "B24 path-node attention + linear rank",
    "B25_hyperbolic_path_depth_attention_mp_code2vec": "B25 depth-aware path-node attention",
    "B26_hyperbolic_path_depth_attention_mp_rank_annealed": "B26 depth-aware attention + linear rank",
    "B27_hyperbolic_path_attention_mp_monotone": "B27 path-node attention + monotone profile",
    "B28_hyperbolic_path_attention_mp_tree_distance": "B28 path-node attention + soft tree-distance calibration",
    "B29_hyperbolic_path_dual_attention_mp_separated": "B29 dual root/detail path-node attention",
    "B30_hyperbolic_path_dual_attention_mp_rank_separated": "B30 dual attention + global rank",
    "B31_hyperbolic_path_dual_attention_mp_soft_rank": "B31 dual attention + soft global rank",
    "B32_lorentz_path_dual_attention_mp_soft_rank": "B32 Lorentz dual attention + soft global rank",
    "B34_hyperbolic_path_dual_attention_mp_adaptive_rank": "B34 dual attention + adaptive global rank",
}

VARIANT_COLORS = {
    "B17_hyperbolic_path_mp_code2vec": "#0072B2",
    "B23_hyperbolic_path_attention_mp_code2vec": "#009E73",
    "B24_hyperbolic_path_attention_mp_rank_annealed": "#D55E00",
    "B25_hyperbolic_path_depth_attention_mp_code2vec": "#CC79A7",
    "B26_hyperbolic_path_depth_attention_mp_rank_annealed": "#000000",
    "B27_hyperbolic_path_attention_mp_monotone": "#56B4E9",
    "B28_hyperbolic_path_attention_mp_tree_distance": "#E69F00",
    "B29_hyperbolic_path_dual_attention_mp_separated": "#882255",
    "B30_hyperbolic_path_dual_attention_mp_rank_separated": "#44AA99",
    "B31_hyperbolic_path_dual_attention_mp_soft_rank": "#AA4499",
    "B32_lorentz_path_dual_attention_mp_soft_rank": "#332288",
    "B34_hyperbolic_path_dual_attention_mp_adaptive_rank": "#117733",
}

VARIANT_MARKERS = {
    "B17_hyperbolic_path_mp_code2vec": "o",
    "B23_hyperbolic_path_attention_mp_code2vec": "s",
    "B24_hyperbolic_path_attention_mp_rank_annealed": "^",
    "B25_hyperbolic_path_depth_attention_mp_code2vec": "D",
    "B26_hyperbolic_path_depth_attention_mp_rank_annealed": "X",
    "B27_hyperbolic_path_attention_mp_monotone": "P",
    "B28_hyperbolic_path_attention_mp_tree_distance": "*",
    "B29_hyperbolic_path_dual_attention_mp_separated": "v",
    "B30_hyperbolic_path_dual_attention_mp_rank_separated": "h",
    "B31_hyperbolic_path_dual_attention_mp_soft_rank": "<",
    "B32_lorentz_path_dual_attention_mp_soft_rank": ">",
    "B34_hyperbolic_path_dual_attention_mp_adaptive_rank": "8",
}


def _load_summary(path: Path) -> dict[str, dict[str, float | int | str]]:
    result = json.loads(path.read_text(encoding="utf-8"))
    return {str(row["variant"]): row for row in summarize_pilot_runs(result)}


def _annotation_offsets(column_label: str) -> dict[str, tuple[int, int]]:
    if column_label == "Original":
        return {
            "B17_hyperbolic_path_mp_code2vec": (5, 5),
            "B23_hyperbolic_path_attention_mp_code2vec": (8, -27),
            "B24_hyperbolic_path_attention_mp_rank_annealed": (5, 5),
            "B25_hyperbolic_path_depth_attention_mp_code2vec": (8, 7),
            "B26_hyperbolic_path_depth_attention_mp_rank_annealed": (8, -7),
            "B27_hyperbolic_path_attention_mp_monotone": (-34, -12),
            "B28_hyperbolic_path_attention_mp_tree_distance": (-34, 9),
            "B29_hyperbolic_path_dual_attention_mp_separated": (8, 16),
            "B30_hyperbolic_path_dual_attention_mp_rank_separated": (8, -18),
            "B31_hyperbolic_path_dual_attention_mp_soft_rank": (8, 8),
            "B32_lorentz_path_dual_attention_mp_soft_rank": (8, -8),
            "B34_hyperbolic_path_dual_attention_mp_adaptive_rank": (-42, -8),
        }
    return {
        "B17_hyperbolic_path_mp_code2vec": (8, -18),
        "B23_hyperbolic_path_attention_mp_code2vec": (8, -28),
        "B24_hyperbolic_path_attention_mp_rank_annealed": (8, 11),
        "B25_hyperbolic_path_depth_attention_mp_code2vec": (8, 5),
        "B26_hyperbolic_path_depth_attention_mp_rank_annealed": (8, -10),
        "B27_hyperbolic_path_attention_mp_monotone": (-35, -12),
        "B28_hyperbolic_path_attention_mp_tree_distance": (-36, 9),
        "B29_hyperbolic_path_dual_attention_mp_separated": (8, 16),
        "B30_hyperbolic_path_dual_attention_mp_rank_separated": (8, -18),
        "B31_hyperbolic_path_dual_attention_mp_soft_rank": (8, 8),
        "B32_lorentz_path_dual_attention_mp_soft_rank": (8, -8),
        "B34_hyperbolic_path_dual_attention_mp_adaptive_rank": (-36, 20),
    }


def _plot_panel(
    ax: plt.Axes,
    row_label: str,
    column_label: str,
    summaries: dict[str, dict[str, float | int | str]],
) -> None:
    offsets = _annotation_offsets(column_label)
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
            fmt=VARIANT_MARKERS[variant],
            color=VARIANT_COLORS[variant],
            ecolor=VARIANT_COLORS[variant],
            markersize=6.4,
            elinewidth=1.0,
            capsize=2.5,
            markeredgecolor="#222222",
            markeredgewidth=0.5,
            label=VARIANT_FULL_LABELS[variant],
        )
        ax.annotate(
            VARIANT_LABELS[variant],
            (x, y),
            xytext=offsets.get(variant, (5, 5)),
            textcoords="offset points",
            fontsize=8.2,
            bbox={"boxstyle": "round,pad=0.14", "facecolor": "white", "edgecolor": "none", "alpha": 0.78},
        )
    ax.axvline(0.0, color="#555555", linestyle="--", linewidth=0.75)
    ax.grid(alpha=0.20, linewidth=0.6)
    ax.set_title(f"{row_label}: {column_label}")
    ax.set_xlabel("AST-distance Spearman")
    ax.set_ylabel("Target-subtoken F1")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_path_attention(inputs: tuple[tuple[str, str, Path], ...], output_prefix: Path) -> None:
    if len(inputs) != 2:
        raise ValueError("path-attention plot expects exactly two input files")
    summaries = {
        (row_label, column_label): _load_summary(path)
        for row_label, column_label, path in inputs
    }

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 9.0,
            "axes.titlesize": 10.0,
            "axes.labelsize": 9.0,
            "legend.fontsize": 8.4,
            "figure.titlesize": 11.5,
            "xtick.labelsize": 8.0,
            "ytick.labelsize": 8.0,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.8), constrained_layout=False)
    fig.subplots_adjust(left=0.08, right=0.74, top=0.84, bottom=0.18, wspace=0.34)

    for ax, (row_label, column_label, _) in zip(axes, inputs, strict=True):
        _plot_panel(ax, row_label, column_label, summaries[(row_label, column_label)])

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center left", bbox_to_anchor=(0.76, 0.50), frameon=False)
    fig.suptitle("Path-node attention in hyperbolic AST-path message passing")
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_prefix.with_suffix(".png"), dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(output_prefix.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot Code2Hyp path-attention F1/Spearman trade-offs.")
    parser.add_argument("--output-prefix", type=Path, default=DEFAULT_OUTPUT_PREFIX)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plot_path_attention(DEFAULT_INPUTS, args.output_prefix)
    print(f"Wrote {args.output_prefix.with_suffix('.png')}")
    print(f"Wrote {args.output_prefix.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
