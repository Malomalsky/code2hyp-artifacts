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


DEFAULT_ORIGINAL_INPUT = PROJECT_ROOT / "outputs/code2hyp_java_small_focused_b20_original_1k_3seeds.json"
DEFAULT_STRUCTURAL_INPUT = PROJECT_ROOT / "outputs/code2hyp_java_small_focused_b20_structural_only_1k_3seeds.json"
DEFAULT_OUTPUT_PREFIX = PROJECT_ROOT / "figures/code2hyp_b20_f1_spearman_tradeoff"

VARIANT_ORDER = (
    "B4_hyperbolic_code2vec",
    "B8_hyperbolic_frechet_code2vec",
    "B17_hyperbolic_path_mp_code2vec",
    "B18_hyperbolic_path_mp_struct_rank",
    "B19_hyperbolic_path_mp_rank_annealed",
    "B20_hyperbolic_path_mp_rank_delayed",
)

VARIANT_LABELS = {
    "B4_hyperbolic_code2vec": "B4",
    "B8_hyperbolic_frechet_code2vec": "B8",
    "B17_hyperbolic_path_mp_code2vec": "B17",
    "B18_hyperbolic_path_mp_struct_rank": "B18",
    "B19_hyperbolic_path_mp_rank_annealed": "B19",
    "B20_hyperbolic_path_mp_rank_delayed": "B20",
}

VARIANT_FULL_LABELS = {
    "B4_hyperbolic_code2vec": "B4 full-context Poincare",
    "B8_hyperbolic_frechet_code2vec": "B8 Frechet aggregation",
    "B17_hyperbolic_path_mp_code2vec": "B17 path message passing",
    "B18_hyperbolic_path_mp_struct_rank": "B18 constant rank loss",
    "B19_hyperbolic_path_mp_rank_annealed": "B19 linear rank schedule",
    "B20_hyperbolic_path_mp_rank_delayed": "B20 delayed rank schedule",
}

# Okabe-Ito-inspired palette; readable in print and under common CVD profiles.
VARIANT_COLORS = {
    "B4_hyperbolic_code2vec": "#D55E00",
    "B8_hyperbolic_frechet_code2vec": "#CC79A7",
    "B17_hyperbolic_path_mp_code2vec": "#0072B2",
    "B18_hyperbolic_path_mp_struct_rank": "#56B4E9",
    "B19_hyperbolic_path_mp_rank_annealed": "#009E73",
    "B20_hyperbolic_path_mp_rank_delayed": "#E69F00",
}


def _load_summary(path: Path) -> dict[str, dict[str, float | int | str]]:
    result = json.loads(path.read_text(encoding="utf-8"))
    return {str(row["variant"]): row for row in summarize_pilot_runs(result)}


def _plot_panel(
    ax: plt.Axes,
    summaries: dict[str, dict[str, float | int | str]],
    title: str,
    label_offsets: dict[str, tuple[int, int]] | None = None,
) -> None:
    plotted_variants = tuple(variant for variant in VARIANT_ORDER if variant in summaries)
    if not plotted_variants:
        raise ValueError(f"no known variants available for panel: {title}")

    for variant in plotted_variants:
        row = summaries[variant]
        x = float(row["validation_structural_spearman_mean"])
        y = float(row["validation_f1_mean"])
        xerr = float(row["validation_structural_spearman_std"])
        yerr = float(row["validation_f1_std"])
        color = VARIANT_COLORS[variant]
        ax.errorbar(
            x,
            y,
            xerr=xerr,
            yerr=yerr,
            fmt="o",
            color=color,
            ecolor=color,
            elinewidth=1.0,
            capsize=2.5,
            markersize=7.0,
            markeredgecolor="#222222",
            markeredgewidth=0.55,
            label=VARIANT_FULL_LABELS[variant],
        )
        ax.annotate(
            VARIANT_LABELS[variant],
            (x, y),
            xytext=(label_offsets or {}).get(variant, (5, 4)),
            textcoords="offset points",
            fontsize=8.5,
            color="#222222",
        )

    ax.axvline(0.0, color="#444444", linewidth=0.8, linestyle="--", alpha=0.7)
    ax.set_title(title)
    ax.set_xlabel("AST-distance Spearman")
    ax.set_ylabel("Target-subtoken F1")
    ax.grid(alpha=0.22, linewidth=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_b20_tradeoff(original_input: Path, structural_input: Path, output_prefix: Path) -> None:
    original = _load_summary(original_input)
    structural = _load_summary(structural_input)

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.labelsize": 9.5,
            "legend.fontsize": 8.5,
            "figure.titlesize": 12,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.8), constrained_layout=False)
    fig.subplots_adjust(left=0.075, right=0.745, top=0.82, bottom=0.18, wspace=0.30)

    _plot_panel(
        axes[0],
        original,
        "Original / obfuscated",
        {
            "B4_hyperbolic_code2vec": (5, 7),
            "B8_hyperbolic_frechet_code2vec": (5, -11),
            "B19_hyperbolic_path_mp_rank_annealed": (4, 10),
            "B20_hyperbolic_path_mp_rank_delayed": (8, -9),
        },
    )
    _plot_panel(
        axes[1],
        structural,
        "Structural-only stress test",
        {
            "B18_hyperbolic_path_mp_struct_rank": (5, 7),
            "B19_hyperbolic_path_mp_rank_annealed": (5, 7),
            "B20_hyperbolic_path_mp_rank_delayed": (5, 7),
        },
    )

    axes[0].set_xlim(0.31, 0.43)
    axes[0].set_ylim(0.11, 0.20)
    axes[1].set_xlim(-0.28, 0.05)
    axes[1].set_ylim(0.08, 0.18)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center left", bbox_to_anchor=(0.765, 0.50), frameon=False)
    fig.suptitle("B20 schedule ablation: task quality versus structural alignment", y=0.98)

    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_prefix.with_suffix(".png"), dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(output_prefix.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot B20 F1/Spearman trade-off.")
    parser.add_argument("--original-input", type=Path, default=DEFAULT_ORIGINAL_INPUT)
    parser.add_argument("--structural-input", type=Path, default=DEFAULT_STRUCTURAL_INPUT)
    parser.add_argument("--output-prefix", type=Path, default=DEFAULT_OUTPUT_PREFIX)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plot_b20_tradeoff(args.original_input, args.structural_input, args.output_prefix)
    print(f"Wrote {args.output_prefix.with_suffix('.png')}")
    print(f"Wrote {args.output_prefix.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
