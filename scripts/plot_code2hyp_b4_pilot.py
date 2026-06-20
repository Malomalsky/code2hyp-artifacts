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
from matplotlib.patches import Patch

from geometry_profile_research.code2hyp_reporting import summarize_pilot_runs


DEFAULT_INPUT = PROJECT_ROOT / "outputs/code2hyp_java_small_focused_b20_original_1k_3seeds.json"
DEFAULT_OUTPUT_PREFIX = PROJECT_ROOT / "figures/code2hyp_b4_java_small_focused_b20_metrics"

VARIANT_ORDER = (
    "B1_euclidean",
    "B2_product_fixed_curvature",
    "B3_product",
    "B4_hyperbolic_code2vec",
    "B4T_hyperbolic_code2vec_trainable_curvature",
    "B8_hyperbolic_frechet_code2vec",
    "B17_hyperbolic_path_mp_code2vec",
    "B18_hyperbolic_path_mp_struct_rank",
    "B19_hyperbolic_path_mp_rank_annealed",
    "B20_hyperbolic_path_mp_rank_delayed",
    "B9_lorentz_code2vec",
    "B15_lorentz_product_code2vec",
    "B10_factorized_product_code2vec",
    "B11_factorized_product_struct_rank",
    "B12_factorized_product_learned_metric_rank",
    "B16_factorized_product_three_metric_rank",
    "B13_factorized_product_channel_mixer_rank",
    "B7_hyperbolic_attention_only",
    "B6_euclidean_metric_code2vec",
    "B14_bounded_euclidean_metric_code2vec",
    "B_tree_euclidean_lca_bias",
    "B5_euclidean_struct_loss",
)

VARIANT_SHORT_LABELS = (
    "B1",
    "B2",
    "B3",
    "B4",
    "B4T",
    "B8",
    "B17",
    "B18",
    "B19",
    "B20",
    "B9",
    "B15",
    "B10",
    "B11",
    "B12",
    "B16",
    "B13",
    "B7",
    "B6",
    "B14",
    "Btree",
    "B5",
)

VARIANT_LEGEND_LABELS = (
    "B1 Euclidean",
    "B2 Product fixed curvature",
    "B3 Product trainable curvature",
    "B4 Hyperbolic code2vec",
    "B4T Hyperbolic code2vec trainable curvature",
    "B8 Hyperbolic Frechet code2vec",
    "B17 Hyperbolic AST-path message passing",
    "B18 Hyperbolic AST-path message passing + rank loss",
    "B19 Hyperbolic AST-path message passing + annealed rank loss",
    "B20 Hyperbolic AST-path message passing + delayed rank loss",
    "B9 Lorentz hyperboloid code2vec",
    "B15 Lorentz product code2vec",
    "B10 Factorized mixed-product code2vec",
    "B11 Factorized mixed-product + structural rank loss",
    "B12 Factorized mixed-product learned metric + rank loss",
    "B16 Factorized mixed-product three-metric + rank loss",
    "B13 Factorized mixed-product nonlinear channel mixer + rank loss",
    "B7 Hyperbolic attention only",
    "B6 Euclidean metric code2vec",
    "B14 Bounded Euclidean metric code2vec",
    "Btree Euclidean LCA/tree bias",
    "B5 Euclidean + structural loss",
)

VARIANT_COLORS = (
    "#4C78A8",
    "#B279A2",
    "#F58518",
    "#D64F45",
    "#9C3A34",
    "#7F3C8D",
    "#A05195",
    "#D45087",
    "#F95D6A",
    "#FF7C43",
    "#6B6ECF",
    "#8D6E63",
    "#2F4B7C",
    "#33658A",
    "#003F5C",
    "#005F73",
    "#1B9E77",
    "#C17C2F",
    "#72B7B2",
    "#00A6A6",
    "#E45756",
    "#54A24B",
)

PANELS = (
    ("validation_f1", "Target-subtoken F1", "higher is better"),
    ("validation_structural_loss", "Structural distance loss", "lower is better"),
    ("validation_structural_spearman", "AST-distance Spearman", "higher is better"),
)


def plot_b4_pilot_metrics(input_path: Path, output_prefix: Path) -> None:
    result = json.loads(input_path.read_text(encoding="utf-8"))
    rows = {str(row["variant"]): row for row in summarize_pilot_runs(result)}
    plotted_variants = tuple(variant for variant in VARIANT_ORDER if variant in rows)
    if not plotted_variants:
        raise ValueError("input result has no variants known to the B4 pilot plotter")
    short_labels = tuple(VARIANT_SHORT_LABELS[VARIANT_ORDER.index(variant)] for variant in plotted_variants)
    legend_labels = tuple(VARIANT_LEGEND_LABELS[VARIANT_ORDER.index(variant)] for variant in plotted_variants)
    colors = tuple(VARIANT_COLORS[VARIANT_ORDER.index(variant)] for variant in plotted_variants)

    plt.rcParams.update({"font.size": 10, "axes.titlesize": 12, "figure.titlesize": 13})
    fig, axes = plt.subplots(1, 3, figsize=(21.3, 4.55), constrained_layout=False)
    fig.subplots_adjust(left=0.055, right=0.725, top=0.82, bottom=0.29, wspace=0.30)

    for ax, (metric_key, title, subtitle) in zip(axes, PANELS, strict=True):
        means = [float(rows[variant][f"{metric_key}_mean"]) for variant in plotted_variants]
        stds = [float(rows[variant][f"{metric_key}_std"]) for variant in plotted_variants]
        ax.bar(
            range(len(plotted_variants)),
            means,
            yerr=stds,
            capsize=3,
            color=colors,
            edgecolor="#222222",
            linewidth=0.5,
        )
        ax.set_title(title)
        ax.set_xlabel(subtitle, labelpad=8)
        ax.set_xticks(range(len(plotted_variants)), short_labels, rotation=32, ha="right")
        ax.set_ylabel("Mean over seeds")
        ax.grid(axis="y", alpha=0.25, linewidth=0.7)
        if metric_key == "validation_structural_spearman":
            ax.axhline(0.0, color="#333333", linewidth=0.8)
            ax.set_ylim(-0.4, 0.45)

    handles = [
        Patch(facecolor=color, edgecolor="#222222", label=label)
        for color, label in zip(colors, legend_labels, strict=True)
    ]
    fig.legend(handles=handles, loc="center left", ncol=1, frameon=False, bbox_to_anchor=(0.765, 0.50))
    fig.suptitle("B4 hyperbolic code2vec pilot on Java-small 1k/256 setting", y=1.04)

    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_prefix.with_suffix(".png"), dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(output_prefix.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot B4 hyperbolic code2vec pilot metrics.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-prefix", type=Path, default=DEFAULT_OUTPUT_PREFIX)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plot_b4_pilot_metrics(args.input, args.output_prefix)
    print(f"Wrote {args.output_prefix.with_suffix('.png')}")
    print(f"Wrote {args.output_prefix.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
