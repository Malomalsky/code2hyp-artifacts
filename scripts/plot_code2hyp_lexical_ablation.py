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
        "Original",
        PROJECT_ROOT / "outputs/code2hyp_java_small_focused_b20_original_1k_3seeds.json",
    ),
    (
        "Obfuscated",
        PROJECT_ROOT / "outputs/code2hyp_java_small_focused_b20_obfuscated_1k_3seeds.json",
    ),
    (
        "Structural only",
        PROJECT_ROOT / "outputs/code2hyp_java_small_focused_b20_structural_only_1k_3seeds.json",
    ),
)

VARIANT_ORDER = (
    "B1_euclidean",
    "B3_product",
    "B4_hyperbolic_code2vec",
    "B6_euclidean_metric_code2vec",
    "B8_hyperbolic_frechet_code2vec",
    "B17_hyperbolic_path_mp_code2vec",
    "B18_hyperbolic_path_mp_struct_rank",
    "B19_hyperbolic_path_mp_rank_annealed",
    "B20_hyperbolic_path_mp_rank_delayed",
    "B15_lorentz_product_code2vec",
    "B10_factorized_product_code2vec",
    "B11_factorized_product_struct_rank",
    "B16_factorized_product_three_metric_rank",
    "B14_bounded_euclidean_metric_code2vec",
    "B_tree_euclidean_lca_bias",
)

VARIANT_LABELS = {
    "B1_euclidean": "B1 Euclidean",
    "B3_product": "B3 Product",
    "B4_hyperbolic_code2vec": "B4 Hyp-code2vec",
    "B6_euclidean_metric_code2vec": "B6 Euc metric",
    "B8_hyperbolic_frechet_code2vec": "B8 Hyp-Frechet",
    "B17_hyperbolic_path_mp_code2vec": "B17 Hyp-path-MP",
    "B18_hyperbolic_path_mp_struct_rank": "B18 Hyp-path-MP rank",
    "B19_hyperbolic_path_mp_rank_annealed": "B19 Hyp-path-MP anneal",
    "B20_hyperbolic_path_mp_rank_delayed": "B20 Hyp-path-MP delayed",
    "B15_lorentz_product_code2vec": "B15 Lorentz-product",
    "B10_factorized_product_code2vec": "B10 Mixed-product",
    "B11_factorized_product_struct_rank": "B11 Mixed + rank",
    "B16_factorized_product_three_metric_rank": "B16 Mixed 3-metric",
    "B14_bounded_euclidean_metric_code2vec": "B14 Bounded Euc",
    "B_tree_euclidean_lca_bias": "Btree LCA-bias",
}

VARIANT_COLORS = {
    "B1_euclidean": "#4C78A8",
    "B3_product": "#F58518",
    "B4_hyperbolic_code2vec": "#D64F45",
    "B6_euclidean_metric_code2vec": "#72B7B2",
    "B8_hyperbolic_frechet_code2vec": "#7F3C8D",
    "B17_hyperbolic_path_mp_code2vec": "#A05195",
    "B18_hyperbolic_path_mp_struct_rank": "#D45087",
    "B19_hyperbolic_path_mp_rank_annealed": "#F95D6A",
    "B20_hyperbolic_path_mp_rank_delayed": "#FF7C43",
    "B15_lorentz_product_code2vec": "#8D6E63",
    "B10_factorized_product_code2vec": "#2F4B7C",
    "B11_factorized_product_struct_rank": "#33658A",
    "B16_factorized_product_three_metric_rank": "#005F73",
    "B14_bounded_euclidean_metric_code2vec": "#00A6A6",
    "B_tree_euclidean_lca_bias": "#E45756",
}

PANELS = (
    ("validation_f1", "Target-subtoken F1", "higher is better"),
    ("validation_structural_spearman", "AST-distance Spearman", "higher is better"),
    ("validation_structural_loss", "Structural distance loss", "lower is better"),
)


def _load_mode_summaries(inputs: tuple[tuple[str, Path], ...]) -> dict[str, dict[str, dict[str, float | int | str]]]:
    summaries: dict[str, dict[str, dict[str, float | int | str]]] = {}
    for label, path in inputs:
        result = json.loads(path.read_text(encoding="utf-8"))
        summaries[label] = {str(row["variant"]): row for row in summarize_pilot_runs(result)}
    return summaries


def plot_lexical_ablation_metrics(inputs: tuple[tuple[str, Path], ...], output_prefix: Path) -> None:
    summaries = _load_mode_summaries(inputs)
    mode_labels = tuple(label for label, _ in inputs)
    x_positions = tuple(range(len(mode_labels)))

    plotted_variants = tuple(
        variant for variant in VARIANT_ORDER if all(variant in summaries[mode_label] for mode_label in mode_labels)
    )
    if not plotted_variants:
        raise ValueError("input results have no common variants known to the lexical-ablation plotter")

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "figure.titlesize": 13,
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(19.6, 4.6), constrained_layout=False)
    fig.subplots_adjust(left=0.055, right=0.765, top=0.82, bottom=0.18, wspace=0.28)

    for ax, (metric_key, title, subtitle) in zip(axes, PANELS, strict=True):
        for variant in plotted_variants:
            means = [
                float(summaries[mode_label][variant][f"{metric_key}_mean"])
                for mode_label in mode_labels
            ]
            ax.plot(
                x_positions,
                means,
                marker="o",
                linewidth=2.0 if variant == "B4_hyperbolic_code2vec" else 1.35,
                markersize=5.5 if variant == "B4_hyperbolic_code2vec" else 4.0,
                color=VARIANT_COLORS[variant],
                alpha=1.0 if variant == "B4_hyperbolic_code2vec" else 0.78,
                label=VARIANT_LABELS[variant],
            )
        ax.set_title(title)
        ax.set_xlabel(subtitle)
        ax.set_xticks(x_positions, mode_labels)
        ax.grid(axis="y", alpha=0.25, linewidth=0.7)
        if metric_key == "validation_structural_spearman":
            ax.axhline(0.0, color="#333333", linewidth=0.8)

    axes[0].set_ylabel("Mean over 3 seeds")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center left", ncol=1, frameon=False, bbox_to_anchor=(0.795, 0.50))
    fig.suptitle("Code2Hyp lexical-ablation stress test on Java-small 1k/256 pilot", y=1.04)

    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_prefix.with_suffix(".png"), dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(output_prefix.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot Code2Hyp lexical-ablation pilot metrics.")
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=PROJECT_ROOT / "figures/code2hyp_lexical_ablation_metrics",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plot_lexical_ablation_metrics(DEFAULT_INPUTS, args.output_prefix)
    print(f"Wrote {args.output_prefix.with_suffix('.png')}")
    print(f"Wrote {args.output_prefix.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
