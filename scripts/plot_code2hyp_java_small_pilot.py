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
        "1k train",
        PROJECT_ROOT / "outputs/code2hyp_java_small_focused_b20_original_1k_3seeds.json",
    ),
    (
        "4k train",
        PROJECT_ROOT / "outputs/code2hyp_java_small_real_pilot_4k_ctx30_gru_midpoint_posw_with_b14_evalfix_3seeds.json",
    ),
)

VARIANT_LABELS = {
    "B1_euclidean": "B1 Euclidean",
    "B2_product_fixed_curvature": "B2 Product fixed-c",
    "B3_product": "B3 Product",
    "B4_hyperbolic_code2vec": "B4 Hyp-code2vec",
    "B4T_hyperbolic_code2vec_trainable_curvature": "B4T Hyp-code2vec train-c",
    "B8_hyperbolic_frechet_code2vec": "B8 Hyp-Frechet",
    "B17_hyperbolic_path_mp_code2vec": "B17 Hyp-path-MP",
    "B18_hyperbolic_path_mp_struct_rank": "B18 Hyp-path-MP rank",
    "B19_hyperbolic_path_mp_rank_annealed": "B19 Hyp-path-MP anneal",
    "B20_hyperbolic_path_mp_rank_delayed": "B20 Hyp-path-MP delayed",
    "B9_lorentz_code2vec": "B9 Lorentz-code2vec",
    "B15_lorentz_product_code2vec": "B15 Lorentz-product",
    "B10_factorized_product_code2vec": "B10 Mixed-product",
    "B11_factorized_product_struct_rank": "B11 Mixed + rank",
    "B12_factorized_product_learned_metric_rank": "B12 Mixed learned",
    "B16_factorized_product_three_metric_rank": "B16 Mixed 3-metric",
    "B13_factorized_product_channel_mixer_rank": "B13 Mixed mixer",
    "B7_hyperbolic_attention_only": "B7 Hyp-attn only",
    "B6_euclidean_metric_code2vec": "B6 Euc metric-code2vec",
    "B14_bounded_euclidean_metric_code2vec": "B14 Bounded Euc metric",
    "B_tree_euclidean_lca_bias": "Btree Euc LCA-bias",
    "B5_euclidean_struct_loss": "B5 Euclidean + L_struct",
}

VARIANT_COLORS = {
    "B1_euclidean": "#4C78A8",
    "B2_product_fixed_curvature": "#B279A2",
    "B3_product": "#F58518",
    "B4_hyperbolic_code2vec": "#D64F45",
    "B4T_hyperbolic_code2vec_trainable_curvature": "#9C3A34",
    "B8_hyperbolic_frechet_code2vec": "#7F3C8D",
    "B17_hyperbolic_path_mp_code2vec": "#A05195",
    "B18_hyperbolic_path_mp_struct_rank": "#D45087",
    "B19_hyperbolic_path_mp_rank_annealed": "#F95D6A",
    "B20_hyperbolic_path_mp_rank_delayed": "#FF7C43",
    "B9_lorentz_code2vec": "#6B6ECF",
    "B15_lorentz_product_code2vec": "#8D6E63",
    "B10_factorized_product_code2vec": "#2F4B7C",
    "B11_factorized_product_struct_rank": "#33658A",
    "B12_factorized_product_learned_metric_rank": "#003F5C",
    "B16_factorized_product_three_metric_rank": "#005F73",
    "B13_factorized_product_channel_mixer_rank": "#1B9E77",
    "B7_hyperbolic_attention_only": "#C17C2F",
    "B6_euclidean_metric_code2vec": "#72B7B2",
    "B14_bounded_euclidean_metric_code2vec": "#00A6A6",
    "B_tree_euclidean_lca_bias": "#E45756",
    "B5_euclidean_struct_loss": "#54A24B",
}

PANELS = (
    ("validation_f1", "Target-subtoken F1", "higher is better"),
    ("validation_structural_loss", "Structural distance loss", "lower is better"),
    ("validation_structural_spearman", "AST-distance Spearman", "higher is better"),
)


def _load_summaries(inputs: tuple[tuple[str, Path], ...]) -> dict[str, dict[str, dict[str, float | int | str]]]:
    summaries: dict[str, dict[str, dict[str, float | int | str]]] = {}
    for label, path in inputs:
        result = json.loads(path.read_text(encoding="utf-8"))
        summaries[label] = {str(row["variant"]): row for row in summarize_pilot_runs(result)}
    return summaries


def plot_pilot_metrics(inputs: tuple[tuple[str, Path], ...], output_prefix: Path) -> None:
    summaries = _load_summaries(inputs)
    dataset_labels = tuple(label for label, _ in inputs)
    variants = tuple(
        variant
        for variant in VARIANT_LABELS
        if all(variant in summaries[dataset_label] for dataset_label in dataset_labels)
    )
    if not variants:
        raise ValueError("no common variants across input pilot files")
    x_positions = list(range(len(dataset_labels)))
    width = 0.049
    center = (len(variants) - 1) / 2.0
    offsets = {variant: (index - center) * width for index, variant in enumerate(variants)}

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "figure.titlesize": 12,
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(20.6, 4.2), constrained_layout=False)
    fig.subplots_adjust(left=0.055, right=0.735, top=0.82, bottom=0.18, wspace=0.30)

    for ax, (metric_key, title, subtitle) in zip(axes, PANELS, strict=True):
        for variant in variants:
            means = [
                float(summaries[dataset_label][variant][f"{metric_key}_mean"])
                for dataset_label in dataset_labels
            ]
            stds = [
                float(summaries[dataset_label][variant][f"{metric_key}_std"])
                for dataset_label in dataset_labels
            ]
            xs = [x + offsets[variant] for x in x_positions]
            ax.bar(
                xs,
                means,
                width=width,
                yerr=stds,
                capsize=3,
                color=VARIANT_COLORS[variant],
                edgecolor="#222222",
                linewidth=0.5,
                label=VARIANT_LABELS[variant],
            )
        ax.set_title(title)
        ax.set_xlabel(subtitle)
        ax.set_xticks(x_positions, dataset_labels)
        ax.grid(axis="y", alpha=0.25, linewidth=0.7)
        if metric_key == "validation_structural_spearman":
            ax.axhline(0.0, color="#333333", linewidth=0.8)
            ax.set_ylim(-0.45, 0.50)

    axes[0].set_ylabel("Mean over seeds")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center left", ncol=1, frameon=False, bbox_to_anchor=(0.765, 0.50))
    fig.suptitle("Code2Hyp Java-small pilot: predictive and structural diagnostics", y=1.04)

    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_prefix.with_suffix(".png"), dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(output_prefix.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot Code2Hyp Java-small pilot metrics.")
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=PROJECT_ROOT / "figures/code2hyp_java_small_pilot_metrics",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plot_pilot_metrics(DEFAULT_INPUTS, args.output_prefix)
    print(f"Wrote {args.output_prefix.with_suffix('.png')}")
    print(f"Wrote {args.output_prefix.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
