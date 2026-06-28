from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt


VARIANT_LABELS = {
    "B62_code2hyp_context_transform_branch_sequence_product_bias_multi_metric_frechet": "B62 branch product",
    "B84_geocodepath_relation_conditioned_product_proxy": "B84 relation-conditioned",
    "B85_geocodepath_relation_conditioned_aux_product_proxy": "B85 auxiliary endpoint",
}


def _load_rows(input_path: Path) -> pd.DataFrame:
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    rows: list[dict[str, object]] = []
    for run in payload["runs"]:
        variant = run["variant"]
        label = VARIANT_LABELS.get(variant, variant)
        for representation, spearman_key, stress_key in (
            (
                "Context transport",
                "validation_method_transport_spearman",
                "validation_method_transport_normalized_stress",
            ),
            (
                "Frechet aggregate",
                "validation_method_aggregate_spearman",
                "validation_method_aggregate_normalized_stress",
            ),
        ):
            rows.append(
                {
                    "variant": label,
                    "representation": representation,
                    "spearman": run[spearman_key],
                    "normalized_stress": run[stress_key],
                    "seed": run["model_seed"],
                }
            )
    return pd.DataFrame(rows)


def plot_method_transport(input_path: Path, output_stem: Path) -> None:
    df = _load_rows(input_path)
    sns.set_theme(style="ticks", context="paper", font_scale=1.15)
    plt.rcParams.update(
        {
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
        }
    )
    palette = {
        "Context transport": "#0072B2",
        "Frechet aggregate": "#D55E00",
    }
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9), sharex=True)

    sns.barplot(
        data=df,
        x="variant",
        y="spearman",
        hue="representation",
        errorbar="sd",
        palette=palette,
        ax=axes[0],
    )
    sns.stripplot(
        data=df,
        x="variant",
        y="spearman",
        hue="representation",
        dodge=True,
        palette=palette,
        edgecolor="black",
        linewidth=0.35,
        size=3.2,
        alpha=0.8,
        legend=False,
        ax=axes[0],
    )
    axes[0].set_title("Method-level structural rank")
    axes[0].set_ylabel("Spearman correlation")
    axes[0].set_xlabel("")
    axes[0].set_ylim(0.35, 0.78)

    sns.barplot(
        data=df,
        x="variant",
        y="normalized_stress",
        hue="representation",
        errorbar="sd",
        palette=palette,
        ax=axes[1],
    )
    sns.stripplot(
        data=df,
        x="variant",
        y="normalized_stress",
        hue="representation",
        dodge=True,
        palette=palette,
        edgecolor="black",
        linewidth=0.35,
        size=3.2,
        alpha=0.8,
        legend=False,
        ax=axes[1],
    )
    axes[1].set_title("Method-level structural distortion")
    axes[1].set_ylabel("Normalized stress")
    axes[1].set_xlabel("")
    axes[1].set_ylim(0.12, 0.29)

    for ax in axes:
        ax.tick_params(axis="x", rotation=20)
        ax.grid(axis="y", color="#D0D0D0", linewidth=0.5, alpha=0.7)
        sns.despine(ax=ax)

    handles, labels = axes[0].get_legend_handles_labels()
    axes[0].legend_.remove()
    if axes[1].legend_ is not None:
        axes[1].legend_.remove()
    fig.legend(
        handles[:2],
        labels[:2],
        frameon=False,
        loc="upper center",
        ncol=2,
        bbox_to_anchor=(0.5, 1.04),
    )

    fig.text(0.01, 0.98, "A", fontsize=11, fontweight="bold", va="top")
    fig.text(0.505, 0.98, "B", fontsize=11, fontweight="bold", va="top")
    fig.tight_layout(pad=1.1, rect=(0, 0, 1, 0.96))

    output_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-stem", type=Path, required=True)
    args = parser.parse_args()
    plot_method_transport(args.input, args.output_stem)


if __name__ == "__main__":
    main()
