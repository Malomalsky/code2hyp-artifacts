from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib-cache"))
os.environ.setdefault("XDG_CACHE_HOME", str(PROJECT_ROOT / ".matplotlib-cache"))
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_INPUTS = (
    (
        "Original",
        PROJECT_ROOT / "outputs/code2hyp_test_benchmark_10k_5epochs_3seeds_with_stress.json",
    ),
    (
        "Record-obfuscated",
        PROJECT_ROOT / "outputs/code2hyp_test_benchmark_10k_5epochs_3seeds_record_obfuscated_with_stress.json",
    ),
    (
        "Structural only",
        PROJECT_ROOT / "outputs/code2hyp_test_benchmark_10k_5epochs_3seeds_structural_only_with_stress.json",
    ),
)

VARIANT_ORDER = (
    "B39_code2vec_context_transform_baseline",
    "B47_code2vec_context_transform_distance_control",
    "B50_code2vec_context_transform_l1_baseline",
    "B51_code2vec_context_transform_l1_distance_control",
    "B48_code2hyp_context_transform_product_bias_no_struct",
    "B49_code2hyp_context_transform_product_bias_near_euclidean",
    "B36_code2hyp_product_frechet_neighbor",
    "B40_code2hyp_context_transform_frechet",
    "B44_code2hyp_context_transform_product_bias_frechet",
)

VARIANT_LABELS = {
    "B39_code2vec_context_transform_baseline": "B39 matched baseline",
    "B47_code2vec_context_transform_distance_control": "B47 Euclidean + distance",
    "B50_code2vec_context_transform_l1_baseline": "B50 L1 baseline",
    "B51_code2vec_context_transform_l1_distance_control": "B51 L1 + distance",
    "B48_code2hyp_context_transform_product_bias_no_struct": "B48 product, no struct.",
    "B49_code2hyp_context_transform_product_bias_near_euclidean": "B49 near-Euclidean",
    "B36_code2hyp_product_frechet_neighbor": "B36 performance",
    "B40_code2hyp_context_transform_frechet": "B40 Frechet",
    "B44_code2hyp_context_transform_product_bias_frechet": "B44 structural",
}

VARIANT_STYLES = {
    "B39_code2vec_context_transform_baseline": ("#4D4D4D", "o", "-"),
    "B47_code2vec_context_transform_distance_control": ("#CC79A7", "P", "--"),
    "B50_code2vec_context_transform_l1_baseline": ("#999999", "v", ":"),
    "B51_code2vec_context_transform_l1_distance_control": ("#7F3C8D", "*", "--"),
    "B48_code2hyp_context_transform_product_bias_no_struct": ("#E69F00", "X", ":"),
    "B49_code2hyp_context_transform_product_bias_near_euclidean": ("#56B4E9", "^", "--"),
    "B36_code2hyp_product_frechet_neighbor": ("#0072B2", "s", "-"),
    "B40_code2hyp_context_transform_frechet": ("#009E73", "^", "--"),
    "B44_code2hyp_context_transform_product_bias_frechet": ("#D55E00", "D", "-."),
}

PANELS = (
    ("validation_f1", "Target-subtoken F1", "higher is better", 100.0),
    ("validation_structural_spearman", "AST-distance Spearman", "higher is better", 1.0),
    ("validation_structural_normalized_stress", "Normalized stress", "lower is better", 1.0),
    ("validation_structural_neighbor_overlap_at_3", "Overlap@3", "higher is better", 1.0),
)


def _load_mode_summary(path: Path) -> dict[str, dict[str, float]]:
    result = json.loads(path.read_text(encoding="utf-8"))
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for run in result["runs"]:
        grouped[str(run["variant"])].append(run)

    summary: dict[str, dict[str, float]] = {}
    for variant, runs in grouped.items():
        variant_summary: dict[str, float] = {}
        for metric, _, _, _ in PANELS:
            values = [float(run[metric]) for run in runs]
            variant_summary[f"{metric}_mean"] = mean(values)
            variant_summary[f"{metric}_sd"] = stdev(values) if len(values) > 1 else 0.0
        summary[variant] = variant_summary
    return summary


def plot_final_controls(inputs: tuple[tuple[str, Path], ...], output_prefix: Path) -> None:
    mode_labels = tuple(label for label, _ in inputs)
    summaries = {label: _load_mode_summary(path) for label, path in inputs}
    active_variants = [
        variant
        for variant in VARIANT_ORDER
        if all(variant in summaries[mode_label] for mode_label in mode_labels)
    ]
    if not active_variants:
        raise ValueError("input results do not share any known variants")

    x_positions = list(range(len(mode_labels)))
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.2), constrained_layout=True)
    flat_axes = tuple(axes.ravel())

    for panel_index, (ax, (metric, title, subtitle, scale)) in enumerate(zip(flat_axes, PANELS, strict=True)):
        for variant in active_variants:
            color, marker, linestyle = VARIANT_STYLES[variant]
            means = [
                summaries[mode_label][variant][f"{metric}_mean"] * scale
                for mode_label in mode_labels
            ]
            sds = [
                summaries[mode_label][variant][f"{metric}_sd"] * scale
                for mode_label in mode_labels
            ]
            ax.errorbar(
                x_positions,
                means,
                yerr=sds,
                marker=marker,
                linestyle=linestyle,
                linewidth=1.4,
                markersize=4.5,
                capsize=2.5,
                color=color,
                label=VARIANT_LABELS[variant],
            )
        ax.set_title(f"{chr(ord('A') + panel_index)}. {title}")
        ax.set_xlabel(subtitle)
        ax.set_xticks(x_positions, mode_labels, rotation=15, ha="right")
        ax.grid(axis="y", alpha=0.22, linewidth=0.6)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if metric == "validation_structural_spearman":
            ax.axhline(0.0, color="#333333", linewidth=0.8, alpha=0.8)
        if metric == "validation_f1":
            ax.set_ylabel("Mean over seeds, %")
        else:
            ax.set_ylabel("Mean over seeds")

    handles, labels = flat_axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=min(4, len(active_variants)),
        frameon=False,
        bbox_to_anchor=(0.5, -0.03),
    )
    fig.suptitle("Code2Hyp final controls on Java-small test split", y=1.02, fontsize=11)

    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_prefix.with_suffix(".png"), dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(output_prefix.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def parse_labeled_input(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("expected LABEL=PATH")
    label, path = value.split("=", 1)
    label = label.strip()
    path = path.strip()
    if not label:
        raise argparse.ArgumentTypeError("input label must not be empty")
    if not path:
        raise argparse.ArgumentTypeError("input path must not be empty")
    return label, Path(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot final Code2Hyp control benchmarks.")
    parser.add_argument(
        "--input",
        action="append",
        type=parse_labeled_input,
        help="Control input as LABEL=PATH. Can be repeated. Defaults to the frozen 10k controls.",
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=PROJECT_ROOT / "figures/code2hyp_final_controls_10k",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    inputs = tuple(args.input) if args.input else DEFAULT_INPUTS
    plot_final_controls(inputs, args.output_prefix)
    print(f"Wrote {args.output_prefix.with_suffix('.png')}")
    print(f"Wrote {args.output_prefix.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
