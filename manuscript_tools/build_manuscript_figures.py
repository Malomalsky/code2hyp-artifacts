#!/usr/bin/env python3
"""Build publication figures for the Code2Hyp manuscript."""

from __future__ import annotations

import json
import math
import os
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = PROJECT_ROOT / "manuscript_figures"

os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib-cache"))
os.environ.setdefault("XDG_CACHE_HOME", str(PROJECT_ROOT / ".matplotlib-cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ORIGINAL_MAIN = PROJECT_ROOT / "outputs/code2hyp_test_benchmark_25k_5epochs_5seeds_original_main_variants_with_stress.json"
ORIGINAL_CONTROLS = PROJECT_ROOT / "outputs/code2hyp_test_benchmark_25k_5epochs_3seeds_original_plus_euclidean_controls.json"
RECORD_OBF = PROJECT_ROOT / "outputs/code2hyp_test_benchmark_25k_5epochs_3seeds_record_obfuscated_resumable_with_stress.json"
STRUCT_ONLY = PROJECT_ROOT / "outputs/code2hyp_test_benchmark_25k_5epochs_3seeds_structural_only_resumable_with_stress.json"

METRICS = (
    "validation_f1",
    "validation_structural_spearman",
    "validation_structural_normalized_stress",
    "validation_structural_neighbor_overlap_at_3",
)

LABELS = {
    "B39_code2vec_context_transform_baseline": "B39\nmatched\nEuclidean",
    "B36_code2hyp_product_frechet_neighbor": "B36\nCode2Hyp\nperformance",
    "B40_code2hyp_context_transform_frechet": "B40\nCode2Hyp\nFrechet",
    "B44_code2hyp_context_transform_product_bias_frechet": "B44\nCode2Hyp\nstructural",
    "B6_euclidean_metric_code2vec": "B6\nEuclidean\nmetric",
    "B14_bounded_euclidean_metric_code2vec": "B14\nbounded\nEuclidean",
    "B_tree_euclidean_lca_bias": "Btree\ntree/LCA\nbias",
}

MAIN_ORDER = (
    "B39_code2vec_context_transform_baseline",
    "B36_code2hyp_product_frechet_neighbor",
    "B40_code2hyp_context_transform_frechet",
    "B44_code2hyp_context_transform_product_bias_frechet",
    "B6_euclidean_metric_code2vec",
    "B14_bounded_euclidean_metric_code2vec",
    "B_tree_euclidean_lca_bias",
)

COLORS = {
    "baseline": "#3F3F3F",
    "blue": "#0072B2",
    "green": "#009E73",
    "orange": "#D55E00",
    "purple": "#CC79A7",
    "sky": "#56B4E9",
    "yellow": "#E69F00",
    "grid": "#D8D8D8",
}


def style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "font.size": 7.4,
            "axes.labelsize": 7.4,
            "axes.titlesize": 8.2,
            "xtick.labelsize": 6.8,
            "ytick.labelsize": 6.8,
            "legend.fontsize": 6.8,
            "figure.titlesize": 9.0,
            "axes.linewidth": 0.8,
            "savefig.dpi": 450,
        }
    )


def load(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))["runs"]


def group(runs: list[dict]) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for run in runs:
        grouped[run["variant"]].append(run)
    return grouped


def stat(values: list[float]) -> tuple[float, float]:
    return mean(values), stdev(values) if len(values) > 1 else 0.0


def summarize_by_variant(runs: list[dict]) -> dict[str, dict[str, tuple[float, float, int]]]:
    result: dict[str, dict[str, tuple[float, float, int]]] = {}
    for variant, items in group(runs).items():
        result[variant] = {}
        for metric in METRICS:
            values = [float(item[metric]) for item in items]
            m, sd = stat(values)
            result[variant][metric] = (m, sd, len(values))
    return result


def paired_delta(runs: list[dict], positive: str, negative: str, metric: str) -> tuple[float, float]:
    by_variant = group(runs)
    pos = {int(r["model_seed"]): float(r[metric]) for r in by_variant[positive]}
    neg = {int(r["model_seed"]): float(r[metric]) for r in by_variant[negative]}
    seeds = sorted(set(pos) & set(neg))
    values = [pos[s] - neg[s] for s in seeds]
    m, sd = stat(values)
    se = sd / math.sqrt(len(values)) if len(values) > 1 else 0.0
    return m, 1.96 * se


def save(fig: plt.Figure, stem: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / f"{stem}.png", dpi=450, bbox_inches="tight", facecolor="white")
    fig.savefig(OUT_DIR / f"{stem}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def add_box(ax, xy, width, height, text, face, edge="#2B2B2B", size=8):
    box = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.02,rounding_size=0.025",
        linewidth=0.9,
        edgecolor=edge,
        facecolor=face,
    )
    ax.add_patch(box)
    ax.text(
        xy[0] + width / 2,
        xy[1] + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=size,
        linespacing=1.2,
    )


def add_arrow(ax, start, end, color="#333333", rad=0.0):
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=10,
        linewidth=1.0,
        color=color,
        connectionstyle=f"arc3,rad={rad}",
    )
    ax.add_patch(arrow)


def plot_architecture() -> None:
    style()
    fig, ax = plt.subplots(figsize=(4.8, 2.85))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    add_box(ax, (0.03, 0.56), 0.16, 0.16, "Java\nmethod", "#F2F2F2", size=7)
    add_box(ax, (0.24, 0.56), 0.20, 0.16, "AST path\ncontexts\n$(x_i,p_i,y_i)$", "#EAF3F8", size=7)
    add_box(ax, (0.51, 0.72), 0.22, 0.13, "lexical\nchannel\n$\\mathbb{E}^{d_e}$", "#E6F2FF", size=6.6)
    add_box(ax, (0.51, 0.39), 0.22, 0.13, "structural\nchannel\n$\\mathbb{H}^{d_h}_c$", "#FFF0E8", size=6.6)
    add_box(ax, (0.80, 0.56), 0.17, 0.16, "product\nrepresentation\n$\\mathbb{E}\\times\\mathbb{H}$", "#F1F7ED", size=6.5)
    add_box(ax, (0.36, 0.15), 0.23, 0.13, "attention\nand decoder", "#F7F4E8", size=6.8)
    add_box(ax, (0.67, 0.15), 0.24, 0.13, "method-name\nsubtokens", "#F2F2F2", size=6.8)

    add_arrow(ax, (0.19, 0.64), (0.24, 0.64))
    add_arrow(ax, (0.44, 0.64), (0.51, 0.78))
    add_arrow(ax, (0.44, 0.64), (0.51, 0.46))
    add_arrow(ax, (0.73, 0.785), (0.80, 0.68))
    add_arrow(ax, (0.73, 0.455), (0.80, 0.60))
    add_arrow(ax, (0.885, 0.56), (0.50, 0.28), rad=-0.12)
    add_arrow(ax, (0.59, 0.215), (0.67, 0.215))

    ax.text(
        0.52,
        0.035,
        "Code2Hyp keeps token semantics Euclidean and tests negative curvature only for AST-path structure.",
        ha="center",
        va="bottom",
        fontsize=6.4,
    )
    save(fig, "figure01_code2hyp_architecture")


def plot_main_results() -> None:
    style()
    main = summarize_by_variant(load(ORIGINAL_MAIN))
    controls = summarize_by_variant(load(ORIGINAL_CONTROLS))
    combined = dict(main)
    for variant in ("B6_euclidean_metric_code2vec", "B14_bounded_euclidean_metric_code2vec", "B_tree_euclidean_lca_bias"):
        combined[variant] = controls[variant]

    palette = [
        COLORS["baseline"],
        COLORS["blue"],
        COLORS["green"],
        COLORS["orange"],
        COLORS["sky"],
        COLORS["purple"],
        COLORS["yellow"],
    ]
    short_labels = {
        "B39_code2vec_context_transform_baseline": "B39 baseline",
        "B36_code2hyp_product_frechet_neighbor": "B36 Code2Hyp",
        "B40_code2hyp_context_transform_frechet": "B40 Frechet",
        "B44_code2hyp_context_transform_product_bias_frechet": "B44 structural",
        "B6_euclidean_metric_code2vec": "B6 Euclidean metric",
        "B14_bounded_euclidean_metric_code2vec": "B14 bounded Euclidean",
        "B_tree_euclidean_lca_bias": "Btree LCA bias",
    }

    fig = plt.figure(figsize=(6.9, 5.15))
    gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.36)
    axes = [fig.add_subplot(gs[i, j]) for i in range(2) for j in range(2)]

    metric_specs = [
        ("validation_f1", "A. Downstream prediction", "Target-subtoken F1, %", 100.0),
        ("validation_structural_spearman", "B. AST-distance rank preservation", "Spearman correlation", 1.0),
        ("validation_structural_normalized_stress", "C. Metric distortion", "Normalized stress", 1.0),
    ]

    y = list(range(len(MAIN_ORDER)))
    for ax, (metric, title, ylabel, scale) in zip(axes[:3], metric_specs, strict=True):
        means = [combined[v][metric][0] * scale for v in MAIN_ORDER]
        sds = [combined[v][metric][1] * scale for v in MAIN_ORDER]
        for yi, value, error, color in zip(y, means, sds, palette, strict=True):
            ax.errorbar(
                value,
                yi,
                xerr=error,
                marker="o",
                markersize=5,
                capsize=2.2,
                color=color,
                markeredgecolor="#222222",
                markeredgewidth=0.4,
                linewidth=1.0,
            )
        ax.set_title(title, loc="left", fontweight="bold")
        ax.set_xlabel(ylabel)
        ax.set_yticks(y, [short_labels[v] for v in MAIN_ORDER])
        ax.invert_yaxis()
        lo = min(m - e for m, e in zip(means, sds, strict=True))
        hi = max(m + e for m, e in zip(means, sds, strict=True))
        pad = max((hi - lo) * 0.10, 0.02 if scale == 1.0 else 0.4)
        ax.set_xlim(lo - pad, hi + pad)
        ax.grid(axis="x", color=COLORS["grid"], linewidth=0.5, alpha=0.75)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if metric == "validation_structural_spearman":
            ax.axvline(0, color="#222222", linewidth=0.7)
        if metric == "validation_structural_normalized_stress":
            ax.text(0.03, 0.05, "lower is better", transform=ax.transAxes, fontsize=6.5)

    ax = axes[3]
    for i, variant in enumerate(MAIN_ORDER):
        f1 = combined[variant]["validation_f1"][0] * 100
        spearman = combined[variant]["validation_structural_spearman"][0]
        stress = combined[variant]["validation_structural_normalized_stress"][0]
        size = 42 + 110 * max(0.0, 1.0 - min(stress, 1.0))
        ax.scatter(f1, spearman, s=size, color=palette[i], edgecolor="#222222", linewidth=0.55, zorder=3)
        offsets = {
            "B39_code2vec_context_transform_baseline": (0.18, 0.03),
            "B36_code2hyp_product_frechet_neighbor": (0.15, 0.03),
            "B40_code2hyp_context_transform_frechet": (0.13, 0.03),
            "B44_code2hyp_context_transform_product_bias_frechet": (0.14, 0.03),
            "B6_euclidean_metric_code2vec": (0.08, 0.08),
            "B14_bounded_euclidean_metric_code2vec": (0.14, 0.02),
            "B_tree_euclidean_lca_bias": (0.08, -0.08),
        }
        dx, dy = offsets[variant]
        ax.text(f1 + dx, spearman + dy, short_labels[variant].split()[0], fontsize=6.7)
    ax.axhline(0, color="#222222", linewidth=0.7)
    ax.set_title("D. Prediction-structure trade-off", loc="left", fontweight="bold")
    ax.set_xlabel("Target-subtoken F1, %")
    ax.set_ylabel("AST-distance Spearman")
    ax.grid(color=COLORS["grid"], linewidth=0.5, alpha=0.75)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.text(0.02, 0.96, "larger marker = lower stress", transform=ax.transAxes, fontsize=6.5, va="top")

    fig.suptitle("Main Code2Hyp results under the 25,000-example local training budget", y=0.995)
    save(fig, "figure02_main_results")


def plot_control_deltas() -> None:
    style()
    modes = [
        ("Original", load(ORIGINAL_MAIN)),
        ("Record-\nobfuscated", load(RECORD_OBF)),
        ("Structural\nonly", load(STRUCT_ONLY)),
    ]
    x = list(range(len(modes)))

    fig, axes = plt.subplots(2, 2, figsize=(6.8, 4.45), constrained_layout=True)
    axes = list(axes.ravel())

    b39 = "B39_code2vec_context_transform_baseline"
    b36 = "B36_code2hyp_product_frechet_neighbor"
    b44 = "B44_code2hyp_context_transform_product_bias_frechet"

    summaries = [summarize_by_variant(runs) for _, runs in modes]

    for variant, color, label in [(b39, COLORS["baseline"], "B39 baseline"), (b36, COLORS["blue"], "B36 Code2Hyp")]:
        means = [summary[variant]["validation_f1"][0] * 100 for summary in summaries]
        sds = [summary[variant]["validation_f1"][1] * 100 for summary in summaries]
        axes[0].errorbar(x, means, yerr=sds, color=color, marker="o", capsize=2.5, linewidth=1.4, label=label)
    axes[0].set_title("A. F1 under lexical controls", loc="left", fontweight="bold")
    axes[0].set_ylabel("F1, %")

    deltas = [paired_delta(runs, b36, b39, "validation_f1") for _, runs in modes]
    axes[1].bar(x, [d[0] * 100 for d in deltas], yerr=[d[1] * 100 for d in deltas], color=COLORS["blue"], capsize=2.5)
    axes[1].axhline(0, color="#222222", linewidth=0.75)
    axes[1].set_title("B. Paired F1 delta", loc="left", fontweight="bold")
    axes[1].set_ylabel("B36 - B39, points")

    for variant, color, label in [(b39, COLORS["baseline"], "B39 baseline"), (b44, COLORS["orange"], "B44 Code2Hyp")]:
        means = [summary[variant]["validation_structural_spearman"][0] for summary in summaries]
        sds = [summary[variant]["validation_structural_spearman"][1] for summary in summaries]
        axes[2].errorbar(x, means, yerr=sds, color=color, marker="o", capsize=2.5, linewidth=1.4, label=label)
    axes[2].axhline(0, color="#222222", linewidth=0.75)
    axes[2].set_title("C. Structural rank preservation", loc="left", fontweight="bold")
    axes[2].set_ylabel("AST-distance Spearman")

    deltas = [paired_delta(runs, b44, b39, "validation_structural_spearman") for _, runs in modes]
    axes[3].bar(x, [d[0] for d in deltas], yerr=[d[1] for d in deltas], color=COLORS["orange"], capsize=2.5)
    axes[3].axhline(0, color="#222222", linewidth=0.75)
    axes[3].set_title("D. Paired structural delta", loc="left", fontweight="bold")
    axes[3].set_ylabel("B44 - B39, Spearman")

    for ax in axes:
        ax.set_xticks(x, [label for label, _ in modes])
        ax.grid(axis="y", color=COLORS["grid"], linewidth=0.5, alpha=0.75)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0].legend(frameon=False, loc="best")
    axes[2].legend(frameon=False, loc="best")
    save(fig, "figure03_lexical_controls")


def main() -> None:
    plot_architecture()
    plot_main_results()
    plot_control_deltas()
    print(f"Wrote figures to {OUT_DIR}")


if __name__ == "__main__":
    main()
