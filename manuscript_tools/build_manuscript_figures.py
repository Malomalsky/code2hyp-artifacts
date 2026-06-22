#!/usr/bin/env python3
"""Build publication figures for the Code2Hyp manuscript."""

from __future__ import annotations

import argparse
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MANUSCRIPT_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = MANUSCRIPT_ROOT / "figures"
GRAYSCALE_DIR = MANUSCRIPT_ROOT / "build" / "grayscale_previews"

os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib-cache"))
os.environ.setdefault("XDG_CACHE_HOME", str(PROJECT_ROOT / ".matplotlib-cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from PIL import Image


POSTREVIEW_RESULT = (
    PROJECT_ROOT
    / "outputs/code2hyp_postreview_benchmark_25k_5epochs_5seeds_with_b49_l1_and_geometry_diagnostics.json"
)

COLORS = {
    "baseline": "#3F3F3F",
    "blue": "#0072B2",
    "green": "#009E73",
    "orange": "#D55E00",
    "purple": "#CC79A7",
    "sky": "#56B4E9",
    "yellow": "#E69F00",
    "brown": "#8A5A44",
    "grid": "#D8D8D8",
}

POSTREVIEW_ORDER = (
    "B39_code2vec_context_transform_baseline",
    "B47_code2vec_context_transform_distance_control",
    "B50_code2vec_context_transform_l1_baseline",
    "B51_code2vec_context_transform_l1_distance_control",
    "B48_code2hyp_context_transform_product_bias_no_struct",
    "B49_code2hyp_context_transform_product_bias_near_euclidean",
    "B36_code2hyp_product_frechet_neighbor",
    "B44_code2hyp_context_transform_product_bias_frechet",
)

POSTREVIEW_CODES = {
    "B39_code2vec_context_transform_baseline": "B39",
    "B47_code2vec_context_transform_distance_control": "B47",
    "B50_code2vec_context_transform_l1_baseline": "B50",
    "B51_code2vec_context_transform_l1_distance_control": "B51",
    "B48_code2hyp_context_transform_product_bias_no_struct": "B48",
    "B49_code2hyp_context_transform_product_bias_near_euclidean": "B49",
    "B36_code2hyp_product_frechet_neighbor": "B36",
    "B44_code2hyp_context_transform_product_bias_frechet": "B44",
}

POSTREVIEW_LABELS = {
    "B39_code2vec_context_transform_baseline": "B39 Euclidean\nbaseline",
    "B47_code2vec_context_transform_distance_control": "B47 Euclidean\nL2 distance",
    "B50_code2vec_context_transform_l1_baseline": "B50 Euclidean\nL1 baseline",
    "B51_code2vec_context_transform_l1_distance_control": "B51 Euclidean\nL1 distance",
    "B48_code2hyp_context_transform_product_bias_no_struct": "B48 product\nno structural loss",
    "B49_code2hyp_context_transform_product_bias_near_euclidean": "B49 same path\nnear Euclidean",
    "B36_code2hyp_product_frechet_neighbor": "B36 Code2Hyp\ndownstream",
    "B44_code2hyp_context_transform_product_bias_frechet": "B44 Code2Hyp\nstructural",
}

POSTREVIEW_PALETTE = {
    "B39": COLORS["baseline"],
    "B47": COLORS["purple"],
    "B50": "#8C8C8C",
    "B51": "#7F3C8D",
    "B48": COLORS["yellow"],
    "B49": COLORS["sky"],
    "B36": COLORS["blue"],
    "B44": COLORS["orange"],
}

CURRENT_FIGURE_STEMS = (
    "figure01_code2hyp_architecture",
    "figure02_main_results",
    "figure03_geometry_diagnostics",
    "figure04_distance_levels",
)


def style() -> None:
    sns.set_theme(
        style="whitegrid",
        context="paper",
        font="DejaVu Serif",
        rc={
            "axes.edgecolor": "#222222",
            "grid.color": COLORS["grid"],
            "grid.linewidth": 0.48,
        },
    )
    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "font.size": 7.5,
            "axes.labelsize": 7.5,
            "axes.titlesize": 8.1,
            "xtick.labelsize": 6.8,
            "ytick.labelsize": 6.8,
            "legend.fontsize": 6.8,
            "figure.titlesize": 8.5,
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


def save(fig: plt.Figure, stem: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / f"{stem}.png", dpi=450, bbox_inches="tight", facecolor="white")
    fig.savefig(OUT_DIR / f"{stem}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def build_grayscale_previews() -> None:
    """Write grayscale PNG previews for print-readability inspection."""
    GRAYSCALE_DIR.mkdir(parents=True, exist_ok=True)
    for stem in CURRENT_FIGURE_STEMS:
        src = OUT_DIR / f"{stem}.png"
        if not src.exists():
            continue
        Image.open(src).convert("L").save(GRAYSCALE_DIR / src.name)


def apply_clean_axes(ax: plt.Axes, *, axis: str = "x") -> None:
    if axis == "both":
        ax.grid(True, axis="both", color=COLORS["grid"], linewidth=0.48, alpha=0.8)
    else:
        ax.grid(True, axis=axis, color=COLORS["grid"], linewidth=0.48, alpha=0.8)
        ax.grid(False, axis="y" if axis == "x" else "x")
    sns.despine(ax=ax, trim=False)


def add_box(ax, xy, width, height, text, face, edge="#2B2B2B", size=8):
    box = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.018,rounding_size=0.018",
        linewidth=0.85,
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
        mutation_scale=9,
        linewidth=0.95,
        color=color,
        connectionstyle=f"arc3,rad={rad}",
    )
    ax.add_patch(arrow)


def plot_architecture() -> None:
    style()
    fig, ax = plt.subplots(figsize=(6.85, 2.75))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    add_box(ax, (0.035, 0.48), 0.13, 0.18, "Java\nmethod", "#F5F5F5", size=7.3)
    add_box(ax, (0.225, 0.47), 0.18, 0.20, "AST path contexts\n$(x_i, p_i, y_i)$", "#EAF3F8", size=7.1)

    add_box(ax, (0.455, 0.68), 0.19, 0.18, "lexical tokens\n$z_E\\in\\mathbb{E}^{d_e}$", "#E7F1FA", size=6.8)
    add_box(ax, (0.455, 0.28), 0.19, 0.18, "AST path structure\n$z_H\\in\\mathbb{H}^{d_h}_{c}$", "#FFF0E6", size=6.8)
    add_box(ax, (0.705, 0.48), 0.14, 0.18, "product vector\n$z=(z_E,z_H)$", "#F0F7EC", size=6.6)
    add_box(ax, (0.705, 0.15), 0.14, 0.13, "attention +\ndecoder", "#F7F4E8", size=6.4)
    add_box(ax, (0.895, 0.15), 0.085, 0.13, "method-name\nsubtokens", "#F5F5F5", size=5.3)

    add_arrow(ax, (0.165, 0.57), (0.225, 0.57))
    add_arrow(ax, (0.405, 0.58), (0.455, 0.77))
    add_arrow(ax, (0.405, 0.55), (0.455, 0.37))
    add_arrow(ax, (0.645, 0.77), (0.705, 0.61))
    add_arrow(ax, (0.645, 0.37), (0.705, 0.53))
    add_arrow(ax, (0.775, 0.48), (0.775, 0.28))
    add_arrow(ax, (0.845, 0.215), (0.895, 0.215))

    ax.text(0.55, 0.91, "Euclidean lexical factor", ha="center", va="center", fontsize=6.2, color="#333333")
    ax.text(0.55, 0.21, "hyperbolic structural factor", ha="center", va="center", fontsize=6.2, color="#333333")
    ax.text(0.50, 0.055, "Only the AST-path channel is assigned negative curvature.", ha="center", va="bottom", fontsize=6.2, color="#333333")
    save(fig, "figure01_code2hyp_architecture")


def load_postreview_runs(path: Path = POSTREVIEW_RESULT) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(
            "Final factorial benchmark JSON is missing. Run "
            "`scripts/run_code2hyp_postreview_benchmark.sh` before rebuilding "
            f"manuscript result figures. Expected: {path}"
        )
    runs = load(path)
    grouped = group(runs)
    missing = [variant for variant in POSTREVIEW_ORDER if variant not in grouped]
    if missing:
        raise ValueError(
            "Final factorial benchmark JSON is incomplete; missing variants: "
            + ", ".join(missing)
        )
    return runs


def postreview_summary(runs: list[dict]) -> dict[str, dict[str, tuple[float, float, int]]]:
    required_metrics = (
        "validation_f1",
        "validation_structural_spearman",
        "validation_structural_normalized_stress",
        "validation_structural_neighbor_exact_overlap_at_3",
    )
    result: dict[str, dict[str, tuple[float, float, int]]] = {}
    grouped = group(runs)
    for variant in POSTREVIEW_ORDER:
        items = grouped[variant]
        result[variant] = {}
        for metric in required_metrics:
            missing = [item.get("model_seed", "?") for item in items if metric not in item]
            if missing:
                raise ValueError(f"{variant} lacks {metric} for seeds {missing}")
            values = [float(item[metric]) for item in items]
            m, sd = stat(values)
            result[variant][metric] = (m, sd, len(values))
    return result


def plot_postreview_main_results(runs: list[dict]) -> None:
    style()
    summary = postreview_summary(runs)
    panels = (
        ("validation_f1", "A. Downstream prediction", "Target-subtoken F1, %", 100.0, True),
        ("validation_structural_spearman", "B. Rank preservation", "Prefix-trie Spearman", 1.0, True),
        ("validation_structural_normalized_stress", "C. Metric distortion", "Normalized stress", 1.0, False),
        ("validation_structural_neighbor_exact_overlap_at_3", "D. Exact local neighborhoods", "Exact Overlap@3", 1.0, True),
    )
    labels = [POSTREVIEW_LABELS[variant] for variant in POSTREVIEW_ORDER]

    fig, axes = plt.subplots(2, 2, figsize=(7.9, 5.65), constrained_layout=True)
    for ax, (metric, title, xlabel, scale, higher_is_better) in zip(axes.ravel(), panels, strict=True):
        rows: list[dict] = []
        for variant in POSTREVIEW_ORDER:
            value, sd, n = summary[variant][metric]
            code = POSTREVIEW_CODES[variant]
            rows.append(
                {
                    "variant": variant,
                    "code": code,
                    "label": POSTREVIEW_LABELS[variant],
                    "value": value * scale,
                    "sd": sd * scale,
                    "n": n,
                }
            )
        df = pd.DataFrame(rows)
        df["label"] = pd.Categorical(df["label"], categories=labels, ordered=True)
        sns.scatterplot(
            data=df,
            x="value",
            y="label",
            hue="code",
            palette=POSTREVIEW_PALETTE,
            s=55,
            edgecolor="#222222",
            linewidth=0.55,
            legend=False,
            zorder=3,
            ax=ax,
        )
        for row in df.itertuples(index=False):
            y = labels.index(str(row.label))
            ax.errorbar(
                row.value,
                y,
                xerr=row.sd,
                fmt="none",
                ecolor=POSTREVIEW_PALETTE[row.code],
                elinewidth=1.0,
                capsize=2.0,
                capthick=1.0,
                zorder=2,
            )
        ax.set_title(title, loc="left", fontweight="bold")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("")
        if metric == "validation_structural_spearman":
            ax.axvline(0, color="#222222", linewidth=0.7)
        if not higher_is_better:
            ax.text(
                0.03,
                0.06,
                "lower is better",
                transform=ax.transAxes,
                fontsize=6.4,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 1.2},
            )
        apply_clean_axes(ax, axis="x")

    fig.suptitle("Factorial control comparison under the 25,000-example budget", y=1.02)
    save(fig, "figure02_main_results")


def optional_metric_summary(runs: list[dict], metric: str) -> list[dict]:
    rows: list[dict] = []
    for variant, items in group(runs).items():
        if variant not in POSTREVIEW_ORDER:
            continue
        values = [
            float(item[metric])
            for item in items
            if metric in item and item[metric] is not None and math.isfinite(float(item[metric]))
        ]
        if not values:
            continue
        m, sd = stat(values)
        code = POSTREVIEW_CODES[variant]
        rows.append(
            {
                "variant": variant,
                "code": code,
                "label": POSTREVIEW_LABELS[variant],
                "value": m,
                "sd": sd,
                "n": len(values),
            }
        )
    rows.sort(key=lambda row: POSTREVIEW_ORDER.index(row["variant"]))
    return rows


def plot_metric_panel(ax: plt.Axes, rows: list[dict], title: str, xlabel: str) -> None:
    if not rows:
        ax.text(0.5, 0.5, "not available", ha="center", va="center", fontsize=8)
        ax.set_axis_off()
        return
    labels = [row["label"] for row in rows]
    df = pd.DataFrame(rows)
    df["label"] = pd.Categorical(df["label"], categories=labels, ordered=True)
    sns.barplot(
        data=df,
        x="value",
        y="label",
        hue="code",
        palette=POSTREVIEW_PALETTE,
        dodge=False,
        edgecolor="#222222",
        linewidth=0.45,
        legend=False,
        ax=ax,
    )
    for row in df.itertuples(index=False):
        y = labels.index(str(row.label))
        ax.errorbar(
            row.value,
            y,
            xerr=row.sd,
            fmt="none",
            ecolor="#222222",
            elinewidth=0.9,
            capsize=1.8,
            capthick=0.9,
            zorder=3,
        )
    ax.set_title(title, loc="left", fontweight="bold")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("")
    apply_clean_axes(ax, axis="x")


def plot_postreview_geometry_diagnostics(runs: list[dict]) -> None:
    style()
    fig, axes = plt.subplots(2, 2, figsize=(7.9, 5.55), constrained_layout=True)
    panels = (
        (
            "validation_poincare_frechet_residual_mean",
            "A. Karcher residual",
            r"$\|\sum_i \alpha_i \log_\mu(h_i)\|$",
        ),
        (
            "validation_poincare_context_radius_ratio_max",
            "B. Radius utilization",
            r"max $\sqrt{c}\|h_i\|$",
        ),
        (
            "validation_poincare_context_near_boundary_rate",
            "C. Near-boundary rate",
            "fraction of context points",
        ),
        ("curvature", "D. Final curvature", "curvature parameter c"),
    )
    for ax, (metric, title, xlabel) in zip(axes.ravel(), panels, strict=True):
        plot_metric_panel(ax, optional_metric_summary(runs, metric), title, xlabel)
    fig.suptitle("Geometry diagnostics for product-manifold variants", y=1.02)
    save(fig, "figure03_geometry_diagnostics")


def load_distance_level_rows(runs: list[dict]) -> dict[str, list[dict[str, float]]]:
    per_variant_level: dict[str, dict[float, list[float]]] = defaultdict(lambda: defaultdict(list))
    for run in runs:
        variant = str(run["variant"])
        summary = run.get("validation_structural_prefix_distance_level_summary")
        if summary is None:
            raise ValueError("benchmark lacks validation_structural_prefix_distance_level_summary")
        for item in summary:
            level = float(item["target_distance"])
            per_variant_level[variant][level].append(float(item["model_distance_mean"]))

    rows: dict[str, list[dict[str, float]]] = {}
    for variant, by_level in per_variant_level.items():
        variant_rows = []
        for level in sorted(by_level):
            values = by_level[level]
            variant_rows.append(
                {
                    "target_distance": level,
                    "model_distance_mean": mean(values),
                    "model_distance_sd": stdev(values) if len(values) > 1 else 0.0,
                }
            )
        rows[variant] = variant_rows
    return rows


def plot_postreview_distance_levels(runs: list[dict]) -> None:
    style()
    rows_by_variant = load_distance_level_rows(runs)
    fig, ax = plt.subplots(figsize=(8.6, 4.45), constrained_layout=True)

    line_styles = {
        "B39": ("#4D4D4D", "o", "-"),
        "B47": ("#CC79A7", "P", "--"),
        "B50": ("#999999", "v", ":"),
        "B51": ("#7F3C8D", "*", "--"),
        "B48": ("#E69F00", "X", ":"),
        "B49": ("#56B4E9", "^", "--"),
        "B36": ("#0072B2", "s", "-"),
        "B44": ("#D55E00", "D", "-."),
    }
    labels = {
        "B39": "B39 Euclidean baseline",
        "B47": "B47 Euclidean distance loss",
        "B50": "B50 L1 baseline",
        "B51": "B51 L1 distance loss",
        "B48": "B48 product, no structural loss",
        "B49": "B49 near-Euclidean same code path",
        "B36": "B36 downstream-oriented Code2Hyp",
        "B44": "B44 structure-oriented Code2Hyp",
    }

    for variant in POSTREVIEW_ORDER:
        if variant not in rows_by_variant:
            continue
        code = POSTREVIEW_CODES[variant]
        color, marker, linestyle = line_styles[code]
        rows = rows_by_variant[variant]
        x_values = [row["target_distance"] for row in rows]
        y_values = [row["model_distance_mean"] for row in rows]
        y_sd = [row["model_distance_sd"] for row in rows]
        lower = [max(0.0, y - sd) for y, sd in zip(y_values, y_sd, strict=True)]
        upper = [y + sd for y, sd in zip(y_values, y_sd, strict=True)]
        ax.plot(
            x_values,
            y_values,
            marker=marker,
            linestyle=linestyle,
            linewidth=1.35,
            markersize=4.1,
            color=color,
            label=labels[code],
        )
        ax.fill_between(x_values, lower, upper, color=color, alpha=0.09, linewidth=0.0)

    ax.set_title("Learned structural distance by prefix-trie target level")
    ax.set_xlabel("Prefix-trie path distance")
    ax.set_ylabel("Mean learned distance")
    ax.grid(axis="both", color=COLORS["grid"], linewidth=0.48, alpha=0.8)
    sns.despine(ax=ax, trim=False)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False, fontsize=6.3, borderaxespad=0.0)
    save(fig, "figure04_distance_levels")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build publication figures for the Code2Hyp manuscript.",
    )
    parser.add_argument(
        "--grayscale-preview",
        action="store_true",
        help="also write grayscale PNG previews under build/grayscale_previews",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    postreview_runs = load_postreview_runs()
    plot_architecture()
    plot_postreview_main_results(postreview_runs)
    plot_postreview_geometry_diagnostics(postreview_runs)
    plot_postreview_distance_levels(postreview_runs)
    print(f"Wrote figures to {OUT_DIR}")
    if args.grayscale_preview:
        build_grayscale_previews()
        print(f"Wrote grayscale previews to {GRAYSCALE_DIR}")


if __name__ == "__main__":
    main()
