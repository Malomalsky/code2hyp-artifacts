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


VARIANT_ORDER = (
    "B39_code2vec_context_transform_baseline",
    "B47_code2vec_context_transform_distance_control",
    "B50_code2vec_context_transform_l1_baseline",
    "B51_code2vec_context_transform_l1_distance_control",
    "B48_code2hyp_context_transform_product_bias_no_struct",
    "B49_code2hyp_context_transform_product_bias_near_euclidean",
    "B36_code2hyp_product_frechet_neighbor",
    "B44_code2hyp_context_transform_product_bias_frechet",
)

VARIANT_LABELS = {
    "B39_code2vec_context_transform_baseline": "B39 Euclidean baseline",
    "B47_code2vec_context_transform_distance_control": "B47 Euclidean distance loss",
    "B50_code2vec_context_transform_l1_baseline": "B50 L1 baseline",
    "B51_code2vec_context_transform_l1_distance_control": "B51 L1 distance loss",
    "B48_code2hyp_context_transform_product_bias_no_struct": "B48 product, no structural loss",
    "B49_code2hyp_context_transform_product_bias_near_euclidean": "B49 near-Euclidean same code path",
    "B36_code2hyp_product_frechet_neighbor": "B36 downstream-oriented Code2Hyp",
    "B44_code2hyp_context_transform_product_bias_frechet": "B44 structure-oriented Code2Hyp",
}

VARIANT_STYLES = {
    "B39_code2vec_context_transform_baseline": ("#4D4D4D", "o", "-"),
    "B47_code2vec_context_transform_distance_control": ("#CC79A7", "P", "--"),
    "B50_code2vec_context_transform_l1_baseline": ("#999999", "v", ":"),
    "B51_code2vec_context_transform_l1_distance_control": ("#7F3C8D", "*", "--"),
    "B48_code2hyp_context_transform_product_bias_no_struct": ("#E69F00", "X", ":"),
    "B49_code2hyp_context_transform_product_bias_near_euclidean": ("#56B4E9", "^", "--"),
    "B36_code2hyp_product_frechet_neighbor": ("#0072B2", "s", "-"),
    "B44_code2hyp_context_transform_product_bias_frechet": ("#D55E00", "D", "-."),
}


def _load_distance_level_rows(path: Path) -> dict[str, list[dict[str, float]]]:
    result = json.loads(path.read_text(encoding="utf-8"))
    per_variant_level: dict[str, dict[float, list[float]]] = defaultdict(lambda: defaultdict(list))
    pair_counts: dict[str, dict[float, list[int]]] = defaultdict(lambda: defaultdict(list))

    for run in result.get("runs", []):
        variant = str(run["variant"])
        summary = run.get("validation_structural_prefix_distance_level_summary")
        if summary is None:
            raise ValueError(
                "input result does not contain validation_structural_prefix_distance_level_summary; "
                "rerun the benchmark with the post-fix code"
            )
        for item in summary:
            level = float(item["target_distance"])
            per_variant_level[variant][level].append(float(item["model_distance_mean"]))
            pair_counts[variant][level].append(int(item["pair_count"]))

    rows: dict[str, list[dict[str, float]]] = {}
    for variant, by_level in per_variant_level.items():
        variant_rows = []
        for level in sorted(by_level):
            values = by_level[level]
            counts = pair_counts[variant][level]
            variant_rows.append(
                {
                    "target_distance": level,
                    "model_distance_mean": mean(values),
                    "model_distance_sd": stdev(values) if len(values) > 1 else 0.0,
                    "seed_count": float(len(values)),
                    "pair_count_mean": mean(counts),
                }
            )
        rows[variant] = variant_rows
    return rows


def plot_distance_levels(input_path: Path, output_prefix: Path) -> None:
    rows_by_variant = _load_distance_level_rows(input_path)
    active_variants = [variant for variant in VARIANT_ORDER if variant in rows_by_variant]
    if not active_variants:
        raise ValueError("input result does not contain known Code2Hyp variants")

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "legend.fontsize": 7.5,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
        }
    )
    fig, ax = plt.subplots(figsize=(7.2, 4.2), constrained_layout=True)

    for variant in active_variants:
        color, marker, linestyle = VARIANT_STYLES[variant]
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
            linewidth=1.45,
            markersize=4.2,
            color=color,
            label=VARIANT_LABELS[variant],
        )
        ax.fill_between(x_values, lower, upper, color=color, alpha=0.10, linewidth=0.0)

    ax.set_title("Learned structural distance by prefix-trie target level")
    ax.set_xlabel("Prefix-trie path distance")
    ax.set_ylabel("Mean learned distance")
    ax.grid(axis="both", alpha=0.22, linewidth=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper left", frameon=False)

    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_prefix.with_suffix(".png"), dpi=220)
    fig.savefig(output_prefix.with_suffix(".pdf"))
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    args = parser.parse_args()

    plot_distance_levels(args.input, args.output_prefix)


if __name__ == "__main__":
    main()
