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
    ("Original", PROJECT_ROOT / "outputs/code2hyp_structural_retrieval_original_1024_3epochs_3seeds.json"),
    (
        "Structural only",
        PROJECT_ROOT / "outputs/code2hyp_structural_retrieval_structural_only_1024_3epochs_3seeds.json",
    ),
)
DEFAULT_OUTPUT_PREFIX = PROJECT_ROOT / "figures/code2hyp_structural_retrieval_overlap"

VARIANT_ORDER = (
    "B1_euclidean",
    "B4_hyperbolic_code2vec",
    "B8_hyperbolic_frechet_code2vec",
    "B29_hyperbolic_path_dual_attention_mp_separated",
    "B31_hyperbolic_path_dual_attention_mp_soft_rank",
    "B34_hyperbolic_path_dual_attention_mp_adaptive_rank",
)

VARIANT_LABELS = {
    "B1_euclidean": "B1 Euclidean",
    "B4_hyperbolic_code2vec": "B4 Hyp.",
    "B8_hyperbolic_frechet_code2vec": "B8 Frechet",
    "B29_hyperbolic_path_dual_attention_mp_separated": "B29 dual",
    "B31_hyperbolic_path_dual_attention_mp_soft_rank": "B31 soft-rank",
    "B34_hyperbolic_path_dual_attention_mp_adaptive_rank": "B34 adaptive",
}

VARIANT_COLORS = {
    "B1_euclidean": "#777777",
    "B4_hyperbolic_code2vec": "#0072B2",
    "B8_hyperbolic_frechet_code2vec": "#009E73",
    "B29_hyperbolic_path_dual_attention_mp_separated": "#882255",
    "B31_hyperbolic_path_dual_attention_mp_soft_rank": "#AA4499",
    "B34_hyperbolic_path_dual_attention_mp_adaptive_rank": "#117733",
}

METRICS = (
    ("validation_f1_mean", "Target-subtoken F1"),
    ("validation_structural_spearman_mean", "AST-distance Spearman"),
    ("validation_structural_neighbor_overlap_at_3_mean", "Local AST-neighbor Overlap@3"),
)


def _load_summary(path: Path) -> dict[str, dict[str, float | int | str]]:
    result = json.loads(path.read_text(encoding="utf-8"))
    return {str(row["variant"]): row for row in summarize_pilot_runs(result)}


def _metric_values(
    summaries: dict[str, dict[str, float | int | str]],
    metric_key: str,
) -> tuple[list[str], list[float], list[str]]:
    labels: list[str] = []
    values: list[float] = []
    colors: list[str] = []
    for variant in VARIANT_ORDER:
        row = summaries.get(variant)
        if row is None or metric_key not in row:
            continue
        labels.append(VARIANT_LABELS[variant])
        values.append(float(row[metric_key]))
        colors.append(VARIANT_COLORS[variant])
    return labels, values, colors


def plot_structural_retrieval(
    inputs: tuple[tuple[str, Path], ...],
    output_prefix: Path,
) -> None:
    if len(inputs) != 2:
        raise ValueError("structural retrieval plot expects exactly two input files")
    summaries = {regime: _load_summary(path) for regime, path in inputs}

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 8.8,
            "axes.titlesize": 10.0,
            "axes.labelsize": 8.8,
            "figure.titlesize": 12.0,
            "xtick.labelsize": 8.0,
            "ytick.labelsize": 8.0,
        }
    )
    fig, axes = plt.subplots(len(METRICS), len(inputs), figsize=(8.6, 7.2), constrained_layout=True)
    fig.suptitle("Code2Hyp local AST-neighborhood diagnostics")

    for column_index, (regime, _) in enumerate(inputs):
        for row_index, (metric_key, metric_label) in enumerate(METRICS):
            ax = axes[row_index, column_index]
            labels, values, colors = _metric_values(summaries[regime], metric_key)
            y_positions = list(range(len(labels)))
            ax.barh(y_positions, values, color=colors, alpha=0.88, edgecolor="#222222", linewidth=0.4)
            ax.set_yticks(y_positions)
            ax.set_yticklabels(labels if column_index == 0 else [])
            ax.invert_yaxis()
            ax.grid(axis="x", alpha=0.24, linewidth=0.6)
            ax.set_title(regime if row_index == 0 else "")
            ax.set_xlabel(metric_label)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            if "spearman" in metric_key:
                ax.axvline(0.0, color="#333333", linestyle="--", linewidth=0.8)
            for y_position, value in zip(y_positions, values, strict=True):
                if value < 0:
                    ax.text(value / 2.0, y_position, f"{value:+.3f}", va="center", ha="center", color="white")
                    continue
                ax.text(value + 0.01, y_position, f"{value:.3f}", va="center", ha="left")

    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_prefix.with_suffix(".png"), dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(output_prefix.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot Code2Hyp structural retrieval diagnostics.")
    parser.add_argument(
        "--input",
        action="append",
        nargs=2,
        metavar=("REGIME", "PATH"),
        help="Regime label and pilot JSON path. Must be supplied exactly twice.",
    )
    parser.add_argument("--output-prefix", type=Path, default=DEFAULT_OUTPUT_PREFIX)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    inputs = tuple((label, Path(path)) for label, path in args.input) if args.input else DEFAULT_INPUTS
    plot_structural_retrieval(inputs, args.output_prefix)
    print(f"Wrote {args.output_prefix.with_suffix('.png')}")
    print(f"Wrote {args.output_prefix.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
