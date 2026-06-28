from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.summarize_code2hyp_paired_variant_deltas import paired_delta_rows


METRIC_LABELS = {
    "validation_f1": "Target-subtoken F1",
    "validation_fixed_top3_f1": "Fixed top-3 F1",
    "validation_structural_spearman": "Prefix-tree Spearman",
    "validation_structural_edit_spearman": "Edit-distance Spearman",
    "validation_structural_jaccard_spearman": "Jaccard-bigram Spearman",
    "validation_structural_normalized_stress": "Normalized stress",
    "validation_method_aggregate_spearman": "Method-aggregate Spearman",
    "validation_method_aggregate_normalized_stress": "Method-aggregate stress",
    "validation_method_transport_spearman": "Method-transport Spearman",
    "validation_method_transport_normalized_stress": "Method-transport stress",
    "validation_method_transport_prefix_spearman": "Transport prefix Spearman",
    "validation_method_transport_prefix_normalized_stress": "Transport prefix stress",
    "validation_method_aggregate_prefix_spearman": "Aggregate prefix Spearman",
    "validation_method_aggregate_prefix_normalized_stress": "Aggregate prefix stress",
    "validation_method_transport_edit_spearman": "Transport edit Spearman",
    "validation_method_transport_edit_normalized_stress": "Transport edit stress",
    "validation_method_aggregate_edit_spearman": "Aggregate edit Spearman",
    "validation_method_aggregate_edit_normalized_stress": "Aggregate edit stress",
    "validation_method_transport_jaccard_spearman": "Transport Jaccard Spearman",
    "validation_method_transport_jaccard_normalized_stress": "Transport Jaccard stress",
    "validation_method_aggregate_jaccard_spearman": "Aggregate Jaccard Spearman",
    "validation_method_aggregate_jaccard_normalized_stress": "Aggregate Jaccard stress",
    "validation_structural_neighbor_overlap_at_3": "Neighbor overlap@3",
}

LOWER_IS_BETTER = {
    "validation_structural_normalized_stress",
    "validation_method_aggregate_normalized_stress",
    "validation_method_transport_normalized_stress",
    "validation_method_transport_prefix_normalized_stress",
    "validation_method_aggregate_prefix_normalized_stress",
    "validation_method_transport_edit_normalized_stress",
    "validation_method_aggregate_edit_normalized_stress",
    "validation_method_transport_jaccard_normalized_stress",
    "validation_method_aggregate_jaccard_normalized_stress",
}


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def _load_runs(paths: list[Path]) -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        runs.extend(payload.get("runs", []))
    return runs


def _is_improvement(metric: str, delta: float) -> bool:
    return delta < 0 if metric in LOWER_IS_BETTER else delta > 0


def plot_paired_variant_deltas(
    inputs: list[Path],
    *,
    baseline: str,
    candidate: str,
    output_prefix: Path,
) -> tuple[Path, Path]:
    runs = _load_runs(inputs)
    rows, matched_seeds, _, _ = paired_delta_rows(runs, baseline=baseline, candidate=candidate)
    if not rows:
        raise ValueError("No matched metric deltas found for the requested variants.")

    labels = [METRIC_LABELS.get(str(row["metric"]), str(row["metric"])) for row in rows]
    deltas = [float(row["delta_mean"]) for row in rows]
    errors = [float(row["delta_sd"]) for row in rows]
    metrics = [str(row["metric"]) for row in rows]
    colors = ["#0072B2" if _is_improvement(metric, delta) else "#D55E00" for metric, delta in zip(metrics, deltas)]

    sns.set_theme(style="ticks", context="paper", font_scale=1.05)
    fig_height = max(3.0, 0.42 * len(rows) + 1.1)
    fig, ax = plt.subplots(figsize=(7.2, fig_height), constrained_layout=True)

    y_positions = list(range(len(rows)))
    ax.barh(
        y_positions,
        deltas,
        xerr=errors,
        color=colors,
        edgecolor="#222222",
        linewidth=0.5,
        capsize=3,
        alpha=0.92,
    )
    ax.axvline(0.0, color="#333333", linewidth=0.9)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("Mean paired delta: candidate - baseline")
    ax.set_title("B86 method-transport regularizer against B85 relation-conditioned baseline", fontsize=10)
    ax.text(
        0.99,
        0.02,
        f"matched seeds: {len(matched_seeds)}; error bars: sd\nstress metrics: lower is better",
        ha="right",
        va="bottom",
        transform=ax.transAxes,
        fontsize=8,
        color="#444444",
    )
    ax.grid(True, axis="x", color="#DDDDDD", linewidth=0.6)
    ax.grid(False, axis="y")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    label_positions = [
        delta + ((error + 0.006) if delta >= 0 else -(error + 0.006))
        for delta, error in zip(deltas, errors)
    ]
    x_min = min([0.0, *[delta - error for delta, error in zip(deltas, errors)], *label_positions])
    x_max = max([0.0, *[delta + error for delta, error in zip(deltas, errors)], *label_positions])
    x_pad = max((x_max - x_min) * 0.08, 0.025)
    ax.set_xlim(x_min - x_pad, x_max + x_pad)
    for y, delta, error in zip(y_positions, deltas, errors):
        ha = "left" if delta >= 0 else "right"
        offset = (error + 0.006) if delta >= 0 else -(error + 0.006)
        ax.text(delta + offset, y, f"{delta:+.3f}", va="center", ha=ha, fontsize=8)

    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    png_path = output_prefix.with_suffix(".png")
    pdf_path = output_prefix.with_suffix(".pdf")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return png_path, pdf_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot paired metric deltas between two Code2Hyp variants.")
    parser.add_argument("--inputs", nargs="+", type=Path, required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    png_path, pdf_path = plot_paired_variant_deltas(
        [_resolve(path) for path in args.inputs],
        baseline=args.baseline,
        candidate=args.candidate,
        output_prefix=_resolve(args.output_prefix),
    )
    print(f"Wrote {png_path}")
    print(f"Wrote {pdf_path}")


if __name__ == "__main__":
    main()
