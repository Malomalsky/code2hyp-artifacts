from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


CODE2SEQ_JAVA_SMALL_BASELINES = (
    ("ConvAttention", 50.25, 24.62, 33.05, "Allamanis et al. 2016; code2seq Table 1"),
    ("Paths+CRFs", 8.39, 5.63, 6.74, "Alon et al. 2018; code2seq Table 1"),
    ("code2vec", 18.51, 18.74, 18.62, "Alon et al. 2019; code2seq Table 1"),
    ("2-layer BiLSTM, no token splitting", 32.40, 20.40, 25.03, "code2seq Table 1"),
    ("2-layer BiLSTM", 42.63, 29.97, 35.20, "code2seq Table 1"),
    ("TreeLSTM", 40.02, 31.84, 35.46, "Tai et al. 2015; code2seq Table 1"),
    ("Transformer", 38.13, 26.70, 31.41, "Vaswani et al. 2017; code2seq Table 1"),
    ("code2seq", 50.64, 37.40, 43.02, "Alon et al. 2019; code2seq Table 1"),
)


VARIANT_LABELS = {
    "B36_code2hyp_product_frechet_neighbor": "Code2Hyp B36 product-Frechet + neighbor",
    "B39_code2vec_context_transform_baseline": "B39 matched code2vec-style baseline",
    "B46_code2vec_context_transform_neighbor_control": "B46 Euclidean context-transform + neighbor",
    "B47_code2vec_context_transform_distance_control": "B47 Euclidean context-transform + distance loss",
    "B50_code2vec_context_transform_l1_baseline": "B50 L1 structural-distance baseline",
    "B51_code2vec_context_transform_l1_distance_control": "B51 L1 structural-distance + distance loss",
    "B48_code2hyp_context_transform_product_bias_no_struct": "B48 hyperbolic product-bias without structural loss",
    "B49_code2hyp_context_transform_product_bias_near_euclidean": "B49 same code path, near-Euclidean curvature",
    "B40_code2hyp_context_transform_frechet": "Code2Hyp B40 context-transform + Frechet",
    "B44_code2hyp_context_transform_product_bias_frechet": "Code2Hyp B44 structural-bias attention",
    "B60_code2hyp_context_transform_branch_sequence_product_bias_frechet": "B60 branch-sequence product manifold",
    "B62_code2hyp_context_transform_branch_sequence_product_bias_multi_metric_frechet": (
        "B62 branch-sequence product manifold + multi-metric loss"
    ),
    "B63_code2hyp_context_transform_product_bias_multi_metric_frechet": (
        "B63 product-bias manifold + multi-metric loss"
    ),
    "B64_code2vec_context_transform_multi_metric_control": "B64 Euclidean context-transform + multi-metric loss",
    "B65_code2vec_context_transform_l1_multi_metric_control": "B65 L1 context-transform + multi-metric loss",
}


def _percent(value: float) -> float:
    return 100.0 * value


def _mean_sd(values: list[float]) -> str:
    if not values:
        return "n/a"
    if len(values) == 1:
        return f"{_percent(values[0]):.2f}"
    return f"{_percent(mean(values)):.2f} +/- {_percent(stdev(values)):.2f}"


def _mean_float(values: list[float]) -> str:
    if not values:
        return "n/a"
    if len(values) == 1:
        return f"{values[0]:.4f}"
    return f"{mean(values):.4f} +/- {stdev(values):.4f}"


def _optional_mean_float(runs: list[dict[str, Any]], key: str) -> str:
    values = [float(run[key]) for run in runs if key in run and run[key] is not None]
    return _mean_float(values)


def _group_runs(runs: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for run in runs:
        grouped[run["variant"]].append(run)
    return dict(grouped)


def build_paper_benchmark_markdown(result: dict[str, Any]) -> str:
    dataset = result.get("dataset", {})
    evaluation = result.get("evaluation", {})
    training = result.get("training", {})
    grouped = _group_runs(result.get("runs", []))

    lines = [
        "# Code2Hyp Java-small benchmark summary",
        "",
        "## Scope",
        "",
        "This report separates external literature numbers from local Code2Hyp runs.",
        "The external numbers are full Java-small literature baselines from code2seq Table 1.",
        "The local numbers are controlled Code2Hyp runs with the training budget recorded below.",
        "",
        "## Local run metadata",
        "",
        f"- evaluation split: `{evaluation.get('split', 'unknown')}`",
        f"- train records used: `{dataset.get('train_records', 'unknown')}`",
        f"- evaluation records loaded: `{dataset.get('validation_records_loaded', 'unknown')}`",
        (
            "- evaluation records after known-target filtering: "
            f"`{dataset.get('validation_records_after_known_target_filter', 'unknown')}`"
        ),
        f"- epochs: `{training.get('epochs', 'unknown')}`",
        f"- batch size: `{training.get('batch_size', 'unknown')}`",
        f"- seeds: `{training.get('model_seeds', 'unknown')}`",
        f"- metric: `{training.get('metric', 'unknown')}`",
        "",
        "## External Java-small literature baselines",
        "",
        "| Model | Precision | Recall | F1 | Source |",
        "|---|---:|---:|---:|---|",
    ]
    for name, precision, recall, f1, source in CODE2SEQ_JAVA_SMALL_BASELINES:
        lines.append(f"| {name} | {precision:.2f} | {recall:.2f} | {f1:.2f} | {source} |")

    lines.extend(
        [
            "",
            "## Local Code2Hyp controlled results",
            "",
            (
                "| Variant | Precision | Recall | F1 | Structural Spearman | "
                "Edit Spearman | Jaccard Spearman | Normalized stress | Edit stress | Jaccard stress | "
                "Overlap@3 | Exact Overlap@3 | Karcher residual | "
                "Radius max | Near-boundary rate | Curvature | rho | n seeds |"
            ),
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for variant in sorted(grouped):
        runs = grouped[variant]
        label = VARIANT_LABELS.get(variant, variant)
        precision = _mean_sd([float(run["validation_precision"]) for run in runs])
        recall = _mean_sd([float(run["validation_recall"]) for run in runs])
        f1 = _mean_sd([float(run["validation_f1"]) for run in runs])
        spearman = _mean_float([float(run["validation_structural_spearman"]) for run in runs])
        edit_spearman = _optional_mean_float(runs, "validation_structural_edit_spearman")
        jaccard_spearman = _optional_mean_float(runs, "validation_structural_jaccard_spearman")
        stress = _optional_mean_float(runs, "validation_structural_normalized_stress")
        edit_stress = _optional_mean_float(runs, "validation_structural_edit_normalized_stress")
        jaccard_stress = _optional_mean_float(runs, "validation_structural_jaccard_normalized_stress")
        overlap_at_3 = _mean_float([float(run["validation_structural_neighbor_overlap_at_3"]) for run in runs])
        exact_overlap_at_3 = _optional_mean_float(runs, "validation_structural_neighbor_exact_overlap_at_3")
        karcher_residual = _optional_mean_float(runs, "validation_poincare_frechet_residual_mean")
        radius_max = _optional_mean_float(runs, "validation_poincare_context_radius_ratio_max")
        near_boundary_rate = _optional_mean_float(runs, "validation_poincare_context_near_boundary_rate")
        curvature = _mean_float([float(run["curvature"]) for run in runs if math.isfinite(float(run["curvature"]))])
        rho = _mean_float([float(run["product_attention_bias_weight"]) for run in runs])
        lines.append(
            f"| {label} | {precision} | {recall} | {f1} | {spearman} | "
            f"{edit_spearman} | {jaccard_spearman} | {stress} | {edit_stress} | {jaccard_stress} | "
            f"{overlap_at_3} | {exact_overlap_at_3} | {karcher_residual} | {radius_max} | "
            f"{near_boundary_rate} | {curvature} | {rho} | {len(runs)} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "Do not compare the local subset run as a direct SOTA claim against full-budget literature models.",
            "Use it to validate the instrument and decide whether a full Java-small run is worth executing.",
            "A manuscript-level claim requires fixed hyperparameters, `--eval-split test`, full or explicitly",
            "bounded training budget, paired seed analysis, and no post-hoc model selection on the test split.",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a paper-oriented Code2Hyp benchmark table.")
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = json.loads(args.input.read_text(encoding="utf-8"))
    markdown = build_paper_benchmark_markdown(result)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(markdown + "\n", encoding="utf-8")
        print(f"Wrote {args.output}")
    else:
        print(markdown)


if __name__ == "__main__":
    main()
