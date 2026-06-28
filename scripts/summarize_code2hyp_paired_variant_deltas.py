from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, stdev
from typing import Any


METRICS = (
    "validation_f1",
    "validation_fixed_top3_f1",
    "validation_structural_spearman",
    "validation_structural_edit_spearman",
    "validation_structural_jaccard_spearman",
    "validation_structural_normalized_stress",
    "validation_method_aggregate_spearman",
    "validation_method_aggregate_normalized_stress",
    "validation_method_transport_spearman",
    "validation_method_transport_normalized_stress",
    "validation_method_transport_prefix_spearman",
    "validation_method_transport_prefix_normalized_stress",
    "validation_method_aggregate_prefix_spearman",
    "validation_method_aggregate_prefix_normalized_stress",
    "validation_method_transport_edit_spearman",
    "validation_method_transport_edit_normalized_stress",
    "validation_method_aggregate_edit_spearman",
    "validation_method_aggregate_edit_normalized_stress",
    "validation_method_transport_jaccard_spearman",
    "validation_method_transport_jaccard_normalized_stress",
    "validation_method_aggregate_jaccard_spearman",
    "validation_method_aggregate_jaccard_normalized_stress",
    "validation_structural_neighbor_overlap_at_3",
)


def _load_runs(paths: list[Path]) -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        runs.extend(payload.get("runs", []))
    return runs


def _mean_sd(values: list[float]) -> tuple[float, float]:
    return mean(values), stdev(values) if len(values) > 1 else 0.0


def _format_seed_list(seeds: set[int]) -> str:
    return ", ".join(str(seed) for seed in sorted(seeds)) if seeds else "none"


def paired_delta_rows(
    runs: list[dict[str, Any]],
    baseline: str,
    candidate: str,
) -> tuple[list[dict[str, float | int | str]], set[int], set[int], set[int]]:
    baseline_by_seed = {
        int(run["model_seed"]): run for run in runs if str(run.get("variant")) == baseline
    }
    candidate_by_seed = {
        int(run["model_seed"]): run for run in runs if str(run.get("variant")) == candidate
    }
    matched_seeds = set(baseline_by_seed) & set(candidate_by_seed)
    rows: list[dict[str, float | int | str]] = []
    for metric in METRICS:
        deltas = [
            float(candidate_by_seed[seed][metric]) - float(baseline_by_seed[seed][metric])
            for seed in sorted(matched_seeds)
            if metric in candidate_by_seed[seed] and metric in baseline_by_seed[seed]
        ]
        if not deltas:
            continue
        metric_mean, metric_sd = _mean_sd(deltas)
        rows.append(
            {
                "metric": metric,
                "pairs": len(deltas),
                "delta_mean": metric_mean,
                "delta_sd": metric_sd,
                "delta_min": min(deltas),
                "delta_max": max(deltas),
            }
        )
    return rows, matched_seeds, set(baseline_by_seed) - matched_seeds, set(candidate_by_seed) - matched_seeds


def render_markdown(
    rows: list[dict[str, float | int | str]],
    inputs: list[Path],
    baseline: str,
    candidate: str,
    matched_seeds: set[int],
    unmatched_baseline_seeds: set[int],
    unmatched_candidate_seeds: set[int],
) -> str:
    lines = [
        "# Code2Hyp paired variant deltas",
        "",
        f"Baseline: `{baseline}`",
        "",
        f"Candidate: `{candidate}`",
        "",
        f"Matched seeds: `{_format_seed_list(matched_seeds)}`",
        "",
        f"Unmatched baseline seeds ignored: `{_format_seed_list(unmatched_baseline_seeds)}`",
        "",
        f"Unmatched candidate seeds ignored: `{_format_seed_list(unmatched_candidate_seeds)}`",
        "",
        "Inputs:",
        "",
    ]
    for path in inputs:
        lines.append(f"- `{path}`")
    lines.extend(
        [
            "",
            "Deltas are computed as `candidate - baseline` on matched seeds only.",
            "",
            "| Metric | Pairs | Delta mean +- sd | Delta min | Delta max |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["metric"]),
                    str(int(row["pairs"])),
                    f"{float(row['delta_mean']):.4f} +- {float(row['delta_sd']):.4f}",
                    f"{float(row['delta_min']):.4f}",
                    f"{float(row['delta_max']):.4f}",
                ]
            )
            + " |"
        )
    lines.append("")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize paired metric deltas between two Code2Hyp variants.")
    parser.add_argument("--inputs", nargs="+", type=Path, required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    runs = _load_runs(args.inputs)
    rows, matched, unmatched_baseline, unmatched_candidate = paired_delta_rows(
        runs,
        baseline=args.baseline,
        candidate=args.candidate,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        render_markdown(
            rows,
            inputs=args.inputs,
            baseline=args.baseline,
            candidate=args.candidate,
            matched_seeds=matched,
            unmatched_baseline_seeds=unmatched_baseline,
            unmatched_candidate_seeds=unmatched_candidate,
        ),
        encoding="utf-8",
    )
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
