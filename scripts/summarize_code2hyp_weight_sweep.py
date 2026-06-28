from __future__ import annotations

import argparse
import json
from collections import defaultdict
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


def _mean_sd(values: list[float]) -> tuple[float, float]:
    return mean(values), stdev(values) if len(values) > 1 else 0.0


def _load_runs(paths: list[Path]) -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        runs.extend(payload.get("runs", []))
    return runs


def summarize_runs(runs: list[dict[str, Any]]) -> list[dict[str, float | int | str | None]]:
    grouped: dict[tuple[str, str, float], list[dict[str, Any]]] = defaultdict(list)
    for run in runs:
        key = (
            str(run.get("variant", "unknown")),
            str(run.get("structural_regularizer", "unknown")),
            float(run["structural_loss_weight"]),
        )
        grouped[key].append(run)

    rows: list[dict[str, float | int | str | None]] = []
    for (variant, regularizer, weight), items in sorted(grouped.items()):
        row: dict[str, float | int | str | None] = {
            "variant": variant,
            "structural_regularizer": regularizer,
            "structural_loss_weight": weight,
            "runs": len(items),
        }
        for metric in METRICS:
            values = [float(item[metric]) for item in items if metric in item]
            if values:
                metric_mean, metric_sd = _mean_sd(values)
                row[f"{metric}_mean"] = metric_mean
                row[f"{metric}_sd"] = metric_sd
            else:
                row[f"{metric}_mean"] = None
                row[f"{metric}_sd"] = None
        rows.append(row)
    return rows


def _format_metric_cell(row: dict[str, float | int | str | None], metric: str) -> str:
    metric_mean = row[f"{metric}_mean"]
    metric_sd = row[f"{metric}_sd"]
    if metric_mean is None or metric_sd is None:
        return "n/a"
    return f"{float(metric_mean):.4f} +- {float(metric_sd):.4f}"


def render_markdown(rows: list[dict[str, float | int | str | None]], inputs: list[Path]) -> str:
    lines = [
        "# Code2Hyp structural-weight sweep summary",
        "",
        "Inputs:",
        "",
    ]
    for path in inputs:
        lines.append(f"- `{path}`")
    lines.extend(
        [
            "",
            "Metrics are grouped by `variant`, `structural_regularizer`, and `structural_loss_weight`; values are reported as mean +- standard deviation over seeds.",
            "",
            "| Variant | Regularizer | Weight | Runs | " + " | ".join(METRICS) + " |",
            "|---|---|---:|---:|" + "|".join("---:" for _ in METRICS) + "|",
        ]
    )
    for row in rows:
        cells = [
            str(row["variant"]),
            str(row["structural_regularizer"]),
            f"{float(row['structural_loss_weight']):.4f}",
            f"{int(row['runs'])}",
        ]
        for metric in METRICS:
            cells.append(_format_metric_cell(row, metric))
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize Code2Hyp structural-weight sweep result JSON files.")
    parser.add_argument("--inputs", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    runs = _load_runs(args.inputs)
    rows = summarize_runs(runs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render_markdown(rows, args.inputs), encoding="utf-8")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
