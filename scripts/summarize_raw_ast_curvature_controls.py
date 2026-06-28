from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence


METRICS = ("recall_at_1", "mrr", "margin_mean")


def summarize_curvature_controls(input_path: Path) -> dict[str, Any]:
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    runs = payload.get("runs", [])
    if not runs:
        raise ValueError(f"no runs found in {input_path}")

    means = _means_by_variant(runs)
    deltas = _paired_deltas(runs)
    return {
        "input": str(input_path),
        "status": payload.get("status"),
        "completed_runs": payload.get("completed_runs"),
        "expected_runs": payload.get("expected_runs"),
        "means": means,
        "paired_deltas": deltas,
    }


def _means_by_variant(runs: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, float, int, str], list[dict[str, Any]]] = defaultdict(list)
    for run in runs:
        grouped[
            (
                str(run["project"]),
                str(run["node_input_mode"]),
                float(run["curvature"]),
                int(run["dim"]),
                str(run["geometry"]),
            )
        ].append(run)

    rows: list[dict[str, Any]] = []
    for (project, node_input_mode, curvature, dim, geometry), items in sorted(grouped.items()):
        row: dict[str, Any] = {
            "project": project,
            "node_input_mode": node_input_mode,
            "geometry": geometry,
            "curvature": curvature,
            "dim": dim,
            "n": len(items),
        }
        for metric in METRICS:
            row[f"mean_{metric}"] = _mean(float(item[metric]) for item in items)
        if geometry == "poincare":
            diagnostics = [item.get("geometry_diagnostics", {}) for item in items]
            row["mean_sqrt_curvature_norm_mean"] = _mean(
                float(item.get("sqrt_curvature_norm_mean", 0.0)) for item in diagnostics
            )
            row["max_sqrt_curvature_norm_max"] = max(
                float(item.get("sqrt_curvature_norm_max", 0.0)) for item in diagnostics
            )
            row["mean_projection_active_fraction"] = _mean(
                float(item.get("projection_active_fraction", 0.0)) for item in diagnostics
            )
            row["mean_near_boundary_fraction"] = _mean(
                float(item.get("near_boundary_fraction", 0.0)) for item in diagnostics
            )
        rows.append(row)
    return rows


def _paired_deltas(runs: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, int, int], dict[tuple[str, float], dict[str, Any]]] = defaultdict(dict)
    for run in runs:
        key = (
            str(run["project"]),
            str(run["node_input_mode"]),
            int(run["dim"]),
            int(run["seed"]),
        )
        grouped[key][(str(run["geometry"]), float(run["curvature"]))] = run

    by_delta: dict[tuple[str, str, int, float], list[dict[str, float]]] = defaultdict(list)
    for (project, node_input_mode, dim, seed), variants in grouped.items():
        euclidean = variants.get(("euclidean", 1.0))
        if euclidean is None:
            continue
        for (geometry, curvature), run in variants.items():
            if geometry != "poincare":
                continue
            by_delta[(project, node_input_mode, dim, curvature)].append(
                {
                    "seed": float(seed),
                    **{metric: float(run[metric]) - float(euclidean[metric]) for metric in METRICS},
                }
            )

    rows: list[dict[str, Any]] = []
    for (project, node_input_mode, dim, curvature), items in sorted(by_delta.items()):
        row: dict[str, Any] = {
            "project": project,
            "node_input_mode": node_input_mode,
            "dim": dim,
            "curvature": curvature,
            "n": len(items),
        }
        for metric in METRICS:
            values = [float(item[metric]) for item in items]
            row[f"mean_delta_{metric}"] = _mean(values)
            row[f"positive_{metric}"] = sum(value > 0.0 for value in values)
            row[f"per_seed_delta_{metric}"] = values
        rows.append(row)
    return rows


def format_markdown(summary: dict[str, Any]) -> str:
    lines = [
        f"# Curvature-control summary",
        "",
        f"Input: `{summary['input']}`",
        "",
        f"Status: `{summary.get('status')}`; completed `{summary.get('completed_runs')}/{summary.get('expected_runs')}`.",
        "",
        "## Mean metrics",
        "",
        "| Project | Mode | Geometry | c | d | n | R@1 | MRR | Margin | mean sqrt(c)||x|| | max sqrt(c)||x|| |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary["means"]:
        lines.append(
            "| {project} | {node_input_mode} | {geometry} | {curvature:g} | {dim} | {n} | "
            "{mean_recall_at_1:.4f} | {mean_mrr:.4f} | {mean_margin_mean:.4f} | {sqrt_mean} | {sqrt_max} |".format(
                **row,
                sqrt_mean=_fmt_optional(row.get("mean_sqrt_curvature_norm_mean")),
                sqrt_max=_fmt_optional(row.get("max_sqrt_curvature_norm_max")),
            )
        )
    lines.extend(
        [
            "",
            "## Paired deltas versus Euclidean",
            "",
            "| Project | Mode | c | d | n | Delta R@1 | Delta MRR | Delta margin | Positive MRR seeds |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in summary["paired_deltas"]:
        lines.append(
            "| {project} | {node_input_mode} | {curvature:g} | {dim} | {n} | "
            "{mean_delta_recall_at_1:+.4f} | {mean_delta_mrr:+.4f} | {mean_delta_margin_mean:+.4f} | "
            "{positive_mrr}/{n} |".format(**row)
        )
    return "\n".join(lines) + "\n"


def _mean(values: Sequence[float] | Any) -> float:
    values = list(values)
    if not values:
        return 0.0
    return sum(values) / len(values)


def _fmt_optional(value: Any) -> str:
    if value is None:
        return "-"
    return f"{float(value):.4f}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize raw-AST Code2Hyp curvature-control matrix outputs.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--json-output", type=Path)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = summarize_curvature_controls(args.input)
    markdown = format_markdown(summary)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(markdown, encoding="utf-8")
    else:
        sys.stdout.write(markdown)
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
