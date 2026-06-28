from __future__ import annotations

import argparse
import json
import random
import statistics as st
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence


def summarize_dta_task_transfer(
    input_path: Path,
    *,
    bootstrap_samples: int = 2000,
    seed: int = 20260624,
) -> dict[str, Any]:
    """Summarize canonical DTA transfer at the task level.

    The DTA exercises are the external uncertainty units. This summary therefore
    reports per-task paired deltas first and macro-averages over tasks second.
    """

    payload = json.loads(input_path.read_text(encoding="utf-8"))
    runs = list(payload.get("runs", []))
    if not runs:
        raise ValueError(f"no runs found in {input_path}")
    task_rows = _task_metric_rows(runs)
    paired_rows = _paired_deltas(runs)
    aggregate_rows = _aggregate_by_curvature(paired_rows, bootstrap_samples=bootstrap_samples, seed=seed)
    return {
        "input": str(input_path),
        "status": payload.get("status"),
        "completed_runs": payload.get("completed_runs"),
        "expected_runs": payload.get("expected_runs"),
        "bootstrap_samples": bootstrap_samples,
        "seed": seed,
        "task_metric_rows": task_rows,
        "paired_delta_rows": paired_rows,
        "macro_delta_rows": aggregate_rows,
        "interpretation": _interpretation(aggregate_rows),
    }


def format_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# DTA Python canonical transfer task-level summary",
        "",
        f"Input: `{summary['input']}`",
        "",
        f"Status: `{summary.get('status')}`; completed `{summary.get('completed_runs')}/{summary.get('expected_runs')}`.",
        "",
        "Unit of external uncertainty: `task`.",
        "",
        "The materialized files are retrieval items inside tasks. They are not treated as independent confirmatory units.",
        "",
        "## Task-level mean metrics",
        "",
        "| Task | Geometry | c | Seeds | MRR mean | MRR sd | Recall@5 mean | Mean rank |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary["task_metric_rows"]:
        lines.append(
            "| {project} | {geometry} | {curvature:g} | {n_seeds} | {mean_mrr:.4f} | "
            "{sd_mrr:.4f} | {mean_recall_at_5:.4f} | {mean_mean_rank:.4f} |".format(**row)
        )
    lines.extend(
        [
            "",
            "## Paired deltas against matched Euclidean",
            "",
            "| Curvature | Task | Mean delta MRR | Positive seeds | Seed deltas |",
            "|---:|---|---:|---:|---|",
        ]
    )
    for row in summary["paired_delta_rows"]:
        seed_deltas = ", ".join(f"{value:+.4f}" for value in row["seed_deltas"])
        row_with_text = dict(row)
        row_with_text["seed_deltas_text"] = seed_deltas
        lines.append(
            "| {curvature:g} | {project} | {mean_delta_mrr:+.4f} | "
            "{positive_seeds}/{n_seeds} | {seed_deltas_text} |".format(**row_with_text)
        )
    lines.extend(
        [
            "",
            "## Macro task-level deltas",
            "",
            "| Curvature | Tasks | Positive tasks | Macro delta MRR | 95% task bootstrap CI |",
            "|---:|---:|---:|---:|---:|",
        ]
    )
    for row in summary["macro_delta_rows"]:
        lines.append(
            "| {curvature:g} | {n_tasks} | {positive_tasks}/{n_tasks} | {macro_delta_mrr:+.4f} | "
            "[{ci_low:+.4f}, {ci_high:+.4f}] |".format(**row)
        )
    lines.extend(["", "## Interpretation", "", summary["interpretation"], ""])
    return "\n".join(lines)


def _task_metric_rows(runs: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, float], list[dict[str, Any]]] = defaultdict(list)
    for run in runs:
        grouped[(str(run["project"]), str(run["geometry"]), float(run["curvature"]))].append(run)
    rows: list[dict[str, Any]] = []
    for (project, geometry, curvature), items in sorted(grouped.items()):
        mrr = [float(item["mrr"]) for item in items]
        recall_at_5 = [float(item.get("recall_at_5", 0.0)) for item in items]
        mean_rank = [float(item.get("mean_rank", 0.0)) for item in items]
        rows.append(
            {
                "project": project,
                "geometry": geometry,
                "curvature": curvature,
                "n_seeds": len(items),
                "mean_mrr": _mean(mrr),
                "sd_mrr": _pstdev(mrr),
                "mean_recall_at_5": _mean(recall_at_5),
                "mean_mean_rank": _mean(mean_rank),
            }
        )
    return rows


def _paired_deltas(runs: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    by_project_seed: dict[tuple[str, int], dict[tuple[str, float], dict[str, Any]]] = defaultdict(dict)
    for run in runs:
        by_project_seed[(str(run["project"]), int(run["seed"]))][(str(run["geometry"]), float(run["curvature"]))] = run
    deltas: dict[tuple[float, str], list[float]] = defaultdict(list)
    for (project, seed), variants in by_project_seed.items():
        baseline = variants.get(("euclidean", 1.0))
        if baseline is None:
            continue
        for (geometry, curvature), run in variants.items():
            if geometry != "poincare":
                continue
            deltas[(curvature, project)].append(float(run["mrr"]) - float(baseline["mrr"]))
    rows: list[dict[str, Any]] = []
    for (curvature, project), values in sorted(deltas.items()):
        rows.append(
            {
                "curvature": curvature,
                "project": project,
                "n_seeds": len(values),
                "seed_deltas": values,
                "mean_delta_mrr": _mean(values),
                "positive_seeds": sum(value > 0.0 for value in values),
            }
        )
    return rows


def _aggregate_by_curvature(
    paired_rows: Sequence[dict[str, Any]],
    *,
    bootstrap_samples: int,
    seed: int,
) -> list[dict[str, Any]]:
    grouped: dict[float, list[dict[str, Any]]] = defaultdict(list)
    for row in paired_rows:
        grouped[float(row["curvature"])].append(dict(row))
    rows: list[dict[str, Any]] = []
    for curvature, items in sorted(grouped.items()):
        task_values = [float(item["mean_delta_mrr"]) for item in items]
        ci_low, ci_high = _bootstrap_ci(task_values, bootstrap_samples=bootstrap_samples, seed=seed)
        rows.append(
            {
                "curvature": curvature,
                "n_tasks": len(items),
                "positive_tasks": sum(value > 0.0 for value in task_values),
                "macro_delta_mrr": _mean(task_values),
                "ci_low": ci_low,
                "ci_high": ci_high,
                "task_deltas": task_values,
            }
        )
    return rows


def _interpretation(aggregate_rows: Sequence[dict[str, Any]]) -> str:
    if not aggregate_rows:
        return "No paired DTA deltas are available yet."
    parts = []
    for row in aggregate_rows:
        curvature = float(row["curvature"])
        positive = int(row["positive_tasks"])
        n_tasks = int(row["n_tasks"])
        delta = float(row["macro_delta_mrr"])
        ci_low = float(row["ci_low"])
        ci_high = float(row["ci_high"])
        if positive == n_tasks and ci_low > 0:
            status = "positive task-level transfer signal"
        elif positive == n_tasks:
            status = "positive task-level direction with wide task bootstrap interval"
        elif delta > 0:
            status = "mixed but positive macro direction"
        else:
            status = "no positive macro transfer signal"
        parts.append(
            f"Poincare c={curvature:g}: {status}; macro delta MRR {delta:+.4f}, "
            f"task bootstrap CI [{ci_low:+.4f}, {ci_high:+.4f}], positive tasks {positive}/{n_tasks}."
        )
    parts.append(
        "This summary supports only task-level transfer claims. It does not isolate negative curvature unless "
        "active Poincare is compared against the near-zero-curvature control under the same implementation path."
    )
    return " ".join(parts)


def _bootstrap_ci(values: Sequence[float], *, bootstrap_samples: int, seed: int) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    if len(values) == 1:
        return float(values[0]), float(values[0])
    rng = random.Random(seed)
    means = []
    values = list(values)
    for _ in range(bootstrap_samples):
        sample = [rng.choice(values) for _ in values]
        means.append(_mean(sample))
    means.sort()
    low_index = int(0.025 * (len(means) - 1))
    high_index = int(0.975 * (len(means) - 1))
    return means[low_index], means[high_index]


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def _pstdev(values: Sequence[float]) -> float:
    return st.pstdev(values) if len(values) > 1 else 0.0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize DTA Python transfer at task level.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--json-output", type=Path)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260624)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = summarize_dta_task_transfer(args.input, bootstrap_samples=args.bootstrap_samples, seed=args.seed)
    markdown = format_markdown(summary)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(markdown, encoding="utf-8")
    else:
        print(markdown)
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()
