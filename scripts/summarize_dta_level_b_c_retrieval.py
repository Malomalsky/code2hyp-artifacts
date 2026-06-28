from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence


def summarize_level_b_c_retrieval(
    input_paths: Sequence[Path],
    *,
    bootstrap_samples: int = 2000,
    seed: int = 20260625,
) -> dict[str, Any]:
    """Summarize DTA Level B/C retrieval with task as uncertainty unit."""

    payloads = [json.loads(path.read_text(encoding="utf-8")) for path in input_paths]
    run_rows = _flatten_runs(payloads)
    task_rows = _task_rows(run_rows)
    contrast_rows = _contrast_rows(task_rows, bootstrap_samples=bootstrap_samples, seed=seed)
    return {
        "inputs": [str(path) for path in input_paths],
        "bootstrap_samples": bootstrap_samples,
        "seed": seed,
        "run_count": len(run_rows),
        "task_metric_rows": task_rows,
        "contrast_rows": contrast_rows,
        "interpretation": _interpretation(contrast_rows),
    }


def format_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# DTA Level B/C task-level summary",
        "",
        "Inputs:",
        *[f"- `{path}`" for path in summary["inputs"]],
        "",
        f"Runs: `{summary['run_count']}`. Bootstrap samples: `{summary['bootstrap_samples']}`.",
        "",
        "Unit of external uncertainty: `task`.",
        "",
        "## Task metrics",
        "",
        "| Level | Seed | Task | Curvature | Queries | MRR | Recall@1 | Recall@5 | Mean rank |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary["task_metric_rows"]:
        lines.append(
            "| {benchmark_level} | {seed} | {task} | {curvature:g} | {query_count} | "
            "{mrr:.4f} | {recall_at_1:.4f} | {recall_at_5:.4f} | {mean_rank:.2f} |".format(**row)
        )
    lines.extend(
        [
            "",
            "## Primary curvature contrasts",
            "",
            "| Level | Contrast | Tasks | Positive tasks | Median task delta | Mean task delta | 95% task bootstrap CI | Exact one-sided sign p |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in summary["contrast_rows"]:
        lines.append(
            "| {benchmark_level} | {contrast} | {n_tasks} | {positive_tasks}/{n_tasks} | "
            "{median_task_delta:+.4f} | {mean_task_delta:+.4f} | "
            "[{ci_low:+.4f}, {ci_high:+.4f}] | {sign_test_p_one_sided:.4g} |".format(**row)
        )
    lines.extend(["", "## Interpretation", "", summary["interpretation"], ""])
    return "\n".join(lines)


def _flatten_runs(payloads: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for payload in payloads:
        level = str(payload.get("benchmark_level", "unknown"))
        for run in payload.get("runs", []):
            row = dict(run)
            row["benchmark_level"] = str(row.get("benchmark_level", level))
            row["seed"] = int(row.get("seed", payload.get("config", {}).get("seed", 0)))
            rows.append(row)
    if not rows:
        raise ValueError("no runs found in input payloads")
    return rows


def _task_rows(run_rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for run in run_rows:
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for record in run.get("query_records", []):
            grouped[str(record["query_task"])].append(record)
        for task, records in sorted(grouped.items()):
            ranks = [int(record["rank"]) for record in records]
            rows.append(
                {
                    "benchmark_level": str(run["benchmark_level"]),
                    "seed": int(run["seed"]),
                    "task": task,
                    "curvature": float(run["curvature"]),
                    "query_count": len(ranks),
                    "mrr": _mean(1.0 / rank for rank in ranks),
                    "recall_at_1": _mean(float(rank <= 1) for rank in ranks),
                    "recall_at_5": _mean(float(rank <= 5) for rank in ranks),
                    "mean_rank": _mean(float(rank) for rank in ranks),
                }
            )
    return rows


def _contrast_rows(
    task_rows: Sequence[dict[str, Any]],
    *,
    bootstrap_samples: int,
    seed: int,
) -> list[dict[str, Any]]:
    by_key: dict[tuple[str, int, str], dict[float, dict[str, Any]]] = defaultdict(dict)
    for row in task_rows:
        by_key[(str(row["benchmark_level"]), int(row["seed"]), str(row["task"]))][float(row["curvature"])] = dict(row)
    deltas: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    contrast_defs = (
        ("param_H1e-4_minus_E", 1e-4, 0.0),
        ("curvature_H1_minus_H1e-4", 1.0, 1e-4),
        ("active_H1_minus_E", 1.0, 0.0),
    )
    for (level, run_seed, task), variants in by_key.items():
        del run_seed
        for contrast_name, high, low in contrast_defs:
            if high in variants and low in variants:
                deltas[(level, contrast_name, task)].append(float(variants[high]["mrr"]) - float(variants[low]["mrr"]))
    rows = []
    by_level_contrast: dict[tuple[str, str], list[tuple[str, float]]] = defaultdict(list)
    for (level, contrast, task), values in deltas.items():
        by_level_contrast[(level, contrast)].append((task, _mean(values)))
    for (level, contrast), task_values in sorted(by_level_contrast.items()):
        values = [value for _, value in sorted(task_values)]
        ci_low, ci_high = _bootstrap_ci(values, bootstrap_samples=bootstrap_samples, seed=seed)
        positive = sum(value > 0.0 for value in values)
        rows.append(
            {
                "benchmark_level": level,
                "contrast": contrast,
                "n_tasks": len(values),
                "positive_tasks": positive,
                "median_task_delta": _median(values),
                "mean_task_delta": _mean(values),
                "ci_low": ci_low,
                "ci_high": ci_high,
                "sign_test_p_one_sided": _sign_test_greater_equal(positive, len(values)),
                "task_deltas": values,
            }
        )
    return rows


def _interpretation(contrast_rows: Sequence[dict[str, Any]]) -> str:
    if not contrast_rows:
        return "No paired curvature contrasts are available."
    parts = []
    for row in contrast_rows:
        level = row["benchmark_level"]
        contrast = row["contrast"]
        positive = int(row["positive_tasks"])
        n_tasks = int(row["n_tasks"])
        mean_delta = float(row["mean_task_delta"])
        ci_low = float(row["ci_low"])
        ci_high = float(row["ci_high"])
        if positive == n_tasks and ci_low > 0:
            status = "positive task-level signal"
        elif mean_delta > 0:
            status = "mixed positive direction"
        elif mean_delta < 0:
            status = "mixed negative direction"
        else:
            status = "no task-level curvature difference"
        parts.append(
            f"{level}, {contrast}: {status}; mean task delta {mean_delta:+.4f}, "
            f"CI [{ci_low:+.4f}, {ci_high:+.4f}], positive tasks {positive}/{n_tasks}."
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
        means.append(_mean(rng.choice(values) for _ in values))
    means.sort()
    return means[int(0.025 * (len(means) - 1))], means[int(0.975 * (len(means) - 1))]


def _sign_test_greater_equal(positive: int, n: int) -> float:
    if n <= 0:
        return 1.0
    return sum(math.comb(n, k) for k in range(positive, n + 1)) / (2**n)


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def _median(values: Sequence[float]) -> float:
    values = sorted(values)
    if not values:
        return 0.0
    middle = len(values) // 2
    if len(values) % 2:
        return float(values[middle])
    return 0.5 * (float(values[middle - 1]) + float(values[middle]))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize DTA Level B/C retrieval at task level.")
    parser.add_argument("--input", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--json-output", type=Path)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260625)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = summarize_level_b_c_retrieval(args.input, bootstrap_samples=args.bootstrap_samples, seed=args.seed)
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
