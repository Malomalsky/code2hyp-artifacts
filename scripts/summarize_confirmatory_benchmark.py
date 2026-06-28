from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@dataclass(frozen=True)
class ConfirmatoryInput:
    dataset: str
    path: Path


METRIC_FIELDS = ("mrr", "recall_at_1", "recall_at_5", "mean_rank")

PRIMARY_CONTRASTS = (
    {
        "contrast": "lca_product_measure_minus_single_point_measure",
        "label": "LCA-product measure - single-point measure",
        "left": {"path_object_mode": "lca_product", "method_aggregation": "measure"},
        "right": {"path_object_mode": "single_point", "method_aggregation": "measure"},
    },
    {
        "contrast": "lca_product_centroid_minus_single_point_centroid",
        "label": "LCA-product centroid - single-point centroid",
        "left": {"path_object_mode": "lca_product", "method_aggregation": "centroid"},
        "right": {"path_object_mode": "single_point", "method_aggregation": "centroid"},
    },
    {
        "contrast": "lca_product_measure_minus_lca_product_centroid",
        "label": "LCA-product measure - LCA-product centroid",
        "left": {"path_object_mode": "lca_product", "method_aggregation": "measure"},
        "right": {"path_object_mode": "lca_product", "method_aggregation": "centroid"},
    },
    {
        "contrast": "single_point_measure_minus_single_point_centroid",
        "label": "single-point measure - single-point centroid",
        "left": {"path_object_mode": "single_point", "method_aggregation": "measure"},
        "right": {"path_object_mode": "single_point", "method_aggregation": "centroid"},
    },
)


def summarize_confirmatory_benchmark(
    inputs: Sequence[ConfirmatoryInput],
    *,
    bootstrap_samples: int = 5000,
    seed: int = 20260628,
) -> dict[str, Any]:
    if not inputs:
        raise ValueError("at least one input is required")
    query_rows = _query_rows(inputs)
    cell_summaries = _cell_summaries(query_rows, bootstrap_samples=bootstrap_samples, seed=seed)
    paired_contrasts = _paired_contrasts(query_rows, bootstrap_samples=bootstrap_samples, seed=seed)
    task_level_contrasts = _task_level_contrasts(query_rows, bootstrap_samples=bootstrap_samples, seed=seed)
    return {
        "inputs": [{"dataset": item.dataset, "path": str(item.path)} for item in inputs],
        "bootstrap_samples": bootstrap_samples,
        "seed": seed,
        "query_rows": query_rows,
        "cell_summaries": cell_summaries,
        "paired_contrasts": paired_contrasts,
        "task_level_contrasts": task_level_contrasts,
        "interpretation": _interpretation(cell_summaries, paired_contrasts),
    }


def _query_rows(inputs: Sequence[ConfirmatoryInput]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in inputs:
        payload = json.loads(item.path.read_text(encoding="utf-8"))
        payload_seed = payload.get("config", {}).get("seed")
        for run_index, run in enumerate(payload.get("runs", [])):
            seed = run.get("seed", payload_seed)
            geometry = str(run.get("geometry", ""))
            cost_mode = str(run.get("cost_mode", ""))
            path_object_mode = str(run.get("path_object_mode", ""))
            method_aggregation = str(run.get("method_aggregation", ""))
            cell_id = _cell_id(item.dataset, geometry, cost_mode, path_object_mode, method_aggregation)
            for record_index, record in enumerate(run.get("query_records", [])):
                rank = int(record["rank"])
                query_id = str(record.get("query_id", f"run-{run_index}:query-{record_index}"))
                query_task = str(record.get("query_task", ""))
                rows.append(
                    {
                        "dataset": item.dataset,
                        "source_file": str(item.path),
                        "seed": seed,
                        "geometry": geometry,
                        "curvature": float(run.get("curvature", 0.0)),
                        "cost_mode": cost_mode,
                        "path_object_mode": path_object_mode,
                        "method_aggregation": method_aggregation,
                        "cell_id": cell_id,
                        "query_key": _query_key(item.dataset, seed, query_task, query_id),
                        "query_id": query_id,
                        "query_task": query_task,
                        "rank": rank,
                        "mrr": 1.0 / rank,
                        "recall_at_1": 1.0 if rank <= 1 else 0.0,
                        "recall_at_5": 1.0 if rank <= 5 else 0.0,
                        "mean_rank": float(rank),
                    }
                )
    return rows


def _cell_summaries(
    query_rows: Sequence[dict[str, Any]],
    *,
    bootstrap_samples: int,
    seed: int,
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in query_rows:
        grouped[row["cell_id"]].append(row)

    summaries: list[dict[str, Any]] = []
    for cell_id, rows in sorted(grouped.items()):
        first = rows[0]
        summary = {
            "cell_id": cell_id,
            "dataset": first["dataset"],
            "geometry": first["geometry"],
            "cost_mode": first["cost_mode"],
            "path_object_mode": first["path_object_mode"],
            "method_aggregation": first["method_aggregation"],
            "query_count": len(rows),
            "seed_count": len({row["seed"] for row in rows}),
            "task_count": len({row["query_task"] for row in rows}),
            "bootstrap_ci": {},
        }
        for metric in METRIC_FIELDS:
            values = [float(row[metric]) for row in rows]
            summary[metric] = _mean(values)
            summary["bootstrap_ci"][metric] = _bootstrap_mean_ci(values, samples=bootstrap_samples, seed=_metric_seed(seed, cell_id, metric))
        summaries.append(summary)
    return summaries


def _paired_contrasts(
    query_rows: Sequence[dict[str, Any]],
    *,
    bootstrap_samples: int,
    seed: int,
) -> list[dict[str, Any]]:
    rows_by_base: dict[tuple[Any, ...], dict[tuple[str, str], dict[str, Any]]] = defaultdict(dict)
    for row in query_rows:
        base = (
            row["dataset"],
            row["geometry"],
            row["cost_mode"],
            row["query_key"],
        )
        rows_by_base[base][(row["path_object_mode"], row["method_aggregation"])] = row

    contrast_rows: list[dict[str, Any]] = []
    for dataset, geometry, cost_mode in sorted({(row["dataset"], row["geometry"], row["cost_mode"]) for row in query_rows}):
        scoped = {
            query_key: cells
            for (row_dataset, row_geometry, row_cost_mode, query_key), cells in rows_by_base.items()
            if (row_dataset, row_geometry, row_cost_mode) == (dataset, geometry, cost_mode)
        }
        for spec in PRIMARY_CONTRASTS:
            left_key = (spec["left"]["path_object_mode"], spec["left"]["method_aggregation"])
            right_key = (spec["right"]["path_object_mode"], spec["right"]["method_aggregation"])
            paired = [(cells[left_key], cells[right_key]) for _, cells in sorted(scoped.items()) if left_key in cells and right_key in cells]
            if not paired:
                continue
            deltas_by_metric = {
                "delta_mrr": [float(left["mrr"]) - float(right["mrr"]) for left, right in paired],
                "delta_recall_at_1": [float(left["recall_at_1"]) - float(right["recall_at_1"]) for left, right in paired],
                "delta_recall_at_5": [float(left["recall_at_5"]) - float(right["recall_at_5"]) for left, right in paired],
                "delta_mean_rank": [float(left["mean_rank"]) - float(right["mean_rank"]) for left, right in paired],
            }
            row = {
                "contrast": spec["contrast"],
                "label": spec["label"],
                "dataset": dataset,
                "geometry": geometry,
                "cost_mode": cost_mode,
                "paired_query_count": len(paired),
                "seed_count": len({left["seed"] for left, _ in paired}),
                "task_count": len({left["query_task"] for left, _ in paired}),
                "bootstrap_ci": {},
            }
            for metric, values in deltas_by_metric.items():
                row[metric] = _mean(values)
                row["bootstrap_ci"][metric] = _bootstrap_mean_ci(
                    values,
                    samples=bootstrap_samples,
                    seed=_metric_seed(seed, f"{dataset}:{geometry}:{cost_mode}:{spec['contrast']}", metric),
                )
            contrast_rows.append(row)
    return contrast_rows


def _task_level_contrasts(
    query_rows: Sequence[dict[str, Any]],
    *,
    bootstrap_samples: int,
    seed: int,
) -> list[dict[str, Any]]:
    rows_by_base: dict[tuple[Any, ...], dict[tuple[str, str], dict[str, Any]]] = defaultdict(dict)
    for row in query_rows:
        base = (
            row["dataset"],
            row["geometry"],
            row["cost_mode"],
            row["query_key"],
        )
        rows_by_base[base][(row["path_object_mode"], row["method_aggregation"])] = row

    contrast_rows: list[dict[str, Any]] = []
    for dataset, geometry, cost_mode in sorted({(row["dataset"], row["geometry"], row["cost_mode"]) for row in query_rows}):
        scoped = {
            query_key: cells
            for (row_dataset, row_geometry, row_cost_mode, query_key), cells in rows_by_base.items()
            if (row_dataset, row_geometry, row_cost_mode) == (dataset, geometry, cost_mode)
        }
        for spec in PRIMARY_CONTRASTS:
            left_key = (spec["left"]["path_object_mode"], spec["left"]["method_aggregation"])
            right_key = (spec["right"]["path_object_mode"], spec["right"]["method_aggregation"])
            paired = [(cells[left_key], cells[right_key]) for _, cells in sorted(scoped.items()) if left_key in cells and right_key in cells]
            if not paired:
                continue
            by_task: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = defaultdict(list)
            for left, right in paired:
                by_task[str(left["query_task"])].append((left, right))

            task_rows: list[dict[str, Any]] = []
            for task, task_pairs in sorted(by_task.items()):
                task_row: dict[str, Any] = {"task": task, "query_count": len(task_pairs)}
                for metric in ("mrr", "recall_at_1", "recall_at_5", "mean_rank"):
                    task_row[f"delta_{metric}"] = _mean([float(left[metric]) - float(right[metric]) for left, right in task_pairs])
                task_rows.append(task_row)

            row: dict[str, Any] = {
                "contrast": spec["contrast"],
                "label": spec["label"],
                "dataset": dataset,
                "geometry": geometry,
                "cost_mode": cost_mode,
                "task_count": len(task_rows),
                "paired_query_count": len(paired),
                "task_rows": task_rows,
                "task_bootstrap_ci": {},
                "task_sign_test_p": {},
            }
            for metric in ("delta_mrr", "delta_recall_at_1", "delta_recall_at_5", "delta_mean_rank"):
                values = [float(task_row[metric]) for task_row in task_rows]
                row[metric] = _mean(values)
                row["task_bootstrap_ci"][metric] = _bootstrap_mean_ci(
                    values,
                    samples=bootstrap_samples,
                    seed=_metric_seed(seed, f"task:{dataset}:{geometry}:{cost_mode}:{spec['contrast']}", metric),
                )
                row["task_sign_test_p"][metric] = _two_sided_sign_test_p(values)
            contrast_rows.append(row)
    return contrast_rows


def _interpretation(
    cell_summaries: Sequence[dict[str, Any]],
    paired_contrasts: Sequence[dict[str, Any]],
) -> list[str]:
    lines = [
        "The retrieval outcome is first computed per query. Because queries from the same task are dependent, the summary also reports task-level paired contrasts with deterministic non-parametric bootstrap over tasks and exact sign tests over task-level deltas.",
        "The paired contrasts use only matched query identifiers within the same dataset, geometry and cost mode. This avoids mixing different train/query/gallery splits when comparing path-object representations.",
    ]
    for row in paired_contrasts:
        verb = "improves MRR by" if row["delta_mrr"] > 0 else "changes MRR by"
        lines.append(
            f"{row['dataset']} / {row['geometry']} / {row['label']}: {verb} "
            f"{row['delta_mrr']:+.4f} with 95% bootstrap CI "
            f"[{row['bootstrap_ci']['delta_mrr'][0]:+.4f}, {row['bootstrap_ci']['delta_mrr'][1]:+.4f}] "
            f"over {row['paired_query_count']} paired queries."
        )
    return lines


def format_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Final confirmatory representation benchmark",
        "",
        "Inputs:",
        *[f"- `{item['dataset']}`: `{item['path']}`" for item in summary["inputs"]],
        "",
        f"Bootstrap samples: `{summary['bootstrap_samples']}`.",
        "",
        "## Cell summaries",
        "",
        "| Dataset | Geometry | Cost | Path object | Aggregation | Queries | MRR, 95% bootstrap CI | Recall@1, 95% bootstrap CI | Recall@5, 95% bootstrap CI | Mean rank, 95% bootstrap CI |",
        "|---|---|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary["cell_summaries"]:
        lines.append(
            f"| {row['dataset']} | {row['geometry']} | {row['cost_mode']} | {row['path_object_mode']} | {row['method_aggregation']} | "
            f"{row['query_count']} | {_fmt_ci(row, 'mrr')} | {_fmt_ci(row, 'recall_at_1')} | {_fmt_ci(row, 'recall_at_5')} | {_fmt_ci(row, 'mean_rank')} |"
        )
    lines.extend(
        [
            "",
            "## Paired query contrasts",
            "",
            "| Dataset | Geometry | Cost | Contrast | Paired queries | Delta MRR, 95% bootstrap CI | Delta Recall@1, 95% bootstrap CI | Delta Recall@5, 95% bootstrap CI | Delta mean rank, 95% bootstrap CI |",
            "|---|---|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in summary["paired_contrasts"]:
        lines.append(
            f"| {row['dataset']} | {row['geometry']} | {row['cost_mode']} | {row['label']} | {row['paired_query_count']} | "
            f"{_fmt_ci(row, 'delta_mrr')} | {_fmt_ci(row, 'delta_recall_at_1')} | {_fmt_ci(row, 'delta_recall_at_5')} | {_fmt_ci(row, 'delta_mean_rank')} |"
        )
    lines.extend(
        [
            "",
            "## Paired task-level contrasts",
            "",
            "| Dataset | Contrast | Tasks | Paired queries | Delta MRR, task-bootstrap 95% CI | Delta Recall@5, task-bootstrap 95% CI | Sign-test p for Delta MRR |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in summary["task_level_contrasts"]:
        lines.append(
            f"| {row['dataset']} | {row['label']} | {row['task_count']} | {row['paired_query_count']} | "
            f"{_fmt_task_ci(row, 'delta_mrr')} | {_fmt_task_ci(row, 'delta_recall_at_5')} | "
            f"{row['task_sign_test_p']['delta_mrr']:.4f} |"
        )
    lines.extend(["", "## Interpretation", ""])
    lines.extend(summary["interpretation"])
    lines.append("")
    return "\n".join(lines)


def _fmt_ci(row: dict[str, Any], metric: str) -> str:
    low, high = row["bootstrap_ci"][metric]
    return f"{row[metric]:.4f} [{low:.4f}, {high:.4f}]"


def _fmt_task_ci(row: dict[str, Any], metric: str) -> str:
    low, high = row["task_bootstrap_ci"][metric]
    return f"{row[metric]:.4f} [{low:.4f}, {high:.4f}]"


def _cell_id(dataset: str, geometry: str, cost_mode: str, path_object_mode: str, method_aggregation: str) -> str:
    return f"{dataset}::{geometry}::{cost_mode}::{path_object_mode}::{method_aggregation}"


def _query_key(dataset: str, seed: Any, query_task: str, query_id: str) -> str:
    return f"{dataset}::{seed}::{query_task}::{query_id}"


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("cannot compute a mean over an empty sequence")
    return sum(values) / len(values)


def _bootstrap_mean_ci(values: Sequence[float], *, samples: int, seed: int, alpha: float = 0.05) -> tuple[float, float]:
    if samples <= 0:
        raise ValueError("bootstrap sample count must be positive")
    if len(values) == 1:
        return values[0], values[0]
    rng = random.Random(seed)
    means = []
    for _ in range(samples):
        sample = [values[rng.randrange(len(values))] for _ in values]
        means.append(_mean(sample))
    means.sort()
    low_index = max(0, int((alpha / 2.0) * samples))
    high_index = min(samples - 1, int((1.0 - alpha / 2.0) * samples))
    return means[low_index], means[high_index]


def _two_sided_sign_test_p(values: Sequence[float]) -> float:
    nonzero = [value for value in values if value != 0.0]
    n = len(nonzero)
    if n == 0:
        return 1.0
    positives = sum(1 for value in nonzero if value > 0.0)
    lower_tail = sum(_binom(n, k) for k in range(0, positives + 1)) / (2**n)
    upper_tail = sum(_binom(n, k) for k in range(positives, n + 1)) / (2**n)
    return min(1.0, 2.0 * min(lower_tail, upper_tail))


def _binom(n: int, k: int) -> int:
    if k < 0 or k > n:
        return 0
    k = min(k, n - k)
    result = 1
    for value in range(1, k + 1):
        result = result * (n - value + 1) // value
    return result


def _metric_seed(seed: int, label: str, metric: str) -> int:
    # Python's built-in hash is intentionally randomized between processes.
    stable = sum((index + 1) * ord(char) for index, char in enumerate(f"{label}:{metric}"))
    return seed + stable


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _parse_inputs(values: Iterable[Sequence[str]]) -> list[ConfirmatoryInput]:
    inputs = []
    for dataset, path in values:
        inputs.append(ConfirmatoryInput(dataset=dataset, path=Path(path)))
    return inputs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize final query-level Code2Hyp confirmatory representation benchmark.")
    parser.add_argument("--input", nargs=2, action="append", metavar=("DATASET", "PATH"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--json-output", type=Path)
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260628)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = summarize_confirmatory_benchmark(
        _parse_inputs(args.input),
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(format_markdown(summary), encoding="utf-8")
    if args.json_output:
        _write_json(args.json_output, summary)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
