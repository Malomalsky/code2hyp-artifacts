#!/usr/bin/env python3
"""Summarize task-level contrasts for Code2Hyp label-encoding sensitivity."""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence


CONTRASTS = (
    ("categorical_minus_scalar_hash", "categorical - scalar hash", "structural_categorical", "structural_scalar_hash"),
    ("categorical_minus_no_label", "categorical - no label", "structural_categorical", "structural_none"),
    ("scalar_hash_minus_no_label", "scalar hash - no label", "structural_scalar_hash", "structural_none"),
)


def summarize(path: Path, *, bootstrap_samples: int = 5000, seed: int = 20260628) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    by_key: dict[tuple[str, str, str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in payload["query_rows"]:
        key = (row["dataset"], str(row["seed"]), row["query_task"], row["query_id"])
        by_key[key][row["variant"]] = row

    contrasts: list[dict[str, Any]] = []
    for contrast_id, label, left_variant, right_variant in CONTRASTS:
        paired = [
            (key, cells[left_variant], cells[right_variant])
            for key, cells in sorted(by_key.items())
            if left_variant in cells and right_variant in cells
        ]
        by_task: dict[tuple[str, str], list[tuple[dict[str, Any], dict[str, Any]]]] = defaultdict(list)
        for (dataset, _seed, task, _query_id), left, right in paired:
            by_task[(dataset, task)].append((left, right))
        for dataset in sorted({key[0] for key in by_task}):
            task_rows = []
            for (task_dataset, task), task_pairs in sorted(by_task.items()):
                if task_dataset != dataset:
                    continue
                task_row = {"task": task, "query_count": len(task_pairs)}
                for metric in ("mrr", "recall_at_1", "recall_at_5", "mean_rank"):
                    task_row[f"delta_{metric}"] = _mean([left[metric] - right[metric] for left, right in task_pairs])
                task_rows.append(task_row)
            row = {
                "contrast": contrast_id,
                "label": label,
                "dataset": dataset,
                "task_count": len(task_rows),
                "paired_query_count": sum(row["query_count"] for row in task_rows),
                "task_rows": task_rows,
                "task_bootstrap_ci": {},
                "task_sign_test_p": {},
            }
            for metric in ("delta_mrr", "delta_recall_at_1", "delta_recall_at_5", "delta_mean_rank"):
                values = [task_row[metric] for task_row in task_rows]
                row[metric] = _mean(values)
                row["task_bootstrap_ci"][metric] = _bootstrap_mean_ci(
                    values,
                    samples=bootstrap_samples,
                    seed=_metric_seed(seed, dataset, contrast_id, metric),
                )
                row["task_sign_test_p"][metric] = _two_sided_sign_test_p(values)
            contrasts.append(row)
    return {
        "input": str(path),
        "max_paths": payload.get("max_paths"),
        "distance_mode": payload.get("distance_mode"),
        "bootstrap_samples": bootstrap_samples,
        "seed": seed,
        "contrasts": contrasts,
    }


def format_markdown(result: dict[str, Any]) -> str:
    lines = [
        "# Code2Hyp label-encoding paired task-level contrasts",
        "",
        f"Input diagnostic: max_paths={result['max_paths']}; distance_mode={result['distance_mode']}.",
        "",
        "| Dataset | Contrast | Tasks | Paired queries | Delta MRR, task-bootstrap 95% CI | Delta R@5, task-bootstrap 95% CI | Sign-test p for Delta MRR |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in result["contrasts"]:
        lines.append(
            f"| {row['dataset']} | {row['label']} | {row['task_count']} | {row['paired_query_count']} | "
            f"{_fmt_ci(row, 'delta_mrr')} | {_fmt_ci(row, 'delta_recall_at_5')} | "
            f"{row['task_sign_test_p']['delta_mrr']:.4f} |"
        )
    lines.extend(
        [
            "",
            "Interpretation: positive categorical-minus-scalar deltas mean that the diagnostic structural signal does not rely on the arbitrary scalar AST-label hash.",
            "",
        ]
    )
    return "\n".join(lines)


def _fmt_ci(row: dict[str, Any], metric: str) -> str:
    low, high = row["task_bootstrap_ci"][metric]
    return f"{row[metric]:.4f} [{low:.4f}, {high:.4f}]"


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("empty mean")
    return sum(values) / len(values)


def _bootstrap_mean_ci(values: Sequence[float], *, samples: int, seed: int, alpha: float = 0.05) -> tuple[float, float]:
    if len(values) == 1:
        return values[0], values[0]
    rng = random.Random(seed)
    means = []
    for _ in range(samples):
        sample = [values[rng.randrange(len(values))] for _ in values]
        means.append(_mean(sample))
    means.sort()
    return means[int((alpha / 2.0) * samples)], means[min(samples - 1, int((1.0 - alpha / 2.0) * samples))]


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


def _metric_seed(seed: int, dataset: str, contrast: str, metric: str) -> int:
    label = f"{dataset}:{contrast}:{metric}"
    return seed + sum((index + 1) * ord(char) for index, char in enumerate(label))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--json-output", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260628)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = summarize(args.input, bootstrap_samples=args.bootstrap_samples, seed=args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(format_markdown(result), encoding="utf-8")
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
