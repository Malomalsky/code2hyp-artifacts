from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


PRIMARY_CONTRASTS = (
    {
        "contrast": "C_path_LCA_product_minus_single_point",
        "label": "LCA-product - single-point",
        "left": {"geometry": "H_1", "path_object_mode": "lca_product", "method_aggregation": "measure"},
        "right": {"geometry": "H_1", "path_object_mode": "single_point", "method_aggregation": "measure"},
    },
    {
        "contrast": "C_measure_measure_minus_centroid",
        "label": "measure - centroid",
        "left": {"geometry": "H_1", "path_object_mode": "lca_product", "method_aggregation": "measure"},
        "right": {"geometry": "H_1", "path_object_mode": "lca_product", "method_aggregation": "centroid"},
    },
    {
        "contrast": "C_param_H1e-4_minus_E",
        "label": "H_1e-4 - E",
        "left": {"geometry": "H_1e-4", "path_object_mode": "lca_product", "method_aggregation": "measure"},
        "right": {"geometry": "E", "path_object_mode": "lca_product", "method_aggregation": "measure"},
    },
    {
        "contrast": "C_curvature_H1_minus_H1e-4",
        "label": "H_1 - H_1e-4",
        "left": {"geometry": "H_1", "path_object_mode": "lca_product", "method_aggregation": "measure"},
        "right": {"geometry": "H_1e-4", "path_object_mode": "lca_product", "method_aggregation": "measure"},
    },
)


def summarize_dta_factor_matrix(
    input_paths: Path | Sequence[Path],
    *,
    bootstrap_samples: int = 5000,
    seed: int = 20260625,
) -> dict[str, Any]:
    paths = (input_paths,) if isinstance(input_paths, Path) else tuple(input_paths)
    if not paths:
        raise ValueError("at least one input path is required")
    payloads = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    runs = []
    for path, payload in zip(paths, payloads):
        payload_seed = payload.get("config", {}).get("seed")
        for run in payload.get("runs", []):
            run = dict(run)
            run.setdefault("seed", payload_seed)
            run.setdefault("source_file", str(path))
            runs.append(run)
    task_rows = _task_metric_rows(runs)
    contrast_rows = _contrast_rows(task_rows, bootstrap_samples=bootstrap_samples, seed=seed)
    return {
        "inputs": [str(path) for path in paths],
        "experiment": payloads[0].get("experiment"),
        "benchmark_level": payloads[0].get("benchmark_level"),
        "bootstrap_samples": bootstrap_samples,
        "seed": seed,
        "task_metric_rows": task_rows,
        "contrast_rows": contrast_rows,
        "interpretation": _interpretation(contrast_rows),
    }


def _task_metric_rows(runs: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for run in runs:
        by_task: dict[str, list[int]] = defaultdict(list)
        for record in run.get("query_records", []):
            by_task[str(record["query_task"])].append(int(record["rank"]))
        for task, ranks in sorted(by_task.items()):
            rows.append(
                {
                    "task": task,
                    "seed": run.get("seed"),
                    "geometry": run["geometry"],
                    "curvature": float(run["curvature"]),
                    "path_object_mode": run["path_object_mode"],
                    "method_aggregation": run["method_aggregation"],
                    "cell_id": run["cell_id"],
                    "query_count": len(ranks),
                    "mrr": sum(1.0 / rank for rank in ranks) / len(ranks),
                    "recall_at_1": sum(1.0 for rank in ranks if rank <= 1) / len(ranks),
                    "recall_at_5": sum(1.0 for rank in ranks if rank <= 5) / len(ranks),
                    "mean_rank": sum(float(rank) for rank in ranks) / len(ranks),
                }
            )
    return rows


def _contrast_rows(
    task_rows: Sequence[dict[str, Any]],
    *,
    bootstrap_samples: int,
    seed: int,
) -> list[dict[str, Any]]:
    by_key = {
        (
            row["seed"],
            row["task"],
            row["geometry"],
            row["path_object_mode"],
            row["method_aggregation"],
        ): row
        for row in task_rows
    }
    rows: list[dict[str, Any]] = []
    raw_rows = []
    for spec in PRIMARY_CONTRASTS:
        deltas = []
        task_deltas = []
        seed_task_deltas = []
        seed_task_pairs = sorted({(row["seed"], row["task"]) for row in task_rows})
        for row_seed, task in seed_task_pairs:
            left_key = _key(row_seed, task, spec["left"])
            right_key = _key(row_seed, task, spec["right"])
            if left_key not in by_key or right_key not in by_key:
                continue
            delta = float(by_key[left_key]["mrr"]) - float(by_key[right_key]["mrr"])
            seed_task_deltas.append({"seed": row_seed, "task": task, "delta_mrr": delta})
        by_task: dict[str, list[float]] = defaultdict(list)
        for item in seed_task_deltas:
            by_task[item["task"]].append(float(item["delta_mrr"]))
        for task, values in sorted(by_task.items()):
            delta = sum(values) / len(values)
            deltas.append(delta)
            task_deltas.append({"task": task, "delta_mrr": delta, "seed_count": len(values)})
        if not deltas:
            continue
        ci_low, ci_high = _bootstrap_mean_ci(deltas, samples=bootstrap_samples, seed=seed)
        positive_tasks = sum(1 for value in deltas if value > 0.0)
        row = {
            "contrast": spec["contrast"],
            "label": spec["label"],
            "n_tasks": len(deltas),
            "n_seed_task_pairs": len(seed_task_deltas),
            "seed_count": len({item["seed"] for item in seed_task_deltas}),
            "positive_tasks": positive_tasks,
            "median_task_delta": _median(deltas),
            "mean_task_delta": sum(deltas) / len(deltas),
            "ci_low": ci_low,
            "ci_high": ci_high,
            "sign_test_p_one_sided": _one_sided_positive_sign_test(deltas),
            "leave_one_task_out": _leave_one_task_out(deltas),
            "task_deltas": task_deltas,
            "seed_task_deltas": seed_task_deltas,
        }
        raw_rows.append(row)
    adjusted = _holm_adjust([row["sign_test_p_one_sided"] for row in raw_rows])
    for row, holm_p in zip(raw_rows, adjusted):
        row["holm_p"] = holm_p
        rows.append(row)
    return rows


def _key(row_seed: Any, task: str, spec: dict[str, str]) -> tuple[Any, str, str, str, str]:
    return (row_seed, task, spec["geometry"], spec["path_object_mode"], spec["method_aggregation"])


def _leave_one_task_out(values: Sequence[float]) -> dict[str, float]:
    if len(values) <= 1:
        mean_value = sum(values) / len(values)
        return {"min_mean_delta": mean_value, "max_mean_delta": mean_value}
    means = []
    for index in range(len(values)):
        kept = [value for current, value in enumerate(values) if current != index]
        means.append(sum(kept) / len(kept))
    return {"min_mean_delta": min(means), "max_mean_delta": max(means)}


def _bootstrap_mean_ci(values: Sequence[float], *, samples: int, seed: int, alpha: float = 0.05) -> tuple[float, float]:
    if len(values) == 1:
        return values[0], values[0]
    rng = random.Random(seed)
    means = []
    for _ in range(samples):
        sample = [values[rng.randrange(len(values))] for _ in values]
        means.append(sum(sample) / len(sample))
    means.sort()
    low_index = max(0, int((alpha / 2.0) * samples))
    high_index = min(samples - 1, int((1.0 - alpha / 2.0) * samples))
    return means[low_index], means[high_index]


def _one_sided_positive_sign_test(values: Sequence[float]) -> float:
    nonzero = [value for value in values if abs(value) > 1e-12]
    n = len(nonzero)
    if n == 0:
        return 1.0
    positives = sum(1 for value in nonzero if value > 0.0)
    return sum(math.comb(n, k) for k in range(positives, n + 1)) / (2**n)


def _holm_adjust(p_values: Sequence[float]) -> list[float]:
    indexed = sorted(enumerate(p_values), key=lambda item: item[1])
    adjusted = [1.0 for _ in p_values]
    running_max = 0.0
    m = len(p_values)
    for rank, (index, p_value) in enumerate(indexed):
        value = min(1.0, (m - rank) * p_value)
        running_max = max(running_max, value)
        adjusted[index] = running_max
    return adjusted


def _median(values: Sequence[float]) -> float:
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return 0.5 * (ordered[middle - 1] + ordered[middle])


def _interpretation(rows: Sequence[dict[str, Any]]) -> list[str]:
    lines = []
    for row in rows:
        direction = "positive" if row["mean_task_delta"] > 0 else "non-positive"
        lines.append(
            f"{row['label']}: {direction} mean task delta {row['mean_task_delta']:+.4f}; "
            f"median {row['median_task_delta']:+.4f}; CI [{row['ci_low']:+.4f}, {row['ci_high']:+.4f}]; "
            f"LOTO mean range [{row['leave_one_task_out']['min_mean_delta']:+.4f}, {row['leave_one_task_out']['max_mean_delta']:+.4f}]; "
            f"positive tasks {row['positive_tasks']}/{row['n_tasks']}; "
            f"sign p={row['sign_test_p_one_sided']:.4g}; Holm p={row['holm_p']:.4g}."
        )
    return lines


def format_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Code2Hyp factor-matrix summary",
        "",
        "Inputs:",
        *[f"- `{path}`" for path in summary["inputs"]],
        f"Benchmark level: `{summary.get('benchmark_level')}`.",
        f"Bootstrap samples: `{summary['bootstrap_samples']}`.",
        "",
        "## Task metrics",
        "",
        "| Seed | Task | Geometry | Path object | Method | Queries | MRR | Recall@1 | Recall@5 | Mean rank |",
        "|---:|---|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary["task_metric_rows"]:
        lines.append(
            f"| {row['seed']} | {row['task']} | {row['geometry']} | {row['path_object_mode']} | {row['method_aggregation']} | "
            f"{row['query_count']} | {row['mrr']:.4f} | {row['recall_at_1']:.4f} | {row['recall_at_5']:.4f} | {row['mean_rank']:.2f} |"
        )
    lines.extend(
        [
            "",
            "## Primary contrasts",
            "",
            "| Contrast | Tasks | Seed-task pairs | Positive tasks | Median task delta | Mean task delta | 95% task bootstrap CI | LOTO mean range | Sign p | Holm p |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in summary["contrast_rows"]:
        lines.append(
            f"| {row['label']} | {row['n_tasks']} | {row['n_seed_task_pairs']} | {row['positive_tasks']}/{row['n_tasks']} | "
            f"{row['median_task_delta']:+.4f} | {row['mean_task_delta']:+.4f} | "
            f"[{row['ci_low']:+.4f}, {row['ci_high']:+.4f}] | "
            f"[{row['leave_one_task_out']['min_mean_delta']:+.4f}, {row['leave_one_task_out']['max_mean_delta']:+.4f}] | "
            f"{row['sign_test_p_one_sided']:.4g} | {row['holm_p']:.4g} |"
        )
    lines.extend(["", "## Interpretation", ""])
    lines.extend(summary["interpretation"])
    lines.append("")
    return "\n".join(lines)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize DTA Code2Hyp factor-matrix primary contrasts.")
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--json-output", type=Path)
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260625)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = summarize_dta_factor_matrix(tuple(args.input), bootstrap_samples=args.bootstrap_samples, seed=args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(format_markdown(summary), encoding="utf-8")
    if args.json_output:
        _write_json(args.json_output, summary)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
