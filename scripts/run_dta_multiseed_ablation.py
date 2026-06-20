from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from geometry_profile_research.experiment_summary import summarize_metric_series
from scripts.run_dta_feature_ablation import run_feature_ablation


def _method_summary(runs: list[dict[str, object]]) -> dict[str, object]:
    method_names = sorted(runs[0]["methods"]) if runs else []
    metric_names = ["top1_accuracy", "mrr", "map", "recall@10"]
    summary: dict[str, object] = {}
    for method_name in method_names:
        summary[method_name] = {
            metric_name: summarize_metric_series(
                [
                    float(run["methods"][method_name]["metrics"][metric_name])
                    for run in runs
                ]
            )
            for metric_name in metric_names
        }
    return summary


def _delta_summary_against_m2(runs: list[dict[str, object]]) -> dict[str, object]:
    if not runs:
        return {}
    baseline_name = next(name for name in runs[0]["methods"] if name.startswith("M2_"))
    method_names = [
        method_name
        for method_name in sorted(runs[0]["methods"])
        if method_name != baseline_name
    ]
    metric_names = ["top1_accuracy", "mrr", "map", "recall@10"]
    summary: dict[str, object] = {}
    for method_name in method_names:
        summary[method_name] = {}
        for metric_name in metric_names:
            values = []
            for run in runs:
                baseline = float(run["methods"][baseline_name]["metrics"][metric_name])
                candidate = float(run["methods"][method_name]["metrics"][metric_name])
                values.append(candidate - baseline)
            summary[method_name][metric_name] = summarize_metric_series(values)
    return summary


def run_multiseed_ablation(
    dataset_dir: Path,
    *,
    limit_per_task: int,
    seeds: list[int],
    markov_weight: float,
    baseline_kind: str,
) -> dict[str, object]:
    runs = [
        run_feature_ablation(
            dataset_dir,
            limit_per_task=limit_per_task,
            sample_seed=seed,
            markov_weight=markov_weight,
            baseline_kind=baseline_kind,
        )
        for seed in seeds
    ]
    return {
        "dataset": {
            "path": str(dataset_dir),
            "limit_per_task": limit_per_task,
            "seeds": seeds,
        },
        "parameters": {
            "markov_weight": markov_weight,
            "feature_weight": 1.0 - markov_weight,
            "baseline_kind": baseline_kind,
        },
        "method_summary": _method_summary(runs),
        "delta_summary_against_M2": _delta_summary_against_m2(runs),
        "runs": runs,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run DTA feature ablation across multiple stratified seeds."
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("data/dta_zenodo_7799972/extracted"),
    )
    parser.add_argument("--limit-per-task", type=int, default=20)
    parser.add_argument("--seeds", nargs="+", type=int, default=[13, 29, 43])
    parser.add_argument("--markov-weight", type=float, default=0.85)
    parser.add_argument(
        "--baseline-kind",
        choices=["flat_markov", "transition_count"],
        default="flat_markov",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/dta_multiseed_ablation_limit20_w085.json"),
    )
    args = parser.parse_args()

    payload = run_multiseed_ablation(
        args.dataset_dir,
        limit_per_task=args.limit_per_task,
        seeds=args.seeds,
        markov_weight=args.markov_weight,
        baseline_kind=args.baseline_kind,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"wrote {args.output}")
    for method_name, metrics in payload["delta_summary_against_M2"].items():
        map_delta = metrics["map"]
        recall_delta = metrics["recall@10"]
        print(
            f"{method_name}: "
            f"MAP delta mean={map_delta['mean']:.4f} "
            f"range=[{map_delta['min']:.4f}, {map_delta['max']:.4f}], "
            f"R@10 delta mean={recall_delta['mean']:.4f} "
            f"range=[{recall_delta['min']:.4f}, {recall_delta['max']:.4f}]"
        )


if __name__ == "__main__":
    main()
