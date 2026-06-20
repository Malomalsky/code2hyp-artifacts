from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from geometry_profile_research.retrieval import (
    combine_distance_matrices,
    euclidean_sparse_distance,
    evaluate_retrieval,
    jensen_shannon_divergence,
    pairwise_distances,
    zscore_feature_vectors,
)
from scripts.run_dta_markov_baselines import _transition_count_vector
from scripts.run_dta_retrieval_experiment import _experiment_records


def run_weight_sweep(
    dataset_dir: Path,
    *,
    limit_per_task: int,
    sample_seed: int,
    baseline_kind: str,
    weights: list[float],
) -> dict[str, object]:
    records, markov_vectors, geometry_vectors, errors = _experiment_records(
        dataset_dir,
        limit_per_task=limit_per_task,
        sample_seed=sample_seed,
    )
    labels = [record.task_id for record in records]
    if baseline_kind == "transition_count":
        markov_vectors = [_transition_count_vector(record.code) for record in records]
    elif baseline_kind != "flat_markov":
        raise ValueError("baseline_kind must be 'flat_markov' or 'transition_count'")

    scaled_geometry_vectors = zscore_feature_vectors(
        [feature_sets["all"] for feature_sets in geometry_vectors]
    )
    markov_distances = pairwise_distances(markov_vectors, jensen_shannon_divergence)
    geometry_distances = pairwise_distances(scaled_geometry_vectors, euclidean_sparse_distance)

    results: list[dict[str, object]] = []
    for weight in weights:
        if weight == 1.0:
            distances = markov_distances
        elif weight == 0.0:
            distances = geometry_distances
        else:
            distances = combine_distance_matrices(
                markov_distances,
                geometry_distances,
                left_weight=weight,
            )
        results.append(
            {
                "markov_weight": weight,
                "geometry_weight": 1.0 - weight,
                "metrics": evaluate_retrieval(labels, distances, k_values=(1, 3, 5, 10)),
            }
        )

    return {
        "dataset": {
            "path": str(dataset_dir),
            "limit_per_task": limit_per_task,
            "sample_seed": sample_seed,
        },
        "parameters": {
            "baseline_kind": baseline_kind,
        },
        "records": {
            "valid": len(records),
            "syntax_errors": len(errors),
            "task_count": len(set(labels)),
        },
        "results": results,
        "errors": errors[:20],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sweep Markov/geometry weights for DTA retrieval."
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("data/dta_zenodo_7799972/extracted"),
    )
    parser.add_argument("--limit-per-task", type=int, default=10)
    parser.add_argument("--sample-seed", type=int, default=13)
    parser.add_argument(
        "--baseline-kind",
        choices=["flat_markov", "transition_count"],
        default="flat_markov",
    )
    parser.add_argument(
        "--weights",
        nargs="+",
        type=float,
        default=[1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.6, 0.5, 0.0],
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/dta_weight_sweep_limit10.json"),
    )
    args = parser.parse_args()

    payload = run_weight_sweep(
        args.dataset_dir,
        limit_per_task=args.limit_per_task,
        sample_seed=args.sample_seed,
        baseline_kind=args.baseline_kind,
        weights=args.weights,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"wrote {args.output}")
    for row in payload["results"]:
        metrics = row["metrics"]
        print(
            f"w={row['markov_weight']:.2f} "
            f"top1={metrics['top1_accuracy']:.4f} "
            f"mrr={metrics['mrr']:.4f} "
            f"map={metrics['map']:.4f} "
            f"r10={metrics['recall@10']:.4f}"
        )


if __name__ == "__main__":
    main()
