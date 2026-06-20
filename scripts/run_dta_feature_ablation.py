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
    paired_bootstrap_ci,
    paired_permutation_p_value,
    pairwise_distances,
    per_query_retrieval_scores,
    zscore_feature_vectors,
)
from scripts.run_dta_retrieval_experiment import _experiment_records
from scripts.run_dta_markov_baselines import _transition_count_vector


def _paired_tests_against_baseline(
    labels: list[int],
    baseline_distances: list[list[float]],
    candidate_distances: list[list[float]],
) -> dict[str, object]:
    baseline_scores = per_query_retrieval_scores(labels, baseline_distances, k_values=(1, 3, 5, 10))
    candidate_scores = per_query_retrieval_scores(labels, candidate_distances, k_values=(1, 3, 5, 10))
    tests: dict[str, object] = {}
    for key, output_key in [
        ("top1", "top1_accuracy"),
        ("reciprocal_rank", "mrr"),
        ("average_precision", "map"),
        ("recall@10", "recall@10"),
    ]:
        baseline_values = [score[key] for score in baseline_scores]
        candidate_values = [score[key] for score in candidate_scores]
        ci_low, ci_high = paired_bootstrap_ci(
            baseline_values,
            candidate_values,
            iterations=5000,
            seed=19,
        )
        tests[output_key] = {
            "mean_delta": sum(
                candidate - baseline
                for baseline, candidate in zip(baseline_values, candidate_values)
            )
            / len(baseline_values)
            if baseline_values
            else 0.0,
            "bootstrap_ci95": [ci_low, ci_high],
            "permutation_p_one_sided": paired_permutation_p_value(
                baseline_values,
                candidate_values,
                iterations=5000,
                seed=19,
            ),
        }
    return tests


def run_feature_ablation(
    dataset_dir: Path,
    *,
    limit_per_task: int,
    sample_seed: int,
    markov_weight: float,
    baseline_kind: str = "flat_markov",
) -> dict[str, object]:
    records, markov_vectors, geometry_feature_sets, errors = _experiment_records(
        dataset_dir,
        limit_per_task=limit_per_task,
        sample_seed=sample_seed,
    )
    if baseline_kind == "transition_count":
        markov_vectors = [_transition_count_vector(record.code) for record in records]
    elif baseline_kind != "flat_markov":
        raise ValueError("baseline_kind must be 'flat_markov' or 'transition_count'")

    labels = [record.task_id for record in records]
    markov_distances = pairwise_distances(markov_vectors, jensen_shannon_divergence)

    methods: dict[str, object] = {
        f"M2_{baseline_kind}_jsd": {
            "metrics": evaluate_retrieval(labels, markov_distances, k_values=(1, 3, 5, 10)),
            "paired_tests_M_minus_M2": None,
        }
    }

    feature_set_specs = {
        "length_only": ["length_only"],
        "size_depth": ["size_depth"],
        "branching": ["branching"],
        "metric_distortion": ["metric_distortion"],
        "shape": ["size_depth", "branching"],
        "all": ["all"],
    }

    for feature_set_name, source_sets in feature_set_specs.items():
        feature_vectors = []
        for record_feature_sets in geometry_feature_sets:
            vector: dict[str, float] = {}
            for source_set in source_sets:
                vector.update(record_feature_sets[source_set])
            feature_vectors.append(vector)
        scaled_feature_vectors = zscore_feature_vectors(feature_vectors)
        feature_distances = pairwise_distances(scaled_feature_vectors, euclidean_sparse_distance)
        combined_distances = combine_distance_matrices(
            markov_distances,
            feature_distances,
            left_weight=markov_weight,
        )
        method_name = f"M4_markov_{feature_set_name}"
        methods[method_name] = {
            "metrics": evaluate_retrieval(labels, combined_distances, k_values=(1, 3, 5, 10)),
            "paired_tests_M_minus_M2": _paired_tests_against_baseline(
                labels,
                markov_distances,
                combined_distances,
            ),
        }

    return {
        "dataset": {
            "path": str(dataset_dir),
            "limit_per_task": limit_per_task,
            "sample_seed": sample_seed,
        },
        "records": {
            "valid": len(records),
            "syntax_errors": len(errors),
            "task_count": len(set(labels)),
        },
        "parameters": {
            "markov_weight": markov_weight,
            "feature_weight": 1.0 - markov_weight,
            "baseline_kind": baseline_kind,
        },
        "methods": methods,
        "errors": errors[:20],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ablate AST geometry feature groups against Markov/JSD baseline."
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("data/dta_zenodo_7799972/extracted"),
    )
    parser.add_argument("--limit-per-task", type=int, default=20)
    parser.add_argument("--sample-seed", type=int, default=13)
    parser.add_argument("--markov-weight", type=float, default=0.85)
    parser.add_argument(
        "--baseline-kind",
        choices=["flat_markov", "transition_count"],
        default="flat_markov",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/dta_feature_ablation_limit20_seed13_w085.json"),
    )
    args = parser.parse_args()

    payload = run_feature_ablation(
        args.dataset_dir,
        limit_per_task=args.limit_per_task,
        sample_seed=args.sample_seed,
        markov_weight=args.markov_weight,
        baseline_kind=args.baseline_kind,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"wrote {args.output}")
    for method_name, method_payload in payload["methods"].items():
        metrics = method_payload["metrics"]
        print(
            f"{method_name}: "
            f"top1={metrics['top1_accuracy']:.4f} "
            f"mrr={metrics['mrr']:.4f} "
            f"map={metrics['map']:.4f} "
            f"r10={metrics['recall@10']:.4f}"
        )


if __name__ == "__main__":
    main()
