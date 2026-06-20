from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from geometry_profile_research.analysis import GeometryProfile, geometry_profile_for_ast_source
from geometry_profile_research.ast_features import ast_markov_probabilities
from geometry_profile_research.dta import DtaRecord, load_dta_records, stratified_sample_records
from geometry_profile_research.geometry_features import geometry_feature_sets
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


def _markov_vector(record: DtaRecord) -> dict[str, float]:
    probabilities = ast_markov_probabilities(record.code)
    return {f"{parent}->{child}": value for (parent, child), value in probabilities.items()}


def _experiment_records(
    dataset_dir: Path,
    *,
    limit_per_task: int,
    sample_seed: int,
) -> tuple[list[DtaRecord], list[dict[str, float]], list[dict[str, dict[str, float]]], list[dict[str, str]]]:
    all_records = load_dta_records(dataset_dir)
    records = stratified_sample_records(
        all_records,
        per_task=limit_per_task,
        seed=sample_seed,
    )
    valid_records: list[DtaRecord] = []
    markov_vectors: list[dict[str, float]] = []
    geometry_vectors: list[dict[str, dict[str, float]]] = []
    errors: list[dict[str, str]] = []

    for record in records:
        try:
            markov_vector = _markov_vector(record)
            profile = geometry_profile_for_ast_source(record.code)
        except SyntaxError as exc:
            errors.append(
                {
                    "record_id": record.record_id,
                    "source_file": record.source_file,
                    "error": str(exc),
                }
            )
            continue

        valid_records.append(record)
        markov_vectors.append(markov_vector)
        geometry_vectors.append(geometry_feature_sets(profile))

    return valid_records, markov_vectors, geometry_vectors, errors


def run_retrieval_experiment(
    dataset_dir: Path,
    *,
    limit_per_task: int,
    combined_markov_weight: float,
    sample_seed: int,
) -> dict[str, object]:
    records, markov_vectors, geometry_vectors, errors = _experiment_records(
        dataset_dir,
        limit_per_task=limit_per_task,
        sample_seed=sample_seed,
    )
    labels = [record.task_id for record in records]

    all_geometry_vectors = [feature_sets["all"] for feature_sets in geometry_vectors]
    scaled_geometry_vectors = zscore_feature_vectors(all_geometry_vectors)
    markov_distances = pairwise_distances(markov_vectors, jensen_shannon_divergence)
    geometry_distances = pairwise_distances(scaled_geometry_vectors, euclidean_sparse_distance)
    combined_distances = combine_distance_matrices(
        markov_distances,
        geometry_distances,
        left_weight=combined_markov_weight,
    )
    k_values = (1, 3, 5, 10)
    markov_scores = per_query_retrieval_scores(labels, markov_distances, k_values=k_values)
    combined_scores = per_query_retrieval_scores(labels, combined_distances, k_values=k_values)

    paired_tests: dict[str, object] = {}
    for baseline_key, candidate_key, output_key in [
        ("top1", "top1", "top1_accuracy"),
        ("reciprocal_rank", "reciprocal_rank", "mrr"),
        ("average_precision", "average_precision", "map"),
        ("recall@10", "recall@10", "recall@10"),
    ]:
        baseline_values = [score[baseline_key] for score in markov_scores]
        candidate_values = [score[candidate_key] for score in combined_scores]
        ci_low, ci_high = paired_bootstrap_ci(
            baseline_values,
            candidate_values,
            iterations=5000,
            seed=17,
        )
        paired_tests[output_key] = {
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
                seed=17,
            ),
        }

    return {
        "dataset": {
            "path": str(dataset_dir),
            "limit_per_task": limit_per_task,
            "sample_seed": sample_seed,
        },
        "records": {
            "loaded": len(records) + len(errors),
            "valid": len(records),
            "syntax_errors": len(errors),
            "task_count": len(set(labels)),
        },
        "methods": {
            "M2_markov_jsd": evaluate_retrieval(labels, markov_distances, k_values=k_values),
            "M3_geometry_only": evaluate_retrieval(
                labels,
                geometry_distances,
                k_values=k_values,
            ),
            "M4_markov_geometry": evaluate_retrieval(
                labels,
                combined_distances,
                k_values=k_values,
            ),
        },
        "paired_tests_M4_minus_M2": paired_tests,
        "parameters": {
            "combined_markov_weight": combined_markov_weight,
            "combined_geometry_weight": 1.0 - combined_markov_weight,
            "markov_distance": "Jensen-Shannon divergence over flattened AST transition probabilities",
            "geometry_distance": "Euclidean distance over z-scored AST geometry profile features",
        },
        "errors": errors[:20],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare DTA retrieval with Markov/JSD, geometry, and combined distances."
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("data/dta_zenodo_7799972/extracted"),
        help="Directory containing task-00.csv ... task-10.csv.",
    )
    parser.add_argument("--limit-per-task", type=int, default=10)
    parser.add_argument("--sample-seed", type=int, default=13)
    parser.add_argument("--combined-markov-weight", type=float, default=0.8)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/dta_retrieval_limit10.json"),
    )
    args = parser.parse_args()

    payload = run_retrieval_experiment(
        args.dataset_dir,
        limit_per_task=args.limit_per_task,
        combined_markov_weight=args.combined_markov_weight,
        sample_seed=args.sample_seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"wrote {args.output}")
    print(json.dumps(payload["records"], ensure_ascii=False, indent=2))
    print(json.dumps(payload["methods"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
