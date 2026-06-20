from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from geometry_profile_research.analysis import geometry_profile_for_ast_source
from geometry_profile_research.dta import (
    DtaRecord,
    load_dta_records,
    stratified_validation_test_split,
)
from geometry_profile_research.geometry_features import geometry_feature_sets
from geometry_profile_research.retrieval import (
    apply_zscore_scaler,
    combine_distance_matrices,
    evaluate_retrieval,
    euclidean_sparse_distance,
    fit_zscore_scaler,
    jensen_shannon_divergence,
    median_nonzero_distance,
    paired_bootstrap_ci,
    paired_permutation_p_value,
    pairwise_distances,
    per_query_retrieval_scores,
)
from scripts.run_dta_markov_baselines import _flatten_markov_probabilities, _transition_count_vector


def _feature_vector_from_sets(
    feature_sets: dict[str, dict[str, float]],
    feature_set_name: str,
) -> dict[str, float]:
    if feature_set_name == "shape":
        return {**feature_sets["size_depth"], **feature_sets["branching"]}
    if feature_set_name not in feature_sets:
        raise ValueError(f"unknown feature set: {feature_set_name}")
    return dict(feature_sets[feature_set_name])


def _baseline_vector(record: DtaRecord, baseline_kind: str) -> dict[str, float]:
    if baseline_kind == "transition_count":
        return _transition_count_vector(record.code)
    if baseline_kind == "flat_markov":
        return _flatten_markov_probabilities(record.code)
    raise ValueError("baseline_kind must be 'transition_count' or 'flat_markov'")


def _vectors_for_records(
    records: list[DtaRecord],
    *,
    baseline_kind: str,
    feature_set_name: str,
) -> tuple[list[int], list[dict[str, float]], list[dict[str, float]], list[dict[str, str]]]:
    labels: list[int] = []
    baseline_vectors: list[dict[str, float]] = []
    feature_vectors: list[dict[str, float]] = []
    errors: list[dict[str, str]] = []

    for record in records:
        try:
            profile = geometry_profile_for_ast_source(record.code)
            feature_sets = geometry_feature_sets(profile)
            baseline_vectors.append(_baseline_vector(record, baseline_kind))
            feature_vectors.append(_feature_vector_from_sets(feature_sets, feature_set_name))
            labels.append(record.task_id)
        except SyntaxError as exc:
            errors.append(
                {
                    "record_id": record.record_id,
                    "source_file": record.source_file,
                    "error": str(exc),
                }
            )
    return labels, baseline_vectors, feature_vectors, errors


def _paired_tests(
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
            seed=23,
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
                seed=23,
            ),
        }
    return tests


def _combined_distances(
    baseline_distances: list[list[float]],
    feature_distances: list[list[float]],
    *,
    weight: float,
    baseline_scale: float,
    feature_scale: float,
) -> list[list[float]]:
    if weight == 1.0:
        return baseline_distances
    return combine_distance_matrices(
        baseline_distances,
        feature_distances,
        left_weight=weight,
        left_scale=baseline_scale,
        right_scale=feature_scale,
    )


def run_confirmatory_split(
    dataset_dir: Path,
    *,
    validation_per_task: int,
    test_per_task: int,
    split_seed: int,
    baseline_kind: str,
    feature_set_name: str,
    weights: list[float],
) -> dict[str, object]:
    split = stratified_validation_test_split(
        load_dta_records(dataset_dir),
        validation_per_task=validation_per_task,
        test_per_task=test_per_task,
        seed=split_seed,
    )

    validation_labels, validation_baseline_vectors, validation_feature_vectors, validation_errors = (
        _vectors_for_records(
            split["validation"],
            baseline_kind=baseline_kind,
            feature_set_name=feature_set_name,
        )
    )
    test_labels, test_baseline_vectors, test_feature_vectors, test_errors = _vectors_for_records(
        split["test"],
        baseline_kind=baseline_kind,
        feature_set_name=feature_set_name,
    )

    scaler = fit_zscore_scaler(validation_feature_vectors)
    scaled_validation_features = apply_zscore_scaler(validation_feature_vectors, scaler)
    scaled_test_features = apply_zscore_scaler(test_feature_vectors, scaler)

    validation_baseline_distances = pairwise_distances(
        validation_baseline_vectors,
        jensen_shannon_divergence,
    )
    validation_feature_distances = pairwise_distances(
        scaled_validation_features,
        euclidean_sparse_distance,
    )
    test_baseline_distances = pairwise_distances(
        test_baseline_vectors,
        jensen_shannon_divergence,
    )
    test_feature_distances = pairwise_distances(
        scaled_test_features,
        euclidean_sparse_distance,
    )

    baseline_scale = median_nonzero_distance(validation_baseline_distances)
    feature_scale = median_nonzero_distance(validation_feature_distances)

    validation_results = []
    for weight in weights:
        distances = _combined_distances(
            validation_baseline_distances,
            validation_feature_distances,
            weight=weight,
            baseline_scale=baseline_scale,
            feature_scale=feature_scale,
        )
        validation_results.append(
            {
                "markov_weight": weight,
                "geometry_weight": 1.0 - weight,
                "metrics": evaluate_retrieval(validation_labels, distances, k_values=(1, 3, 5, 10)),
            }
        )

    selected = max(
        validation_results,
        key=lambda row: (
            row["metrics"]["map"],
            row["metrics"]["recall@10"],
            row["markov_weight"],
        ),
    )
    selected_weight = float(selected["markov_weight"])
    test_candidate_distances = _combined_distances(
        test_baseline_distances,
        test_feature_distances,
        weight=selected_weight,
        baseline_scale=baseline_scale,
        feature_scale=feature_scale,
    )

    return {
        "protocol": {
            "status": "confirmatory_split_after_exploratory_design",
            "selection_rule": "choose markov_weight maximizing validation MAP; break ties by Recall@10, then larger Markov weight",
            "test_rule": "apply selected weight to disjoint test split; do not retune on test",
            "scaling_rule": "fit z-score scaler and distance median scales on validation only",
        },
        "dataset": {
            "path": str(dataset_dir),
            "validation_per_task": validation_per_task,
            "test_per_task": test_per_task,
            "split_seed": split_seed,
        },
        "parameters": {
            "baseline_kind": baseline_kind,
            "feature_set_name": feature_set_name,
            "candidate_weights": weights,
            "selected_markov_weight": selected_weight,
            "selected_geometry_weight": 1.0 - selected_weight,
            "validation_baseline_scale": baseline_scale,
            "validation_feature_scale": feature_scale,
        },
        "records": {
            "validation": len(validation_labels),
            "test": len(test_labels),
            "validation_syntax_errors": len(validation_errors),
            "test_syntax_errors": len(test_errors),
            "task_count_validation": len(set(validation_labels)),
            "task_count_test": len(set(test_labels)),
        },
        "validation_results": validation_results,
        "test_results": {
            "baseline": evaluate_retrieval(test_labels, test_baseline_distances, k_values=(1, 3, 5, 10)),
            "candidate": evaluate_retrieval(test_labels, test_candidate_distances, k_values=(1, 3, 5, 10)),
            "paired_tests_candidate_minus_baseline": _paired_tests(
                test_labels,
                test_baseline_distances,
                test_candidate_distances,
            ),
        },
        "errors": {
            "validation": validation_errors[:20],
            "test": test_errors[:20],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a validation/test confirmatory split for DTA retrieval."
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("data/dta_zenodo_7799972/extracted"),
    )
    parser.add_argument("--validation-per-task", type=int, default=20)
    parser.add_argument("--test-per-task", type=int, default=50)
    parser.add_argument("--split-seed", type=int, default=101)
    parser.add_argument(
        "--baseline-kind",
        choices=["transition_count", "flat_markov"],
        default="transition_count",
    )
    parser.add_argument(
        "--feature-set",
        choices=["length_only", "size_depth", "branching", "metric_distortion", "shape", "all"],
        default="all",
    )
    parser.add_argument(
        "--weights",
        nargs="+",
        type=float,
        default=[1.0, 0.99, 0.98, 0.97, 0.95, 0.9, 0.85, 0.8],
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/dta_confirmatory_split_transition_count_all_v20_t50_seed101.json"),
    )
    args = parser.parse_args()

    payload = run_confirmatory_split(
        args.dataset_dir,
        validation_per_task=args.validation_per_task,
        test_per_task=args.test_per_task,
        split_seed=args.split_seed,
        baseline_kind=args.baseline_kind,
        feature_set_name=args.feature_set,
        weights=args.weights,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"wrote {args.output}")
    print(
        "selected weight:",
        payload["parameters"]["selected_markov_weight"],
        "geometry:",
        payload["parameters"]["selected_geometry_weight"],
    )
    print("test baseline:", json.dumps(payload["test_results"]["baseline"], ensure_ascii=False))
    print("test candidate:", json.dumps(payload["test_results"]["candidate"], ensure_ascii=False))
    print(
        "test paired:",
        json.dumps(
            payload["test_results"]["paired_tests_candidate_minus_baseline"],
            ensure_ascii=False,
        ),
    )


if __name__ == "__main__":
    main()
