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
    evaluate_retrieval_by_label,
    euclidean_sparse_distance,
    fit_zscore_scaler,
    jensen_shannon_divergence,
    median_nonzero_distance,
    pairwise_distances,
)
from scripts.run_dta_confirmatory_split import (
    _baseline_vector,
    _combined_distances,
    _feature_vector_from_sets,
    _paired_tests,
)


def _profile_records(
    records: list[DtaRecord],
    *,
    baseline_kind: str,
) -> tuple[list[int], list[dict[str, float]], list[dict[str, dict[str, float]]], list[dict[str, str]]]:
    labels: list[int] = []
    baseline_vectors: list[dict[str, float]] = []
    feature_sets_by_record: list[dict[str, dict[str, float]]] = []
    errors: list[dict[str, str]] = []

    for record in records:
        try:
            profile = geometry_profile_for_ast_source(record.code)
            labels.append(record.task_id)
            baseline_vectors.append(_baseline_vector(record, baseline_kind))
            feature_sets_by_record.append(geometry_feature_sets(profile))
        except SyntaxError as exc:
            errors.append(
                {
                    "record_id": record.record_id,
                    "source_file": record.source_file,
                    "error": str(exc),
                }
            )
    return labels, baseline_vectors, feature_sets_by_record, errors


def _select_weight(
    validation_labels: list[int],
    validation_baseline_distances: list[list[float]],
    validation_feature_distances: list[list[float]],
    *,
    weights: list[float],
    baseline_scale: float,
    feature_scale: float,
) -> tuple[float, list[dict[str, object]]]:
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
    return float(selected["markov_weight"]), validation_results


def run_confirmatory_feature_sweep(
    dataset_dir: Path,
    *,
    validation_per_task: int,
    test_per_task: int,
    split_seed: int,
    baseline_kind: str,
    feature_set_names: list[str],
    weights: list[float],
) -> dict[str, object]:
    split = stratified_validation_test_split(
        load_dta_records(dataset_dir),
        validation_per_task=validation_per_task,
        test_per_task=test_per_task,
        seed=split_seed,
    )

    validation_labels, validation_baseline_vectors, validation_feature_sets, validation_errors = (
        _profile_records(split["validation"], baseline_kind=baseline_kind)
    )
    test_labels, test_baseline_vectors, test_feature_sets, test_errors = _profile_records(
        split["test"],
        baseline_kind=baseline_kind,
    )

    validation_baseline_distances = pairwise_distances(
        validation_baseline_vectors,
        jensen_shannon_divergence,
    )
    test_baseline_distances = pairwise_distances(
        test_baseline_vectors,
        jensen_shannon_divergence,
    )
    baseline_scale = median_nonzero_distance(validation_baseline_distances)

    feature_set_results: dict[str, object] = {}
    for feature_set_name in feature_set_names:
        validation_feature_vectors = [
            _feature_vector_from_sets(feature_sets, feature_set_name)
            for feature_sets in validation_feature_sets
        ]
        test_feature_vectors = [
            _feature_vector_from_sets(feature_sets, feature_set_name)
            for feature_sets in test_feature_sets
        ]
        scaler = fit_zscore_scaler(validation_feature_vectors)
        scaled_validation_features = apply_zscore_scaler(validation_feature_vectors, scaler)
        scaled_test_features = apply_zscore_scaler(test_feature_vectors, scaler)
        validation_feature_distances = pairwise_distances(
            scaled_validation_features,
            euclidean_sparse_distance,
        )
        test_feature_distances = pairwise_distances(
            scaled_test_features,
            euclidean_sparse_distance,
        )
        feature_scale = median_nonzero_distance(validation_feature_distances)
        selected_weight, validation_results = _select_weight(
            validation_labels,
            validation_baseline_distances,
            validation_feature_distances,
            weights=weights,
            baseline_scale=baseline_scale,
            feature_scale=feature_scale,
        )
        test_candidate_distances = combine_distance_matrices(
            test_baseline_distances,
            test_feature_distances,
            left_weight=selected_weight,
            left_scale=baseline_scale,
            right_scale=feature_scale,
        )
        feature_set_results[feature_set_name] = {
            "selected_markov_weight": selected_weight,
            "selected_geometry_weight": 1.0 - selected_weight,
            "validation_feature_scale": feature_scale,
            "validation_results": validation_results,
            "test_results": {
                "candidate": evaluate_retrieval(
                    test_labels,
                    test_candidate_distances,
                    k_values=(1, 3, 5, 10),
                ),
                "candidate_by_task": evaluate_retrieval_by_label(
                    test_labels,
                    test_candidate_distances,
                    k_values=(1, 3, 5, 10),
                ),
                "paired_tests_candidate_minus_baseline": _paired_tests(
                    test_labels,
                    test_baseline_distances,
                    test_candidate_distances,
                ),
            },
        }

    return {
        "protocol": {
            "status": "confirmatory_feature_set_control_after_exploratory_design",
            "selection_rule": "for each feature set choose markov_weight maximizing validation MAP; break ties by Recall@10, then larger Markov weight",
            "test_rule": "apply each selected weight to the same disjoint test split; do not retune on test",
            "scaling_rule": "fit each z-score scaler and distance median scale on validation only",
        },
        "dataset": {
            "path": str(dataset_dir),
            "validation_per_task": validation_per_task,
            "test_per_task": test_per_task,
            "split_seed": split_seed,
        },
        "parameters": {
            "baseline_kind": baseline_kind,
            "feature_set_names": feature_set_names,
            "candidate_weights": weights,
            "validation_baseline_scale": baseline_scale,
        },
        "records": {
            "validation": len(validation_labels),
            "test": len(test_labels),
            "validation_syntax_errors": len(validation_errors),
            "test_syntax_errors": len(test_errors),
            "task_count_validation": len(set(validation_labels)),
            "task_count_test": len(set(test_labels)),
        },
        "baseline": {
            "validation": evaluate_retrieval(
                validation_labels,
                validation_baseline_distances,
                k_values=(1, 3, 5, 10),
            ),
            "test": evaluate_retrieval(
                test_labels,
                test_baseline_distances,
                k_values=(1, 3, 5, 10),
            ),
            "test_by_task": evaluate_retrieval_by_label(
                test_labels,
                test_baseline_distances,
                k_values=(1, 3, 5, 10),
            ),
        },
        "feature_set_results": feature_set_results,
        "errors": {
            "validation": validation_errors[:20],
            "test": test_errors[:20],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run confirmatory validation/test controls for multiple AST geometry feature sets."
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
        "--feature-sets",
        nargs="+",
        default=["length_only", "size_depth", "branching", "metric_distortion", "shape", "all"],
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
        default=Path("outputs/dta_confirmatory_feature_sweep_transition_count_v20_t50_seed101.json"),
    )
    args = parser.parse_args()

    payload = run_confirmatory_feature_sweep(
        args.dataset_dir,
        validation_per_task=args.validation_per_task,
        test_per_task=args.test_per_task,
        split_seed=args.split_seed,
        baseline_kind=args.baseline_kind,
        feature_set_names=args.feature_sets,
        weights=args.weights,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"wrote {args.output}")
    print("test baseline:", json.dumps(payload["baseline"]["test"], ensure_ascii=False))
    for feature_set_name, result in payload["feature_set_results"].items():
        print(
            feature_set_name,
            "selected_weight=",
            result["selected_markov_weight"],
            "test=",
            json.dumps(result["test_results"]["candidate"], ensure_ascii=False),
        )


if __name__ == "__main__":
    main()
