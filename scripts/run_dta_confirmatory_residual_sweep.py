from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from geometry_profile_research.retrieval import (
    apply_residualizer,
    apply_zscore_scaler,
    combine_distance_matrices,
    euclidean_sparse_distance,
    fit_residualizer,
    fit_zscore_scaler,
    jensen_shannon_divergence,
    median_nonzero_distance,
    pairwise_distances,
)
from scripts.run_dta_confirmatory_feature_sweep import _profile_records, _select_weight
from scripts.run_dta_confirmatory_split import _paired_tests
from geometry_profile_research.dta import load_dta_records, stratified_validation_test_split
from geometry_profile_research.retrieval import evaluate_retrieval


def _without_keys(vector: dict[str, float], keys_to_remove: set[str]) -> dict[str, float]:
    return {key: value for key, value in vector.items() if key not in keys_to_remove}


def _target_vector(feature_sets: dict[str, dict[str, float]], target_name: str) -> dict[str, float]:
    length_keys = set(feature_sets["length_only"])
    if target_name == "residual_size_depth":
        return _without_keys(feature_sets["size_depth"], length_keys)
    if target_name == "residual_branching":
        return dict(feature_sets["branching"])
    if target_name == "residual_metric_distortion":
        return dict(feature_sets["metric_distortion"])
    if target_name == "residual_shape":
        return {
            **_without_keys(feature_sets["size_depth"], length_keys),
            **feature_sets["branching"],
        }
    if target_name == "residual_all_nonlength":
        return {
            **_without_keys(feature_sets["size_depth"], length_keys),
            **feature_sets["branching"],
            **feature_sets["metric_distortion"],
        }
    raise ValueError(f"unknown residual target: {target_name}")


def run_confirmatory_residual_sweep(
    dataset_dir: Path,
    *,
    validation_per_task: int,
    test_per_task: int,
    split_seed: int,
    baseline_kind: str,
    residual_targets: list[str],
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

    validation_controls = [feature_sets["length_only"] for feature_sets in validation_feature_sets]
    test_controls = [feature_sets["length_only"] for feature_sets in test_feature_sets]

    feature_set_results: dict[str, object] = {}
    for target_name in residual_targets:
        validation_targets = [
            _target_vector(feature_sets, target_name)
            for feature_sets in validation_feature_sets
        ]
        test_targets = [
            _target_vector(feature_sets, target_name)
            for feature_sets in test_feature_sets
        ]
        residualizer = fit_residualizer(validation_controls, validation_targets)
        validation_residuals = apply_residualizer(
            validation_controls,
            validation_targets,
            residualizer,
        )
        test_residuals = apply_residualizer(test_controls, test_targets, residualizer)

        scaler = fit_zscore_scaler(validation_residuals)
        scaled_validation_features = apply_zscore_scaler(validation_residuals, scaler)
        scaled_test_features = apply_zscore_scaler(test_residuals, scaler)
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
        feature_set_results[target_name] = {
            "control_set": "length_only",
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
                "paired_tests_candidate_minus_baseline": _paired_tests(
                    test_labels,
                    test_baseline_distances,
                    test_candidate_distances,
                ),
            },
        }

    return {
        "protocol": {
            "status": "confirmatory_residual_feature_set_control",
            "control_set": "length_only",
            "selection_rule": "for each residual feature set choose markov_weight maximizing validation MAP; break ties by Recall@10, then larger Markov weight",
            "test_rule": "fit residualizer/scaler on validation only and apply to disjoint test split",
        },
        "dataset": {
            "path": str(dataset_dir),
            "validation_per_task": validation_per_task,
            "test_per_task": test_per_task,
            "split_seed": split_seed,
        },
        "parameters": {
            "baseline_kind": baseline_kind,
            "residual_targets": residual_targets,
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
        },
        "feature_set_results": feature_set_results,
        "errors": {
            "validation": validation_errors[:20],
            "test": test_errors[:20],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run confirmatory residual controls after removing length_only effects."
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
        "--residual-targets",
        nargs="+",
        default=[
            "residual_size_depth",
            "residual_branching",
            "residual_metric_distortion",
            "residual_shape",
            "residual_all_nonlength",
        ],
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
        default=Path("outputs/dta_confirmatory_residual_sweep_transition_count_v20_t50_seed101.json"),
    )
    args = parser.parse_args()
    payload = run_confirmatory_residual_sweep(
        args.dataset_dir,
        validation_per_task=args.validation_per_task,
        test_per_task=args.test_per_task,
        split_seed=args.split_seed,
        baseline_kind=args.baseline_kind,
        residual_targets=args.residual_targets,
        weights=args.weights,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {args.output}")
    print("test baseline:", json.dumps(payload["baseline"]["test"], ensure_ascii=False))
    for target_name, result in payload["feature_set_results"].items():
        print(
            target_name,
            "selected_weight=",
            result["selected_markov_weight"],
            "test=",
            json.dumps(result["test_results"]["candidate"], ensure_ascii=False),
        )


if __name__ == "__main__":
    main()
