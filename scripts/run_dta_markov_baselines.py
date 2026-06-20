from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from geometry_profile_research.ast_features import (
    ast_markov_probabilities,
    ast_markov_rows,
    ast_node_histogram,
    ast_transition_counts,
)
from geometry_profile_research.dta import load_dta_records, stratified_sample_records
from geometry_profile_research.retrieval import (
    evaluate_retrieval,
    euclidean_sparse_distance,
    jensen_shannon_distance,
    jensen_shannon_divergence,
    pairwise_distances,
    rowwise_markov_jsd,
    zscore_feature_vectors,
)


def _flatten_markov_probabilities(code: str) -> dict[str, float]:
    return {
        f"{parent}->{child}": value
        for (parent, child), value in ast_markov_probabilities(code).items()
    }


def _transition_count_vector(code: str) -> dict[str, float]:
    return {
        f"{parent}->{child}": float(value)
        for (parent, child), value in ast_transition_counts(code).items()
    }


def _node_histogram_vector(code: str) -> dict[str, float]:
    return {node_type: float(count) for node_type, count in ast_node_histogram(code).items()}


def run_markov_baseline_comparison(
    dataset_dir: Path,
    *,
    limit_per_task: int,
    sample_seed: int,
) -> dict[str, object]:
    records = stratified_sample_records(
        load_dta_records(dataset_dir),
        per_task=limit_per_task,
        seed=sample_seed,
    )
    valid_records = []
    labels: list[int] = []
    flat_markov_vectors: list[dict[str, float]] = []
    transition_count_vectors: list[dict[str, float]] = []
    histogram_vectors: list[dict[str, float]] = []
    markov_rows: list[dict[str, dict[str, float]]] = []
    errors: list[dict[str, str]] = []

    for record in records:
        try:
            flat_markov_vectors.append(_flatten_markov_probabilities(record.code))
            transition_count_vectors.append(_transition_count_vector(record.code))
            histogram_vectors.append(_node_histogram_vector(record.code))
            markov_rows.append(ast_markov_rows(record.code))
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
        labels.append(record.task_id)

    scaled_histograms = zscore_feature_vectors(histogram_vectors)
    methods = {
        "flat_markov_jsd_divergence": pairwise_distances(
            flat_markov_vectors,
            jensen_shannon_divergence,
        ),
        "flat_markov_jsd_distance": pairwise_distances(
            flat_markov_vectors,
            jensen_shannon_distance,
        ),
        "transition_count_jsd_divergence": pairwise_distances(
            transition_count_vectors,
            jensen_shannon_divergence,
        ),
        "transition_count_jsd_distance": pairwise_distances(
            transition_count_vectors,
            jensen_shannon_distance,
        ),
        "rowwise_markov_jsd": pairwise_distances(
            markov_rows,
            rowwise_markov_jsd,
        ),
        "ast_histogram_euclidean": pairwise_distances(
            scaled_histograms,
            euclidean_sparse_distance,
        ),
    }

    return {
        "dataset": {
            "path": str(dataset_dir),
            "limit_per_task": limit_per_task,
            "sample_seed": sample_seed,
        },
        "records": {
            "valid": len(valid_records),
            "syntax_errors": len(errors),
            "task_count": len(set(labels)),
        },
        "methods": {
            method_name: evaluate_retrieval(labels, distances, k_values=(1, 3, 5, 10))
            for method_name, distances in methods.items()
        },
        "errors": errors[:20],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare AST histogram and Markov-chain distance definitions."
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("data/dta_zenodo_7799972/extracted"),
    )
    parser.add_argument("--limit-per-task", type=int, default=50)
    parser.add_argument("--sample-seed", type=int, default=13)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/dta_markov_baselines_limit50_seed13.json"),
    )
    args = parser.parse_args()

    payload = run_markov_baseline_comparison(
        args.dataset_dir,
        limit_per_task=args.limit_per_task,
        sample_seed=args.sample_seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"wrote {args.output}")
    for method_name, metrics in payload["methods"].items():
        print(
            f"{method_name}: "
            f"top1={metrics['top1_accuracy']:.4f} "
            f"mrr={metrics['mrr']:.4f} "
            f"map={metrics['map']:.4f} "
            f"r10={metrics['recall@10']:.4f}"
        )


if __name__ == "__main__":
    main()
