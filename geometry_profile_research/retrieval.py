from __future__ import annotations

import math
import random
from itertools import combinations
from statistics import fmean, pstdev
from typing import Callable, Mapping, Sequence


SparseVector = Mapping[str, float]
DistanceMatrix = list[list[float]]
ZScoreScaler = dict[str, dict[str, float] | list[str]]
Residualizer = dict[str, object]


def _positive_normalized(vector: SparseVector) -> dict[str, float]:
    values = {key: float(value) for key, value in vector.items() if value > 0.0}
    total = sum(values.values())
    if total <= 0.0:
        return {}
    return {key: value / total for key, value in values.items()}


def _kl_divergence_base2(left: Mapping[str, float], right: Mapping[str, float]) -> float:
    total = 0.0
    for key, probability in left.items():
        if probability <= 0.0:
            continue
        reference = right.get(key, 0.0)
        if reference <= 0.0:
            continue
        total += probability * math.log2(probability / reference)
    return total


def jensen_shannon_divergence(left: SparseVector, right: SparseVector) -> float:
    """Compute Jensen-Shannon divergence for sparse non-negative vectors.

    The vectors are normalized to probability distributions before comparison.
    With base-2 logarithms the result is bounded by 1.0.
    """
    p = _positive_normalized(left)
    q = _positive_normalized(right)
    if not p and not q:
        return 0.0
    keys = set(p) | set(q)
    midpoint = {key: 0.5 * p.get(key, 0.0) + 0.5 * q.get(key, 0.0) for key in keys}
    return 0.5 * _kl_divergence_base2(p, midpoint) + 0.5 * _kl_divergence_base2(q, midpoint)


def jensen_shannon_distance(left: SparseVector, right: SparseVector) -> float:
    """Metric Jensen-Shannon distance: square root of JSD."""
    return math.sqrt(jensen_shannon_divergence(left, right))


def rowwise_markov_jsd(
    left: Mapping[str, SparseVector],
    right: Mapping[str, SparseVector],
) -> float:
    """Average JSD between transition rows of two Markov chains.

    Each parent AST type contributes one row-level divergence. This keeps the
    Markov-chain interpretation explicit instead of flattening all transitions
    into one global distribution.
    """
    states = sorted(set(left) | set(right))
    if not states:
        return 0.0
    return fmean(
        jensen_shannon_divergence(left.get(state, {}), right.get(state, {}))
        for state in states
    )


def zscore_feature_vectors(vectors: Sequence[SparseVector]) -> list[dict[str, float]]:
    """Column-standardize sparse numeric feature vectors."""
    return apply_zscore_scaler(vectors, fit_zscore_scaler(vectors))


def fit_zscore_scaler(vectors: Sequence[SparseVector]) -> ZScoreScaler:
    """Fit column means and scales for sparse numeric feature vectors."""
    keys = sorted({key for vector in vectors for key in vector})
    if not vectors:
        return {"keys": [], "means": {}, "scales": {}}

    means: dict[str, float] = {}
    scales: dict[str, float] = {}
    for key in keys:
        column = [float(vector.get(key, 0.0)) for vector in vectors]
        means[key] = fmean(column)
        scale = pstdev(column)
        scales[key] = scale if scale > 1e-12 else 1.0

    return {"keys": keys, "means": means, "scales": scales}


def apply_zscore_scaler(
    vectors: Sequence[SparseVector],
    scaler: ZScoreScaler,
) -> list[dict[str, float]]:
    """Apply a previously fitted z-score scaler."""
    keys = list(scaler["keys"])
    means = scaler["means"]
    scales = scaler["scales"]
    return [
        {
            key: (float(vector.get(key, 0.0)) - means[key]) / scales[key]
            for key in keys
        }
        for vector in vectors
    ]


def _solve_linear_system(matrix: list[list[float]], rhs: list[float]) -> list[float]:
    size = len(rhs)
    augmented = [row[:] + [rhs_value] for row, rhs_value in zip(matrix, rhs)]
    for col in range(size):
        pivot = max(range(col, size), key=lambda row: abs(augmented[row][col]))
        if abs(augmented[pivot][col]) < 1e-12:
            continue
        augmented[col], augmented[pivot] = augmented[pivot], augmented[col]
        pivot_value = augmented[col][col]
        augmented[col] = [value / pivot_value for value in augmented[col]]
        for row in range(size):
            if row == col:
                continue
            factor = augmented[row][col]
            if abs(factor) < 1e-12:
                continue
            augmented[row] = [
                row_value - factor * pivot_row_value
                for row_value, pivot_row_value in zip(augmented[row], augmented[col])
            ]
    return [augmented[row][-1] for row in range(size)]


def fit_residualizer(
    controls: Sequence[SparseVector],
    targets: Sequence[SparseVector],
    *,
    ridge: float = 1e-8,
) -> Residualizer:
    """Fit linear residualization of target features against control features.

    The returned object stores validation-fitted coefficients. Apply it to
    test vectors with `apply_residualizer` to avoid fitting on the test split.
    """
    if len(controls) != len(targets):
        raise ValueError("controls and targets must have the same length")
    control_keys = sorted({key for vector in controls for key in vector})
    target_keys = sorted({key for vector in targets for key in vector})
    design = [
        [1.0, *[float(vector.get(key, 0.0)) for key in control_keys]]
        for vector in controls
    ]
    width = len(control_keys) + 1
    coefficients: dict[str, list[float]] = {}
    for target_key in target_keys:
        xtx = [[0.0 for _ in range(width)] for _ in range(width)]
        xty = [0.0 for _ in range(width)]
        for row, target_vector in zip(design, targets):
            y = float(target_vector.get(target_key, 0.0))
            for i in range(width):
                xty[i] += row[i] * y
                for j in range(width):
                    xtx[i][j] += row[i] * row[j]
        for i in range(width):
            xtx[i][i] += ridge
        coefficients[target_key] = _solve_linear_system(xtx, xty)
    return {
        "control_keys": control_keys,
        "target_keys": target_keys,
        "coefficients": coefficients,
    }


def apply_residualizer(
    controls: Sequence[SparseVector],
    targets: Sequence[SparseVector],
    residualizer: Residualizer,
) -> list[dict[str, float]]:
    """Apply validation-fitted residualization to target feature vectors."""
    if len(controls) != len(targets):
        raise ValueError("controls and targets must have the same length")
    control_keys = list(residualizer["control_keys"])
    target_keys = list(residualizer["target_keys"])
    coefficients = residualizer["coefficients"]
    residuals: list[dict[str, float]] = []
    for control_vector, target_vector in zip(controls, targets):
        row = [1.0, *[float(control_vector.get(key, 0.0)) for key in control_keys]]
        residual_vector: dict[str, float] = {}
        for target_key in target_keys:
            predicted = sum(
                coefficient * row_value
                for coefficient, row_value in zip(coefficients[target_key], row)
            )
            residual_vector[target_key] = float(target_vector.get(target_key, 0.0)) - predicted
        residuals.append(residual_vector)
    return residuals


def euclidean_sparse_distance(left: SparseVector, right: SparseVector) -> float:
    keys = set(left) | set(right)
    return math.sqrt(
        sum((float(left.get(key, 0.0)) - float(right.get(key, 0.0))) ** 2 for key in keys)
    )


def pairwise_distances(
    vectors: Sequence[SparseVector],
    distance: Callable[[SparseVector, SparseVector], float],
) -> DistanceMatrix:
    matrix = [[0.0 for _ in vectors] for _ in vectors]
    for left, right in combinations(range(len(vectors)), 2):
        value = float(distance(vectors[left], vectors[right]))
        matrix[left][right] = value
        matrix[right][left] = value
    return matrix


def _median_nonzero_distance(matrix: DistanceMatrix) -> float:
    values = sorted(
        matrix[row][col]
        for row in range(len(matrix))
        for col in range(row + 1, len(matrix))
        if matrix[row][col] > 0.0
    )
    if not values:
        return 1.0
    middle = len(values) // 2
    if len(values) % 2:
        return values[middle]
    return 0.5 * (values[middle - 1] + values[middle])


def median_nonzero_distance(matrix: DistanceMatrix) -> float:
    """Public wrapper for median non-zero pairwise distance."""
    return _median_nonzero_distance(matrix)


def combine_distance_matrices(
    left: DistanceMatrix,
    right: DistanceMatrix,
    *,
    left_weight: float,
    left_scale: float | None = None,
    right_scale: float | None = None,
) -> DistanceMatrix:
    """Combine two distance matrices after median non-zero scaling."""
    if len(left) != len(right):
        raise ValueError("distance matrices must have the same size")
    if not 0.0 <= left_weight <= 1.0:
        raise ValueError("left_weight must be in [0, 1]")

    left_scale = left_scale if left_scale is not None else _median_nonzero_distance(left)
    right_scale = right_scale if right_scale is not None else _median_nonzero_distance(right)
    if left_scale <= 0.0 or right_scale <= 0.0:
        raise ValueError("distance scales must be positive")
    right_weight = 1.0 - left_weight

    combined: DistanceMatrix = [[0.0 for _ in left] for _ in left]
    for row in range(len(left)):
        if len(left[row]) != len(left) or len(right[row]) != len(right):
            raise ValueError("distance matrices must be square")
        for col in range(len(left)):
            if row == col:
                continue
            combined[row][col] = (
                left_weight * left[row][col] / left_scale
                + right_weight * right[row][col] / right_scale
            )
    return combined


def evaluate_retrieval(
    labels: Sequence[object],
    distances: DistanceMatrix,
    *,
    k_values: Sequence[int] = (1, 5, 10),
) -> dict[str, float | int]:
    """Evaluate leave-one-out same-label retrieval."""
    scores = per_query_retrieval_scores(labels, distances, k_values=k_values)
    metrics: dict[str, float | int] = {
        "queries": len(scores),
        "top1_accuracy": fmean([score["top1"] for score in scores]) if scores else 0.0,
        "mrr": fmean([score["reciprocal_rank"] for score in scores]) if scores else 0.0,
        "map": fmean([score["average_precision"] for score in scores]) if scores else 0.0,
    }
    for k in k_values:
        key = f"recall@{int(k)}"
        metrics[key] = fmean([score[key] for score in scores]) if scores else 0.0
    return metrics


def evaluate_retrieval_by_label(
    labels: Sequence[object],
    distances: DistanceMatrix,
    *,
    k_values: Sequence[int] = (1, 5, 10),
) -> dict[str, dict[str, float | int]]:
    """Evaluate leave-one-out retrieval separately for each query label."""
    scores = per_query_retrieval_scores(labels, distances, k_values=k_values)
    eligible_labels = [
        labels[query_index]
        for query_index, query_label in enumerate(labels)
        if any(index != query_index and label == query_label for index, label in enumerate(labels))
    ]
    grouped: dict[str, list[dict[str, float]]] = {}
    for label, score in zip(eligible_labels, scores):
        grouped.setdefault(str(label), []).append(score)

    by_label: dict[str, dict[str, float | int]] = {}
    for label, label_scores in sorted(grouped.items()):
        metrics: dict[str, float | int] = {
            "queries": len(label_scores),
            "top1_accuracy": fmean([score["top1"] for score in label_scores]),
            "mrr": fmean([score["reciprocal_rank"] for score in label_scores]),
            "map": fmean([score["average_precision"] for score in label_scores]),
        }
        for k in k_values:
            key = f"recall@{int(k)}"
            metrics[key] = fmean([score[key] for score in label_scores])
        by_label[label] = metrics
    return by_label


def per_query_retrieval_records(
    labels: Sequence[object],
    distances: DistanceMatrix,
    *,
    k_values: Sequence[int] = (1, 5, 10),
) -> list[dict[str, float | int | str]]:
    """Return leave-one-out retrieval records with query identity preserved."""
    records: list[dict[str, float | int | str]] = []
    for query_index, score in _iter_query_retrieval_scores(
        labels,
        distances,
        k_values=k_values,
    ):
        records.append(
            {
                "query_index": query_index,
                "query_label": str(labels[query_index]),
                **score,
            }
        )
    return records


def per_query_retrieval_scores(
    labels: Sequence[object],
    distances: DistanceMatrix,
    *,
    k_values: Sequence[int] = (1, 5, 10),
) -> list[dict[str, float]]:
    """Return leave-one-out retrieval scores for every query with positives."""
    return [
        score
        for _, score in _iter_query_retrieval_scores(
            labels,
            distances,
            k_values=k_values,
        )
    ]


def _iter_query_retrieval_scores(
    labels: Sequence[object],
    distances: DistanceMatrix,
    *,
    k_values: Sequence[int] = (1, 5, 10),
) -> list[tuple[int, dict[str, float]]]:
    if len(labels) != len(distances):
        raise ValueError("labels and distance matrix must have the same length")

    scores: list[tuple[int, dict[str, float]]] = []
    normalized_k_values = [int(k) for k in k_values]

    for query_index, query_label in enumerate(labels):
        relevant = {
            index
            for index, label in enumerate(labels)
            if index != query_index and label == query_label
        }
        if not relevant:
            continue

        ranking = sorted(
            (index for index in range(len(labels)) if index != query_index),
            key=lambda index: (distances[query_index][index], index),
        )
        query_scores: dict[str, float] = {
            "top1": 1.0 if ranking[0] in relevant else 0.0,
        }

        first_rank = next(
            rank
            for rank, candidate_index in enumerate(ranking, start=1)
            if candidate_index in relevant
        )
        query_scores["reciprocal_rank"] = 1.0 / first_rank

        relevant_seen = 0
        precisions: list[float] = []
        for rank, candidate_index in enumerate(ranking, start=1):
            if candidate_index not in relevant:
                continue
            relevant_seen += 1
            precisions.append(relevant_seen / rank)
        query_scores["average_precision"] = fmean(precisions) if precisions else 0.0

        for k in normalized_k_values:
            top_k = set(ranking[:k])
            query_scores[f"recall@{k}"] = len(top_k & relevant) / len(relevant)

        scores.append((query_index, query_scores))

    return scores


def _paired_differences(
    baseline: Sequence[float],
    candidate: Sequence[float],
) -> list[float]:
    if len(baseline) != len(candidate):
        raise ValueError("paired samples must have the same length")
    return [float(candidate_value) - float(baseline_value) for baseline_value, candidate_value in zip(baseline, candidate)]


def paired_bootstrap_ci(
    baseline: Sequence[float],
    candidate: Sequence[float],
    *,
    iterations: int = 5000,
    confidence: float = 0.95,
    seed: int = 13,
) -> tuple[float, float]:
    """Bootstrap confidence interval for mean paired improvement."""
    differences = _paired_differences(baseline, candidate)
    if not differences:
        return (0.0, 0.0)

    rng = random.Random(seed)
    estimates: list[float] = []
    for _ in range(iterations):
        sample = [rng.choice(differences) for _ in differences]
        estimates.append(fmean(sample))
    estimates.sort()

    alpha = 1.0 - confidence
    low_index = max(0, min(len(estimates) - 1, int((alpha / 2.0) * len(estimates))))
    high_index = max(0, min(len(estimates) - 1, int((1.0 - alpha / 2.0) * len(estimates)) - 1))
    return (estimates[low_index], estimates[high_index])


def paired_permutation_p_value(
    baseline: Sequence[float],
    candidate: Sequence[float],
    *,
    iterations: int = 5000,
    seed: int = 13,
) -> float:
    """Approximate one-sided paired sign-flip permutation test.

    The null hypothesis is that the candidate is not better than the baseline.
    Smaller values therefore support a positive paired improvement.
    """
    differences = _paired_differences(baseline, candidate)
    if not differences:
        return 1.0

    observed = fmean(differences)
    if observed <= 0.0:
        return 1.0

    rng = random.Random(seed)
    extreme_or_equal = 1
    for _ in range(iterations):
        flipped = [
            difference if rng.random() < 0.5 else -difference
            for difference in differences
        ]
        if fmean(flipped) >= observed:
            extreme_or_equal += 1
    return extreme_or_equal / (iterations + 1)
