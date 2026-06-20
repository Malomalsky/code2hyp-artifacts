from __future__ import annotations

import math
import random
from collections import defaultdict
from statistics import fmean, pstdev
from typing import Any, Mapping, Sequence


def summarize_numeric_features_by_label(
    labels: Sequence[object],
    vectors: Sequence[Mapping[str, float]],
) -> list[dict[str, float | int | str]]:
    """Summarize numeric feature distributions within each task label."""
    if len(labels) != len(vectors):
        raise ValueError("labels and vectors must have the same length")

    grouped: dict[str, list[Mapping[str, float]]] = defaultdict(list)
    for label, vector in zip(labels, vectors):
        grouped[str(label)].append(vector)

    keys = sorted({key for vector in vectors for key in vector})
    rows: list[dict[str, float | int | str]] = []
    for label, group_vectors in sorted(grouped.items(), key=lambda item: int(item[0])):
        row: dict[str, float | int | str] = {"task_label": label, "n": len(group_vectors)}
        for key in keys:
            values = [float(vector.get(key, 0.0)) for vector in group_vectors]
            row[f"{key}_mean"] = fmean(values)
            row[f"{key}_std"] = pstdev(values) if len(values) > 1 else 0.0
            row[f"{key}_min"] = min(values)
            row[f"{key}_max"] = max(values)
        rows.append(row)
    return rows


def _average_ranks(values: Sequence[float]) -> list[float]:
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0 for _ in values]
    start = 0
    while start < len(indexed):
        end = start + 1
        while end < len(indexed) and indexed[end][1] == indexed[start][1]:
            end += 1
        average_rank = (start + 1 + end) / 2.0
        for index_pos in range(start, end):
            original_index = indexed[index_pos][0]
            ranks[original_index] = average_rank
        start = end
    return ranks


def _pearson_correlation(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right):
        raise ValueError("vectors must have the same length")
    if len(left) < 2:
        return 0.0
    left_mean = fmean(left)
    right_mean = fmean(right)
    numerator = sum((x - left_mean) * (y - right_mean) for x, y in zip(left, right))
    left_denominator = math.sqrt(sum((x - left_mean) ** 2 for x in left))
    right_denominator = math.sqrt(sum((y - right_mean) ** 2 for y in right))
    denominator = left_denominator * right_denominator
    if denominator <= 1e-12:
        return 0.0
    return numerator / denominator


def spearman_correlation(left: Sequence[float], right: Sequence[float]) -> float:
    """Compute Spearman rank correlation with average ranks for ties."""
    return _pearson_correlation(_average_ranks(left), _average_ranks(right))


def spearman_permutation_p_value(
    left: Sequence[float],
    right: Sequence[float],
    *,
    iterations: int = 10_000,
    seed: int = 13,
) -> float:
    """Two-sided Monte Carlo permutation p-value for Spearman correlation."""
    if len(left) != len(right):
        raise ValueError("vectors must have the same length")
    if len(left) < 3:
        return 1.0
    observed = abs(spearman_correlation(left, right))
    rng = random.Random(seed)
    shuffled = list(right)
    extreme = 1
    for _ in range(iterations):
        rng.shuffle(shuffled)
        if abs(spearman_correlation(left, shuffled)) >= observed:
            extreme += 1
    return extreme / (iterations + 1)


def bootstrap_spearman_ci(
    left: Sequence[float],
    right: Sequence[float],
    *,
    iterations: int = 10_000,
    seed: int = 13,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Bootstrap percentile CI for Spearman correlation over paired observations."""
    if len(left) != len(right):
        raise ValueError("vectors must have the same length")
    if len(left) < 3:
        return (0.0, 0.0)
    rng = random.Random(seed)
    values: list[float] = []
    indices = list(range(len(left)))
    for _ in range(iterations):
        sample = [rng.choice(indices) for _ in indices]
        sampled_left = [left[index] for index in sample]
        sampled_right = [right[index] for index in sample]
        values.append(spearman_correlation(sampled_left, sampled_right))
    values.sort()
    low_index = max(0, int((alpha / 2.0) * len(values)))
    high_index = min(len(values) - 1, int((1.0 - alpha / 2.0) * len(values)))
    return values[low_index], values[high_index]


def leave_one_out_spearman(
    left: Sequence[float],
    right: Sequence[float],
    *,
    labels: Sequence[Any] | None = None,
) -> dict[str, Any]:
    """Assess Spearman sensitivity by omitting each observation once."""
    if len(left) != len(right):
        raise ValueError("vectors must have the same length")
    if len(left) < 4:
        return {
            "n": len(left),
            "rho_min": spearman_correlation(left, right),
            "rho_max": spearman_correlation(left, right),
            "omit_label_at_min": "",
            "omit_label_at_max": "",
        }
    labels = labels if labels is not None else list(range(len(left)))
    results: list[tuple[float, Any]] = []
    for omitted_index in range(len(left)):
        kept_left = [value for index, value in enumerate(left) if index != omitted_index]
        kept_right = [value for index, value in enumerate(right) if index != omitted_index]
        results.append((spearman_correlation(kept_left, kept_right), labels[omitted_index]))
    min_rho, min_label = min(results, key=lambda item: item[0])
    max_rho, max_label = max(results, key=lambda item: item[0])
    return {
        "n": len(left),
        "rho_min": min_rho,
        "rho_max": max_rho,
        "omit_label_at_min": str(min_label),
        "omit_label_at_max": str(max_label),
    }


def benjamini_hochberg_q_values(p_values: Sequence[float]) -> list[float]:
    """Return Benjamini-Hochberg FDR-adjusted q-values in original order."""
    if not p_values:
        return []
    indexed = sorted(enumerate(float(value) for value in p_values), key=lambda item: item[1])
    total = len(indexed)
    adjusted = [1.0 for _ in indexed]
    running_min = 1.0
    for reverse_rank, (original_index, p_value) in enumerate(reversed(indexed), start=1):
        rank = total - reverse_rank + 1
        candidate = min(1.0, p_value * total / rank)
        running_min = min(running_min, candidate)
        adjusted[original_index] = running_min
    return adjusted
