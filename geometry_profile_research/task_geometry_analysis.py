from __future__ import annotations

from collections import defaultdict
import random
from statistics import fmean, pstdev
from typing import Any, Iterable


DEFAULT_TASK_GEOMETRY_METRICS = [
    "node_count",
    "ball_size_mean_r3",
    "forman_mean",
    "forman_negative_mass",
    "forman_positive_mass",
    "ollivier_mean",
    "ollivier_negative_mass",
    "ollivier_near_zero_mass",
]

DEFAULT_CURVATURE_RESPONSE_METRICS = [
    "forman_mean",
    "forman_negative_mass",
    "forman_positive_mass",
    "ollivier_mean",
    "ollivier_negative_mass",
    "ollivier_near_zero_mass",
]

DEFAULT_SIZE_CONTROL_METRICS = [
    "node_count",
    "ball_size_mean_r3",
]


def summarize_task_geometry(
    rows: Iterable[dict[str, Any]],
    metrics: Iterable[str] = DEFAULT_TASK_GEOMETRY_METRICS,
) -> list[dict[str, Any]]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[int(row["task_id"])].append(row)

    summary: list[dict[str, Any]] = []
    for task_id in sorted(grouped):
        task_rows = grouped[task_id]
        summary_row: dict[str, Any] = {"task_id": task_id, "n": len(task_rows)}
        for metric in metrics:
            values = [_as_float(row[metric]) for row in task_rows]
            summary_row[f"{metric}_mean"] = fmean(values)
            summary_row[f"{metric}_std"] = pstdev(values) if len(values) > 1 else 0.0
            summary_row[f"{metric}_min"] = min(values)
            summary_row[f"{metric}_max"] = max(values)
        summary.append(summary_row)
    return summary


def compute_metric_effect_sizes(
    rows: Iterable[dict[str, Any]],
    metrics: Iterable[str] = DEFAULT_TASK_GEOMETRY_METRICS,
) -> list[dict[str, Any]]:
    materialized = list(rows)
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in materialized:
        grouped[int(row["task_id"])].append(row)

    effect_rows: list[dict[str, Any]] = []
    for metric in metrics:
        values = [_as_float(row[metric]) for row in materialized]
        overall_mean = fmean(values)
        total_ss = sum((value - overall_mean) ** 2 for value in values)
        between_ss = 0.0
        task_means: list[float] = []
        for task_rows in grouped.values():
            task_values = [_as_float(row[metric]) for row in task_rows]
            task_mean = fmean(task_values)
            task_means.append(task_mean)
            between_ss += len(task_values) * (task_mean - overall_mean) ** 2

        eta_squared = between_ss / total_ss if total_ss else 0.0
        effect_rows.append(
            {
                "metric": metric,
                "n": len(values),
                "n_tasks": len(grouped),
                "overall_mean": overall_mean,
                "overall_std": pstdev(values) if len(values) > 1 else 0.0,
                "min_task_mean": min(task_means),
                "max_task_mean": max(task_means),
                "range_task_mean": max(task_means) - min(task_means),
                "eta_squared_task": eta_squared,
            }
        )
    return effect_rows


def compute_residual_effect_sizes(
    rows: Iterable[dict[str, Any]],
    response_metrics: Iterable[str] = DEFAULT_CURVATURE_RESPONSE_METRICS,
    covariates: Iterable[str] = DEFAULT_SIZE_CONTROL_METRICS,
) -> list[dict[str, Any]]:
    materialized = list(rows)
    covariate_list = list(covariates)
    output: list[dict[str, Any]] = []
    for metric in response_metrics:
        residuals, r_squared, coefficients = residualize_metric(
            materialized,
            response_metric=metric,
            covariates=covariate_list,
        )
        groups = [int(row["task_id"]) for row in materialized]
        output.append(
            {
                "metric": metric,
                "covariates": ",".join(covariate_list),
                "n": len(materialized),
                "n_tasks": len(set(groups)),
                "covariate_r_squared": r_squared,
                "eta_squared_task_residual": eta_squared_by_group(residuals, groups),
                "residual_std": pstdev(residuals) if len(residuals) > 1 else 0.0,
                **{
                    f"beta_{index}": coefficient
                    for index, coefficient in enumerate(coefficients)
                },
            }
        )
    return output


def compute_permutation_tests(
    rows: Iterable[dict[str, Any]],
    metrics: Iterable[str] = DEFAULT_TASK_GEOMETRY_METRICS,
    residual_response_metrics: Iterable[str] = DEFAULT_CURVATURE_RESPONSE_METRICS,
    covariates: Iterable[str] = DEFAULT_SIZE_CONTROL_METRICS,
    *,
    permutations: int = 1000,
    seed: int = 13,
) -> list[dict[str, Any]]:
    materialized = list(rows)
    groups = [int(row["task_id"]) for row in materialized]
    rng = random.Random(seed)

    output: list[dict[str, Any]] = []
    for metric in metrics:
        values = [_as_float(row[metric]) for row in materialized]
        observed, p_value = permutation_p_value_eta_squared(
            values,
            groups,
            permutations=permutations,
            rng=rng,
        )
        output.append(
            {
                "analysis": "raw",
                "metric": metric,
                "covariates": "",
                "n": len(values),
                "n_tasks": len(set(groups)),
                "permutations": permutations,
                "seed": seed,
                "observed_eta_squared": observed,
                "p_value": p_value,
            }
        )

    covariate_list = list(covariates)
    for metric in residual_response_metrics:
        residuals, r_squared, _coefficients = residualize_metric(
            materialized,
            response_metric=metric,
            covariates=covariate_list,
        )
        observed, p_value = permutation_p_value_eta_squared(
            residuals,
            groups,
            permutations=permutations,
            rng=rng,
        )
        output.append(
            {
                "analysis": "residual",
                "metric": metric,
                "covariates": ",".join(covariate_list),
                "n": len(residuals),
                "n_tasks": len(set(groups)),
                "permutations": permutations,
                "seed": seed,
                "covariate_r_squared": r_squared,
                "observed_eta_squared": observed,
                "p_value": p_value,
            }
        )

    return add_holm_correction(output)


def permutation_p_value_eta_squared(
    values: Iterable[float],
    groups: Iterable[int],
    *,
    permutations: int,
    rng: random.Random,
) -> tuple[float, float]:
    materialized_values = [float(value) for value in values]
    materialized_groups = [int(group) for group in groups]
    observed = eta_squared_by_group(materialized_values, materialized_groups)
    exceedances = 0
    shuffled_groups = materialized_groups[:]
    for _ in range(permutations):
        rng.shuffle(shuffled_groups)
        permuted = eta_squared_by_group(materialized_values, shuffled_groups)
        if permuted >= observed:
            exceedances += 1
    p_value = (exceedances + 1) / (permutations + 1)
    return observed, p_value


def add_holm_correction(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not rows:
        return []
    indexed = sorted(
        enumerate(rows),
        key=lambda item: float(item[1].get("p_value", 1.0)),
    )
    adjusted = [1.0 for _ in rows]
    running_max = 0.0
    m = len(rows)
    for rank, (original_index, row) in enumerate(indexed, start=1):
        candidate = min(1.0, (m - rank + 1) * float(row["p_value"]))
        running_max = max(running_max, candidate)
        adjusted[original_index] = running_max

    output: list[dict[str, Any]] = []
    for row, p_adjusted in zip(rows, adjusted):
        copied = dict(row)
        copied["p_value_holm"] = p_adjusted
        output.append(copied)
    return output


def residualize_metric(
    rows: Iterable[dict[str, Any]],
    response_metric: str,
    covariates: Iterable[str],
) -> tuple[list[float], float, list[float]]:
    materialized = list(rows)
    covariate_list = list(covariates)
    y = [_as_float(row[response_metric]) for row in materialized]
    design = [
        [1.0, *[_as_float(row[covariate]) for covariate in covariate_list]]
        for row in materialized
    ]
    coefficients = _least_squares_coefficients(design, y)
    fitted = [
        sum(coefficient * value for coefficient, value in zip(coefficients, row))
        for row in design
    ]
    residuals = [target - prediction for target, prediction in zip(y, fitted)]
    mean_y = fmean(y)
    total_ss = sum((value - mean_y) ** 2 for value in y)
    residual_ss = sum(value**2 for value in residuals)
    r_squared = 1.0 - residual_ss / total_ss if total_ss else 0.0
    return residuals, r_squared, coefficients


def eta_squared_by_group(values: Iterable[float], groups: Iterable[int]) -> float:
    materialized_values = list(values)
    materialized_groups = list(groups)
    if len(materialized_values) != len(materialized_groups):
        raise ValueError("values and groups must have the same length")
    overall_mean = fmean(materialized_values)
    total_ss = sum((value - overall_mean) ** 2 for value in materialized_values)
    if total_ss < 1e-20:
        return 0.0

    grouped: dict[int, list[float]] = defaultdict(list)
    for value, group in zip(materialized_values, materialized_groups):
        grouped[int(group)].append(float(value))
    between_ss = sum(
        len(group_values) * (fmean(group_values) - overall_mean) ** 2
        for group_values in grouped.values()
    )
    return between_ss / total_ss


def zscore_task_means(
    task_summary: list[dict[str, Any]],
    metrics: Iterable[str] = DEFAULT_TASK_GEOMETRY_METRICS,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in task_summary:
        output.append({"task_id": row["task_id"]})

    for metric in metrics:
        key = f"{metric}_mean"
        means = [_as_float(row[key]) for row in task_summary]
        mean = fmean(means)
        std = pstdev(means) if len(means) > 1 else 0.0
        for index, value in enumerate(means):
            output[index][metric] = (value - mean) / std if std else 0.0
    return output


def _as_float(value: Any) -> float:
    return float(value)


def _least_squares_coefficients(design: list[list[float]], y: list[float]) -> list[float]:
    if not design:
        return []
    n_features = len(design[0])
    xtx = [[0.0 for _ in range(n_features)] for _ in range(n_features)]
    xty = [0.0 for _ in range(n_features)]
    for row, target in zip(design, y):
        for i in range(n_features):
            xty[i] += row[i] * target
            for j in range(n_features):
                xtx[i][j] += row[i] * row[j]

    for i in range(n_features):
        xtx[i][i] += 1e-12
    return _solve_linear_system(xtx, xty)


def _solve_linear_system(matrix: list[list[float]], vector: list[float]) -> list[float]:
    n = len(vector)
    augmented = [row[:] + [value] for row, value in zip(matrix, vector)]
    for column in range(n):
        pivot = max(range(column, n), key=lambda row: abs(augmented[row][column]))
        if abs(augmented[pivot][column]) < 1e-15:
            raise ValueError("linear system is singular")
        if pivot != column:
            augmented[column], augmented[pivot] = augmented[pivot], augmented[column]

        pivot_value = augmented[column][column]
        for j in range(column, n + 1):
            augmented[column][j] /= pivot_value

        for row in range(n):
            if row == column:
                continue
            factor = augmented[row][column]
            for j in range(column, n + 1):
                augmented[row][j] -= factor * augmented[column][j]
    return [augmented[row][n] for row in range(n)]
