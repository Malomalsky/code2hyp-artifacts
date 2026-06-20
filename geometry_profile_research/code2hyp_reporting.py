from __future__ import annotations

import itertools
import math
import random
import statistics
from collections import defaultdict
from collections.abc import Mapping
from typing import Any


METRIC_KEYS = (
    "validation_f1",
    "validation_structural_loss",
    "validation_structural_rank_loss",
    "validation_structural_spearman",
    "validation_structural_neighbor_overlap_at_1",
    "validation_structural_neighbor_overlap_at_3",
    "validation_structural_neighbor_recall_at_1",
    "validation_structural_neighbor_recall_at_3",
)

LEGACY_METRIC_ALIASES = {
    "validation_structural_neighbor_overlap_at_1": "validation_structural_neighbor_recall_at_1",
    "validation_structural_neighbor_overlap_at_3": "validation_structural_neighbor_recall_at_3",
}


def summarize_pilot_runs(result: Mapping[str, Any]) -> list[dict[str, float | int | str]]:
    """Group Code2Hyp pilot JSON runs by variant and summarize key metrics."""
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for run in result.get("runs", []):
        grouped[str(run["variant"])].append(run)

    summaries: list[dict[str, float | int | str]] = []
    for variant in sorted(grouped):
        runs = grouped[variant]
        summary: dict[str, float | int | str] = {
            "variant": variant,
            "n": len(runs),
        }
        for key in METRIC_KEYS:
            legacy_key = LEGACY_METRIC_ALIASES.get(key)
            values = [
                float(run[key] if key in run else run[legacy_key])
                for run in runs
                if key in run or (legacy_key is not None and legacy_key in run)
            ]
            if not values:
                continue
            summary[f"{key}_mean"] = statistics.mean(values)
            summary[f"{key}_std"] = statistics.pstdev(values) if len(values) > 1 else 0.0
        summaries.append(summary)
    return summaries


def pareto_frontier(
    rows: list[Mapping[str, Any]],
    objectives: tuple[tuple[str, str], ...],
) -> list[dict[str, Any]]:
    """Return non-dominated rows for the requested max/min objectives."""
    if not objectives:
        raise ValueError("objectives must not be empty")
    normalized_objectives = tuple(_normalize_objective_direction(direction) for _, direction in objectives)
    complete_rows = [
        dict(row)
        for row in rows
        if all(metric_key in row for metric_key, _ in objectives)
    ]
    frontier = []
    for candidate in complete_rows:
        dominated = False
        for challenger in complete_rows:
            if challenger is candidate:
                continue
            if _dominates(challenger, candidate, objectives, normalized_objectives):
                dominated = True
                break
        if not dominated:
            frontier.append(candidate)
    return sorted(frontier, key=lambda row: str(row.get("variant", "")))


def multi_objective_variant_selection(
    result: Mapping[str, Any],
    objectives: tuple[tuple[str, str, float], ...] = (
        ("validation_f1_mean", "max", 0.5),
        ("validation_structural_spearman_mean", "max", 0.5),
    ),
    variant_filter: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """Rank variant summaries by normalized weighted multi-objective utility."""
    if not objectives:
        raise ValueError("objectives must not be empty")
    if any(weight < 0.0 for _, _, weight in objectives):
        raise ValueError("objective weights must be non-negative")
    total_weight = sum(weight for _, _, weight in objectives)
    if total_weight <= 0.0:
        raise ValueError("at least one objective weight must be positive")

    summaries = summarize_pilot_runs(result)
    if variant_filter is not None:
        allowed = set(variant_filter)
        summaries = [row for row in summaries if str(row["variant"]) in allowed]
    complete_rows = [
        dict(row)
        for row in summaries
        if all(metric_key in row for metric_key, _, _ in objectives)
    ]
    if not complete_rows:
        raise ValueError("no variants contain all requested objective metrics")

    normalized_values: dict[tuple[str, str], float] = {}
    for metric_key, direction, _ in objectives:
        direction = _normalize_objective_direction(direction)
        values = [float(row[metric_key]) for row in complete_rows]
        minimum = min(values)
        maximum = max(values)
        span = maximum - minimum
        for row in complete_rows:
            raw_value = float(row[metric_key])
            if span <= 0.0:
                normalized = 0.5
            elif direction == "max":
                normalized = (raw_value - minimum) / span
            else:
                normalized = (maximum - raw_value) / span
            normalized_values[(str(row["variant"]), metric_key)] = normalized

    frontier_variants = {
        str(row["variant"])
        for row in pareto_frontier(
            complete_rows,
            objectives=tuple((metric_key, direction) for metric_key, direction, _ in objectives),
        )
    }
    ranked = []
    for row in complete_rows:
        variant = str(row["variant"])
        score = sum(
            weight * normalized_values[(variant, metric_key)]
            for metric_key, _, weight in objectives
        ) / total_weight
        ranked_row = dict(row)
        ranked_row["multi_objective_score"] = score
        ranked_row["pareto_frontier"] = variant in frontier_variants
        for metric_key, _, _ in objectives:
            ranked_row[f"{metric_key}_normalized"] = normalized_values[(variant, metric_key)]
        ranked.append(ranked_row)

    ranked.sort(key=lambda row: (-float(row["multi_objective_score"]), str(row["variant"])))
    return {
        "objectives": [
            {"metric": metric_key, "direction": _normalize_objective_direction(direction), "weight": weight}
            for metric_key, direction, weight in objectives
        ],
        "ranked": ranked,
        "best": ranked[0],
        "pareto_frontier": [row for row in ranked if bool(row["pareto_frontier"])],
    }


def _normalize_objective_direction(direction: str) -> str:
    if direction not in ("max", "min"):
        raise ValueError("objective direction must be 'max' or 'min'")
    return direction


def _dominates(
    challenger: Mapping[str, Any],
    candidate: Mapping[str, Any],
    objectives: tuple[tuple[str, str], ...],
    normalized_directions: tuple[str, ...],
) -> bool:
    no_worse = True
    strictly_better = False
    for (metric_key, _), direction in zip(objectives, normalized_directions, strict=True):
        challenger_value = float(challenger[metric_key])
        candidate_value = float(candidate[metric_key])
        if direction == "max":
            if challenger_value < candidate_value:
                no_worse = False
                break
            if challenger_value > candidate_value:
                strictly_better = True
        else:
            if challenger_value > candidate_value:
                no_worse = False
                break
            if challenger_value < candidate_value:
                strictly_better = True
    return no_worse and strictly_better


def paired_metric_comparison(
    result: Mapping[str, Any],
    left_variant: str,
    right_variant: str,
    metric_key: str,
    pairing_key: str = "model_seed",
    confidence: float = 0.95,
    bootstrap_resamples: int = 10_000,
    bootstrap_seed: int = 0,
) -> dict[str, Any]:
    """Compare two variants using only matched runs with the same pairing key.

    The sign test is exact and two-sided. The bootstrap confidence interval is a
    percentile interval over paired deltas; for small n it enumerates all
    resamples exactly instead of using stochastic sampling.
    """
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between 0 and 1")
    left_by_key = _runs_by_pairing_key(result, left_variant, metric_key, pairing_key)
    right_by_key = _runs_by_pairing_key(result, right_variant, metric_key, pairing_key)
    paired_keys = sorted(set(left_by_key) & set(right_by_key))
    if not paired_keys:
        raise ValueError(f"no matched runs for {left_variant} and {right_variant} by {pairing_key}")

    deltas = [left_by_key[key] - right_by_key[key] for key in paired_keys]
    positive = sum(1 for delta in deltas if delta > 0)
    negative = sum(1 for delta in deltas if delta < 0)
    zero = len(deltas) - positive - negative
    ci_low, ci_high = _bootstrap_mean_ci(
        deltas,
        confidence=confidence,
        resamples=bootstrap_resamples,
        seed=bootstrap_seed,
    )
    return {
        "left_variant": left_variant,
        "right_variant": right_variant,
        "metric_key": metric_key,
        "pairing_key": pairing_key,
        "paired_keys": paired_keys,
        "n": len(deltas),
        "deltas": deltas,
        "mean_delta": statistics.mean(deltas),
        "median_delta": statistics.median(deltas),
        "positive_deltas": positive,
        "negative_deltas": negative,
        "zero_deltas": zero,
        "sign_test_p_two_sided": _two_sided_sign_test_p_value(positive, negative),
        "evidence_status": _paired_evidence_status(len(deltas), positive, negative),
        "bootstrap_confidence": confidence,
        "bootstrap_ci_low": ci_low,
        "bootstrap_ci_high": ci_high,
    }


def _runs_by_pairing_key(
    result: Mapping[str, Any],
    variant: str,
    metric_key: str,
    pairing_key: str,
) -> dict[Any, float]:
    values: dict[Any, float] = {}
    for run in result.get("runs", []):
        if str(run.get("variant")) != variant:
            continue
        if pairing_key not in run:
            continue
        value = _run_metric_value(run, metric_key)
        if value is None:
            continue
        key = run[pairing_key]
        if key in values:
            raise ValueError(f"duplicate run for variant={variant}, {pairing_key}={key}")
        values[key] = value
    return values


def _run_metric_value(run: Mapping[str, Any], metric_key: str) -> float | None:
    if metric_key in run:
        return float(run[metric_key])
    legacy_key = LEGACY_METRIC_ALIASES.get(metric_key)
    if legacy_key is not None and legacy_key in run:
        return float(run[legacy_key])
    return None


def _two_sided_sign_test_p_value(positive: int, negative: int) -> float:
    trials = positive + negative
    if trials == 0:
        return 1.0
    extreme = min(positive, negative)
    tail = sum(math.comb(trials, k) for k in range(extreme + 1)) / (2**trials)
    return min(1.0, 2.0 * tail)


def _paired_evidence_status(sample_size: int, positive: int, negative: int) -> str:
    if sample_size < 5:
        return "exploratory_low_power"
    if positive > 0 and negative == 0:
        return "directionally_consistent"
    if negative > 0 and positive == 0:
        return "directionally_consistent_negative"
    return "mixed_direction"


def _bootstrap_mean_ci(
    values: list[float],
    confidence: float,
    resamples: int,
    seed: int,
) -> tuple[float, float]:
    if not values:
        raise ValueError("values must not be empty")
    sample_size = len(values)
    exact_count = sample_size**sample_size
    if exact_count <= 50_000:
        means = [
            statistics.mean(values[index] for index in indices)
            for indices in itertools.product(range(sample_size), repeat=sample_size)
        ]
    else:
        if resamples <= 0:
            raise ValueError("bootstrap_resamples must be positive")
        rng = random.Random(seed)
        means = [
            statistics.mean(values[rng.randrange(sample_size)] for _ in range(sample_size))
            for _ in range(resamples)
        ]
    means.sort()
    alpha = 1.0 - confidence
    return _percentile(means, alpha / 2.0), _percentile(means, 1.0 - alpha / 2.0)


def _percentile(sorted_values: list[float], probability: float) -> float:
    if not sorted_values:
        raise ValueError("sorted_values must not be empty")
    if probability <= 0:
        return sorted_values[0]
    if probability >= 1:
        return sorted_values[-1]
    position = probability * (len(sorted_values) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    fraction = position - lower
    return sorted_values[lower] * (1.0 - fraction) + sorted_values[upper] * fraction
