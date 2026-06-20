from __future__ import annotations

from statistics import mean, median
from typing import Iterable

from .analysis import GeometryProfile


def _quantile(sorted_values: list[float], probability: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = (len(sorted_values) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = position - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def _describe(values: Iterable[float]) -> dict[str, float]:
    series = sorted(float(value) for value in values)
    if not series:
        return {
            "mean": 0.0,
            "median": 0.0,
            "p25": 0.0,
            "p75": 0.0,
            "min": 0.0,
            "max": 0.0,
        }
    return {
        "mean": mean(series),
        "median": median(series),
        "p25": _quantile(series, 0.25),
        "p75": _quantile(series, 0.75),
        "min": series[0],
        "max": series[-1],
    }


def summarize_geometry_profiles(
    profiles: Iterable[GeometryProfile],
) -> dict[str, object]:
    """Aggregate per-graph geometry profiles into corpus-level statistics."""
    items = list(profiles)
    return {
        "profile_count": len(items),
        "path_count": _describe(profile.path_count for profile in items),
        "node_count": _describe(profile.node_count for profile in items),
        "max_depth": _describe(profile.max_depth for profile in items),
        "leaf_fraction": _describe(profile.leaf_fraction for profile in items),
        "mean_branching_factor": _describe(profile.mean_branching_factor for profile in items),
        "max_branching_factor": _describe(profile.max_branching_factor for profile in items),
        "branching_entropy": _describe(profile.branching_entropy for profile in items),
        "delta_norm": _describe(profile.hyperbolicity.delta_norm for profile in items),
        "euclidean_stress": _describe(profile.euclidean.stress for profile in items),
        "hyperbolic_stress": _describe(profile.hyperbolic.stress for profile in items),
        "geometry_advantage": _describe(profile.geometry_advantage for profile in items),
    }
