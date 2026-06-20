from __future__ import annotations

from statistics import fmean, pstdev


def summarize_metric_series(values: list[float]) -> dict[str, float | int]:
    """Summarize repeated experimental measurements across seeds."""
    if not values:
        return {
            "count": 0,
            "mean": 0.0,
            "std": 0.0,
            "min": 0.0,
            "max": 0.0,
        }
    return {
        "count": len(values),
        "mean": fmean(values),
        "std": pstdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }
