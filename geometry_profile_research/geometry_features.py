from __future__ import annotations

import math

from .analysis import GeometryProfile


def length_only_features(profile: GeometryProfile) -> dict[str, float]:
    """Minimal size control: AST node/edge counts only."""
    return {
        "log_node_count": math.log1p(profile.node_count),
        "log_edge_count": math.log1p(profile.edge_count),
    }


def size_depth_features(profile: GeometryProfile) -> dict[str, float]:
    """Basic AST size and depth features without metric-distortion information."""
    length_only = length_only_features(profile)
    return {
        **length_only,
        "max_depth": float(profile.max_depth),
        "diameter": float(profile.hyperbolicity.diameter),
    }


def branching_features(profile: GeometryProfile) -> dict[str, float]:
    """Rooted-tree shape features beyond raw size and depth."""
    return {
        "leaf_fraction": profile.leaf_fraction,
        "mean_branching_factor": profile.mean_branching_factor,
        "max_branching_factor": float(profile.max_branching_factor),
        "branching_entropy": profile.branching_entropy,
    }


def metric_distortion_features(profile: GeometryProfile) -> dict[str, float]:
    """Metric-preservation features comparing Euclidean and hyperbolic distances."""
    return {
        "euclidean_stress": profile.euclidean.stress,
        "hyperbolic_stress": profile.hyperbolic.stress,
        "geometry_advantage": profile.geometry_advantage,
    }


def geometry_feature_sets(profile: GeometryProfile) -> dict[str, dict[str, float]]:
    """Return feature groups for ablation experiments."""
    length_only = length_only_features(profile)
    size_depth = size_depth_features(profile)
    branching = branching_features(profile)
    metric_distortion = metric_distortion_features(profile)
    return {
        "length_only": length_only,
        "size_depth": size_depth,
        "branching": branching,
        "metric_distortion": metric_distortion,
        "all": {**length_only, **size_depth, **branching, **metric_distortion},
    }
