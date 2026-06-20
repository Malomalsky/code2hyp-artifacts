from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from itertools import combinations
from statistics import median
from typing import Callable, Mapping

from .embeddings import Point2D
from .graphs import SimpleGraph


@dataclass(frozen=True)
class HyperbolicityResult:
    delta: float
    delta_norm: float
    diameter: float
    quadruples: int


@dataclass(frozen=True)
class DistortionSummary:
    alpha: float
    median_relative_error: float
    stress: float
    pairs: int


def _single_source_shortest_paths(graph: SimpleGraph, source: str) -> dict[str, int]:
    distances = {source: 0}
    queue: deque[str] = deque([source])

    while queue:
        current = queue.popleft()
        for neighbor in graph.neighbors(current):
            if neighbor in distances:
                continue
            distances[neighbor] = distances[current] + 1
            queue.append(neighbor)

    return distances


def all_pairs_shortest_paths(graph: SimpleGraph) -> dict[str, dict[str, int]]:
    return {node: _single_source_shortest_paths(graph, node) for node in graph.nodes}


def graph_diameter(distances: Mapping[str, Mapping[str, float]]) -> float:
    values = [distance for row in distances.values() for distance in row.values()]
    return max(values) if values else 0.0


def gromov_hyperbolicity(
    distances: Mapping[str, Mapping[str, float]]
) -> HyperbolicityResult:
    """Compute exact four-point Gromov hyperbolicity for a small graph."""
    nodes = sorted(distances)
    diameter = graph_diameter(distances)
    if len(nodes) < 4 or diameter == 0:
        return HyperbolicityResult(delta=0.0, delta_norm=0.0, diameter=diameter, quadruples=0)

    max_delta = 0.0
    quadruple_count = 0

    for a, b, c, d in combinations(nodes, 4):
        sums = [
            distances[a][b] + distances[c][d],
            distances[a][c] + distances[b][d],
            distances[a][d] + distances[b][c],
        ]
        sums.sort()
        delta = (sums[2] - sums[1]) / 2.0
        max_delta = max(max_delta, delta)
        quadruple_count += 1

    return HyperbolicityResult(
        delta=max_delta,
        delta_norm=max_delta / diameter if diameter else 0.0,
        diameter=diameter,
        quadruples=quadruple_count,
    )


def _distance_pairs(
    distances: Mapping[str, Mapping[str, float]],
    embedding: Mapping[str, Point2D],
    metric: Callable[[Point2D, Point2D], float],
) -> list[tuple[float, float]]:
    pairs: list[tuple[float, float]] = []
    for left, right in combinations(sorted(embedding), 2):
        if left not in distances or right not in distances[left]:
            continue
        graph_distance = float(distances[left][right])
        if graph_distance <= 0:
            continue
        model_distance = float(metric(embedding[left], embedding[right]))
        pairs.append((graph_distance, model_distance))
    return pairs


def _least_squares_scale(pairs: list[tuple[float, float]]) -> float:
    denominator = sum(model * model for _, model in pairs)
    if denominator <= 1e-12:
        return 0.0
    return sum(graph * model for graph, model in pairs) / denominator


def distortion_summary(
    distances: Mapping[str, Mapping[str, float]],
    embedding: Mapping[str, Point2D],
    metric: Callable[[Point2D, Point2D], float],
) -> DistortionSummary:
    """Summarize how well embedding distances preserve graph distances."""
    pairs = _distance_pairs(distances, embedding, metric)
    if not pairs:
        return DistortionSummary(alpha=0.0, median_relative_error=0.0, stress=0.0, pairs=0)

    alpha = _least_squares_scale(pairs)
    relative_errors = [
        abs(graph_distance - alpha * model_distance) / graph_distance
        for graph_distance, model_distance in pairs
    ]
    numerator = sum(
        (graph_distance - alpha * model_distance) ** 2
        for graph_distance, model_distance in pairs
    )
    denominator = sum(graph_distance * graph_distance for graph_distance, _ in pairs)
    stress = math.sqrt(numerator / denominator) if denominator else 0.0

    return DistortionSummary(
        alpha=alpha,
        median_relative_error=median(relative_errors),
        stress=stress,
        pairs=len(pairs),
    )
