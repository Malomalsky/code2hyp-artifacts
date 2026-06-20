from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Iterable

from .ast_features import ast_root_paths
from .embeddings import euclidean_distance, path_to_poincare, poincare_distance
from .graphs import build_file_tree_graph
from .metrics import (
    DistortionSummary,
    HyperbolicityResult,
    all_pairs_shortest_paths,
    distortion_summary,
    graph_diameter,
    gromov_hyperbolicity,
)


@dataclass(frozen=True)
class GeometryProfile:
    node_count: int
    edge_count: int
    path_count: int
    max_depth: int
    leaf_fraction: float
    mean_branching_factor: float
    max_branching_factor: int
    branching_entropy: float
    hyperbolicity: HyperbolicityResult
    euclidean: DistortionSummary
    hyperbolic: DistortionSummary
    geometry_advantage: float

    def to_dict(self) -> dict[str, object]:
        return {
            "node_count": self.node_count,
            "edge_count": self.edge_count,
            "path_count": self.path_count,
            "max_depth": self.max_depth,
            "leaf_fraction": self.leaf_fraction,
            "mean_branching_factor": self.mean_branching_factor,
            "max_branching_factor": self.max_branching_factor,
            "branching_entropy": self.branching_entropy,
            "hyperbolicity": asdict(self.hyperbolicity),
            "euclidean": asdict(self.euclidean),
            "hyperbolic": asdict(self.hyperbolic),
            "geometry_advantage": self.geometry_advantage,
        }


def _graph_edge_count(nodes: Iterable[str], neighbors) -> int:
    return sum(len(neighbors(node)) for node in nodes) // 2


def _child_counts_for_rooted_tree(graph, nodes: Iterable[str]) -> dict[str, int]:
    child_counts: dict[str, int] = {}
    for node in nodes:
        node_depth = graph.depth(node)
        child_counts[node] = sum(
            1
            for neighbor in graph.neighbors(node)
            if graph.depth(neighbor) > node_depth
        )
    return child_counts


def _branching_entropy(child_counts: Iterable[int]) -> float:
    values = list(child_counts)
    if not values:
        return 0.0
    frequencies: dict[int, int] = {}
    for value in values:
        frequencies[value] = frequencies.get(value, 0) + 1
    total = len(values)
    return -sum(
        (count / total) * math.log2(count / total)
        for count in frequencies.values()
        if count > 0
    )


def geometry_profile_for_paths(
    paths: Iterable[str],
    *,
    beta: float = 0.45,
    gamma: float = 0.0,
    curvature: float = 1.0,
    assume_tree_hyperbolicity: bool = False,
) -> GeometryProfile:
    """Compute a reproducible first-order geometry profile for file paths.

    Euclidean and hyperbolic summaries use the same deterministic coordinates
    in the Poincare disk. The comparison therefore isolates the metric effect:
    Euclidean distance saturates near the disk boundary, while hyperbolic
    geodesic distance expands radial hierarchy.
    """
    unique_paths = sorted({path for path in paths if path})
    graph = build_file_tree_graph(unique_paths)
    nodes = sorted(graph.nodes)
    distances = all_pairs_shortest_paths(graph)
    embedding = {
        node: path_to_poincare(node, beta=beta, gamma=gamma)
        for node in nodes
    }

    if assume_tree_hyperbolicity:
        diameter = graph_diameter(distances)
        hyperbolicity = HyperbolicityResult(
            delta=0.0,
            delta_norm=0.0,
            diameter=diameter,
            quadruples=0,
        )
    else:
        hyperbolicity = gromov_hyperbolicity(distances)
    euclidean = distortion_summary(distances, embedding, euclidean_distance)
    hyperbolic = distortion_summary(
        distances,
        embedding,
        lambda left, right: poincare_distance(left, right, curvature=curvature),
    )

    denominator = max(euclidean.stress, hyperbolic.stress, 1e-12)
    advantage = (euclidean.stress - hyperbolic.stress) / denominator
    child_counts_by_node = _child_counts_for_rooted_tree(graph, nodes)
    child_counts = list(child_counts_by_node.values())
    internal_child_counts = [count for count in child_counts if count > 0]
    leaf_count = sum(1 for count in child_counts if count == 0)

    return GeometryProfile(
        node_count=len(nodes),
        edge_count=_graph_edge_count(nodes, graph.neighbors),
        path_count=len(unique_paths),
        max_depth=max((graph.depth(node) for node in nodes), default=0),
        leaf_fraction=leaf_count / len(nodes) if nodes else 0.0,
        mean_branching_factor=(
            sum(internal_child_counts) / len(internal_child_counts)
            if internal_child_counts
            else 0.0
        ),
        max_branching_factor=max(child_counts, default=0),
        branching_entropy=_branching_entropy(child_counts),
        hyperbolicity=hyperbolicity,
        euclidean=euclidean,
        hyperbolic=hyperbolic,
        geometry_advantage=advantage,
    )


def geometry_profile_for_ast_source(
    code: str,
    *,
    beta: float = 0.45,
    gamma: float = 0.0,
    curvature: float = 1.0,
) -> GeometryProfile:
    """Compute a tree-geometry profile for a Python program AST."""
    return geometry_profile_for_paths(
        ast_root_paths(code),
        beta=beta,
        gamma=gamma,
        curvature=curvature,
        assume_tree_hyperbolicity=True,
    )
