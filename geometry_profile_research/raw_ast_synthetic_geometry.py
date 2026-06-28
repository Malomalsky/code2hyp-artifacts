from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, Sequence

import torch
from torch import Tensor

from geometry_profile_research.code2hyp_torch import torch_expmap0, torch_logmap0, torch_poincare_distance
from geometry_profile_research.lca_path_measure import (
    EuclideanPathMeasure,
    PoincarePathMeasure,
    euclidean_lca_anchored_product_cost_matrix,
    lca_anchored_product_cost_matrix,
)
from geometry_profile_research.raw_ast import RawAstPath, RawAstTree, lca_anchored_product_distance, terminal_to_terminal_paths


Geometry = Literal["euclidean", "poincare"]
PathObjectMode = Literal["single_point", "lca_product"]


@dataclass(frozen=True)
class SyntheticTreeCase:
    name: str
    tree: RawAstTree
    expected: str


def synthetic_tree_suite() -> tuple[SyntheticTreeCase, ...]:
    """Return deterministic tree families recommended for mechanism probes."""

    return (
        SyntheticTreeCase("comb_chain", _comb_chain_tree(depth=7), "low branching; hyperbolic advantage should be small"),
        SyntheticTreeCase("star", _star_tree(leaves=12), "shallow high branching; LCA-product should matter more than curvature"),
        SyntheticTreeCase("balanced_binary", _balanced_tree(branching=2, depth=4), "deep branching hierarchy"),
        SyntheticTreeCase("repeated_labels", _balanced_tree(branching=3, depth=3, repeated_labels=True), "branch structure with weak labels"),
        SyntheticTreeCase("two_axis_product", _two_axis_product_tree(rows=4, columns=4), "product-like terminal organization"),
    )


def fit_node_embeddings(
    tree: RawAstTree,
    *,
    geometry: Geometry,
    dim: int,
    curvature: float = 1.0,
    steps: int = 300,
    learning_rate: float = 0.05,
    seed: int = 20260624,
) -> Tensor:
    """Fit node embeddings to the tree metric for synthetic stress probes."""

    if geometry not in {"euclidean", "poincare"}:
        raise ValueError(f"unknown geometry: {geometry!r}")
    if dim <= 0:
        raise ValueError("dim must be positive")
    if curvature <= 0:
        raise ValueError("curvature must be positive")
    generator = torch.Generator().manual_seed(seed)
    order = tree.preorder()
    node_to_row = {node: row for row, node in enumerate(order)}
    target = _tree_distance_matrix(tree, order)
    pair_mask = torch.triu(torch.ones_like(target, dtype=torch.bool), diagonal=1)
    parameters = torch.randn((len(order), dim), generator=generator, dtype=torch.float32) * 0.01
    parameters.requires_grad_(True)
    optimizer = torch.optim.Adam([parameters], lr=learning_rate)
    for _ in range(max(0, steps)):
        optimizer.zero_grad(set_to_none=True)
        points = _points_from_parameters(parameters, geometry=geometry, curvature=curvature)
        distances = _pairwise_distances(points, geometry=geometry, curvature=curvature)
        loss = torch.mean((distances[pair_mask] - target[pair_mask]).square())
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        fitted = _points_from_parameters(parameters, geometry=geometry, curvature=curvature)
    max_node_id = max(order)
    output = torch.zeros((max_node_id + 1, dim), dtype=torch.float32)
    for node, row in node_to_row.items():
        output[node] = fitted[row]
    return output


def synthetic_path_distortion_rows(
    *,
    cases: Sequence[SyntheticTreeCase] | None = None,
    geometries: Sequence[Geometry] = ("euclidean", "poincare"),
    path_object_modes: Sequence[PathObjectMode] = ("single_point", "lca_product"),
    dims: Sequence[int] = (2, 4, 8),
    poincare_curvatures: Sequence[float] = (1.0,),
    steps: int = 300,
    learning_rate: float = 0.05,
    max_paths: int = 48,
    seed: int = 20260624,
) -> list[dict[str, float | int | str]]:
    """Evaluate node and path metric distortion on controlled tree families."""

    selected_cases = tuple(cases or synthetic_tree_suite())
    rows: list[dict[str, float | int | str]] = []
    for case in selected_cases:
        paths = terminal_to_terminal_paths(case.tree, max_paths=max_paths)
        if len(paths) < 2:
            continue
        raw_path_distances = _raw_path_distance_matrix(case.tree, paths)
        for dim in dims:
            for geometry in geometries:
                for curvature in _curvature_values(geometry, poincare_curvatures):
                    embeddings = fit_node_embeddings(
                        case.tree,
                        geometry=geometry,
                        dim=dim,
                        curvature=curvature,
                        steps=steps,
                        learning_rate=learning_rate,
                        seed=seed,
                    )
                    node_distances = _node_distance_matrix(case.tree, embeddings, geometry=geometry, curvature=curvature)
                    node_target = _tree_distance_matrix(case.tree, case.tree.preorder())
                    node_stress = metric_stress(node_target, node_distances)
                    node_spearman = spearman_correlation(_upper_values(node_target), _upper_values(node_distances))
                    for path_object_mode in path_object_modes:
                        represented = _represented_path_distances(
                            case.tree,
                            paths,
                            embeddings,
                            geometry=geometry,
                            path_object_mode=path_object_mode,
                            curvature=curvature,
                        )
                        rows.append(
                            {
                                "case": case.name,
                                "expected": case.expected,
                                "geometry": geometry,
                                "curvature": curvature if geometry == "poincare" else 0.0,
                                "path_object_mode": path_object_mode,
                                "dim": dim,
                                "node_count": len(case.tree.preorder()),
                                "path_count": len(paths),
                                "node_stress": node_stress,
                                "node_spearman": node_spearman,
                                "path_stress": metric_stress(raw_path_distances, represented),
                                "path_spearman": spearman_correlation(_upper_values(raw_path_distances), _upper_values(represented)),
                            }
                        )
    return rows


def _curvature_values(geometry: Geometry, poincare_curvatures: Sequence[float]) -> tuple[float, ...]:
    if geometry == "euclidean":
        return (1.0,)
    values = tuple(float(value) for value in poincare_curvatures)
    if not values:
        raise ValueError("poincare_curvatures must not be empty when Poincare geometry is requested")
    if any(value <= 0.0 for value in values):
        raise ValueError("Poincare curvatures must be positive")
    return values


def metric_stress(target: Tensor, represented: Tensor) -> float:
    """Scale-invariant Kruskal-style stress over upper-triangular distances."""

    target_values = _upper_values(target).detach()
    represented_values = _upper_values(represented).detach()
    if target_values.numel() == 0:
        return 0.0
    denominator = torch.sum(represented_values.square())
    if float(denominator) <= 1e-12:
        return float("inf")
    scale = torch.sum(target_values * represented_values) / denominator
    residual = target_values - scale * represented_values
    target_norm = torch.clamp(torch.sum(target_values.square()), min=1e-12)
    return float(torch.sqrt(torch.sum(residual.square()) / target_norm))


def spearman_correlation(left: Tensor, right: Tensor) -> float:
    """Spearman correlation with average-free deterministic ranks."""

    left = left.detach()
    right = right.detach()
    if left.numel() != right.numel():
        raise ValueError("ranked vectors must have the same length")
    if left.numel() < 2:
        return 0.0
    left_rank = _ordinal_rank(left)
    right_rank = _ordinal_rank(right)
    left_centered = left_rank - left_rank.mean()
    right_centered = right_rank - right_rank.mean()
    denominator = torch.linalg.vector_norm(left_centered) * torch.linalg.vector_norm(right_centered)
    if float(denominator) <= 1e-12:
        return 0.0
    value = torch.dot(left_centered, right_centered) / denominator
    return float(torch.clamp(value, min=-1.0, max=1.0))


def _comb_chain_tree(*, depth: int) -> RawAstTree:
    edges: list[tuple[int, int]] = []
    labels: dict[int, str] = {0: "Root"}
    current = 0
    next_id = 1
    for level in range(depth):
        internal = next_id
        leaf = next_id + 1
        next_id += 2
        edges.append((current, internal))
        edges.append((current, leaf))
        labels[internal] = f"Spine{level}"
        labels[leaf] = "Terminal"
        current = internal
    labels[current] = "Terminal"
    return RawAstTree.from_edges(root_id=0, edges=tuple(edges), labels=labels)


def _star_tree(*, leaves: int) -> RawAstTree:
    edges = tuple((0, node) for node in range(1, leaves + 1))
    labels = {0: "Root", **{node: "Terminal" for node in range(1, leaves + 1)}}
    return RawAstTree.from_edges(root_id=0, edges=edges, labels=labels)


def _balanced_tree(*, branching: int, depth: int, repeated_labels: bool = False) -> RawAstTree:
    edges: list[tuple[int, int]] = []
    labels: dict[int, str] = {0: "Root"}
    frontier = [0]
    next_id = 1
    for level in range(1, depth + 1):
        new_frontier = []
        for parent in frontier:
            for child_index in range(branching):
                child = next_id
                next_id += 1
                edges.append((parent, child))
                new_frontier.append(child)
                if level == depth:
                    labels[child] = "Terminal" if repeated_labels else f"Terminal{child_index}"
                else:
                    labels[child] = "Internal" if repeated_labels else f"Internal{level}_{child_index}"
        frontier = new_frontier
    return RawAstTree.from_edges(root_id=0, edges=tuple(edges), labels=labels)


def _two_axis_product_tree(*, rows: int, columns: int) -> RawAstTree:
    edges: list[tuple[int, int]] = []
    labels: dict[int, str] = {0: "Root"}
    next_id = 1
    for row in range(rows):
        row_node = next_id
        next_id += 1
        edges.append((0, row_node))
        labels[row_node] = f"Row{row}"
        for column in range(columns):
            column_node = next_id
            terminal = next_id + 1
            next_id += 2
            edges.append((row_node, column_node))
            edges.append((column_node, terminal))
            labels[column_node] = f"Column{column}"
            labels[terminal] = "Terminal"
    return RawAstTree.from_edges(root_id=0, edges=tuple(edges), labels=labels)


def _points_from_parameters(parameters: Tensor, *, geometry: Geometry, curvature: float) -> Tensor:
    if geometry == "poincare":
        return torch_expmap0(parameters, curvature)
    return parameters


def _pairwise_distances(points: Tensor, *, geometry: Geometry, curvature: float) -> Tensor:
    if geometry == "poincare":
        return torch_poincare_distance(points.unsqueeze(1), points.unsqueeze(0), curvature=curvature)
    return torch.linalg.vector_norm(points.unsqueeze(1) - points.unsqueeze(0), dim=-1)


def _node_distance_matrix(tree: RawAstTree, embeddings: Tensor, *, geometry: Geometry, curvature: float) -> Tensor:
    order = tree.preorder()
    points = torch.stack([embeddings[node] for node in order], dim=0)
    return _pairwise_distances(points, geometry=geometry, curvature=curvature)


def _tree_distance_matrix(tree: RawAstTree, order: Sequence[int]) -> Tensor:
    rows = []
    for left in order:
        rows.append([float(tree.tree_distance(left, right)) for right in order])
    return torch.tensor(rows, dtype=torch.float32)


def _raw_path_distance_matrix(tree: RawAstTree, paths: Sequence[RawAstPath]) -> Tensor:
    rows = []
    for left in paths:
        rows.append([float(lca_anchored_product_distance(tree, left, right)) for right in paths])
    return torch.tensor(rows, dtype=torch.float32)


def _represented_path_distances(
    tree: RawAstTree,
    paths: Sequence[RawAstPath],
    embeddings: Tensor,
    *,
    geometry: Geometry,
    path_object_mode: PathObjectMode,
    curvature: float,
) -> Tensor:
    if path_object_mode == "single_point":
        points = _single_path_points(tree, paths, embeddings, geometry=geometry, curvature=curvature)
        return _pairwise_distances(points, geometry=geometry, curvature=curvature)
    if geometry == "poincare":
        measure = _poincare_measure(tree, paths, embeddings, curvature=curvature)
        return torch.sqrt(torch.clamp(lca_anchored_product_cost_matrix(measure, measure, unoriented=True), min=0.0))
    measure = _euclidean_measure(tree, paths, embeddings)
    return torch.sqrt(torch.clamp(euclidean_lca_anchored_product_cost_matrix(measure, measure, unoriented=True), min=0.0))


def _single_path_points(
    tree: RawAstTree,
    paths: Sequence[RawAstPath],
    embeddings: Tensor,
    *,
    geometry: Geometry,
    curvature: float,
) -> Tensor:
    triples = _path_triples(tree, paths, embeddings)
    if geometry == "poincare":
        tangent = torch_logmap0(triples, curvature)
        return torch_expmap0(torch.mean(tangent, dim=1), curvature)
    return torch.mean(triples, dim=1)


def _poincare_measure(tree: RawAstTree, paths: Sequence[RawAstPath], embeddings: Tensor, *, curvature: float) -> PoincarePathMeasure:
    return PoincarePathMeasure(points=_path_triples(tree, paths, embeddings), mass=torch.ones((len(paths),)), curvature=curvature)


def _euclidean_measure(tree: RawAstTree, paths: Sequence[RawAstPath], embeddings: Tensor) -> EuclideanPathMeasure:
    return EuclideanPathMeasure(points=_path_triples(tree, paths, embeddings), mass=torch.ones((len(paths),)))


def _path_triples(tree: RawAstTree, paths: Sequence[RawAstPath], embeddings: Tensor) -> Tensor:
    triples = []
    for path in paths:
        triples.append(torch.stack([embeddings[path.lca(tree)], embeddings[path.start], embeddings[path.end]], dim=0))
    return torch.stack(triples, dim=0)


def _upper_values(matrix: Tensor) -> Tensor:
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("distance matrix must be square")
    if matrix.shape[0] < 2:
        return torch.zeros((0,), dtype=matrix.dtype, device=matrix.device)
    mask = torch.triu(torch.ones_like(matrix, dtype=torch.bool), diagonal=1)
    return matrix[mask]


def _ordinal_rank(values: Tensor) -> Tensor:
    order = torch.argsort(values, stable=True)
    ranks = torch.empty_like(order, dtype=torch.float32)
    ranks[order] = torch.arange(values.numel(), dtype=torch.float32, device=values.device)
    return ranks
