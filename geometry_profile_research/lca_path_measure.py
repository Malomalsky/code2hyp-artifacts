from __future__ import annotations

import math
from hashlib import blake2b
from dataclasses import dataclass
from typing import Literal, Sequence

import torch
from torch import Tensor

from geometry_profile_research.code2hyp_torch import torch_poincare_distance, torch_project_to_ball
from geometry_profile_research.gromov_wasserstein import sinkhorn_divergence
from geometry_profile_research.raw_ast import RawAstPath, RawAstTree


@dataclass(frozen=True)
class PoincarePathMeasure:
    """Finite measure over LCA-anchored product-hyperbolic AST path objects.

    ``points[i]`` stores ``(x_lca, x_start, x_end)`` for path ``i``.
    The ambient path object lives in ``H x H x H``; the method is represented
    as a probability measure over these path objects.
    """

    points: Tensor
    mass: Tensor
    curvature: float = 1.0

    def __post_init__(self) -> None:
        points = torch.as_tensor(self.points, dtype=torch.float32)
        if points.ndim != 3 or points.shape[1] != 3:
            raise ValueError("points must have shape (n_paths, 3, dim)")
        mass = torch.as_tensor(self.mass, dtype=points.dtype, device=points.device)
        if mass.ndim != 1 or mass.numel() != points.shape[0]:
            raise ValueError("mass must contain one value per path")
        if bool((mass < 0).any()):
            raise ValueError("mass values must be non-negative")
        total = torch.sum(mass)
        if float(total) <= 0.0:
            raise ValueError("mass must have positive total")
        mass = mass / total
        object.__setattr__(self, "points", torch_project_to_ball(points, self.curvature))
        object.__setattr__(self, "mass", mass)


@dataclass(frozen=True)
class EuclideanPathMeasure:
    """Finite measure over LCA-anchored product-Euclidean AST path objects.

    This is the matched Euclidean control for the Poincare product protocol:
    the path object is still ``(x_lca, x_start, x_end)``, but each factor lives
    in ordinary Euclidean space.
    """

    points: Tensor
    mass: Tensor

    def __post_init__(self) -> None:
        points = torch.as_tensor(self.points, dtype=torch.float32)
        if points.ndim != 3 or points.shape[1] != 3:
            raise ValueError("points must have shape (n_paths, 3, dim)")
        mass = torch.as_tensor(self.mass, dtype=points.dtype, device=points.device)
        if mass.ndim != 1 or mass.numel() != points.shape[0]:
            raise ValueError("mass must contain one value per path")
        if bool((mass < 0).any()):
            raise ValueError("mass values must be non-negative")
        total = torch.sum(mass)
        if float(total) <= 0.0:
            raise ValueError("mass must have positive total")
        object.__setattr__(self, "points", points)
        object.__setattr__(self, "mass", mass / total)


def poincare_node_embeddings_from_tree(
    tree: RawAstTree,
    *,
    curvature: float = 1.0,
    radial_scale: float = 0.35,
    angle_mode: Literal["branch_sector", "label_hash", "path_signature_hash", "depth_only"] = "branch_sector",
    eps: float = 1e-5,
) -> Tensor:
    """Deterministically embed rooted tree nodes into a two-dimensional Poincare disk.

    Radius is monotone in tree depth. ``branch_sector`` recursively splits local
    tree sectors and is useful for within-tree geometry probes. ``label_hash``
    uses stable AST-label angles. ``path_signature_hash`` hashes the typed
    root-to-node path and is a globally defined branch-aware control.
    ``depth_only`` places every node on the same ray and is a radial-only
    ablation for LCA-depth relations.
    """

    if curvature <= 0:
        raise ValueError("curvature must be positive")
    if angle_mode not in {"branch_sector", "label_hash", "path_signature_hash", "depth_only"}:
        raise ValueError(f"unknown angle_mode: {angle_mode!r}")
    order = tree.preorder()
    max_node_id = max(order)
    embeddings = torch.zeros((max_node_id + 1, 2), dtype=torch.float32)
    sqrt_c = math.sqrt(curvature)

    sectors: dict[int, tuple[float, float]] = {tree.root_id: (0.0, 2.0 * math.pi)}
    stack = [tree.root_id]
    while stack:
        node = stack.pop()
        start_angle, end_angle = sectors[node]
        if angle_mode == "label_hash":
            angle = _stable_label_angle(tree.labels.get(node, ""))
        elif angle_mode == "path_signature_hash":
            angle = _stable_label_angle(_node_path_signature(tree, node))
        elif angle_mode == "depth_only":
            angle = 0.0
        else:
            angle = 0.5 * (start_angle + end_angle)
        depth = tree.depth(node)
        radius = math.tanh(radial_scale * depth) / sqrt_c
        radius = min(radius, (1.0 - eps) / sqrt_c)
        embeddings[node] = torch.tensor(
            [radius * math.cos(angle), radius * math.sin(angle)],
            dtype=torch.float32,
        )

        children = tree.children_by_node.get(node, ())
        if not children:
            continue
        width = (end_angle - start_angle) / len(children)
        for index, child in enumerate(children):
            child_start = start_angle + index * width
            child_end = child_start + width
            sectors[child] = (child_start, child_end)
        stack.extend(reversed(children))

    return torch_project_to_ball(embeddings, curvature, eps=eps)


def _stable_label_angle(label: str) -> float:
    digest = blake2b(label.encode("utf-8"), digest_size=8).digest()
    value = int.from_bytes(digest, byteorder="big", signed=False)
    return (value / float(2**64 - 1)) * 2.0 * math.pi


def _node_path_signature(tree: RawAstTree, node: int) -> str:
    ancestors = tuple(reversed(tree.ancestors(node)))
    parts = []
    for current in ancestors:
        parent = tree.parent(current)
        if parent is None:
            child_index = 0
        else:
            child_index = tree.children_by_node[parent].index(current)
        parts.append(f"{tree.labels.get(current, '')}:{child_index}")
    return "/".join(parts)


def poincare_path_measure_from_paths(
    tree: RawAstTree,
    paths: Sequence[RawAstPath],
    *,
    node_embeddings: Tensor | None = None,
    curvature: float = 1.0,
    mass: Tensor | Sequence[float] | None = None,
) -> PoincarePathMeasure:
    """Represent raw AST paths as ``(LCA, start, end)`` product-hyperbolic points."""

    if not paths:
        raise ValueError("at least one AST path is required")
    embeddings = (
        poincare_node_embeddings_from_tree(tree, curvature=curvature)
        if node_embeddings is None
        else torch.as_tensor(node_embeddings, dtype=torch.float32)
    )
    triples = []
    for path in paths:
        lca = path.lca(tree)
        triples.append(torch.stack([embeddings[lca], embeddings[path.start], embeddings[path.end]], dim=0))
    points = torch.stack(triples, dim=0)
    if mass is None:
        mass_tensor = torch.ones((len(paths),), dtype=torch.float32)
    else:
        mass_tensor = torch.as_tensor(mass, dtype=torch.float32)
    return PoincarePathMeasure(points=points, mass=mass_tensor, curvature=curvature)


def poincare_gromov_product_at_origin(left: Tensor, right: Tensor, *, curvature: float = 1.0) -> Tensor:
    """Gromov product ``(left | right)_o`` in the Poincare ball at the origin.

    For a rooted tree, ``(u | v)_root`` equals the depth of ``LCA(u, v)``.
    This function is the hyperbolic counterpart used to supervise whether
    embedded path endpoints preserve the LCA/depth relation.
    """

    left = torch.as_tensor(left, dtype=torch.float32)
    right = torch.as_tensor(right, dtype=left.dtype, device=left.device)
    origin = torch.zeros_like(left)
    return 0.5 * (
        torch_poincare_distance(origin, left, curvature=curvature)
        + torch_poincare_distance(origin, right, curvature=curvature)
        - torch_poincare_distance(left, right, curvature=curvature)
    )


def lca_anchored_product_cost_matrix(
    left: PoincarePathMeasure,
    right: PoincarePathMeasure,
    *,
    lca_weight: float = 1.0,
    start_weight: float = 1.0,
    end_weight: float = 1.0,
    unoriented: bool = True,
) -> Tensor:
    """Squared product-hyperbolic ground cost between LCA-anchored path objects."""

    weights = torch.stack(
        [
            torch.as_tensor(lca_weight, dtype=left.points.dtype, device=left.points.device),
            torch.as_tensor(start_weight, dtype=left.points.dtype, device=left.points.device),
            torch.as_tensor(end_weight, dtype=left.points.dtype, device=left.points.device),
        ]
    )
    if bool((weights < 0.0).any()):
        raise ValueError("path-object weights must be non-negative")
    if left.points.shape[-1] != right.points.shape[-1]:
        raise ValueError("left and right path measures must have the same dimension")
    curvature = left.curvature
    if abs(curvature - right.curvature) > 1e-12:
        raise ValueError("left and right path measures must use the same curvature")

    left_lca = left.points[:, 0].unsqueeze(1)
    left_start = left.points[:, 1].unsqueeze(1)
    left_end = left.points[:, 2].unsqueeze(1)
    right_lca = right.points[:, 0].unsqueeze(0)
    right_start = right.points[:, 1].unsqueeze(0)
    right_end = right.points[:, 2].unsqueeze(0)

    lca_term = weights[0] * torch_poincare_distance(left_lca, right_lca, curvature=curvature).square()
    direct = (
        weights[1] * torch_poincare_distance(left_start, right_start, curvature=curvature).square()
        + weights[2] * torch_poincare_distance(left_end, right_end, curvature=curvature).square()
    )
    if not unoriented:
        return lca_term + direct
    reversed_direct = (
        weights[1] * torch_poincare_distance(left_start, right_end, curvature=curvature).square()
        + weights[2] * torch_poincare_distance(left_end, right_start, curvature=curvature).square()
    )
    return lca_term + torch.minimum(direct, reversed_direct)


def endpoint_product_cost_matrix(
    left: PoincarePathMeasure,
    right: PoincarePathMeasure,
    *,
    start_weight: float = 1.0,
    end_weight: float = 1.0,
    unoriented: bool = True,
) -> Tensor:
    """Endpoint-only product-hyperbolic ground cost used as an ablation."""

    return lca_anchored_product_cost_matrix(
        left,
        right,
        lca_weight=0.0,
        start_weight=start_weight,
        end_weight=end_weight,
        unoriented=unoriented,
    )


def euclidean_lca_anchored_product_cost_matrix(
    left: EuclideanPathMeasure,
    right: EuclideanPathMeasure,
    *,
    lca_weight: float = 1.0,
    start_weight: float = 1.0,
    end_weight: float = 1.0,
    unoriented: bool = True,
) -> Tensor:
    """Squared product-Euclidean ground cost for the matched control."""

    weights = torch.stack(
        [
            torch.as_tensor(lca_weight, dtype=left.points.dtype, device=left.points.device),
            torch.as_tensor(start_weight, dtype=left.points.dtype, device=left.points.device),
            torch.as_tensor(end_weight, dtype=left.points.dtype, device=left.points.device),
        ]
    )
    if bool((weights < 0.0).any()):
        raise ValueError("path-object weights must be non-negative")
    if left.points.shape[-1] != right.points.shape[-1]:
        raise ValueError("left and right path measures must have the same dimension")

    left_lca = left.points[:, 0].unsqueeze(1)
    left_start = left.points[:, 1].unsqueeze(1)
    left_end = left.points[:, 2].unsqueeze(1)
    right_lca = right.points[:, 0].unsqueeze(0)
    right_start = right.points[:, 1].unsqueeze(0)
    right_end = right.points[:, 2].unsqueeze(0)

    lca_term = weights[0] * torch.sum((left_lca - right_lca).square(), dim=-1)
    direct = (
        weights[1] * torch.sum((left_start - right_start).square(), dim=-1)
        + weights[2] * torch.sum((left_end - right_end).square(), dim=-1)
    )
    if not unoriented:
        return lca_term + direct
    reversed_direct = (
        weights[1] * torch.sum((left_start - right_end).square(), dim=-1)
        + weights[2] * torch.sum((left_end - right_start).square(), dim=-1)
    )
    return lca_term + torch.minimum(direct, reversed_direct)


def sinkhorn_path_measure_distance(
    left: PoincarePathMeasure,
    right: PoincarePathMeasure,
    *,
    epsilon: float = 0.05,
    iterations: int = 128,
    lca_weight: float = 1.0,
    start_weight: float = 1.0,
    end_weight: float = 1.0,
    unoriented: bool = True,
    normalize_cost: bool = True,
) -> Tensor:
    """Debiased Sinkhorn divergence over LCA-anchored product-hyperbolic ground cost."""

    cross = lca_anchored_product_cost_matrix(
        left,
        right,
        lca_weight=lca_weight,
        start_weight=start_weight,
        end_weight=end_weight,
        unoriented=unoriented,
    )
    left_self = lca_anchored_product_cost_matrix(
        left,
        left,
        lca_weight=lca_weight,
        start_weight=start_weight,
        end_weight=end_weight,
        unoriented=unoriented,
    )
    right_self = lca_anchored_product_cost_matrix(
        right,
        right,
        lca_weight=lca_weight,
        start_weight=start_weight,
        end_weight=end_weight,
        unoriented=unoriented,
    )
    if normalize_cost:
        scale = torch.max(torch.stack([cross.max(), left_self.max(), right_self.max()]))
        if float(scale.detach()) > 1e-12:
            cross = cross / scale
            left_self = left_self / scale
            right_self = right_self / scale
    return sinkhorn_divergence(
        cross,
        left.mass,
        right.mass,
        left_self_cost=left_self,
        right_self_cost=right_self,
        epsilon=epsilon,
        iterations=iterations,
    )


def sinkhorn_euclidean_path_measure_distance(
    left: EuclideanPathMeasure,
    right: EuclideanPathMeasure,
    *,
    epsilon: float = 0.05,
    iterations: int = 128,
    lca_weight: float = 1.0,
    start_weight: float = 1.0,
    end_weight: float = 1.0,
    unoriented: bool = True,
    normalize_cost: bool = True,
) -> Tensor:
    """Debiased Sinkhorn divergence over matched product-Euclidean ground cost."""

    cross = euclidean_lca_anchored_product_cost_matrix(
        left,
        right,
        lca_weight=lca_weight,
        start_weight=start_weight,
        end_weight=end_weight,
        unoriented=unoriented,
    )
    left_self = euclidean_lca_anchored_product_cost_matrix(
        left,
        left,
        lca_weight=lca_weight,
        start_weight=start_weight,
        end_weight=end_weight,
        unoriented=unoriented,
    )
    right_self = euclidean_lca_anchored_product_cost_matrix(
        right,
        right,
        lca_weight=lca_weight,
        start_weight=start_weight,
        end_weight=end_weight,
        unoriented=unoriented,
    )
    if normalize_cost:
        scale = torch.max(torch.stack([cross.max(), left_self.max(), right_self.max()]))
        if float(scale.detach()) > 1e-12:
            cross = cross / scale
            left_self = left_self / scale
            right_self = right_self / scale
    return sinkhorn_divergence(
        cross,
        left.mass,
        right.mass,
        left_self_cost=left_self,
        right_self_cost=right_self,
        epsilon=epsilon,
        iterations=iterations,
    )
