from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Literal, Mapping, Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from geometry_profile_research.code2hyp_torch import (
    torch_expmap0,
    torch_logmap0,
    torch_poincare_distance,
    torch_project_to_ball,
)
from geometry_profile_research.gromov_wasserstein import sinkhorn_divergence
from geometry_profile_research.raw_ast import NodeId, RawAstPath, RawAstTree, terminal_to_terminal_paths


Manifold = Literal["poincare", "euclidean"]
TerminalPolicy = Literal["type", "class", "value"]
NodeInputMode = Literal["label_only", "label_depth", "label_depth_prefix"]
PathObjectMode = Literal["single_point", "lca_product"]
AnchorMode = Literal["true_lca", "zero_anchor", "root_anchor", "depth_matched_shuffled"]
MethodAggregation = Literal["centroid", "measure"]
PathCostOrientation = Literal["directed", "unoriented"]


@dataclass(frozen=True)
class RawASTMethodMeasure:
    """Uniform measure over canonical Code2Hyp path objects.

    In ``lca_product`` mode, ``points[i]`` stores
    ``(x_lca, x_start, x_end)``. In ``single_point`` mode it stores one pooled
    path point. ``left_branch`` and ``right_branch`` store order-aware branch
    codes for the two sides of the terminal-to-terminal raw-AST path.
    """

    points: Tensor
    left_branch: Tensor
    right_branch: Tensor
    mass: Tensor
    manifold: Manifold
    curvature: float = 1.0
    path_object_mode: PathObjectMode = "lca_product"

    def __post_init__(self) -> None:
        points = torch.as_tensor(self.points, dtype=torch.float32)
        if points.ndim != 3:
            raise ValueError("points must have shape (n_paths, n_factors, dim)")
        if self.path_object_mode == "lca_product" and points.shape[1] != 3:
            raise ValueError("lca_product points must have shape (n_paths, 3, dim)")
        if self.path_object_mode == "single_point" and points.shape[1] != 1:
            raise ValueError("single_point points must have shape (n_paths, 1, dim)")
        if self.path_object_mode not in {"single_point", "lca_product"}:
            raise ValueError(f"unknown path_object_mode: {self.path_object_mode!r}")
        left_branch = torch.as_tensor(self.left_branch, dtype=points.dtype, device=points.device)
        right_branch = torch.as_tensor(self.right_branch, dtype=points.dtype, device=points.device)
        if left_branch.shape != right_branch.shape or left_branch.shape[0] != points.shape[0]:
            raise ValueError("branch tensors must have shape (n_paths, branch_dim)")
        mass = torch.as_tensor(self.mass, dtype=points.dtype, device=points.device)
        if mass.ndim != 1 or mass.numel() != points.shape[0]:
            raise ValueError("mass must contain one value per path")
        if bool((mass < 0.0).any()):
            raise ValueError("mass values must be non-negative")
        total = torch.sum(mass)
        if float(total) <= 0.0:
            raise ValueError("mass must have positive total")
        if self.manifold == "poincare":
            points = torch_project_to_ball(points, self.curvature)
        object.__setattr__(self, "points", points)
        object.__setattr__(self, "left_branch", left_branch)
        object.__setattr__(self, "right_branch", right_branch)
        object.__setattr__(self, "mass", mass / total)


def build_raw_ast_token_vocab(
    trees: Sequence[RawAstTree],
    *,
    terminal_policy: TerminalPolicy = "type",
    node_input_mode: NodeInputMode = "label_depth_prefix",
) -> dict[str, int]:
    """Build a small token vocabulary for raw-AST root-to-node sequences."""

    if terminal_policy not in {"type", "class", "value"}:
        raise ValueError(f"unknown terminal_policy: {terminal_policy!r}")
    if node_input_mode not in {"label_only", "label_depth", "label_depth_prefix"}:
        raise ValueError(f"unknown node_input_mode: {node_input_mode!r}")
    vocab = {"<pad>": 0, "<unk>": 1}
    for tree in trees:
        for node in tree.preorder():
            for token in _root_to_node_tokens(
                tree,
                node,
                terminal_policy=terminal_policy,
                input_mode=node_input_mode,
            ):
                vocab.setdefault(token, len(vocab))
            parent = tree.parent(node)
            if parent is not None:
                edge = _edge_token(tree, node)
                vocab.setdefault(edge, len(vocab))
                vocab.setdefault(f"edge_up:{edge}", len(vocab))
                vocab.setdefault(f"edge_down:{edge}", len(vocab))
    return vocab


class RawASTCode2Hyp(nn.Module):
    """Canonical raw-AST Code2Hyp model.

    This is the canonical model family:

    raw AST node -> shared root-to-node GRU point;
    AST path -> LCA/start/end product object plus two branch GRU codes;
    method -> uniform measure over path objects;
    method distance -> debiased Sinkhorn divergence.
    """

    def __init__(
        self,
        token_to_id: Mapping[str, int],
        *,
        dim: int = 8,
        token_dim: int | None = None,
        manifold: Manifold = "poincare",
        curvature: float = 1.0,
        max_paths: int | None = 32,
        terminal_policy: TerminalPolicy = "type",
        node_input_mode: NodeInputMode = "label_depth_prefix",
        path_object_mode: PathObjectMode = "lca_product",
        method_aggregation: MethodAggregation = "measure",
        path_cost_orientation: PathCostOrientation = "directed",
        path_selection_policy: str = "preorder_first",
        anchor_mode: AnchorMode = "true_lca",
    ) -> None:
        super().__init__()
        if manifold not in {"poincare", "euclidean"}:
            raise ValueError(f"unknown manifold: {manifold!r}")
        if terminal_policy not in {"type", "class", "value"}:
            raise ValueError(f"unknown terminal_policy: {terminal_policy!r}")
        if node_input_mode not in {"label_only", "label_depth", "label_depth_prefix"}:
            raise ValueError(f"unknown node_input_mode: {node_input_mode!r}")
        if path_object_mode not in {"single_point", "lca_product"}:
            raise ValueError(f"unknown path_object_mode: {path_object_mode!r}")
        if method_aggregation not in {"centroid", "measure"}:
            raise ValueError(f"unknown method_aggregation: {method_aggregation!r}")
        if path_cost_orientation not in {"directed", "unoriented"}:
            raise ValueError(f"unknown path_cost_orientation: {path_cost_orientation!r}")
        if path_selection_policy not in {"preorder_first", "hash_sorted", "lca_depth_stratified"}:
            raise ValueError(f"unknown path_selection_policy: {path_selection_policy!r}")
        if anchor_mode not in {"true_lca", "zero_anchor", "root_anchor", "depth_matched_shuffled"}:
            raise ValueError(f"unknown anchor_mode: {anchor_mode!r}")
        if dim <= 0:
            raise ValueError("dim must be positive")
        if curvature <= 0:
            raise ValueError("curvature must be positive")
        self.token_to_id = dict(token_to_id)
        self.dim = int(dim)
        self.token_dim = int(token_dim or dim)
        self.manifold = manifold
        self.curvature = float(curvature)
        self.max_paths = max_paths
        self.terminal_policy = terminal_policy
        self.node_input_mode = node_input_mode
        self.path_object_mode = path_object_mode
        self.method_aggregation = method_aggregation
        self.path_cost_orientation = path_cost_orientation
        self.path_selection_policy = path_selection_policy
        self.anchor_mode = anchor_mode
        self.embedding = nn.Embedding(len(self.token_to_id), self.token_dim, padding_idx=0)
        self.node_gru = nn.GRU(self.token_dim, self.dim, batch_first=True)
        self.branch_gru = nn.GRU(self.token_dim, self.dim, batch_first=True)
        self.node_projection = nn.Linear(self.dim, self.dim)

    def encode_method(self, tree: RawAstTree, paths: Sequence[RawAstPath] | None = None) -> RawASTMethodMeasure:
        """Encode one raw-AST callable scope as a uniform measure over path objects."""

        raw_paths = (
            tuple(paths)
            if paths is not None
            else terminal_to_terminal_paths(
                tree,
                max_paths=self.max_paths,
                selection_policy=self.path_selection_policy,
            )
        )
        if not raw_paths:
            raise ValueError("encode_method requires at least one terminal-to-terminal path")
        node_points = self.encode_nodes(tree)
        triples = []
        left_branches = []
        right_branches = []
        for path in raw_paths:
            lca = path.lca(tree)
            anchor = self._anchor_point(tree, path, node_points=node_points, true_lca=lca)
            lca_product = torch.stack([anchor, node_points[path.start], node_points[path.end]], dim=0)
            if self.path_object_mode == "lca_product":
                triples.append(lca_product)
            else:
                triples.append(self._single_path_point(lca_product).unsqueeze(0))
            left_ids, right_ids = _branch_node_ids(tree, path)
            left_branches.append(self._encode_branch(tree, left_ids))
            right_branches.append(self._encode_branch(tree, right_ids))
        path_count = len(raw_paths)
        return RawASTMethodMeasure(
            points=torch.stack(triples, dim=0),
            left_branch=torch.stack(left_branches, dim=0),
            right_branch=torch.stack(right_branches, dim=0),
            mass=torch.full((path_count,), 1.0 / path_count, dtype=torch.float32),
            manifold=self.manifold,
            curvature=self.curvature,
            path_object_mode=self.path_object_mode,
        )

    def _anchor_point(self, tree: RawAstTree, path: RawAstPath, *, node_points: Tensor, true_lca: NodeId) -> Tensor:
        if self.anchor_mode == "true_lca":
            return node_points[true_lca]
        if self.anchor_mode == "zero_anchor":
            return torch.zeros_like(node_points[true_lca])
        if self.anchor_mode == "root_anchor":
            return node_points[tree.root_id]

        depth = tree.depth(true_lca)
        candidates = [node for node in tree.preorder() if tree.depth(node) == depth and node != true_lca]
        if not candidates:
            return torch.zeros_like(node_points[true_lca])
        candidates.sort(key=lambda node: (_node_structural_signature(tree, node), node))
        payload = "|".join(
            (
                _node_structural_signature(tree, path.start),
                _node_structural_signature(tree, path.end),
                str(depth),
            )
        ).encode("utf-8")
        index = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % len(candidates)
        return node_points[candidates[index]]

    def encode_nodes(self, tree: RawAstTree) -> Tensor:
        """Encode every node id in the tree into the selected manifold."""

        order = tree.preorder()
        sequences = [
            _root_to_node_tokens(
                tree,
                node,
                terminal_policy=self.terminal_policy,
                input_mode=self.node_input_mode,
            )
            for node in order
        ]
        encoded = self._encode_token_sequences(sequences, self.node_gru)
        tangent = self.node_projection(encoded)
        if self.manifold == "poincare":
            points = torch_expmap0(tangent, self.curvature)
        else:
            points = tangent
        max_node_id = max(order)
        output = torch.zeros((max_node_id + 1, self.dim), dtype=points.dtype, device=points.device)
        for row, node in enumerate(order):
            output[node] = points[row]
        return output

    def method_distance(
        self,
        left: RawASTMethodMeasure,
        right: RawASTMethodMeasure,
        *,
        epsilon: float = 0.05,
        sinkhorn_iterations: int = 80,
        normalize_cost: bool = False,
    ) -> Tensor:
        """Distance between two method representations.

        ``method_aggregation="measure"`` uses debiased Sinkhorn divergence over
        path measures. ``method_aggregation="centroid"`` collapses each method
        to a weighted path-object centroid and gives the matched control needed
        for the reviewer-recommended factor matrix. ``normalize_cost=True`` is
        a legacy exploratory mode; confirmatory runs should use a train-split
        scale for ``epsilon`` instead of pair-local cost normalization.
        """

        if self.method_aggregation == "centroid":
            return self._centroid_distance(left, right)

        cross = self._path_cost_matrix(left, right)
        left_self = self._path_cost_matrix(left, left)
        right_self = self._path_cost_matrix(right, right)
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
            iterations=sinkhorn_iterations,
        )

    def training_loss(
        self,
        triples: Sequence[tuple[RawAstTree, RawAstTree, RawAstTree]],
        *,
        margin: float = 0.1,
        lambda_edge: float = 0.1,
        lambda_gromov: float = 0.1,
        lambda_branch: float = 0.1,
        lambda_reversal: float = 0.1,
        lambda_retrieval: float = 1.0,
        sinkhorn_epsilon: float = 0.05,
        sinkhorn_iterations: int = 40,
    ) -> dict[str, Tensor]:
        """Compute retrieval plus hierarchy regularizers for tree triples.

        Each triple is ``(anchor, positive, hard_negative)``. Positive examples
        are expected to be safe transformations of the same method; hard
        negatives are lexically or semantically similar but structurally
        different methods.
        """

        retrieval_terms = []
        for anchor_tree, positive_tree, negative_tree in triples:
            anchor = self.encode_method(anchor_tree)
            positive = self.encode_method(positive_tree)
            negative = self.encode_method(negative_tree)
            positive_distance = self.method_distance(
                anchor,
                positive,
                epsilon=sinkhorn_epsilon,
                sinkhorn_iterations=sinkhorn_iterations,
            )
            negative_distance = self.method_distance(
                anchor,
                negative,
                epsilon=sinkhorn_epsilon,
                sinkhorn_iterations=sinkhorn_iterations,
            )
            retrieval_terms.append(F.relu(margin + positive_distance - negative_distance))
        retrieval = torch.stack(retrieval_terms).mean() if retrieval_terms else torch.zeros(())
        unique_trees = []
        seen = set()
        for triple in triples:
            for tree in triple:
                key = id(tree)
                if key not in seen:
                    seen.add(key)
                    unique_trees.append(tree)
        edge = torch.stack([self.edge_length_loss(tree) for tree in unique_trees]).mean()
        gromov = torch.stack([self.gromov_lca_loss(tree) for tree in unique_trees]).mean()
        gromov_residuals = tuple(self.gromov_lca_residuals(tree) for tree in unique_trees)
        nonempty_residuals = tuple(residual for residual in gromov_residuals if residual.numel() > 0)
        if nonempty_residuals:
            gromov_abs_residual = torch.cat(nonempty_residuals).abs().mean()
        else:
            gromov_abs_residual = gromov.new_tensor(0.0)
        branch = torch.stack([self.branch_length_loss(tree) for tree in unique_trees]).mean()
        reversal = torch.stack([self.reversal_equivariance_loss(tree) for tree in unique_trees]).mean()
        total = (
            lambda_retrieval * retrieval
            + lambda_edge * edge
            + lambda_gromov * gromov
            + lambda_branch * branch
            + lambda_reversal * reversal
        )
        return {
            "loss": total,
            "retrieval": retrieval,
            "edge": edge,
            "gromov_lca": gromov,
            "gromov_lca_mean_abs_residual": gromov_abs_residual,
            "branch_length": branch,
            "reversal": reversal,
            "retrieval_weight": retrieval.new_tensor(lambda_retrieval),
        }

    def edge_length_loss(self, tree: RawAstTree) -> Tensor:
        points = self.encode_nodes(tree)
        losses = []
        for parent, children in tree.children_by_node.items():
            for child in children:
                distance = self._point_distance(points[parent].unsqueeze(0), points[child].unsqueeze(0)).squeeze()
                losses.append((distance - 1.0).square())
        if not losses:
            return torch.zeros((), dtype=points.dtype, device=points.device)
        return torch.stack(losses).mean()

    def gromov_lca_loss(self, tree: RawAstTree, paths: Sequence[RawAstPath] | None = None) -> Tensor:
        residuals = self.gromov_lca_residuals(tree, paths)
        if residuals.numel() == 0:
            return torch.zeros(())
        return residuals.square().mean()

    def gromov_lca_residuals(self, tree: RawAstTree, paths: Sequence[RawAstPath] | None = None) -> Tensor:
        """Return soft Gromov-product residuals against raw LCA depths.

        These residuals are a distortion diagnostic, not a hard isometry
        certificate. A strict zero target is impossible for simple branching
        configurations in negatively curved space, so experiments should report
        residual magnitudes instead of claiming exact tree embedding.
        """

        raw_paths = tuple(paths) if paths is not None else terminal_to_terminal_paths(tree, max_paths=self.max_paths)
        if not raw_paths:
            return torch.empty(0)
        points = self.encode_nodes(tree)
        root = points[tree.root_id]
        residuals = []
        for path in raw_paths:
            lca_depth = float(tree.depth(path.lca(tree)))
            start = points[path.start]
            end = points[path.end]
            product = 0.5 * (
                self._point_distance(root.unsqueeze(0), start.unsqueeze(0)).squeeze()
                + self._point_distance(root.unsqueeze(0), end.unsqueeze(0)).squeeze()
                - self._point_distance(start.unsqueeze(0), end.unsqueeze(0)).squeeze()
            )
            residuals.append(product - lca_depth)
        return torch.stack(residuals)

    def gromov_lca_diagnostics(self, tree: RawAstTree, paths: Sequence[RawAstPath] | None = None) -> dict[str, float]:
        """Summarize LCA/Gromov residuals for reporting."""

        with torch.no_grad():
            residuals = self.gromov_lca_residuals(tree, paths)
            if residuals.numel() == 0:
                return {
                    "path_count": 0.0,
                    "mean_abs_residual": 0.0,
                    "max_abs_residual": 0.0,
                    "mse": 0.0,
                }
            absolute = residuals.abs()
            return {
                "path_count": float(residuals.numel()),
                "mean_abs_residual": float(absolute.mean().detach()),
                "max_abs_residual": float(absolute.max().detach()),
                "mse": float(residuals.square().mean().detach()),
            }

    def branch_length_loss(self, tree: RawAstTree, paths: Sequence[RawAstPath] | None = None) -> Tensor:
        """Match LCA-to-endpoint geodesic lengths to raw AST branch lengths."""

        raw_paths = tuple(paths) if paths is not None else terminal_to_terminal_paths(tree, max_paths=self.max_paths)
        if not raw_paths:
            return torch.zeros(())
        points = self.encode_nodes(tree)
        losses = []
        for path in raw_paths:
            lca = path.lca(tree)
            left_distance = self._point_distance(points[lca].unsqueeze(0), points[path.start].unsqueeze(0)).squeeze()
            right_distance = self._point_distance(points[lca].unsqueeze(0), points[path.end].unsqueeze(0)).squeeze()
            left_target = float(tree.tree_distance(lca, path.start))
            right_target = float(tree.tree_distance(lca, path.end))
            losses.append((left_distance - left_target).square())
            losses.append((right_distance - right_target).square())
        return torch.stack(losses).mean()

    def reversal_equivariance_loss(self, tree: RawAstTree, paths: Sequence[RawAstPath] | None = None) -> Tensor:
        """Check that reversing an AST path corresponds to swapping path sides."""

        raw_paths = tuple(paths) if paths is not None else terminal_to_terminal_paths(tree, max_paths=self.max_paths)
        if not raw_paths:
            return torch.zeros(())
        forward = self.encode_method(tree, raw_paths)
        reversed_measure = self.encode_method(tree, tuple(path.reversed() for path in raw_paths))
        swapped_forward = self._swap_path_orientation(forward)
        cost = self._path_cost_matrix(swapped_forward, reversed_measure)
        return torch.diagonal(cost).mean()

    def _path_cost_matrix(self, left: RawASTMethodMeasure, right: RawASTMethodMeasure) -> Tensor:
        if left.manifold != self.manifold or right.manifold != self.manifold:
            raise ValueError("method measures must match the model manifold")
        if left.path_object_mode != self.path_object_mode or right.path_object_mode != self.path_object_mode:
            raise ValueError("method measures must match the model path_object_mode")
        if self.path_object_mode == "lca_product":
            lca = self._point_distance(left.points[:, 0].unsqueeze(1), right.points[:, 0].unsqueeze(0)).square()
            direct_endpoints = (
                self._point_distance(left.points[:, 1].unsqueeze(1), right.points[:, 1].unsqueeze(0)).square()
                + self._point_distance(left.points[:, 2].unsqueeze(1), right.points[:, 2].unsqueeze(0)).square()
            )
            reversed_endpoints = (
                self._point_distance(left.points[:, 1].unsqueeze(1), right.points[:, 2].unsqueeze(0)).square()
                + self._point_distance(left.points[:, 2].unsqueeze(1), right.points[:, 1].unsqueeze(0)).square()
            )
        else:
            point_cost = self._point_distance(left.points[:, 0].unsqueeze(1), right.points[:, 0].unsqueeze(0)).square()
        direct_branch = (
            torch.sum((left.left_branch.unsqueeze(1) - right.left_branch.unsqueeze(0)).square(), dim=-1)
            + torch.sum((left.right_branch.unsqueeze(1) - right.right_branch.unsqueeze(0)).square(), dim=-1)
        )
        if self.path_cost_orientation == "directed":
            if self.path_object_mode == "lca_product":
                point_cost = lca + direct_endpoints
            return point_cost + direct_branch
        reversed_branch = (
            torch.sum((left.left_branch.unsqueeze(1) - right.right_branch.unsqueeze(0)).square(), dim=-1)
            + torch.sum((left.right_branch.unsqueeze(1) - right.left_branch.unsqueeze(0)).square(), dim=-1)
        )
        if self.path_object_mode == "lca_product":
            return lca + torch.minimum(direct_endpoints + direct_branch, reversed_endpoints + reversed_branch)
        return point_cost + torch.minimum(direct_branch, reversed_branch)

    def _centroid_distance(self, left: RawASTMethodMeasure, right: RawASTMethodMeasure) -> Tensor:
        if left.manifold != self.manifold or right.manifold != self.manifold:
            raise ValueError("method measures must match the model manifold")
        if left.path_object_mode != self.path_object_mode or right.path_object_mode != self.path_object_mode:
            raise ValueError("method measures must match the model path_object_mode")
        left_points, left_branch, right_branch_left = self._measure_centroid(left)
        right_points, right_left_branch, right_right_branch = self._measure_centroid(right)
        direct_branch = torch.sum((left_branch - right_left_branch).square()) + torch.sum((right_branch_left - right_right_branch).square())
        if self.path_object_mode == "lca_product":
            lca = self._point_distance(left_points[0].unsqueeze(0), right_points[0].unsqueeze(0)).square().squeeze()
            direct_endpoints = (
                self._point_distance(left_points[1].unsqueeze(0), right_points[1].unsqueeze(0)).square().squeeze()
                + self._point_distance(left_points[2].unsqueeze(0), right_points[2].unsqueeze(0)).square().squeeze()
            )
            if self.path_cost_orientation == "directed":
                return lca + direct_endpoints + direct_branch
            reversed_endpoints = (
                self._point_distance(left_points[1].unsqueeze(0), right_points[2].unsqueeze(0)).square().squeeze()
                + self._point_distance(left_points[2].unsqueeze(0), right_points[1].unsqueeze(0)).square().squeeze()
            )
            reversed_branch = torch.sum((left_branch - right_right_branch).square()) + torch.sum((right_branch_left - right_left_branch).square())
            return lca + torch.minimum(direct_endpoints + direct_branch, reversed_endpoints + reversed_branch)
        point_cost = self._point_distance(left_points, right_points).square().sum()
        if self.path_cost_orientation == "directed":
            return point_cost + direct_branch
        reversed_branch = torch.sum((left_branch - right_right_branch).square()) + torch.sum((right_branch_left - right_left_branch).square())
        return point_cost + torch.minimum(direct_branch, reversed_branch)

    def _measure_centroid(self, measure: RawASTMethodMeasure) -> tuple[Tensor, Tensor, Tensor]:
        point_weights = measure.mass.view(-1, 1, 1)
        if self.manifold == "poincare":
            tangent = torch_logmap0(measure.points, self.curvature)
            point_centroid = torch_expmap0(torch.sum(point_weights * tangent, dim=0), self.curvature)
        else:
            point_centroid = torch.sum(point_weights * measure.points, dim=0)
        branch_weights = measure.mass.view(-1, 1)
        left_branch = torch.sum(branch_weights * measure.left_branch, dim=0)
        right_branch = torch.sum(branch_weights * measure.right_branch, dim=0)
        return point_centroid, left_branch, right_branch

    def _swap_path_orientation(self, measure: RawASTMethodMeasure) -> RawASTMethodMeasure:
        if measure.path_object_mode == "lca_product":
            points = torch.stack((measure.points[:, 0], measure.points[:, 2], measure.points[:, 1]), dim=1)
        else:
            points = measure.points
        return RawASTMethodMeasure(
            points=points,
            left_branch=measure.right_branch,
            right_branch=measure.left_branch,
            mass=measure.mass,
            manifold=measure.manifold,
            curvature=measure.curvature,
            path_object_mode=measure.path_object_mode,
        )

    def _single_path_point(self, lca_product: Tensor) -> Tensor:
        if self.manifold == "poincare":
            tangent = torch_logmap0(lca_product, self.curvature)
            return torch_expmap0(torch.mean(tangent, dim=0), self.curvature)
        return torch.mean(lca_product, dim=0)

    def _point_distance(self, left: Tensor, right: Tensor) -> Tensor:
        if self.manifold == "poincare":
            return torch_poincare_distance(left, right, curvature=self.curvature)
        return torch.linalg.vector_norm(left - right, dim=-1)

    def _encode_branch(self, tree: RawAstTree, node_ids: Sequence[NodeId]) -> Tensor:
        tokens = _branch_tokens(tree, node_ids, terminal_policy=self.terminal_policy)
        return self._encode_token_sequences((tuple(tokens),), self.branch_gru)[0]

    def _encode_token_sequences(self, sequences: Sequence[Sequence[str]], gru: nn.GRU) -> Tensor:
        max_length = max(len(sequence) for sequence in sequences)
        ids = torch.zeros((len(sequences), max_length), dtype=torch.long, device=self.embedding.weight.device)
        lengths = []
        for row, sequence in enumerate(sequences):
            lengths.append(len(sequence))
            for column, token in enumerate(sequence):
                ids[row, column] = self.token_to_id.get(token, self.token_to_id.get("<unk>", 1))
        embedded = self.embedding(ids)
        output, hidden = gru(embedded)
        del output
        return hidden[-1]


def _node_token(tree: RawAstTree, node: NodeId, *, terminal_policy: TerminalPolicy = "type") -> str:
    attributes = tree.attributes.get(node, {})
    terminal = attributes.get("terminal_token")
    if tree.labels.get(node) == "TerminalToken" and terminal is not None:
        if terminal_policy == "value":
            return f"terminal:{terminal}"
        if terminal_policy == "class":
            return f"terminal_class:{_terminal_class(terminal)}"
    return f"node:{tree.labels.get(node, '')}"


def _node_structural_signature(tree: RawAstTree, node: NodeId) -> str:
    return "/".join(
        _root_to_node_tokens(
            tree,
            node,
            terminal_policy="class",
            input_mode="label_depth_prefix",
        )
    )


def _edge_token(tree: RawAstTree, node: NodeId) -> str:
    parent = tree.parent(node)
    if parent is None:
        return "edge:root"
    attributes = tree.attributes.get(node, {})
    edge_type = attributes.get("edge_type")
    child_index_value = attributes.get("child_index")
    if edge_type is not None:
        if child_index_value is None:
            child_index_value = str(tree.children_by_node[parent].index(node))
        return f"edge:{edge_type}:{child_index_value}"
    child_index = tree.children_by_node[parent].index(node)
    return f"edge:child_{child_index}"


def _root_to_node_tokens(
    tree: RawAstTree,
    node: NodeId,
    *,
    terminal_policy: TerminalPolicy = "type",
    input_mode: NodeInputMode = "label_depth_prefix",
) -> tuple[str, ...]:
    if input_mode not in {"label_only", "label_depth", "label_depth_prefix"}:
        raise ValueError(f"unknown node input mode: {input_mode!r}")
    node_token = _node_token(tree, node, terminal_policy=terminal_policy)
    if input_mode == "label_only":
        return (node_token,)
    depth_token = f"depth:{tree.depth(node)}"
    if input_mode == "label_depth":
        return (node_token, depth_token)
    ancestors = tuple(reversed(tree.ancestors(node)))
    tokens = [_node_token(tree, ancestors[0], terminal_policy=terminal_policy)]
    for current in ancestors[1:]:
        tokens.append(_edge_token(tree, current))
        tokens.append(_node_token(tree, current, terminal_policy=terminal_policy))
    tokens.append(depth_token)
    return tuple(tokens)


def _branch_node_ids(tree: RawAstTree, path: RawAstPath) -> tuple[tuple[NodeId, ...], tuple[NodeId, ...]]:
    lca = path.lca(tree)
    lca_index = path.nodes.index(lca)
    left = tuple(path.nodes[: lca_index + 1])
    right = tuple(reversed(path.nodes[lca_index:]))
    return left, right


def _branch_tokens(
    tree: RawAstTree,
    node_ids: Sequence[NodeId],
    *,
    terminal_policy: TerminalPolicy = "type",
) -> tuple[str, ...]:
    if not node_ids:
        return (_node_token(tree, tree.root_id, terminal_policy=terminal_policy),)
    tokens = [_node_token(tree, node_ids[0], terminal_policy=terminal_policy)]
    for child, parent in zip(node_ids, node_ids[1:]):
        if tree.parent(child) == parent:
            tokens.append(f"edge_up:{_edge_token(tree, child)}")
        elif tree.parent(parent) == child:
            tokens.append(f"edge_down:{_edge_token(tree, parent)}")
        else:
            raise ValueError("branch nodes must be adjacent in the raw AST")
        tokens.append(_node_token(tree, parent, terminal_policy=terminal_policy))
    return tuple(tokens)


def _terminal_class(token: str) -> str:
    if token == "":
        return "empty"
    if token.isidentifier():
        return "identifier"
    try:
        float(token)
    except ValueError:
        return "symbol"
    return "number"
