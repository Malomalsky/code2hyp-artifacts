from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Sequence

import torch
from torch import Tensor, nn

from geometry_profile_research.code2hyp_torch import (
    structural_distance_loss,
    structural_normalized_stress,
    torch_expmap0,
    torch_poincare_distance,
)
from geometry_profile_research.geodesic_path_space import poincare_gromov_product_at_origin


AxiomProbeGeometry = Literal["poincare", "euclidean"]


@dataclass(frozen=True)
class AxiomPathRecord:
    scope_index: int
    start: int
    end: int
    lca: int
    length: int
    lca_depth: int

    @property
    def start_key(self) -> tuple[int, int]:
        return (self.scope_index, self.start)

    @property
    def end_key(self) -> tuple[int, int]:
        return (self.scope_index, self.end)

    @property
    def lca_key(self) -> tuple[int, int]:
        return (self.scope_index, self.lca)


@dataclass(frozen=True)
class AxiomEdgeRecord:
    scope_index: int
    parent: int
    child: int

    @property
    def parent_key(self) -> tuple[int, int]:
        return (self.scope_index, self.parent)

    @property
    def child_key(self) -> tuple[int, int]:
        return (self.scope_index, self.child)


@dataclass(frozen=True)
class EncodedAxiomProbeDataset:
    node_count: int
    train_node_index: Tensor
    train_node_depth: Tensor
    train_edge_left: Tensor
    train_edge_right: Tensor
    train_path_start: Tensor
    train_path_end: Tensor
    train_path_lca: Tensor
    train_path_length: Tensor
    train_path_lca_depth: Tensor
    eval_path_start: Tensor
    eval_path_end: Tensor
    eval_path_lca: Tensor
    eval_path_length: Tensor
    eval_path_lca_depth: Tensor


@dataclass(frozen=True)
class AxiomProbeResult:
    geometry: str
    dim: int
    seed: int
    node_count: int
    train_edge_count: int
    train_path_count: int
    eval_path_count: int
    eval_length_spearman: float
    eval_lca_depth_spearman: float
    eval_length_stress: float
    eval_lca_depth_stress: float
    eval_additivity_residual_mean: float
    eval_lca_radial_depth_spearman: float
    train_edge_distance_mean: float

    def as_dict(self) -> dict[str, int | float]:
        return {
            "geometry": self.geometry,
            "dim": self.dim,
            "seed": self.seed,
            "node_count": self.node_count,
            "train_edge_count": self.train_edge_count,
            "train_path_count": self.train_path_count,
            "eval_path_count": self.eval_path_count,
            "eval_length_spearman": self.eval_length_spearman,
            "eval_lca_depth_spearman": self.eval_lca_depth_spearman,
            "eval_length_stress": self.eval_length_stress,
            "eval_lca_depth_stress": self.eval_lca_depth_stress,
            "eval_additivity_residual_mean": self.eval_additivity_residual_mean,
            "eval_lca_radial_depth_spearman": self.eval_lca_radial_depth_spearman,
            "train_edge_distance_mean": self.train_edge_distance_mean,
        }


def _average_ranks(values: Sequence[float]) -> list[float]:
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    index = 0
    while index < len(indexed):
        end = index + 1
        while end < len(indexed) and indexed[end][1] == indexed[index][1]:
            end += 1
        average_rank = (index + end - 1) / 2.0
        for position in range(index, end):
            ranks[indexed[position][0]] = average_rank
        index = end
    return ranks


def spearman_correlation(predicted: Sequence[float], target: Sequence[float]) -> float:
    if len(predicted) != len(target):
        raise ValueError("predicted and target must have the same length")
    if len(predicted) < 2:
        return 0.0
    left = _average_ranks([float(value) for value in predicted])
    right = _average_ranks([float(value) for value in target])
    left_mean = sum(left) / len(left)
    right_mean = sum(right) / len(right)
    numerator = sum((x - left_mean) * (y - right_mean) for x, y in zip(left, right))
    left_var = sum((x - left_mean) ** 2 for x in left)
    right_var = sum((y - right_mean) ** 2 for y in right)
    denominator = math.sqrt(left_var * right_var)
    if denominator == 0.0:
        return 0.0
    return numerator / denominator


def load_axiom_probe_dataset(
    path: Path,
    *,
    max_scopes: int | None = None,
) -> EncodedAxiomProbeDataset:
    """Load raw-AST tree axioms from the extraction JSONL.

    Direct parent-child AST edges are used as the metric training substrate.
    Extracted leaf-to-leaf AST paths are split deterministically into train and
    held-out evaluation paths, so the probe tests transfer from local tree
    geometry to unseen path-level tree axioms.
    """

    edges: list[AxiomEdgeRecord] = []
    train_paths: list[AxiomPathRecord] = []
    eval_paths: list[AxiomPathRecord] = []
    node_depths: dict[tuple[int, int], int] = {}
    path_counter = 0

    for scope_index, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        if max_scopes is not None and scope_index >= max_scopes:
            break
        if not line.strip():
            continue
        payload = json.loads(line)
        if payload.get("status") != "ok":
            continue
        for record in payload.get("order_records", ()):
            if "ancestor_depth" in record:
                node_depths[(scope_index, int(record["ancestor"]))] = int(record["ancestor_depth"])
            if "descendant_depth" in record:
                node_depths[(scope_index, int(record["descendant"]))] = int(record["descendant_depth"])
            if int(record.get("label", 0)) == 1 and bool(record.get("is_direct_edge", False)):
                edges.append(
                    AxiomEdgeRecord(
                        scope_index=scope_index,
                        parent=int(record["ancestor"]),
                        child=int(record["descendant"]),
                    )
                )
        for record in payload.get("paths", ()):
            path_record = AxiomPathRecord(
                scope_index=scope_index,
                start=int(record["start"]),
                end=int(record["end"]),
                lca=int(record["lca"]),
                length=int(record["length"]),
                lca_depth=int(record["lca_depth"]),
            )
            if path_counter % 2 == 0:
                train_paths.append(path_record)
            else:
                eval_paths.append(path_record)
            path_counter += 1

    node_to_index: dict[tuple[int, int], int] = {}

    def node_index(key: tuple[int, int]) -> int:
        if key not in node_to_index:
            node_to_index[key] = len(node_to_index)
        return node_to_index[key]

    def encode_edges(records: Sequence[AxiomEdgeRecord]) -> tuple[list[int], list[int]]:
        left: list[int] = []
        right: list[int] = []
        for record in records:
            left.append(node_index(record.parent_key))
            right.append(node_index(record.child_key))
        return left, right

    def encode_paths(records: Sequence[AxiomPathRecord]) -> tuple[list[int], list[int], list[int], list[float], list[float]]:
        start: list[int] = []
        end: list[int] = []
        lca: list[int] = []
        length: list[float] = []
        lca_depth: list[float] = []
        for record in records:
            start.append(node_index(record.start_key))
            end.append(node_index(record.end_key))
            lca.append(node_index(record.lca_key))
            length.append(float(record.length))
            lca_depth.append(float(record.lca_depth))
        return start, end, lca, length, lca_depth

    edge_left, edge_right = encode_edges(edges)
    train_start, train_end, train_lca, train_length, train_lca_depth = encode_paths(train_paths)
    eval_start, eval_end, eval_lca, eval_length, eval_lca_depth = encode_paths(eval_paths)
    node_depth_items = sorted(node_depths.items())
    train_node_index = [node_index(key) for key, _depth in node_depth_items]
    train_node_depth = [float(depth) for _key, depth in node_depth_items]

    return EncodedAxiomProbeDataset(
        node_count=len(node_to_index),
        train_node_index=torch.tensor(train_node_index, dtype=torch.long),
        train_node_depth=torch.tensor(train_node_depth, dtype=torch.float32),
        train_edge_left=torch.tensor(edge_left, dtype=torch.long),
        train_edge_right=torch.tensor(edge_right, dtype=torch.long),
        train_path_start=torch.tensor(train_start, dtype=torch.long),
        train_path_end=torch.tensor(train_end, dtype=torch.long),
        train_path_lca=torch.tensor(train_lca, dtype=torch.long),
        train_path_length=torch.tensor(train_length, dtype=torch.float32),
        train_path_lca_depth=torch.tensor(train_lca_depth, dtype=torch.float32),
        eval_path_start=torch.tensor(eval_start, dtype=torch.long),
        eval_path_end=torch.tensor(eval_end, dtype=torch.long),
        eval_path_lca=torch.tensor(eval_lca, dtype=torch.long),
        eval_path_length=torch.tensor(eval_length, dtype=torch.float32),
        eval_path_lca_depth=torch.tensor(eval_lca_depth, dtype=torch.float32),
    )


class _AxiomNodeEmbeddingProbe(nn.Module):
    def __init__(self, node_count: int, dim: int, *, geometry: AxiomProbeGeometry, curvature: float) -> None:
        super().__init__()
        self.geometry = geometry
        self.curvature = curvature
        self.embedding = nn.Embedding(node_count, dim)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.01)

    def points(self, indices: Tensor) -> Tensor:
        embedding = self.embedding(indices)
        if self.geometry == "poincare":
            return torch_expmap0(embedding, curvature=self.curvature)
        if self.geometry == "euclidean":
            return embedding
        raise ValueError(f"unknown axiom probe geometry: {self.geometry!r}")

    def distance(self, left: Tensor, right: Tensor) -> Tensor:
        left_points = self.points(left)
        right_points = self.points(right)
        if self.geometry == "poincare":
            return torch_poincare_distance(left_points, right_points, curvature=self.curvature)
        if self.geometry == "euclidean":
            return torch.linalg.vector_norm(left_points - right_points, dim=-1)
        raise ValueError(f"unknown axiom probe geometry: {self.geometry!r}")


def _gromov_product_at_origin(probe: _AxiomNodeEmbeddingProbe, start_points: Tensor, end_points: Tensor) -> Tensor:
    if probe.geometry == "poincare":
        return poincare_gromov_product_at_origin(start_points, end_points, curvature=probe.curvature)
    if probe.geometry == "euclidean":
        origin = torch.zeros_like(start_points)
        return 0.5 * (
            torch.linalg.vector_norm(start_points - origin, dim=-1)
            + torch.linalg.vector_norm(end_points - origin, dim=-1)
            - torch.linalg.vector_norm(start_points - end_points, dim=-1)
        )
    raise ValueError(f"unknown axiom probe geometry: {probe.geometry!r}")


def _point_distance(probe: _AxiomNodeEmbeddingProbe, left: Tensor, right: Tensor) -> Tensor:
    if probe.geometry == "poincare":
        return torch_poincare_distance(left, right, curvature=probe.curvature)
    if probe.geometry == "euclidean":
        return torch.linalg.vector_norm(left - right, dim=-1)
    raise ValueError(f"unknown axiom probe geometry: {probe.geometry!r}")


def _origin_distance(probe: _AxiomNodeEmbeddingProbe, points: Tensor) -> Tensor:
    origin = torch.zeros_like(points)
    return _point_distance(probe, origin, points)


def _path_distances(probe: _AxiomNodeEmbeddingProbe, start: Tensor, end: Tensor, lca: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    start_points = probe.points(start)
    end_points = probe.points(end)
    lca_points = probe.points(lca)
    start_end = _point_distance(probe, start_points, end_points)
    start_lca = _point_distance(probe, start_points, lca_points)
    lca_end = _point_distance(probe, lca_points, end_points)
    gromov = _gromov_product_at_origin(probe, start_points, end_points)
    return start_end, start_lca, lca_end, gromov


def run_axiom_probe(
    dataset: EncodedAxiomProbeDataset,
    *,
    geometry: AxiomProbeGeometry = "poincare",
    dim: int,
    seed: int,
    epochs: int = 80,
    learning_rate: float = 0.05,
    curvature: float = 1.0,
    edge_weight: float = 1.0,
    length_weight: float = 1.0,
    lca_weight: float = 0.5,
    depth_weight: float = 0.5,
    additivity_weight: float = 0.5,
) -> AxiomProbeResult:
    if dataset.node_count == 0:
        raise ValueError("dataset must contain at least one node")
    if int(dataset.eval_path_start.numel()) == 0:
        raise ValueError("dataset must contain held-out evaluation paths")

    torch.manual_seed(seed)
    random.seed(seed)
    probe = _AxiomNodeEmbeddingProbe(dataset.node_count, dim=dim, geometry=geometry, curvature=curvature)
    optimizer = torch.optim.Adam(probe.parameters(), lr=learning_rate)

    for _ in range(epochs):
        losses: list[Tensor] = []
        if int(dataset.train_edge_left.numel()) > 0:
            edge_distance = probe.distance(dataset.train_edge_left, dataset.train_edge_right)
            losses.append(edge_weight * (edge_distance - 1.0).square().mean())
        if int(dataset.train_node_index.numel()) > 0:
            node_points = probe.points(dataset.train_node_index)
            radial_depth = _origin_distance(probe, node_points)
            scale_loss = structural_distance_loss(radial_depth, dataset.train_node_depth)
            root_mask = dataset.train_node_depth == 0
            root_loss = radial_depth[root_mask].square().mean() if bool(root_mask.any()) else radial_depth.new_tensor(0.0)
            losses.append(depth_weight * (scale_loss + root_loss))
        if int(dataset.train_path_start.numel()) > 0:
            path_distance, start_lca, lca_end, gromov = _path_distances(
                probe,
                dataset.train_path_start,
                dataset.train_path_end,
                dataset.train_path_lca,
            )
            losses.append(length_weight * structural_distance_loss(path_distance, dataset.train_path_length))
            losses.append(lca_weight * structural_distance_loss(gromov, dataset.train_path_lca_depth))
            losses.append(additivity_weight * (path_distance - start_lca - lca_end).square().mean())
        loss = torch.stack(losses).sum()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        eval_distance, eval_start_lca, eval_lca_end, eval_gromov = _path_distances(
            probe,
            dataset.eval_path_start,
            dataset.eval_path_end,
            dataset.eval_path_lca,
        )
        eval_lca_points = probe.points(dataset.eval_path_lca)
        eval_lca_radial_depth = _origin_distance(probe, eval_lca_points)
        if int(dataset.train_edge_left.numel()) > 0:
            train_edge_distance = probe.distance(dataset.train_edge_left, dataset.train_edge_right)
            edge_mean = float(train_edge_distance.mean().detach().cpu())
        else:
            edge_mean = 0.0
        return AxiomProbeResult(
            geometry=geometry,
            dim=dim,
            seed=seed,
            node_count=dataset.node_count,
            train_edge_count=int(dataset.train_edge_left.numel()),
            train_path_count=int(dataset.train_path_start.numel()),
            eval_path_count=int(dataset.eval_path_start.numel()),
            eval_length_spearman=spearman_correlation(
                eval_distance.detach().cpu().tolist(),
                dataset.eval_path_length.detach().cpu().tolist(),
            ),
            eval_lca_depth_spearman=spearman_correlation(
                eval_gromov.detach().cpu().tolist(),
                dataset.eval_path_lca_depth.detach().cpu().tolist(),
            ),
            eval_length_stress=float(structural_normalized_stress(eval_distance, dataset.eval_path_length).detach().cpu()),
            eval_lca_depth_stress=float(
                structural_normalized_stress(eval_gromov, dataset.eval_path_lca_depth).detach().cpu()
            ),
            eval_additivity_residual_mean=float(
                torch.abs(eval_distance - eval_start_lca - eval_lca_end).mean().detach().cpu()
            ),
            eval_lca_radial_depth_spearman=spearman_correlation(
                eval_lca_radial_depth.detach().cpu().tolist(),
                dataset.eval_path_lca_depth.detach().cpu().tolist(),
            ),
            train_edge_distance_mean=edge_mean,
        )
