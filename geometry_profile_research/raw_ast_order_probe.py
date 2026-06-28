from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from geometry_profile_research.code2hyp_torch import (
    torch_expmap0,
    torch_poincare_entailment_cone_energy,
)


OrderProbeModel = Literal["poincare_cone", "euclidean_order"]


@dataclass(frozen=True)
class OrderProbeRecord:
    scope_index: int
    ancestor: int
    descendant: int
    label: int
    is_direct_edge: bool
    tree_distance: int

    @property
    def ancestor_key(self) -> tuple[int, int]:
        return (self.scope_index, self.ancestor)

    @property
    def descendant_key(self) -> tuple[int, int]:
        return (self.scope_index, self.descendant)


@dataclass(frozen=True)
class OrderProbeSplit:
    train_positive: tuple[OrderProbeRecord, ...]
    train_negative: tuple[OrderProbeRecord, ...]
    eval_positive: tuple[OrderProbeRecord, ...]
    eval_negative: tuple[OrderProbeRecord, ...]


@dataclass(frozen=True)
class EncodedOrderProbeSplit:
    node_count: int
    train_left: Tensor
    train_right: Tensor
    train_label: Tensor
    eval_left: Tensor
    eval_right: Tensor
    eval_label: Tensor


@dataclass(frozen=True)
class OrderProbeResult:
    model: str
    dim: int
    seed: int
    train_size: int
    eval_size: int
    eval_auc: float
    eval_accuracy: float
    positive_energy_mean: float
    negative_energy_mean: float

    def as_dict(self) -> dict[str, int | float | str]:
        return {
            "model": self.model,
            "dim": self.dim,
            "seed": self.seed,
            "train_size": self.train_size,
            "eval_size": self.eval_size,
            "eval_auc": self.eval_auc,
            "eval_accuracy": self.eval_accuracy,
            "positive_energy_mean": self.positive_energy_mean,
            "negative_energy_mean": self.negative_energy_mean,
        }


def load_order_probe_records(path: Path) -> tuple[OrderProbeRecord, ...]:
    records: list[OrderProbeRecord] = []
    for scope_index, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        if not line.strip():
            continue
        payload = json.loads(line)
        if payload.get("status") != "ok":
            continue
        for record in payload.get("order_records", ()):
            records.append(
                OrderProbeRecord(
                    scope_index=scope_index,
                    ancestor=int(record["ancestor"]),
                    descendant=int(record["descendant"]),
                    label=int(record["label"]),
                    is_direct_edge=bool(record["is_direct_edge"]),
                    tree_distance=int(record["tree_distance"]),
                )
            )
    return tuple(records)


def split_order_probe_records(records: Sequence[OrderProbeRecord]) -> OrderProbeSplit:
    train_positive: list[OrderProbeRecord] = []
    train_negative: list[OrderProbeRecord] = []
    eval_positive: list[OrderProbeRecord] = []
    eval_negative: list[OrderProbeRecord] = []

    negative_index = 0
    for record in records:
        if record.label == 1 and record.is_direct_edge:
            train_positive.append(record)
        elif record.label == 1:
            eval_positive.append(record)
        elif record.label == 0:
            if negative_index % 2 == 0:
                train_negative.append(record)
            else:
                eval_negative.append(record)
            negative_index += 1
        else:
            raise ValueError(f"unknown order label: {record.label!r}")

    return OrderProbeSplit(
        train_positive=tuple(train_positive),
        train_negative=tuple(train_negative),
        eval_positive=tuple(eval_positive),
        eval_negative=tuple(eval_negative),
    )


def binary_auc(scores: Sequence[float], labels: Sequence[int]) -> float:
    positives = [score for score, label in zip(scores, labels) if label == 1]
    negatives = [score for score, label in zip(scores, labels) if label == 0]
    if not positives or not negatives:
        raise ValueError("AUC requires at least one positive and one negative label")
    wins = 0.0
    for positive in positives:
        for negative in negatives:
            if positive > negative:
                wins += 1.0
            elif positive == negative:
                wins += 0.5
    return wins / (len(positives) * len(negatives))


def encode_order_probe_split(split: OrderProbeSplit) -> EncodedOrderProbeSplit:
    node_to_index: dict[tuple[int, int], int] = {}

    def node_index(key: tuple[int, int]) -> int:
        if key not in node_to_index:
            node_to_index[key] = len(node_to_index)
        return node_to_index[key]

    def encode(records: Sequence[OrderProbeRecord]) -> tuple[list[int], list[int], list[int]]:
        left: list[int] = []
        right: list[int] = []
        labels: list[int] = []
        for record in records:
            left.append(node_index(record.ancestor_key))
            right.append(node_index(record.descendant_key))
            labels.append(record.label)
        return left, right, labels

    train_records = tuple(split.train_positive + split.train_negative)
    eval_records = tuple(split.eval_positive + split.eval_negative)
    train_left, train_right, train_label = encode(train_records)
    eval_left, eval_right, eval_label = encode(eval_records)
    return EncodedOrderProbeSplit(
        node_count=len(node_to_index),
        train_left=torch.tensor(train_left, dtype=torch.long),
        train_right=torch.tensor(train_right, dtype=torch.long),
        train_label=torch.tensor(train_label, dtype=torch.float32),
        eval_left=torch.tensor(eval_left, dtype=torch.long),
        eval_right=torch.tensor(eval_right, dtype=torch.long),
        eval_label=torch.tensor(eval_label, dtype=torch.float32),
    )


class _OrderEmbeddingProbe(nn.Module):
    def __init__(self, node_count: int, dim: int, model: OrderProbeModel, curvature: float, cone_k: float) -> None:
        super().__init__()
        self.model = model
        self.curvature = curvature
        self.cone_k = cone_k
        self.embedding = nn.Embedding(node_count, dim)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.01)

    def energy(self, left: Tensor, right: Tensor) -> Tensor:
        left_embedding = self.embedding(left)
        right_embedding = self.embedding(right)
        if self.model == "poincare_cone":
            left_point = torch_expmap0(left_embedding, curvature=self.curvature)
            right_point = torch_expmap0(right_embedding, curvature=self.curvature)
            return torch_poincare_entailment_cone_energy(
                left_point,
                right_point,
                curvature=self.curvature,
                cone_k=self.cone_k,
            )
        if self.model == "euclidean_order":
            return torch.linalg.vector_norm(F.relu(left_embedding - right_embedding), dim=-1)
        raise ValueError(f"unknown order probe model: {self.model!r}")


def _probe_loss(energy: Tensor, labels: Tensor, margin: float) -> Tensor:
    positive = labels > 0.5
    negative = ~positive
    losses: list[Tensor] = []
    if bool(positive.any()):
        losses.append(energy[positive].mean())
    if bool(negative.any()):
        losses.append(F.relu(margin - energy[negative]).mean())
    if not losses:
        return energy.new_tensor(0.0)
    return torch.stack(losses).sum()


def run_order_probe(
    split: EncodedOrderProbeSplit,
    *,
    model: OrderProbeModel,
    dim: int,
    seed: int,
    epochs: int = 80,
    batch_size: int = 1024,
    learning_rate: float = 0.05,
    margin: float = 0.25,
    curvature: float = 1.0,
    cone_k: float = 0.1,
) -> OrderProbeResult:
    torch.manual_seed(seed)
    random.seed(seed)
    probe = _OrderEmbeddingProbe(split.node_count, dim=dim, model=model, curvature=curvature, cone_k=cone_k)
    optimizer = torch.optim.Adam(probe.parameters(), lr=learning_rate)
    indices = list(range(int(split.train_label.numel())))

    for _ in range(epochs):
        random.shuffle(indices)
        for start in range(0, len(indices), batch_size):
            batch_index = torch.tensor(indices[start : start + batch_size], dtype=torch.long)
            energy = probe.energy(split.train_left[batch_index], split.train_right[batch_index])
            loss = _probe_loss(energy, split.train_label[batch_index], margin=margin)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    with torch.no_grad():
        eval_energy = probe.energy(split.eval_left, split.eval_right)
        scores = (-eval_energy).detach().cpu().tolist()
        labels = [int(value) for value in split.eval_label.detach().cpu().tolist()]
        auc = binary_auc(scores, labels)
        predictions = [1 if score >= -margin else 0 for score in scores]
        accuracy = sum(int(pred == label) for pred, label in zip(predictions, labels)) / len(labels)
        positive_energy = eval_energy[split.eval_label > 0.5]
        negative_energy = eval_energy[split.eval_label <= 0.5]

    return OrderProbeResult(
        model=model,
        dim=dim,
        seed=seed,
        train_size=int(split.train_label.numel()),
        eval_size=int(split.eval_label.numel()),
        eval_auc=float(auc),
        eval_accuracy=float(accuracy),
        positive_energy_mean=float(positive_energy.mean().item()),
        negative_energy_mean=float(negative_energy.mean().item()),
    )
