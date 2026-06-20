from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterator

import torch
from torch import Tensor
from torch.nn import functional as F

from .code2hyp_torch import (
    Code2HypBatch,
    Code2HypTorchModel,
    batch_structural_distance_regularizer,
    batch_structural_neighbor_distribution_regularizer,
    batch_structural_rank_regularizer,
    path_attention_tree_distance_loss,
    path_dual_attention_separation_loss,
)


ADAPTIVE_RANK_MIN_WEIGHT = 0.10
ADAPTIVE_RANK_MAX_WEIGHT = 0.50


@dataclass(frozen=True)
class SupervisedRunSummary:
    history: list[dict[str, float]]
    final_accuracy: float


def slice_batch(batch: Code2HypBatch, indices: Tensor) -> Code2HypBatch:
    context_tree_features = None
    if batch.context_tree_features is not None:
        context_tree_features = batch.context_tree_features[indices]
    return Code2HypBatch(
        start_tokens=batch.start_tokens[indices],
        end_tokens=batch.end_tokens[indices],
        ast_paths=batch.ast_paths[indices],
        ast_path_mask=batch.ast_path_mask[indices],
        context_mask=batch.context_mask[indices],
        context_tree_features=context_tree_features,
    )


def make_minibatches(
    batch: Code2HypBatch,
    labels: Tensor,
    batch_size: int,
    shuffle: bool,
    seed: int = 0,
) -> Iterator[tuple[Code2HypBatch, Tensor]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    examples = labels.shape[0]
    if shuffle:
        generator = torch.Generator(device=labels.device).manual_seed(seed)
        order = torch.randperm(examples, generator=generator, device=labels.device)
    else:
        order = torch.arange(examples, device=labels.device)

    for start in range(0, examples, batch_size):
        indices = order[start : start + batch_size]
        yield slice_batch(batch, indices), labels[indices]


def accuracy_from_logits(logits: Tensor, labels: Tensor) -> float:
    predictions = torch.argmax(logits, dim=-1)
    return float((predictions == labels).float().mean().detach())


def compute_multilabel_pos_weight(labels: Tensor, max_weight: float = 20.0) -> Tensor:
    if max_weight < 1.0:
        raise ValueError("max_weight must be at least 1.0")
    labels = labels.float()
    positives = labels.sum(dim=0)
    negatives = labels.shape[0] - positives
    raw_weights = negatives / torch.clamp(positives, min=1.0)
    weights = torch.where(positives > 0, raw_weights, torch.ones_like(raw_weights))
    return torch.clamp(weights, min=1.0, max=max_weight)


def multilabel_metrics_from_logits(
    logits: Tensor,
    labels: Tensor,
    target_sizes: Tensor,
) -> dict[str, float]:
    predicted = torch.zeros_like(labels)
    for row_index, target_size in enumerate(target_sizes.tolist()):
        k = max(1, min(int(target_size), logits.shape[1]))
        indices = torch.topk(logits[row_index], k=k).indices
        predicted[row_index, indices] = 1.0

    true_positive = float((predicted * labels).sum().detach())
    false_positive = float((predicted * (1.0 - labels)).sum().detach())
    false_negative = float(((1.0 - predicted) * labels).sum().detach())
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def train_step(
    model: Code2HypTorchModel,
    optimizer: torch.optim.Optimizer,
    batch: Code2HypBatch,
    labels: Tensor,
    structural_loss_weight: float = 0.0,
    structural_regularizer: str = "distance",
) -> dict[str, float]:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    output = model(batch)
    task_loss = F.cross_entropy(output.logits, labels)
    if structural_loss_weight:
        structural_loss = _structural_regularizer_loss(output, batch, structural_regularizer)
    else:
        structural_loss = task_loss.new_tensor(0.0)
    loss = task_loss + structural_loss_weight * structural_loss
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
    optimizer.step()
    return {
        "loss": float(loss.detach()),
        "task_loss": float(task_loss.detach()),
        "structural_loss": float(structural_loss.detach()),
        "accuracy": accuracy_from_logits(output.logits.detach(), labels),
        "curvature": float(output.curvature.detach()),
    }


def train_multilabel_step(
    model: Code2HypTorchModel,
    optimizer: torch.optim.Optimizer,
    batch: Code2HypBatch,
    labels: Tensor,
    target_sizes: Tensor,
    structural_loss_weight: float = 0.0,
    structural_regularizer: str = "distance",
    pos_weight: Tensor | None = None,
) -> dict[str, float]:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    output = model(batch)
    if pos_weight is not None:
        pos_weight = pos_weight.to(device=labels.device, dtype=labels.dtype)
    task_loss = F.binary_cross_entropy_with_logits(output.logits, labels, pos_weight=pos_weight)
    if structural_loss_weight:
        structural_loss = _structural_regularizer_loss(output, batch, structural_regularizer)
    else:
        structural_loss = task_loss.new_tensor(0.0)
    loss = task_loss + structural_loss_weight * structural_loss
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
    optimizer.step()
    metrics = multilabel_metrics_from_logits(output.logits.detach(), labels, target_sizes)
    return {
        "loss": float(loss.detach()),
        "task_loss": float(task_loss.detach()),
        "structural_loss": float(structural_loss.detach()),
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
        "curvature": float(output.curvature.detach()),
    }


def scheduled_structural_loss_weight(
    base_weight: float,
    epoch_index: int,
    epochs: int,
    schedule: str = "constant",
) -> float:
    if base_weight < 0.0:
        raise ValueError("base_weight must be non-negative")
    if epoch_index < 0:
        raise ValueError("epoch_index must be non-negative")
    if epochs <= 0:
        raise ValueError("epochs must be positive")
    if schedule == "constant":
        return base_weight
    if schedule == "linear":
        return base_weight * float(epoch_index + 1) / float(epochs)
    if schedule == "delayed_linear":
        if epochs == 1:
            return base_weight
        return base_weight * float(epoch_index) / float(epochs - 1)
    if schedule == "cosine":
        if epochs == 1:
            return base_weight
        progress = float(epoch_index) / float(epochs - 1)
        return base_weight * 0.5 * (1.0 - math.cos(math.pi * progress))
    if schedule == "warmup_decay":
        if epochs <= 2:
            return base_weight
        progress = float(epoch_index) / float(epochs - 1)
        return base_weight * max(0.0, 1.0 - abs(2.0 * progress - 1.0))
    raise ValueError(f"unknown structural_loss_schedule: {schedule}")


@torch.no_grad()
def evaluate_accuracy(
    model: Code2HypTorchModel,
    batch: Code2HypBatch,
    labels: Tensor,
    batch_size: int = 64,
) -> float:
    model.eval()
    correct = 0
    total = 0
    for minibatch, minibatch_labels in make_minibatches(batch, labels, batch_size=batch_size, shuffle=False):
        output = model(minibatch)
        predictions = torch.argmax(output.logits, dim=-1)
        correct += int((predictions == minibatch_labels).sum())
        total += int(minibatch_labels.numel())
    return correct / total if total else 0.0


@torch.no_grad()
def evaluate_multilabel_metrics(
    model: Code2HypTorchModel,
    batch: Code2HypBatch,
    labels: Tensor,
    target_sizes: Tensor,
    batch_size: int = 64,
) -> dict[str, float]:
    model.eval()
    logits = []
    all_labels = []
    all_target_sizes = []
    offset = 0
    for minibatch, minibatch_labels in make_minibatches(batch, labels, batch_size=batch_size, shuffle=False):
        start = offset
        end = start + int(minibatch_labels.shape[0])
        offset = end
        output = model(minibatch)
        logits.append(output.logits)
        all_labels.append(minibatch_labels)
        all_target_sizes.append(target_sizes[start:end])
    return multilabel_metrics_from_logits(
        torch.cat(logits, dim=0),
        torch.cat(all_labels, dim=0),
        torch.cat(all_target_sizes, dim=0),
    )


def fit_supervised(
    model: Code2HypTorchModel,
    batch: Code2HypBatch,
    labels: Tensor,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    structural_loss_weight: float = 0.0,
    structural_loss_schedule: str = "constant",
    structural_regularizer: str = "distance",
    seed: int = 0,
) -> list[dict[str, float]]:
    if epochs <= 0:
        raise ValueError("epochs must be positive")

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    history: list[dict[str, float]] = []
    for epoch in range(epochs):
        epoch_losses = []
        epoch_task_losses = []
        epoch_structural_losses = []
        epoch_accuracies = []
        curvatures = []
        effective_structural_loss_weight = scheduled_structural_loss_weight(
            structural_loss_weight,
            epoch,
            epochs,
            schedule=structural_loss_schedule,
        )
        for minibatch, minibatch_labels in make_minibatches(
            batch,
            labels,
            batch_size=batch_size,
            shuffle=True,
            seed=seed + epoch,
        ):
            metrics = train_step(
                model,
                optimizer,
                minibatch,
                minibatch_labels,
                structural_loss_weight=effective_structural_loss_weight,
                structural_regularizer=structural_regularizer,
            )
            epoch_losses.append(metrics["loss"])
            epoch_task_losses.append(metrics["task_loss"])
            epoch_structural_losses.append(metrics["structural_loss"])
            epoch_accuracies.append(metrics["accuracy"])
            curvatures.append(metrics["curvature"])
        history.append(
            {
                "epoch": float(epoch + 1),
                "loss": sum(epoch_losses) / len(epoch_losses),
                "task_loss": sum(epoch_task_losses) / len(epoch_task_losses),
                "structural_loss": sum(epoch_structural_losses) / len(epoch_structural_losses),
                "accuracy": sum(epoch_accuracies) / len(epoch_accuracies),
                "curvature": sum(curvatures) / len(curvatures),
                "structural_loss_weight": effective_structural_loss_weight,
            }
        )
    return history


def fit_multilabel_supervised(
    model: Code2HypTorchModel,
    batch: Code2HypBatch,
    labels: Tensor,
    target_sizes: Tensor,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    structural_loss_weight: float = 0.0,
    structural_loss_schedule: str = "constant",
    structural_regularizer: str = "distance",
    pos_weight: Tensor | None = None,
    seed: int = 0,
) -> list[dict[str, float]]:
    if epochs <= 0:
        raise ValueError("epochs must be positive")

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    history: list[dict[str, float]] = []
    for epoch in range(epochs):
        epoch_losses = []
        epoch_task_losses = []
        epoch_structural_losses = []
        epoch_precisions = []
        epoch_recalls = []
        epoch_f1s = []
        curvatures = []
        effective_structural_loss_weight = scheduled_structural_loss_weight(
            structural_loss_weight,
            epoch,
            epochs,
            schedule=structural_loss_schedule,
        )
        for minibatch, minibatch_labels in make_minibatches(
            batch,
            labels,
            batch_size=batch_size,
            shuffle=True,
            seed=seed + epoch,
        ):
            # target_sizes are aligned only for full-batch training metrics. For
            # minibatches we use the true label cardinality, which is invariant.
            minibatch_target_sizes = minibatch_labels.sum(dim=1).to(dtype=torch.long)
            metrics = train_multilabel_step(
                model,
                optimizer,
                minibatch,
                minibatch_labels,
                minibatch_target_sizes,
                structural_loss_weight=effective_structural_loss_weight,
                structural_regularizer=structural_regularizer,
                pos_weight=pos_weight,
            )
            epoch_losses.append(metrics["loss"])
            epoch_task_losses.append(metrics["task_loss"])
            epoch_structural_losses.append(metrics["structural_loss"])
            epoch_precisions.append(metrics["precision"])
            epoch_recalls.append(metrics["recall"])
            epoch_f1s.append(metrics["f1"])
            curvatures.append(metrics["curvature"])
        history.append(
            {
                "epoch": float(epoch + 1),
                "loss": sum(epoch_losses) / len(epoch_losses),
                "task_loss": sum(epoch_task_losses) / len(epoch_task_losses),
                "structural_loss": sum(epoch_structural_losses) / len(epoch_structural_losses),
                "precision": sum(epoch_precisions) / len(epoch_precisions),
                "recall": sum(epoch_recalls) / len(epoch_recalls),
                "f1": sum(epoch_f1s) / len(epoch_f1s),
                "curvature": sum(curvatures) / len(curvatures),
                "structural_loss_weight": effective_structural_loss_weight,
            }
        )
    return history


def _structural_regularizer_loss(
    output,
    batch: Code2HypBatch,
    structural_regularizer: str,
) -> Tensor:
    if structural_regularizer == "distance":
        return batch_structural_distance_regularizer(output, batch)
    if structural_regularizer == "rank":
        return batch_structural_rank_regularizer(output, batch)
    if structural_regularizer == "neighbor_distribution":
        return batch_structural_neighbor_distribution_regularizer(output, batch)
    if structural_regularizer == "path_attention_monotone":
        if output.path_node_attention_monotonicity_loss is None:
            raise RuntimeError("path_attention_monotone requires a path-node attention model output")
        return output.path_node_attention_monotonicity_loss
    if structural_regularizer == "path_attention_tree_distance":
        if output.path_node_attention is None:
            raise RuntimeError("path_attention_tree_distance requires a path-node attention model output")
        return path_attention_tree_distance_loss(
            output.path_node_attention,
            batch.ast_paths,
            batch.ast_path_mask,
            batch.context_mask,
        )
    if structural_regularizer == "path_dual_attention_separation":
        if output.path_node_attention_pair is None:
            raise RuntimeError("path_dual_attention_separation requires a dual path-node attention model output")
        return path_dual_attention_separation_loss(output.path_node_attention_pair, batch.ast_path_mask)
    if structural_regularizer == "path_dual_attention_separation_rank":
        if output.path_node_attention_pair is None:
            raise RuntimeError("path_dual_attention_separation_rank requires a dual path-node attention model output")
        return (
            path_dual_attention_separation_loss(output.path_node_attention_pair, batch.ast_path_mask)
            + batch_structural_rank_regularizer(output, batch)
        )
    if structural_regularizer == "path_dual_attention_separation_soft_rank":
        if output.path_node_attention_pair is None:
            raise RuntimeError(
                "path_dual_attention_separation_soft_rank requires a dual path-node attention model output"
            )
        return (
            path_dual_attention_separation_loss(output.path_node_attention_pair, batch.ast_path_mask)
            + 0.25 * batch_structural_rank_regularizer(output, batch)
        )
    if structural_regularizer == "path_dual_attention_separation_adaptive_rank":
        if output.path_node_attention_pair is None:
            raise RuntimeError(
                "path_dual_attention_separation_adaptive_rank requires a dual path-node attention model output"
            )
        separation_loss = path_dual_attention_separation_loss(output.path_node_attention_pair, batch.ast_path_mask)
        rank_loss = batch_structural_rank_regularizer(output, batch)
        rank_weight = torch.clamp(
            separation_loss.detach() / torch.clamp(
                separation_loss.detach() + rank_loss.detach(),
                min=1e-12,
            ),
            min=ADAPTIVE_RANK_MIN_WEIGHT,
            max=ADAPTIVE_RANK_MAX_WEIGHT,
        )
        return separation_loss + rank_weight * rank_loss
    raise ValueError(f"unknown structural_regularizer: {structural_regularizer}")
