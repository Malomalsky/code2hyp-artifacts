from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from geometry_profile_research.code2hyp_data import (
    Vocabulary,
    encode_records_to_multilabel_batch,
    filter_records_by_known_label_subtokens,
    label_subtoken_coverage,
    load_code2vec_records,
    sample_code2vec_records,
)
from geometry_profile_research.code2hyp_torch import (
    Code2HypBatch,
    torch_expmap0,
    torch_logmap0,
    torch_poincare_distance,
)
from geometry_profile_research.code2hyp_training import (
    compute_multilabel_pos_weight,
    make_minibatches,
    multilabel_metrics_from_logits,
)


SupervisedVariant = Literal["euclidean", "poincare", "poincare_near_zero"]
Geometry = Literal["euclidean", "poincare"]


@dataclass(frozen=True)
class Code2HypSupervisedConfig:
    token_vocab_size: int
    ast_node_vocab_size: int
    target_vocab_size: int
    token_dim: int = 32
    structural_dim: int = 32
    geometry: Geometry = "euclidean"
    curvature: float = 1.0
    path_encoder: Literal["mean", "gru"] = "gru"
    eps: float = 1e-5

    @property
    def representation_dim(self) -> int:
        return 2 * self.token_dim + self.structural_dim


@dataclass(frozen=True)
class Code2HypSupervisedOutput:
    logits: Tensor
    representation: Tensor
    attention: Tensor
    path_vectors: Tensor
    path_points: Tensor | None
    curvature: Tensor


class Code2VecCompatibleCode2Hyp(nn.Module):
    """Minimal matched code2vec-style supervised model.

    Lexical endpoints stay Euclidean. The structural AST-path channel is either
    Euclidean or Poincare, while parameter count and inputs are matched.
    """

    def __init__(self, config: Code2HypSupervisedConfig) -> None:
        super().__init__()
        if config.geometry not in {"euclidean", "poincare"}:
            raise ValueError(f"unknown geometry: {config.geometry}")
        if config.curvature <= 0:
            raise ValueError("curvature must be positive")
        self.config = config
        self.token_embeddings = nn.Embedding(config.token_vocab_size, config.token_dim)
        self.ast_node_embeddings = nn.Embedding(config.ast_node_vocab_size, config.structural_dim)
        if config.path_encoder == "gru":
            self.path_encoder = nn.GRU(config.structural_dim, config.structural_dim, batch_first=True)
        elif config.path_encoder == "mean":
            self.path_encoder = None
        else:
            raise ValueError(f"unknown path_encoder: {config.path_encoder}")

        self.context_transform = nn.Linear(config.representation_dim, config.representation_dim)
        self.lexical_start_query = nn.Parameter(torch.empty(config.token_dim))
        self.lexical_end_query = nn.Parameter(torch.empty(config.token_dim))
        self.path_query = nn.Parameter(torch.empty(config.structural_dim))
        self.channel_log_weights = nn.Parameter(torch.zeros(3))
        self.decoder = nn.Linear(config.representation_dim, config.target_vocab_size)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.token_embeddings.weight, mean=0.0, std=0.05)
        nn.init.normal_(self.ast_node_embeddings.weight, mean=0.0, std=0.05)
        if self.path_encoder is not None:
            for name, parameter in self.path_encoder.named_parameters():
                if "weight" in name:
                    nn.init.xavier_uniform_(parameter)
                elif "bias" in name:
                    nn.init.zeros_(parameter)
        nn.init.xavier_uniform_(self.context_transform.weight)
        nn.init.zeros_(self.context_transform.bias)
        nn.init.normal_(self.lexical_start_query, mean=0.0, std=0.05)
        nn.init.normal_(self.lexical_end_query, mean=0.0, std=0.05)
        nn.init.normal_(self.path_query, mean=0.0, std=0.05)
        nn.init.xavier_uniform_(self.decoder.weight)
        nn.init.zeros_(self.decoder.bias)

    def forward(self, batch: Code2HypBatch) -> Code2HypSupervisedOutput:
        start_vectors = self.token_embeddings(batch.start_tokens)
        end_vectors = self.token_embeddings(batch.end_tokens)
        path_vectors = self._encode_paths(batch.ast_paths, batch.ast_path_mask)
        curvature = start_vectors.new_tensor(self.config.curvature)

        if self.config.geometry == "poincare":
            path_points = torch_expmap0(path_vectors, curvature=curvature, eps=self.config.eps)
            path_decoder_vectors = torch_logmap0(path_points, curvature=curvature)
            path_query_point = torch_expmap0(self.path_query, curvature=curvature, eps=self.config.eps)
            path_distance = torch_poincare_distance(path_points, path_query_point.view(1, 1, -1), curvature=curvature)
        else:
            path_points = None
            path_decoder_vectors = path_vectors
            path_distance = torch.linalg.vector_norm(path_vectors - self.path_query.view(1, 1, -1), dim=-1)

        transformed_contexts = torch.tanh(
            self.context_transform(torch.cat([start_vectors, path_decoder_vectors, end_vectors], dim=-1))
        )
        channel_weights = F.softplus(self.channel_log_weights) + self.config.eps
        start_distance = torch.linalg.vector_norm(start_vectors - self.lexical_start_query.view(1, 1, -1), dim=-1)
        end_distance = torch.linalg.vector_norm(end_vectors - self.lexical_end_query.view(1, 1, -1), dim=-1)
        scores = -(
            channel_weights[0] * start_distance.square()
            + channel_weights[1] * path_distance.square()
            + channel_weights[2] * end_distance.square()
        )
        scores = scores.masked_fill(~batch.context_mask, -1e9)
        attention = torch.softmax(scores, dim=1) * batch.context_mask.to(dtype=scores.dtype)
        attention = attention / torch.clamp(attention.sum(dim=1, keepdim=True), min=1e-12)
        representation = torch.sum(transformed_contexts * attention.unsqueeze(-1), dim=1)
        logits = self.decoder(representation)
        return Code2HypSupervisedOutput(
            logits=logits,
            representation=representation,
            attention=attention,
            path_vectors=path_vectors,
            path_points=path_points,
            curvature=curvature,
        )

    def _encode_paths(self, ast_paths: Tensor, ast_path_mask: Tensor) -> Tensor:
        node_vectors = self.ast_node_embeddings(ast_paths)
        masked = node_vectors * ast_path_mask.unsqueeze(-1).to(dtype=node_vectors.dtype)
        lengths = torch.clamp(ast_path_mask.sum(dim=-1), min=1).to(dtype=node_vectors.dtype)
        if self.path_encoder is None:
            return masked.sum(dim=-2) / lengths.unsqueeze(-1)

        batch_shape = ast_paths.shape[:2]
        flat_vectors = masked.reshape(-1, ast_paths.shape[-1], self.config.structural_dim)
        _, hidden = self.path_encoder(flat_vectors)
        encoded = hidden[-1].reshape(*batch_shape, self.config.structural_dim)
        return encoded * ast_path_mask.any(dim=-1, keepdim=True).to(dtype=encoded.dtype)


def train_supervised_c2s_variants(
    *,
    train_path: Path,
    validation_path: Path,
    output_path: Path,
    variants: Sequence[SupervisedVariant] = ("euclidean", "poincare_near_zero", "poincare"),
    train_limit: int = 1024,
    validation_limit: int = 512,
    max_contexts: int = 64,
    max_path_length: int = 16,
    token_dim: int = 32,
    structural_dim: int = 32,
    epochs: int = 3,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    seed: int = 20260623,
    pos_weight_max: float = 20.0,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    random.seed(seed)
    train_records = sample_code2vec_records(train_path, limit=train_limit, seed=seed)
    validation_records_raw = load_code2vec_records(validation_path, limit=validation_limit)
    train_encoded = encode_records_to_multilabel_batch(
        train_records,
        max_contexts=max_contexts,
        max_path_length=max_path_length,
        token_dim=token_dim,
        structural_dim=structural_dim,
        context_sample_seed=seed,
    )
    validation_coverage = label_subtoken_coverage(validation_records_raw, train_encoded.target_vocab)
    validation_records = filter_records_by_known_label_subtokens(validation_records_raw, train_encoded.target_vocab)
    validation_encoded = encode_records_to_multilabel_batch(
        validation_records,
        max_contexts=max_contexts,
        max_path_length=max_path_length,
        token_dim=token_dim,
        structural_dim=structural_dim,
        token_vocab=train_encoded.token_vocab,
        ast_node_vocab=train_encoded.ast_node_vocab,
        target_vocab=train_encoded.target_vocab,
        context_sample_seed=seed,
    )
    pos_weight = compute_multilabel_pos_weight(train_encoded.labels, max_weight=pos_weight_max)

    runs = []
    for variant in variants:
        variant_seed = seed + _variant_seed_offset(variant)
        torch.manual_seed(variant_seed)
        geometry, curvature = _variant_geometry_and_curvature(variant)
        config = Code2HypSupervisedConfig(
            token_vocab_size=len(train_encoded.token_vocab),
            ast_node_vocab_size=len(train_encoded.ast_node_vocab),
            target_vocab_size=len(train_encoded.target_vocab),
            token_dim=token_dim,
            structural_dim=structural_dim,
            geometry=geometry,
            curvature=curvature,
        )
        model = Code2VecCompatibleCode2Hyp(config)
        history = _fit_model(
            model,
            train_encoded.batch,
            train_encoded.labels,
            train_encoded.target_sizes,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            pos_weight=pos_weight,
            seed=variant_seed,
        )
        metrics = _evaluate_model(
            model,
            validation_encoded.batch,
            validation_encoded.labels,
            validation_encoded.target_sizes,
            batch_size=batch_size,
        )
        runs.append(
            {
                "variant": variant,
                "geometry": geometry,
                "curvature": curvature,
                "model_seed": variant_seed,
                "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
                "history": history,
                "validation_precision": metrics["precision"],
                "validation_recall": metrics["recall"],
                "validation_f1": metrics["f1"],
                "validation_fixed_top3_precision": metrics["fixed_top3_precision"],
                "validation_fixed_top3_recall": metrics["fixed_top3_recall"],
                "validation_fixed_top3_f1": metrics["fixed_top3_f1"],
            }
        )

    payload = {
        "experiment": "code2vec_compatible_code2hyp_supervised",
        "config": {
            "train_path": str(train_path),
            "validation_path": str(validation_path),
            "train_limit": train_limit,
            "validation_limit": validation_limit,
            "max_contexts": max_contexts,
            "max_path_length": max_path_length,
            "token_dim": token_dim,
            "structural_dim": structural_dim,
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "seed": seed,
            "pos_weight_max": pos_weight_max,
        },
        "train_record_count": len(train_records),
        "validation_record_count_raw": len(validation_records_raw),
        "validation_record_count_closed_vocab": len(validation_records),
        "validation_coverage": validation_coverage,
        "token_vocab_size": len(train_encoded.token_vocab),
        "ast_node_vocab_size": len(train_encoded.ast_node_vocab),
        "target_vocab_size": len(train_encoded.target_vocab),
        "runs": runs,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return payload


def _fit_model(
    model: Code2VecCompatibleCode2Hyp,
    batch: Code2HypBatch,
    labels: Tensor,
    target_sizes: Tensor,
    *,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    pos_weight: Tensor,
    seed: int,
) -> list[dict[str, float]]:
    if epochs <= 0:
        raise ValueError("epochs must be positive")
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    history: list[dict[str, float]] = []
    for epoch in range(epochs):
        model.train()
        losses = []
        f1s = []
        for minibatch, minibatch_labels in make_minibatches(
            batch,
            labels,
            batch_size=batch_size,
            shuffle=True,
            seed=seed + epoch,
        ):
            optimizer.zero_grad(set_to_none=True)
            output = model(minibatch)
            loss = F.binary_cross_entropy_with_logits(
                output.logits,
                minibatch_labels,
                pos_weight=pos_weight.to(device=minibatch_labels.device, dtype=minibatch_labels.dtype),
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            losses.append(float(loss.detach()))
            target_size = minibatch_labels.sum(dim=1).to(dtype=torch.long)
            f1s.append(multilabel_metrics_from_logits(output.logits.detach(), minibatch_labels, target_size)["f1"])
        history.append(
            {
                "epoch": float(epoch + 1),
                "loss": sum(losses) / len(losses),
                "f1": sum(f1s) / len(f1s),
            }
        )
    return history


@torch.no_grad()
def _evaluate_model(
    model: Code2VecCompatibleCode2Hyp,
    batch: Code2HypBatch,
    labels: Tensor,
    target_sizes: Tensor,
    *,
    batch_size: int,
) -> dict[str, float]:
    model.eval()
    logits = []
    all_labels = []
    all_target_sizes = []
    offset = 0
    for minibatch, minibatch_labels in make_minibatches(batch, labels, batch_size=batch_size, shuffle=False):
        output = model(minibatch)
        logits.append(output.logits)
        all_labels.append(minibatch_labels)
        end = offset + int(minibatch_labels.shape[0])
        all_target_sizes.append(target_sizes[offset:end])
        offset = end
    logits_tensor = torch.cat(logits, dim=0)
    labels_tensor = torch.cat(all_labels, dim=0)
    target_sizes_tensor = torch.cat(all_target_sizes, dim=0)
    oracle = multilabel_metrics_from_logits(logits_tensor, labels_tensor, target_sizes_tensor, selection="oracle_topk")
    fixed = multilabel_metrics_from_logits(
        logits_tensor,
        labels_tensor,
        target_sizes_tensor,
        selection="fixed_topk",
        fixed_k=3,
    )
    return {
        "precision": oracle["precision"],
        "recall": oracle["recall"],
        "f1": oracle["f1"],
        "fixed_top3_precision": fixed["precision"],
        "fixed_top3_recall": fixed["recall"],
        "fixed_top3_f1": fixed["f1"],
    }


def _variant_geometry_and_curvature(variant: SupervisedVariant) -> tuple[Geometry, float]:
    if variant == "euclidean":
        return "euclidean", 1.0
    if variant == "poincare_near_zero":
        return "poincare", 0.05
    if variant == "poincare":
        return "poincare", 1.0
    raise ValueError(f"unknown supervised variant: {variant}")


def _variant_seed_offset(variant: SupervisedVariant) -> int:
    return {"euclidean": 0, "poincare_near_zero": 10_000, "poincare": 20_000}[variant]
