from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np


Array = np.ndarray


@dataclass(frozen=True)
class PathContext:
    """One code2vec-style AST path context.

    The context follows the original path-based interface:
    `(start token, AST path, end token)`.
    """

    start_token: int
    ast_path: tuple[int, ...]
    end_token: int


@dataclass(frozen=True)
class Code2HypConfig:
    token_vocab_size: int
    ast_node_vocab_size: int
    label_vocab_size: int
    token_dim: int = 32
    structural_dim: int = 32
    curvature: float = 1.0
    eps: float = 1e-5

    @property
    def representation_dim(self) -> int:
        return 2 * self.token_dim + self.structural_dim


@dataclass(frozen=True)
class ModelOutput:
    logits: Array
    representation: Array
    attention: Array
    structural_points: Array | None = None


def _safe_norm(x: Array, axis: int = -1, keepdims: bool = False) -> Array:
    return np.linalg.norm(x, axis=axis, keepdims=keepdims)


def project_to_ball(x: Array, curvature: float, eps: float = 1e-5) -> Array:
    """Project vectors into the open Poincare ball with sectional curvature `-c`."""
    if curvature <= 0:
        raise ValueError("curvature must be positive")
    # The Poincare ball is open; keep a tiny margin beyond the user-facing eps.
    radius = (1.0 - eps) / math.sqrt(curvature) * (1.0 - 1e-12)
    norm = _safe_norm(x, axis=-1, keepdims=True)
    scale = np.minimum(1.0, radius / np.maximum(norm, 1e-15))
    return x * scale


def expmap0(v: Array, curvature: float, eps: float = 1e-5) -> Array:
    """Exponential map at the origin of the Poincare ball."""
    if curvature <= 0:
        raise ValueError("curvature must be positive")
    sqrt_c = math.sqrt(curvature)
    norm = _safe_norm(v, axis=-1, keepdims=True)
    scaled = np.tanh(sqrt_c * norm) * v / np.maximum(sqrt_c * norm, 1e-15)
    return project_to_ball(scaled, curvature=curvature, eps=eps)


def logmap0(x: Array, curvature: float) -> Array:
    """Logarithmic map at the origin of the Poincare ball."""
    if curvature <= 0:
        raise ValueError("curvature must be positive")
    sqrt_c = math.sqrt(curvature)
    x = project_to_ball(x, curvature=curvature)
    norm = _safe_norm(x, axis=-1, keepdims=True)
    scaled_norm = np.clip(sqrt_c * norm, 0.0, 1.0 - 1e-12)
    return np.arctanh(scaled_norm) * x / np.maximum(sqrt_c * norm, 1e-15)


def poincare_distance(left: Array, right: Array, curvature: float) -> Array:
    """Geodesic distance in the Poincare ball."""
    if curvature <= 0:
        raise ValueError("curvature must be positive")
    sqrt_c = math.sqrt(curvature)
    left = project_to_ball(left, curvature=curvature)
    right = project_to_ball(right, curvature=curvature)
    left_norm2 = np.sum(left * left, axis=-1)
    right_norm2 = np.sum(right * right, axis=-1)
    diff_norm2 = np.sum((left - right) ** 2, axis=-1)
    denominator = np.maximum((1.0 - curvature * left_norm2) * (1.0 - curvature * right_norm2), 1e-15)
    argument = 1.0 + 2.0 * curvature * diff_norm2 / denominator
    return np.arccosh(np.maximum(argument, 1.0)) / sqrt_c


def _softmax(values: Array) -> Array:
    shifted = values - np.max(values)
    exp_values = np.exp(shifted)
    return exp_values / np.sum(exp_values)


class ProductCode2HypModel:
    """Minimal executable core for B1 and B3 Code2Hyp comparisons.

    This is intentionally a NumPy forward-pass model. The trainable PyTorch
    model should preserve these interfaces and invariants.
    """

    def __init__(
        self,
        config: Code2HypConfig,
        token_embeddings: Array,
        ast_node_embeddings: Array,
        attention_query: Array,
        decoder_weight: Array,
        decoder_bias: Array,
    ) -> None:
        self.config = config
        self.token_embeddings = token_embeddings
        self.ast_node_embeddings = ast_node_embeddings
        self.attention_query = attention_query
        self.decoder_weight = decoder_weight
        self.decoder_bias = decoder_bias

    @classmethod
    def random(cls, config: Code2HypConfig, seed: int = 0) -> "ProductCode2HypModel":
        rng = np.random.default_rng(seed)
        scale = 0.05
        token_embeddings = rng.normal(0.0, scale, size=(config.token_vocab_size, config.token_dim))
        ast_node_embeddings = rng.normal(0.0, scale, size=(config.ast_node_vocab_size, config.structural_dim))
        attention_query = rng.normal(0.0, scale, size=(config.representation_dim,))
        decoder_weight = rng.normal(0.0, scale, size=(config.representation_dim, config.label_vocab_size))
        decoder_bias = np.zeros(config.label_vocab_size, dtype=float)
        return cls(
            config=config,
            token_embeddings=token_embeddings,
            ast_node_embeddings=ast_node_embeddings,
            attention_query=attention_query,
            decoder_weight=decoder_weight,
            decoder_bias=decoder_bias,
        )

    def forward_euclidean(self, batch: Sequence[Sequence[PathContext]]) -> ModelOutput:
        representations: list[Array] = []
        attentions: list[Array] = []
        max_contexts = max((len(contexts) for contexts in batch), default=0)

        for contexts in batch:
            context_representations = np.vstack(
                [self._euclidean_context_representation(context) for context in contexts]
            )
            attention = self._attention(context_representations)
            representations.append(np.sum(context_representations * attention[:, None], axis=0))
            attentions.append(self._pad_attention(attention, max_contexts))

        representation_matrix = np.vstack(representations)
        logits = representation_matrix @ self.decoder_weight + self.decoder_bias
        return ModelOutput(
            logits=logits,
            representation=representation_matrix,
            attention=np.vstack(attentions),
            structural_points=None,
        )

    def forward_product(self, batch: Sequence[Sequence[PathContext]]) -> ModelOutput:
        representations: list[Array] = []
        attentions: list[Array] = []
        structural_points: list[Array] = []
        max_contexts = max((len(contexts) for contexts in batch), default=0)

        for contexts in batch:
            start_vectors = np.vstack([self._token_embedding(context.start_token) for context in contexts])
            end_vectors = np.vstack([self._token_embedding(context.end_token) for context in contexts])
            path_tangents = np.vstack([self._path_tangent(context.ast_path) for context in contexts])
            path_points = expmap0(path_tangents, curvature=self.config.curvature, eps=self.config.eps)
            path_logs = logmap0(path_points, curvature=self.config.curvature)
            context_representations = np.hstack([start_vectors, path_logs, end_vectors])
            attention = self._attention(context_representations)

            start_centroid = np.sum(start_vectors * attention[:, None], axis=0)
            end_centroid = np.sum(end_vectors * attention[:, None], axis=0)
            structural_tangent_centroid = np.sum(path_logs * attention[:, None], axis=0)
            structural_point = expmap0(
                structural_tangent_centroid[None, :],
                curvature=self.config.curvature,
                eps=self.config.eps,
            )[0]
            structural_log = logmap0(structural_point[None, :], curvature=self.config.curvature)[0]

            representations.append(np.concatenate([start_centroid, structural_log, end_centroid]))
            structural_points.append(structural_point)
            attentions.append(self._pad_attention(attention, max_contexts))

        representation_matrix = np.vstack(representations)
        logits = representation_matrix @ self.decoder_weight + self.decoder_bias
        return ModelOutput(
            logits=logits,
            representation=representation_matrix,
            attention=np.vstack(attentions),
            structural_points=np.vstack(structural_points),
        )

    def _token_embedding(self, token_id: int) -> Array:
        if token_id < 0 or token_id >= self.config.token_vocab_size:
            raise IndexError(f"token id out of vocabulary: {token_id}")
        return self.token_embeddings[token_id]

    def _path_tangent(self, ast_path: Sequence[int]) -> Array:
        if not ast_path:
            return np.zeros(self.config.structural_dim, dtype=float)
        embeddings = []
        for node_id in ast_path:
            if node_id < 0 or node_id >= self.config.ast_node_vocab_size:
                raise IndexError(f"AST node id out of vocabulary: {node_id}")
            embeddings.append(self.ast_node_embeddings[node_id])
        return np.mean(np.vstack(embeddings), axis=0)

    def _euclidean_context_representation(self, context: PathContext) -> Array:
        return np.concatenate(
            [
                self._token_embedding(context.start_token),
                self._path_tangent(context.ast_path),
                self._token_embedding(context.end_token),
            ]
        )

    def _attention(self, context_representations: Array) -> Array:
        scores = context_representations @ self.attention_query
        return _softmax(scores)

    @staticmethod
    def _pad_attention(attention: Array, max_contexts: int) -> Array:
        padded = np.zeros(max_contexts, dtype=float)
        padded[: len(attention)] = attention
        return padded
