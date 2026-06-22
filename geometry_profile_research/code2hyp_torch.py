from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor, nn
from torch.nn import functional as F


Variant = Literal[
    "euclidean",
    "euclidean_metric",
    "bounded_euclidean_metric",
    "euclidean_tree",
    "product",
    "hyperbolic",
    "hyperbolic_attention",
    "hyperbolic_frechet",
    "hyperbolic_path_message_passing",
    "hyperbolic_path_attention_message_passing",
    "hyperbolic_path_depth_attention_message_passing",
    "hyperbolic_path_dual_attention_message_passing",
    "lorentz_path_dual_attention_message_passing",
    "lorentz",
    "lorentz_product",
    "factorized_product",
    "factorized_product_learned_metric",
    "factorized_product_three_metric",
    "factorized_product_channel_mixer",
    "code2hyp_product_frechet",
    "code2hyp_code2vec_attention_frechet",
    "code2vec_context_transform",
    "code2vec_context_transform_l1",
    "code2hyp_context_transform_frechet",
    "code2hyp_product_context_transform_frechet",
    "code2hyp_context_transform_product_bias_frechet",
    "code2hyp_branch_product_context_transform_frechet",
    "code2hyp_context_transform_branch_product_bias_frechet",
    "code2hyp_context_transform_latent_lca_branch_product_bias_frechet",
    "code2hyp_context_transform_latent_lca_prior_branch_product_bias_frechet",
    "code2hyp_context_transform_branch_sequence_product_bias_frechet",
]
PathEncoder = Literal["mean", "gru"]
RepresentationTransform = Literal["identity", "tanh"]
StructuralGeometry = Literal["poincare", "lorentz", "poincare_product"]
StructuralEmbeddingMetric = Literal["l2", "l1"]
StructuralTargetDistance = Literal["prefix_tree", "edit", "jaccard_bigrams"]
ProductDistanceMetric = Literal["riemannian_l2", "l1_sum"]


@dataclass(frozen=True)
class Code2HypTorchConfig:
    token_vocab_size: int
    ast_node_vocab_size: int
    label_vocab_size: int
    token_dim: int = 32
    structural_dim: int = 32
    curvature: float = 1.0
    trainable_curvature: bool = False
    path_encoder: PathEncoder = "mean"
    representation_transform: RepresentationTransform = "identity"
    frechet_steps: int = 4
    frechet_step_size: float = 0.5
    factorized_mixer_rank: int = 4
    path_message_passing_steps: int = 1
    eps: float = 1e-5

    @property
    def representation_dim(self) -> int:
        return 2 * self.token_dim + self.structural_dim


@dataclass(frozen=True)
class Code2HypBatch:
    start_tokens: Tensor
    end_tokens: Tensor
    ast_paths: Tensor
    ast_path_mask: Tensor
    context_mask: Tensor
    context_tree_features: Tensor | None = None


@dataclass(frozen=True)
class Code2HypTorchOutput:
    logits: Tensor
    representation: Tensor
    attention: Tensor
    curvature: Tensor
    structural_points: Tensor | None = None
    context_structural_embeddings: Tensor | None = None
    context_structural_points: Tensor | None = None
    structural_product_points: tuple[Tensor, ...] | None = None
    context_structural_product_points: tuple[Tensor, ...] | None = None
    structural_product_distance_metric: ProductDistanceMetric = "riemannian_l2"
    context_tree_features: Tensor | None = None
    structural_geometry: StructuralGeometry | None = None
    structural_embedding_metric: StructuralEmbeddingMetric = "l2"
    path_node_attention: Tensor | None = None
    path_node_attention_pair: Tensor | None = None
    path_node_attention_monotonicity_loss: Tensor | None = None


def _safe_norm(x: Tensor, dim: int = -1, keepdim: bool = False) -> Tensor:
    return torch.linalg.vector_norm(x, dim=dim, keepdim=keepdim)


def torch_project_to_ball(x: Tensor, curvature: Tensor | float, eps: float = 1e-5) -> Tensor:
    curvature_tensor = torch.as_tensor(curvature, dtype=x.dtype, device=x.device)
    if bool((curvature_tensor <= 0).any()):
        raise ValueError("curvature must be positive")
    radius = ((1.0 - eps) / torch.sqrt(curvature_tensor)) * (1.0 - 1e-12)
    norm = _safe_norm(x, dim=-1, keepdim=True)
    scale = torch.minimum(torch.ones_like(norm), radius / torch.clamp(norm, min=1e-15))
    return x * scale


def torch_expmap0(v: Tensor, curvature: Tensor | float, eps: float = 1e-5) -> Tensor:
    curvature_tensor = torch.as_tensor(curvature, dtype=v.dtype, device=v.device)
    if bool((curvature_tensor <= 0).any()):
        raise ValueError("curvature must be positive")
    sqrt_c = torch.sqrt(curvature_tensor)
    norm = _safe_norm(v, dim=-1, keepdim=True)
    denominator = torch.clamp(sqrt_c * norm, min=1e-15)
    mapped = torch.tanh(sqrt_c * norm) * v / denominator
    return torch_project_to_ball(mapped, curvature_tensor, eps=eps)


def torch_logmap0(x: Tensor, curvature: Tensor | float) -> Tensor:
    curvature_tensor = torch.as_tensor(curvature, dtype=x.dtype, device=x.device)
    if bool((curvature_tensor <= 0).any()):
        raise ValueError("curvature must be positive")
    x = torch_project_to_ball(x, curvature_tensor)
    sqrt_c = torch.sqrt(curvature_tensor)
    norm = _safe_norm(x, dim=-1, keepdim=True)
    scaled_norm = torch.clamp(sqrt_c * norm, min=0.0, max=1.0 - 1e-12)
    return torch.atanh(scaled_norm) * x / torch.clamp(sqrt_c * norm, min=1e-15)


def _torch_mobius_add(left: Tensor, right: Tensor, curvature: Tensor | float, eps: float = 1e-5) -> Tensor:
    curvature_tensor = torch.as_tensor(curvature, dtype=left.dtype, device=left.device)
    if bool((curvature_tensor <= 0).any()):
        raise ValueError("curvature must be positive")
    left = torch_project_to_ball(left, curvature_tensor, eps=eps)
    right = torch_project_to_ball(right, curvature_tensor, eps=eps)
    left_norm2 = torch.sum(left * left, dim=-1, keepdim=True)
    right_norm2 = torch.sum(right * right, dim=-1, keepdim=True)
    inner = torch.sum(left * right, dim=-1, keepdim=True)
    numerator = (
        (1.0 + 2.0 * curvature_tensor * inner + curvature_tensor * right_norm2) * left
        + (1.0 - curvature_tensor * left_norm2) * right
    )
    denominator = torch.clamp(
        1.0 + 2.0 * curvature_tensor * inner + curvature_tensor * curvature_tensor * left_norm2 * right_norm2,
        min=1e-15,
    )
    return torch_project_to_ball(numerator / denominator, curvature_tensor, eps=eps)


def _torch_lambda_x(x: Tensor, curvature: Tensor | float) -> Tensor:
    curvature_tensor = torch.as_tensor(curvature, dtype=x.dtype, device=x.device)
    x_norm2 = torch.sum(x * x, dim=-1, keepdim=True)
    return 2.0 / torch.clamp(1.0 - curvature_tensor * x_norm2, min=1e-15)


def torch_expmap(base: Tensor, tangent: Tensor, curvature: Tensor | float, eps: float = 1e-5) -> Tensor:
    """Exponential map in the Poincare ball at an arbitrary base point."""
    curvature_tensor = torch.as_tensor(curvature, dtype=base.dtype, device=base.device)
    if bool((curvature_tensor <= 0).any()):
        raise ValueError("curvature must be positive")
    base = torch_project_to_ball(base, curvature_tensor, eps=eps)
    sqrt_c = torch.sqrt(curvature_tensor)
    tangent_norm = _safe_norm(tangent, dim=-1, keepdim=True)
    lambda_base = _torch_lambda_x(base, curvature_tensor)
    second = torch.tanh(sqrt_c * lambda_base * tangent_norm / 2.0) * tangent / torch.clamp(
        sqrt_c * tangent_norm,
        min=1e-15,
    )
    return _torch_mobius_add(base, second, curvature_tensor, eps=eps)


def torch_logmap(base: Tensor, point: Tensor, curvature: Tensor | float, eps: float = 1e-5) -> Tensor:
    """Logarithmic map in the Poincare ball at an arbitrary base point."""
    curvature_tensor = torch.as_tensor(curvature, dtype=base.dtype, device=base.device)
    if bool((curvature_tensor <= 0).any()):
        raise ValueError("curvature must be positive")
    base = torch_project_to_ball(base, curvature_tensor, eps=eps)
    point = torch_project_to_ball(point, curvature_tensor, eps=eps)
    sqrt_c = torch.sqrt(curvature_tensor)
    delta = _torch_mobius_add(-base, point, curvature_tensor, eps=eps)
    delta_norm = _safe_norm(delta, dim=-1, keepdim=True)
    lambda_base = _torch_lambda_x(base, curvature_tensor)
    scaled_norm = torch.clamp(sqrt_c * delta_norm, min=0.0, max=1.0 - 1e-12)
    factor = 2.0 * torch.atanh(scaled_norm) / torch.clamp(sqrt_c * lambda_base * delta_norm, min=1e-15)
    return factor * delta


def torch_poincare_distance(left: Tensor, right: Tensor, curvature: Tensor | float) -> Tensor:
    """Geodesic distance in the Poincare ball with curvature ``-curvature``.

    The equivalent ``acosh(1 + z)`` expression is poorly conditioned near
    ``z = 0`` and requires a positive clamp that makes ``d(x, x)`` non-zero.
    The ``asinh`` form keeps the diagonal exactly zero while remaining stable
    for small separations.
    """
    curvature_tensor = torch.as_tensor(curvature, dtype=left.dtype, device=left.device)
    if bool((curvature_tensor <= 0).any()):
        raise ValueError("curvature must be positive")
    sqrt_c = torch.sqrt(curvature_tensor)
    left = torch_project_to_ball(left, curvature_tensor)
    right = torch_project_to_ball(right, curvature_tensor)
    left_norm2 = torch.sum(left * left, dim=-1)
    right_norm2 = torch.sum(right * right, dim=-1)
    diff_norm = torch.linalg.vector_norm(left - right, dim=-1)
    denominator = torch.sqrt(
        torch.clamp(
            (1.0 - curvature_tensor * left_norm2) * (1.0 - curvature_tensor * right_norm2),
            min=1e-30,
        )
    )
    argument = sqrt_c * diff_norm / denominator
    return 2.0 * torch.asinh(argument) / sqrt_c


def torch_lorentz_expmap0(v: Tensor, curvature: Tensor | float) -> Tensor:
    """Exponential map from the origin tangent space to the Lorentz hyperboloid.

    The hyperboloid is represented in R^{d+1} with Minkowski norm
    -x_0^2 + ||x_{1:d}||^2 = -1 / c.
    """
    curvature_tensor = torch.as_tensor(curvature, dtype=v.dtype, device=v.device)
    if bool((curvature_tensor <= 0).any()):
        raise ValueError("curvature must be positive")
    sqrt_c = torch.sqrt(curvature_tensor)
    tangent_norm = _safe_norm(v, dim=-1, keepdim=True)
    scaled_norm = torch.clamp(sqrt_c * tangent_norm, max=15.0)
    time = torch.cosh(scaled_norm) / sqrt_c
    space = torch.sinh(scaled_norm) * v / torch.clamp(sqrt_c * tangent_norm, min=1e-15)
    return torch.cat([time, space], dim=-1)


def torch_lorentz_logmap0(point: Tensor, curvature: Tensor | float) -> Tensor:
    """Logarithmic map from the Lorentz hyperboloid to the origin tangent space."""
    curvature_tensor = torch.as_tensor(curvature, dtype=point.dtype, device=point.device)
    if bool((curvature_tensor <= 0).any()):
        raise ValueError("curvature must be positive")
    sqrt_c = torch.sqrt(curvature_tensor)
    time = point[..., :1]
    space = point[..., 1:]
    spatial_norm = _safe_norm(space, dim=-1, keepdim=True)
    argument = torch.clamp(sqrt_c * time, min=1.0 + 1e-12)
    distance = torch.acosh(argument) / sqrt_c
    return distance * space / torch.clamp(spatial_norm, min=1e-15)


def torch_lorentz_minkowski_inner(left: Tensor, right: Tensor) -> Tensor:
    """Minkowski inner product with signature (-, +, ..., +)."""
    return -left[..., 0] * right[..., 0] + torch.sum(left[..., 1:] * right[..., 1:], dim=-1)


def torch_lorentz_distance(left: Tensor, right: Tensor, curvature: Tensor | float) -> Tensor:
    """Geodesic distance on the Lorentz hyperboloid.

    The direct ``acosh(-c <x,y>_L)`` form is unstable near the diagonal if the
    argument is clamped above 1. The equivalent ``asinh`` form keeps
    ``d(x, x)`` exactly zero, which is important for small-curvature controls.
    """
    curvature_tensor = torch.as_tensor(curvature, dtype=left.dtype, device=left.device)
    if bool((curvature_tensor <= 0).any()):
        raise ValueError("curvature must be positive")
    sqrt_c = torch.sqrt(curvature_tensor)
    difference = left - right
    spacelike_separation = torch.clamp(torch_lorentz_minkowski_inner(difference, difference), min=0.0)
    argument = torch.sqrt(torch.clamp(curvature_tensor * spacelike_separation / 4.0, min=0.0))
    return 2.0 * torch.asinh(argument) / sqrt_c


def torch_lorentz_weighted_centroid(points: Tensor, weights: Tensor, curvature: Tensor | float) -> Tensor:
    """Weighted Lorentz centroid via ambient averaging and hyperboloid projection.

    This is a stable centroid-like aggregation control for the hyperboloid
    model. It is deliberately parameter-free and keeps the result exactly on
    the Lorentz manifold.
    """
    curvature_tensor = torch.as_tensor(curvature, dtype=points.dtype, device=points.device)
    if bool((curvature_tensor <= 0).any()):
        raise ValueError("curvature must be positive")
    ambient_mean = torch.sum(points * weights.unsqueeze(-1), dim=1)
    spatial = ambient_mean[..., 1:]
    time = torch.sqrt(1.0 / curvature_tensor + torch.sum(spatial * spatial, dim=-1, keepdim=True))
    return torch.cat([time, spatial], dim=-1)


def torch_poincare_weighted_midpoint(
    points: Tensor,
    weights: Tensor,
    curvature: Tensor | float,
    eps: float = 1e-5,
) -> Tensor:
    """Differentiable weighted Einstein midpoint in the Poincare ball.

    The implementation maps Poincare points to Klein coordinates, computes the
    weighted Einstein midpoint, and maps it back to Poincare coordinates. This
    makes product aggregation genuinely hyperbolic instead of an exp/log
    identity through the origin tangent space.
    """
    curvature_tensor = torch.as_tensor(curvature, dtype=points.dtype, device=points.device)
    points = torch_project_to_ball(points, curvature_tensor, eps=eps)
    point_norm2 = torch.sum(points * points, dim=-1, keepdim=True)
    klein = 2.0 * points / torch.clamp(1.0 + curvature_tensor * point_norm2, min=1e-15)
    klein_norm2 = torch.sum(klein * klein, dim=-1, keepdim=True)
    gamma = torch.rsqrt(torch.clamp(1.0 - curvature_tensor * klein_norm2, min=1e-12))
    weighted_gamma = weights.unsqueeze(-1) * gamma
    numerator = torch.sum(weighted_gamma * klein, dim=1)
    denominator = torch.clamp(torch.sum(weighted_gamma, dim=1), min=1e-15)
    klein_midpoint = numerator / denominator
    midpoint_norm2 = torch.sum(klein_midpoint * klein_midpoint, dim=-1, keepdim=True)
    poincare_midpoint = klein_midpoint / torch.clamp(
        1.0 + torch.sqrt(torch.clamp(1.0 - curvature_tensor * midpoint_norm2, min=1e-12)),
        min=1e-15,
    )
    return torch_project_to_ball(poincare_midpoint, curvature_tensor, eps=eps)


def torch_poincare_frechet_mean(
    points: Tensor,
    weights: Tensor,
    curvature: Tensor | float,
    steps: int = 4,
    step_size: float = 0.5,
    eps: float = 1e-5,
) -> Tensor:
    """Unrolled Karcher mean refinement for weighted points in the Poincare ball.

    The weighted Einstein midpoint is used as a stable initialization. Each
    iteration moves along the weighted average of logarithmic maps at the
    current estimate, which approximates the minimizer of
    sum_i w_i d_c(mu, x_i)^2 without introducing new trainable parameters.
    """
    if steps < 0:
        raise ValueError("steps must be non-negative")
    curvature_tensor = torch.as_tensor(curvature, dtype=points.dtype, device=points.device)
    mean = torch_poincare_weighted_midpoint(points, weights, curvature=curvature_tensor, eps=eps)
    for _ in range(steps):
        tangent_updates = torch_logmap(mean.unsqueeze(1), points, curvature=curvature_tensor, eps=eps)
        update = torch.sum(weights.unsqueeze(-1) * tangent_updates, dim=1)
        mean = torch_expmap(mean, step_size * update, curvature=curvature_tensor, eps=eps)
    return torch_project_to_ball(mean, curvature_tensor, eps=eps)


def torch_poincare_frechet_residual(
    points: Tensor,
    weights: Tensor,
    mean: Tensor,
    curvature: Tensor | float,
    eps: float = 1e-5,
) -> Tensor:
    """Norm of the weighted Karcher first-order residual at ``mean``.

    For the weighted Frechet objective ``sum_i w_i d_c(mu, x_i)^2``, an exact
    Karcher mean satisfies ``sum_i w_i log_mu(x_i) = 0``. This diagnostic
    reports the norm of that tangent-space residual for each batch item.
    """
    curvature_tensor = torch.as_tensor(curvature, dtype=points.dtype, device=points.device)
    tangent_updates = torch_logmap(mean.unsqueeze(1), points, curvature=curvature_tensor, eps=eps)
    residual = torch.sum(weights.unsqueeze(-1) * tangent_updates, dim=1)
    return torch.linalg.vector_norm(residual, dim=-1)


def torch_poincare_frechet_objective(
    points: Tensor,
    weights: Tensor,
    mean: Tensor,
    curvature: Tensor | float,
) -> Tensor:
    """Weighted squared-distance Frechet objective at ``mean``."""
    curvature_tensor = torch.as_tensor(curvature, dtype=points.dtype, device=points.device)
    distances = torch_poincare_distance(mean.unsqueeze(1), points, curvature=curvature_tensor)
    return torch.sum(weights * distances.square(), dim=1)


def structural_distance_loss(embedding_distances: Tensor, ast_distances: Tensor) -> Tensor:
    """Scale-invariant MSE between embedding distances and AST/tree distances."""
    embedding_distances = embedding_distances.float()
    ast_distances = ast_distances.float()
    valid = ast_distances > 0
    if not bool(valid.any()):
        return embedding_distances.new_tensor(0.0)
    emb = embedding_distances[valid]
    ast = ast_distances[valid]
    alpha = torch.sum(ast * emb) / torch.clamp(torch.sum(emb * emb), min=1e-12)
    residual = ast - alpha * emb
    return torch.mean(residual * residual) / torch.clamp(torch.mean(ast * ast), min=1e-12)


def structural_normalized_stress(embedding_distances: Tensor, ast_distances: Tensor) -> Tensor:
    """Scale-invariant normalized stress between learned and AST distances.

    The optimal scalar alpha aligns learned distances to AST distances before
    measuring distortion:

        stress = sqrt(sum_i (d_AST_i - alpha d_model_i)^2 / sum_i d_AST_i^2).

    This is a diagnostic metric, while `structural_distance_loss` is its
    squared training-loss counterpart.
    """
    squared_stress = structural_distance_loss(embedding_distances, ast_distances)
    return torch.sqrt(torch.clamp(squared_stress, min=0.0))


def structural_rank_loss(embedding_distances: Tensor, ast_distances: Tensor, margin: float = 0.05) -> Tensor:
    """Adjacent ranking loss: larger AST distances should have larger embedding distances."""
    embedding_distances = embedding_distances.float()
    ast_distances = ast_distances.float()
    valid = ast_distances > 0
    if int(valid.sum()) < 2:
        return embedding_distances.new_tensor(0.0)
    emb = embedding_distances[valid]
    ast = ast_distances[valid]
    order = torch.argsort(ast)
    sorted_ast = ast[order]
    sorted_emb = emb[order]
    ast_diff = sorted_ast[1:] - sorted_ast[:-1]
    emb_diff = sorted_emb[1:] - sorted_emb[:-1]
    ordered = ast_diff > 0
    if not bool(ordered.any()):
        return embedding_distances.new_tensor(0.0)
    return torch.relu(margin - emb_diff[ordered]).mean()


def _average_ranks(values: Tensor) -> Tensor:
    """Average ranks for tied one-dimensional values, zero-based."""
    if values.ndim != 1:
        raise ValueError("values must be one-dimensional")
    sorted_values, order = torch.sort(values)
    ranks = torch.empty_like(sorted_values)
    unique_values, counts = torch.unique_consecutive(sorted_values, return_counts=True)
    del unique_values
    start = 0
    for count_tensor in counts:
        count = int(count_tensor.item())
        end = start + count
        average_rank = (start + end - 1) / 2.0
        ranks[start:end] = average_rank
        start = end
    inverse_order = torch.empty_like(order)
    inverse_order[order] = torch.arange(order.numel(), device=order.device)
    return ranks[inverse_order]


def structural_spearman_correlation(embedding_distances: Tensor, ast_distances: Tensor) -> Tensor:
    """Spearman rank correlation between representation distances and AST distances.

    This is a diagnostic statistic, not a training loss. It measures whether
    pairs that are farther in the AST are also farther in the learned geometry.
    Degenerate constant-distance cases return 0 instead of NaN.
    """
    embedding_distances = embedding_distances.float()
    ast_distances = ast_distances.float()
    valid = ast_distances > 0
    if int(valid.sum()) < 2:
        return embedding_distances.new_tensor(0.0)
    emb_ranks = _average_ranks(embedding_distances[valid])
    ast_ranks = _average_ranks(ast_distances[valid])
    emb_centered = emb_ranks - emb_ranks.mean()
    ast_centered = ast_ranks - ast_ranks.mean()
    denominator = torch.linalg.vector_norm(emb_centered) * torch.linalg.vector_norm(ast_centered)
    if float(denominator.detach()) <= 1e-12:
        return embedding_distances.new_tensor(0.0)
    return torch.sum(emb_centered * ast_centered) / denominator


def longest_common_prefix_length(left: Tensor, left_mask: Tensor, right: Tensor, right_mask: Tensor) -> Tensor:
    """Length of the common prefix in serialized AST-label path sequences."""
    max_len = min(left.shape[-1], right.shape[-1])
    shared = (left[..., :max_len] == right[..., :max_len]) & left_mask[..., :max_len] & right_mask[..., :max_len]
    if max_len > 1:
        shared_prefix = torch.cumprod(shared.long(), dim=-1).bool()
    else:
        shared_prefix = shared
    return shared_prefix.long().sum(dim=-1).float()


def prefix_trie_distance(left: Tensor, left_mask: Tensor, right: Tensor, right_mask: Tensor) -> Tensor:
    """Tree distance in the prefix trie of serialized AST-label path sequences.

    This operates on the sequence representation available in code2vec/code2seq
    preprocessed files. It is not a true source-AST LCA distance unless the
    extractor also preserves endpoint node identifiers and parent links.
    """
    left_len = left_mask.long().sum(dim=-1)
    right_len = right_mask.long().sum(dim=-1)
    lcp_length = longest_common_prefix_length(left, left_mask, right, right_mask)
    return (left_len + right_len - 2 * lcp_length).float()


def ast_sequence_lca_depth(left: Tensor, left_mask: Tensor, right: Tensor, right_mask: Tensor) -> Tensor:
    """Backward-compatible alias for ``longest_common_prefix_length``.

    The old name is retained for existing experiment artifacts. New code should
    use ``longest_common_prefix_length`` to avoid implying access to true AST
    LCA nodes.
    """
    return longest_common_prefix_length(left, left_mask, right, right_mask)


def ast_sequence_tree_distance(left: Tensor, left_mask: Tensor, right: Tensor, right_mask: Tensor) -> Tensor:
    """Backward-compatible alias for ``prefix_trie_distance``."""
    return prefix_trie_distance(left, left_mask, right, right_mask)


def ast_path_midpoint_branch_masks(ast_path_mask: Tensor) -> tuple[Tensor, Tensor]:
    """Split a serialized terminal-to-terminal AST path into two branch masks.

    The original code2seq Java-small files expose serialized AST-node labels,
    not persistent AST node identifiers. Therefore this split uses the middle
    observed path position as an LCA proxy. The pivot is included in both
    branches, matching the interpretation of an LCA shared by the two branches.
    """
    if ast_path_mask.ndim < 1:
        raise ValueError("ast_path_mask must have at least one dimension")
    path_length = ast_path_mask.shape[-1]
    positions = torch.arange(path_length, device=ast_path_mask.device).view(*([1] * (ast_path_mask.ndim - 1)), -1)
    lengths = ast_path_mask.long().sum(dim=-1, keepdim=True)
    pivot = torch.clamp((lengths - 1) // 2, min=0)
    left_mask = ast_path_mask & (positions <= pivot) & (lengths > 0)
    right_mask = ast_path_mask & (positions >= pivot) & (lengths > 0)
    return left_mask, right_mask


def _masked_sequence(row: Tensor, mask: Tensor) -> tuple[int, ...]:
    return tuple(int(token) for token, keep in zip(row.detach().cpu().tolist(), mask.detach().cpu().tolist()) if keep)


def _levenshtein_distance(left: tuple[int, ...], right: tuple[int, ...]) -> int:
    if not left:
        return len(right)
    if not right:
        return len(left)
    previous = list(range(len(right) + 1))
    for left_index, left_token in enumerate(left, start=1):
        current = [left_index]
        for right_index, right_token in enumerate(right, start=1):
            substitution = previous[right_index - 1] + (0 if left_token == right_token else 1)
            insertion = current[right_index - 1] + 1
            deletion = previous[right_index] + 1
            current.append(min(substitution, insertion, deletion))
        previous = current
    return previous[-1]


def ast_sequence_edit_distance(left: Tensor, left_mask: Tensor, right: Tensor, right_mask: Tensor) -> Tensor:
    """Levenshtein distance between masked AST-path label sequences.

    This is a diagnostic proxy, not a differentiable training objective.
    """
    flat_left = left.reshape(-1, left.shape[-1])
    flat_left_mask = left_mask.reshape(-1, left_mask.shape[-1])
    flat_right = right.reshape(-1, right.shape[-1])
    flat_right_mask = right_mask.reshape(-1, right_mask.shape[-1])
    distances = [
        float(_levenshtein_distance(_masked_sequence(l_row, l_mask), _masked_sequence(r_row, r_mask)))
        for l_row, l_mask, r_row, r_mask in zip(flat_left, flat_left_mask, flat_right, flat_right_mask)
    ]
    return left.new_tensor(distances, dtype=torch.float32).reshape(left.shape[:-1])


def _ngrams(sequence: tuple[int, ...], n: int = 2) -> set[tuple[int, ...]]:
    if not sequence:
        return set()
    if len(sequence) < n:
        return {sequence}
    return {sequence[index : index + n] for index in range(len(sequence) - n + 1)}


def ast_sequence_jaccard_distance(
    left: Tensor,
    left_mask: Tensor,
    right: Tensor,
    right_mask: Tensor,
    ngram: int = 2,
) -> Tensor:
    """Set-Jaccard distance over directed AST-label n-grams.

    Multiplicity is intentionally ignored. Use a multiset or weighted Jaccard
    sensitivity analysis before making claims about repeated grammar patterns.
    """
    if ngram <= 0:
        raise ValueError("ngram must be positive")
    flat_left = left.reshape(-1, left.shape[-1])
    flat_left_mask = left_mask.reshape(-1, left_mask.shape[-1])
    flat_right = right.reshape(-1, right.shape[-1])
    flat_right_mask = right_mask.reshape(-1, right_mask.shape[-1])
    distances: list[float] = []
    for l_row, l_mask, r_row, r_mask in zip(flat_left, flat_left_mask, flat_right, flat_right_mask):
        left_ngrams = _ngrams(_masked_sequence(l_row, l_mask), n=ngram)
        right_ngrams = _ngrams(_masked_sequence(r_row, r_mask), n=ngram)
        union = left_ngrams | right_ngrams
        if not union:
            distances.append(0.0)
        else:
            distances.append(1.0 - (len(left_ngrams & right_ngrams) / len(union)))
    return left.new_tensor(distances, dtype=torch.float32).reshape(left.shape[:-1])


def ast_sequence_distance(
    left: Tensor,
    left_mask: Tensor,
    right: Tensor,
    right_mask: Tensor,
    target_distance: StructuralTargetDistance = "prefix_tree",
) -> Tensor:
    if target_distance == "prefix_tree":
        return prefix_trie_distance(left, left_mask, right, right_mask)
    if target_distance == "edit":
        return ast_sequence_edit_distance(left, left_mask, right, right_mask)
    if target_distance == "jaccard_bigrams":
        return ast_sequence_jaccard_distance(left, left_mask, right, right_mask, ngram=2)
    raise ValueError(f"unknown structural target distance: {target_distance}")


def tree_context_features(batch: Code2HypBatch) -> Tensor:
    """Explicit non-hyperbolic prefix-trie features for each path context.

    Features are normalized to a stable [0, 1]-scale:
    path length, mean prefix-trie distance to other contexts, max prefix-trie
    distance to other contexts, and mean longest-common-prefix length.
    """
    if batch.context_tree_features is not None:
        return batch.context_tree_features

    batch_size, context_count, max_path_length = batch.ast_paths.shape
    dtype = torch.float32
    device = batch.ast_paths.device
    normalizer = float(max(max_path_length, 1))
    distance_normalizer = float(max(2 * max_path_length, 1))

    path_lengths = batch.ast_path_mask.to(dtype=dtype).sum(dim=-1)
    path_length_feature = path_lengths / normalizer
    distance_sum = torch.zeros(batch_size, context_count, dtype=dtype, device=device)
    distance_max = torch.zeros(batch_size, context_count, dtype=dtype, device=device)
    lcp_sum = torch.zeros(batch_size, context_count, dtype=dtype, device=device)
    pair_counts = torch.zeros(batch_size, context_count, dtype=dtype, device=device)

    for left_index in range(context_count):
        for right_index in range(context_count):
            if left_index == right_index:
                continue
            valid_pair = batch.context_mask[:, left_index] & batch.context_mask[:, right_index]
            if not bool(valid_pair.any()):
                continue
            distance = prefix_trie_distance(
                batch.ast_paths[:, left_index],
                batch.ast_path_mask[:, left_index],
                batch.ast_paths[:, right_index],
                batch.ast_path_mask[:, right_index],
            ).to(dtype=dtype) / distance_normalizer
            lcp_length = longest_common_prefix_length(
                batch.ast_paths[:, left_index],
                batch.ast_path_mask[:, left_index],
                batch.ast_paths[:, right_index],
                batch.ast_path_mask[:, right_index],
            ).to(dtype=dtype) / normalizer
            valid_float = valid_pair.to(dtype=dtype)
            distance_sum[:, left_index] = distance_sum[:, left_index] + distance * valid_float
            distance_max[:, left_index] = torch.maximum(distance_max[:, left_index], distance * valid_float)
            lcp_sum[:, left_index] = lcp_sum[:, left_index] + lcp_length * valid_float
            pair_counts[:, left_index] = pair_counts[:, left_index] + valid_float

    denominator = torch.clamp(pair_counts, min=1.0)
    features = torch.stack(
        [
            path_length_feature,
            distance_sum / denominator,
            distance_max,
            lcp_sum / denominator,
        ],
        dim=-1,
    )
    return features * batch.context_mask.unsqueeze(-1).to(dtype=dtype)


def path_node_attention_monotonicity_loss(weights: Tensor, mask: Tensor) -> Tensor:
    """Bidirectional monotonicity prior for attention along root-to-leaf AST paths.

    The loss is zero when a path attention profile is monotone either from root
    to leaf or from leaf to root. This keeps the regularizer structural and
    interpretable without hard-coding whether a task should prefer abstract
    root-near nodes or concrete leaf-near nodes.
    """
    if weights.shape != mask.shape:
        raise ValueError("weights and mask must have the same shape")
    if weights.shape[-1] < 2:
        return weights.new_tensor(0.0)

    valid_pairs = mask[..., :-1] & mask[..., 1:]
    pair_counts = valid_pairs.to(dtype=weights.dtype).sum(dim=-1)
    valid_paths = pair_counts > 0
    if not bool(valid_paths.any()):
        return weights.new_tensor(0.0)

    left = weights[..., :-1]
    right = weights[..., 1:]
    pair_weight = valid_pairs.to(dtype=weights.dtype)
    increasing_violation = torch.relu(left - right) * pair_weight
    decreasing_violation = torch.relu(right - left) * pair_weight
    denominator = torch.clamp(pair_counts, min=1.0)
    increasing_per_path = increasing_violation.sum(dim=-1) / denominator
    decreasing_per_path = decreasing_violation.sum(dim=-1) / denominator
    bidirectional_per_path = torch.minimum(increasing_per_path, decreasing_per_path)
    valid_weight = valid_paths.to(dtype=weights.dtype)
    return torch.sum(bidirectional_per_path * valid_weight) / torch.clamp(valid_weight.sum(), min=1.0)


def path_attention_soft_tree_distances(
    weights: Tensor,
    ast_paths: Tensor,
    ast_path_mask: Tensor,
    context_mask: Tensor,
) -> tuple[Tensor, Tensor]:
    """Attention-induced tree distances and ordinary leaf-to-leaf AST distances.

    For two AST paths p and q with node-attention distributions alpha and beta,
    the soft tree distance is

        sum_i sum_j alpha_i beta_j d(prefix_i(p), prefix_j(q)).

    This makes the path-node attention itself geometrically testable: if the
    attention mass is concentrated on leaf nodes, the soft distance recovers
    the usual leaf-to-leaf tree distance; if attention collapses to roots, the
    induced distances lose tree-scale information.
    """
    if weights.shape != ast_path_mask.shape:
        raise ValueError("weights and ast_path_mask must have the same shape")
    if ast_paths.shape != ast_path_mask.shape:
        raise ValueError("ast_paths and ast_path_mask must have the same shape")
    if context_mask.shape != weights.shape[:2]:
        raise ValueError("context_mask must match the batch and context dimensions")

    masked_weights = weights * ast_path_mask.to(dtype=weights.dtype)
    masked_weights = masked_weights / torch.clamp(masked_weights.sum(dim=-1, keepdim=True), min=1e-12)

    soft_distances: list[Tensor] = []
    leaf_distances: list[Tensor] = []
    context_count = context_mask.shape[1]
    path_length = weights.shape[-1]
    for left_index in range(context_count):
        for right_index in range(left_index + 1, context_count):
            valid_pair = context_mask[:, left_index] & context_mask[:, right_index]
            if not bool(valid_pair.any()):
                continue

            left_paths = ast_paths[valid_pair, left_index]
            right_paths = ast_paths[valid_pair, right_index]
            left_mask = ast_path_mask[valid_pair, left_index]
            right_mask = ast_path_mask[valid_pair, right_index]
            left_weights = masked_weights[valid_pair, left_index]
            right_weights = masked_weights[valid_pair, right_index]

            expected_distance = weights.new_zeros(left_paths.shape[0])
            for left_node_index in range(path_length):
                for right_node_index in range(path_length):
                    valid_node_pair = left_mask[:, left_node_index] & right_mask[:, right_node_index]
                    if not bool(valid_node_pair.any()):
                        continue

                    left_prefix_mask = left_mask[:, : left_node_index + 1]
                    right_prefix_mask = right_mask[:, : right_node_index + 1]
                    lcp_length = longest_common_prefix_length(
                        left_paths[:, : left_node_index + 1],
                        left_prefix_mask,
                        right_paths[:, : right_node_index + 1],
                        right_prefix_mask,
                    ).to(dtype=weights.dtype)
                    left_depth = left_prefix_mask.to(dtype=weights.dtype).sum(dim=-1)
                    right_depth = right_prefix_mask.to(dtype=weights.dtype).sum(dim=-1)
                    node_distance = left_depth + right_depth - 2.0 * lcp_length
                    node_weight = left_weights[:, left_node_index] * right_weights[:, right_node_index]
                    expected_distance = expected_distance + node_weight * node_distance

            soft_distances.append(expected_distance)
            leaf_distances.append(
                prefix_trie_distance(left_paths, left_mask, right_paths, right_mask).to(dtype=weights.dtype)
            )

    if not soft_distances:
        empty = weights.new_tensor([])
        return empty, empty
    return torch.cat(soft_distances), torch.cat(leaf_distances)


def path_attention_tree_distance_loss(
    weights: Tensor,
    ast_paths: Tensor,
    ast_path_mask: Tensor,
    context_mask: Tensor,
) -> Tensor:
    """Scale-invariant calibration loss for attention-induced AST distances."""
    soft_distances, leaf_distances = path_attention_soft_tree_distances(
        weights,
        ast_paths,
        ast_path_mask,
        context_mask,
    )
    if soft_distances.numel() == 0:
        return weights.new_tensor(0.0)
    return structural_distance_loss(soft_distances, leaf_distances)


def path_dual_attention_separation_loss(
    attention_pair: Tensor,
    mask: Tensor,
    margin: float = 0.25,
) -> Tensor:
    """Encourage root/detail attention heads to occupy different path depths.

    `attention_pair[..., 0, :]` is the root/abstract channel and
    `attention_pair[..., 1, :]` is the leaf/detail channel. The loss is zero
    when the expected leaf-channel depth exceeds the root-channel depth by at
    least `margin`.
    """
    if attention_pair.shape[:-2] + attention_pair.shape[-1:] != mask.shape:
        raise ValueError("attention_pair must have shape (*mask_prefix, 2, path_length)")
    if attention_pair.shape[-2] != 2:
        raise ValueError("attention_pair must contain exactly two attention heads")
    if margin < 0.0:
        raise ValueError("margin must be non-negative")

    path_length = mask.shape[-1]
    positions = torch.arange(path_length, dtype=attention_pair.dtype, device=attention_pair.device)
    lengths = mask.to(dtype=attention_pair.dtype).sum(dim=-1, keepdim=True)
    denominator = torch.clamp(lengths - 1.0, min=1.0)
    normalized_depth = positions.view(*([1] * (mask.ndim - 1)), path_length) / denominator
    normalized_depth = torch.where(lengths > 1.0, normalized_depth, torch.zeros_like(normalized_depth))

    valid_paths = mask.any(dim=-1)
    if not bool(valid_paths.any()):
        return attention_pair.new_tensor(0.0)

    masked_attention = attention_pair * mask.unsqueeze(-2).to(dtype=attention_pair.dtype)
    masked_attention = masked_attention / torch.clamp(masked_attention.sum(dim=-1, keepdim=True), min=1e-12)
    expected_depth = torch.sum(masked_attention * normalized_depth.unsqueeze(-2), dim=-1)
    root_depth = expected_depth[..., 0]
    leaf_depth = expected_depth[..., 1]
    loss_per_path = torch.relu(margin - (leaf_depth - root_depth))
    valid_weight = valid_paths.to(dtype=attention_pair.dtype)
    return torch.sum(loss_per_path * valid_weight) / torch.clamp(valid_weight.sum(), min=1.0)


def _collect_structural_pair_distances(
    output: Code2HypTorchOutput,
    batch: Code2HypBatch,
    target_distance: StructuralTargetDistance = "prefix_tree",
) -> tuple[Tensor, Tensor]:
    if (
        output.context_structural_embeddings is None
        and output.context_structural_points is None
        and output.context_structural_product_points is None
    ):
        raise ValueError("model output does not contain context structural embeddings")

    embedding_distances = []
    ast_distances = []
    context_count = batch.context_mask.shape[1]
    for left_index in range(context_count):
        for right_index in range(left_index + 1, context_count):
            valid_pair = batch.context_mask[:, left_index] & batch.context_mask[:, right_index]
            if not bool(valid_pair.any()):
                continue

            left_ast = batch.ast_paths[valid_pair, left_index]
            right_ast = batch.ast_paths[valid_pair, right_index]
            left_mask = batch.ast_path_mask[valid_pair, left_index]
            right_mask = batch.ast_path_mask[valid_pair, right_index]
            ast_distances.append(ast_sequence_distance(left_ast, left_mask, right_ast, right_mask, target_distance))

            if output.context_structural_product_points is not None:
                left_factors = tuple(factor[valid_pair, left_index] for factor in output.context_structural_product_points)
                right_factors = tuple(factor[valid_pair, right_index] for factor in output.context_structural_product_points)
                embedding_distances.append(
                    _poincare_product_distance(
                        left_factors,
                        right_factors,
                        output.curvature,
                        metric=output.structural_product_distance_metric,
                    )
                )
            elif output.context_structural_points is not None:
                left_point = output.context_structural_points[valid_pair, left_index]
                right_point = output.context_structural_points[valid_pair, right_index]
                if output.structural_geometry == "lorentz":
                    embedding_distances.append(torch_lorentz_distance(left_point, right_point, output.curvature))
                else:
                    embedding_distances.append(torch_poincare_distance(left_point, right_point, output.curvature))
            else:
                left_embedding = output.context_structural_embeddings[valid_pair, left_index]
                right_embedding = output.context_structural_embeddings[valid_pair, right_index]
                embedding_distances.append(
                    _structural_embedding_distance(
                        left_embedding,
                        right_embedding,
                        metric=output.structural_embedding_metric,
                    )
                )

    if not embedding_distances:
        empty = output.logits.new_tensor([])
        return empty, empty
    return torch.cat(embedding_distances), torch.cat(ast_distances)


def _context_embedding_distance_matrix(output: Code2HypTorchOutput, sample_index: int, valid_contexts: Tensor) -> Tensor:
    if output.context_structural_product_points is not None:
        factors = tuple(factor[sample_index, valid_contexts] for factor in output.context_structural_product_points)
        left_factors = tuple(factor.unsqueeze(1) for factor in factors)
        right_factors = tuple(factor.unsqueeze(0) for factor in factors)
        return _poincare_product_distance(
            left_factors,
            right_factors,
            output.curvature,
            metric=output.structural_product_distance_metric,
        )
    if output.context_structural_points is not None:
        points = output.context_structural_points[sample_index, valid_contexts]
        left = points.unsqueeze(1)
        right = points.unsqueeze(0)
        if output.structural_geometry == "lorentz":
            return torch_lorentz_distance(left, right, output.curvature)
        return torch_poincare_distance(left, right, output.curvature)
    if output.context_structural_embeddings is None:
        raise ValueError("model output does not contain context structural embeddings")
    embeddings = output.context_structural_embeddings[sample_index, valid_contexts]
    return _structural_embedding_distance(
        embeddings.unsqueeze(1),
        embeddings.unsqueeze(0),
        metric=output.structural_embedding_metric,
    )


def _structural_embedding_distance(
    left: Tensor,
    right: Tensor,
    metric: StructuralEmbeddingMetric = "l2",
) -> Tensor:
    """Distance between Euclidean structural embeddings.

    The default is Euclidean L2, matching the original code2vec-style controls.
    The L1 option is a reviewer-requested control for path spaces whose natural
    relation is closer to an incidence/symmetric-difference metric.
    """
    difference = left - right
    if metric == "l2":
        return torch.linalg.vector_norm(difference, dim=-1)
    if metric == "l1":
        return torch.sum(torch.abs(difference), dim=-1)
    raise ValueError(f"unknown structural embedding metric: {metric}")


def _poincare_product_distance(
    left_factors: tuple[Tensor, ...],
    right_factors: tuple[Tensor, ...],
    curvature: Tensor,
    metric: ProductDistanceMetric = "riemannian_l2",
) -> Tensor:
    if len(left_factors) != len(right_factors):
        raise ValueError("product factors must have the same number of components")
    if not left_factors:
        raise ValueError("product distance requires at least one factor")
    factor_distances = [
        torch_poincare_distance(left, right, curvature=curvature)
        for left, right in zip(left_factors, right_factors)
    ]
    stacked = torch.stack(factor_distances, dim=0)
    if metric == "riemannian_l2":
        return torch.sqrt(torch.sum(stacked.square(), dim=0) + 1e-12)
    if metric == "l1_sum":
        return torch.sum(stacked, dim=0)
    raise ValueError(f"unknown product distance metric: {metric}")


def _context_ast_distance_matrix(batch: Code2HypBatch, sample_index: int, valid_contexts: Tensor) -> Tensor:
    paths = batch.ast_paths[sample_index, valid_contexts]
    masks = batch.ast_path_mask[sample_index, valid_contexts]
    return prefix_trie_distance(
        paths.unsqueeze(1),
        masks.unsqueeze(1),
        paths.unsqueeze(0),
        masks.unsqueeze(0),
    )


def _topk_indices_with_index_tiebreak(distances: Tensor, k: int) -> Tensor:
    """Return nearest-neighbor indices with deterministic index tie-breaking."""
    context_count = distances.shape[-1]
    tie_break = torch.arange(context_count, dtype=distances.dtype, device=distances.device)
    tie_break = tie_break * torch.finfo(distances.dtype).eps
    return torch.argsort(distances + tie_break.view(1, context_count), dim=-1)[:, :k]


def batch_structural_neighbor_overlap_at_k(
    output: Code2HypTorchOutput,
    batch: Code2HypBatch,
    k: int = 1,
) -> Tensor:
    """Tie-tolerant top-k overlap for local AST-neighborhood preservation.

    For each valid path-context anchor, the predicted neighbors are the k
    nearest contexts in the learned structural geometry. The target set is
    tie-tolerant: all contexts with AST distance no larger than the kth nearest
    AST distance are accepted as structurally correct neighbors.

    The denominator is the number of predicted neighbors, not the full size of
    the tie-expanded target set. Thus the metric is a top-k neighborhood overlap
    diagnostic, not classical information-retrieval recall.
    """
    if k <= 0:
        raise ValueError("k must be positive")
    if (
        output.context_structural_embeddings is None
        and output.context_structural_points is None
        and output.context_structural_product_points is None
    ):
        raise ValueError("model output does not contain structural context representations")

    scores: list[Tensor] = []
    for sample_index in range(batch.context_mask.shape[0]):
        valid_contexts = torch.nonzero(batch.context_mask[sample_index], as_tuple=False).flatten()
        context_count = int(valid_contexts.numel())
        if context_count <= 1:
            continue
        effective_k = min(k, context_count - 1)
        embedding_distances = _context_embedding_distance_matrix(output, sample_index, valid_contexts)
        ast_distances = _context_ast_distance_matrix(batch, sample_index, valid_contexts).to(
            dtype=embedding_distances.dtype,
            device=embedding_distances.device,
        )
        diagonal = torch.eye(context_count, dtype=torch.bool, device=embedding_distances.device)
        embedding_distances = embedding_distances.masked_fill(diagonal, float("inf"))
        ast_distances = ast_distances.masked_fill(diagonal, float("inf"))

        predicted_neighbors = torch.topk(
            embedding_distances,
            k=effective_k,
            largest=False,
            dim=-1,
        ).indices
        kth_ast_distance = torch.topk(ast_distances, k=effective_k, largest=False, dim=-1).values[:, -1:]
        relevant_neighbors = ast_distances <= kth_ast_distance
        hits = torch.gather(relevant_neighbors, dim=1, index=predicted_neighbors)
        scores.append(hits.to(dtype=embedding_distances.dtype).mean(dim=-1))

    if not scores:
        return output.logits.new_tensor(0.0)
    return torch.cat(scores).mean()


def batch_structural_neighbor_exact_overlap_at_k(
    output: Code2HypTorchOutput,
    batch: Code2HypBatch,
    k: int = 1,
) -> Tensor:
    """Exact top-k overlap with deterministic tie-breaking.

    Unlike :func:`batch_structural_neighbor_overlap_at_k`, this diagnostic does
    not expand the target set when several contexts share the kth AST distance.
    It reports overlap against exactly k target neighbors after sorting by
    `(distance, context_index)`. This makes the metric stricter, but also more
    sensitive to arbitrary ties in discrete prefix-trie distances.
    """
    if k <= 0:
        raise ValueError("k must be positive")
    if (
        output.context_structural_embeddings is None
        and output.context_structural_points is None
        and output.context_structural_product_points is None
    ):
        raise ValueError("model output does not contain structural context representations")

    scores: list[Tensor] = []
    for sample_index in range(batch.context_mask.shape[0]):
        valid_contexts = torch.nonzero(batch.context_mask[sample_index], as_tuple=False).flatten()
        context_count = int(valid_contexts.numel())
        if context_count <= 1:
            continue
        effective_k = min(k, context_count - 1)
        embedding_distances = _context_embedding_distance_matrix(output, sample_index, valid_contexts)
        ast_distances = _context_ast_distance_matrix(batch, sample_index, valid_contexts).to(
            dtype=embedding_distances.dtype,
            device=embedding_distances.device,
        )
        diagonal = torch.eye(context_count, dtype=torch.bool, device=embedding_distances.device)
        embedding_distances = embedding_distances.masked_fill(diagonal, float("inf"))
        ast_distances = ast_distances.masked_fill(diagonal, float("inf"))

        predicted_neighbors = _topk_indices_with_index_tiebreak(embedding_distances, effective_k)
        target_neighbors = _topk_indices_with_index_tiebreak(ast_distances, effective_k)
        hits = (predicted_neighbors.unsqueeze(-1) == target_neighbors.unsqueeze(1)).any(dim=-1)
        scores.append(hits.to(dtype=embedding_distances.dtype).mean(dim=-1))

    if not scores:
        return output.logits.new_tensor(0.0)
    return torch.cat(scores).mean()


def batch_structural_distance_level_summary(
    output: Code2HypTorchOutput,
    batch: Code2HypBatch,
    target_distance: StructuralTargetDistance = "prefix_tree",
) -> list[dict[str, float | int]]:
    """Aggregate learned distances conditional on each target-distance level."""
    embedding_distances, target_distances = _collect_structural_pair_distances(output, batch, target_distance)
    if embedding_distances.numel() == 0:
        return []

    summaries: list[dict[str, float | int]] = []
    unique_levels = torch.unique(target_distances.detach()).sort().values
    for level in unique_levels:
        level_mask = target_distances == level
        level_distances = embedding_distances[level_mask]
        pair_count = int(level_distances.numel())
        if pair_count == 0:
            continue
        summaries.append(
            {
                "target_distance": float(level.detach().cpu()),
                "pair_count": pair_count,
                "model_distance_mean": float(level_distances.mean().detach().cpu()),
                "model_distance_std": float(level_distances.std(unbiased=False).detach().cpu()),
            }
        )
    return summaries


def batch_poincare_frechet_diagnostics(
    output: Code2HypTorchOutput,
) -> dict[str, Tensor] | None:
    """Frechet/Karcher residual diagnostics for Poincare aggregation outputs."""
    if output.structural_geometry == "poincare_product":
        if output.context_structural_product_points is None or output.structural_product_points is None:
            return None
        residuals = []
        objectives = []
        for context_points, aggregate_points in zip(
            output.context_structural_product_points,
            output.structural_product_points,
        ):
            residuals.append(
                torch_poincare_frechet_residual(
                    context_points,
                    output.attention,
                    aggregate_points,
                    curvature=output.curvature,
                )
            )
            objectives.append(
                torch_poincare_frechet_objective(
                    context_points,
                    output.attention,
                    aggregate_points,
                    curvature=output.curvature,
                )
            )
        residual = torch.stack(residuals, dim=0)
        objective = torch.stack(objectives, dim=0)
        return {
            "residual_mean": residual.mean(),
            "residual_max": residual.max(),
            "objective_mean": objective.mean(),
        }
    if output.structural_geometry != "poincare":
        return None
    if output.context_structural_points is None or output.structural_points is None:
        return None

    residual = torch_poincare_frechet_residual(
        output.context_structural_points,
        output.attention,
        output.structural_points,
        curvature=output.curvature,
    )
    objective = torch_poincare_frechet_objective(
        output.context_structural_points,
        output.attention,
        output.structural_points,
        curvature=output.curvature,
    )
    return {
        "residual_mean": residual.mean(),
        "residual_max": residual.max(),
        "objective_mean": objective.mean(),
    }


def batch_poincare_radius_utilization(
    output: Code2HypTorchOutput,
    batch: Code2HypBatch,
    near_boundary_threshold: float = 0.95,
) -> dict[str, Tensor] | None:
    """Radius-utilization diagnostics for Poincare context and aggregate points."""
    if output.structural_geometry == "poincare_product":
        if output.context_structural_product_points is None or output.structural_product_points is None:
            return None
        if not 0.0 < near_boundary_threshold < 1.0:
            raise ValueError("near_boundary_threshold must be in (0, 1)")
        curvature = torch.as_tensor(
            output.curvature,
            dtype=output.context_structural_product_points[0].dtype,
            device=output.context_structural_product_points[0].device,
        )
        radius = 1.0 / torch.sqrt(curvature)
        context_ratios = [
            torch.linalg.vector_norm(factor[batch.context_mask], dim=-1) / radius
            for factor in output.context_structural_product_points
            if factor[batch.context_mask].numel() > 0
        ]
        if not context_ratios:
            return None
        aggregate_ratios = [
            torch.linalg.vector_norm(factor, dim=-1) / radius
            for factor in output.structural_product_points
        ]
        context_ratio = torch.cat(context_ratios)
        aggregate_ratio = torch.cat(aggregate_ratios)
        return {
            "context_radius_ratio_mean": context_ratio.mean(),
            "context_radius_ratio_max": context_ratio.max(),
            "context_near_boundary_rate": (context_ratio > near_boundary_threshold).to(dtype=context_ratio.dtype).mean(),
            "aggregate_radius_ratio_mean": aggregate_ratio.mean(),
            "aggregate_radius_ratio_max": aggregate_ratio.max(),
        }
    if output.structural_geometry != "poincare":
        return None
    if output.context_structural_points is None or output.structural_points is None:
        return None
    if not 0.0 < near_boundary_threshold < 1.0:
        raise ValueError("near_boundary_threshold must be in (0, 1)")

    curvature = torch.as_tensor(
        output.curvature,
        dtype=output.context_structural_points.dtype,
        device=output.context_structural_points.device,
    )
    radius = 1.0 / torch.sqrt(curvature)
    valid_context_points = output.context_structural_points[batch.context_mask]
    if valid_context_points.numel() == 0:
        return None

    context_ratio = torch.linalg.vector_norm(valid_context_points, dim=-1) / radius
    aggregate_ratio = torch.linalg.vector_norm(output.structural_points, dim=-1) / radius
    return {
        "context_radius_ratio_mean": context_ratio.mean(),
        "context_radius_ratio_max": context_ratio.max(),
        "context_near_boundary_rate": (context_ratio > near_boundary_threshold).to(dtype=context_ratio.dtype).mean(),
        "aggregate_radius_ratio_mean": aggregate_ratio.mean(),
        "aggregate_radius_ratio_max": aggregate_ratio.max(),
    }


def batch_structural_neighbor_recall_at_k(
    output: Code2HypTorchOutput,
    batch: Code2HypBatch,
    k: int = 1,
) -> Tensor:
    """Backward-compatible alias for the former metric name.

    New code should prefer :func:`batch_structural_neighbor_overlap_at_k`.
    """
    return batch_structural_neighbor_overlap_at_k(output, batch, k=k)


def batch_structural_neighbor_distribution_regularizer(
    output: Code2HypTorchOutput,
    batch: Code2HypBatch,
    tree_temperature: float = 1.0,
    embedding_temperature: float = 1.0,
    eps: float = 1e-12,
) -> Tensor:
    """Local-neighborhood preservation loss over path contexts within each method.

    For every anchor context i, AST distances define a target neighbor
    distribution q(j|i) = softmax(-d_AST(i,j) / tau_T), while learned structural
    distances define p(j|i) = softmax(-d_model(i,j) / tau_E). The loss averages
    KL(q(.|i) || p(.|i)) over valid anchors. This is intentionally local:
    it penalizes wrong nearest-neighbor geometry even when global pairwise rank
    correlation looks acceptable.
    """
    if tree_temperature <= 0.0:
        raise ValueError("tree_temperature must be positive")
    if embedding_temperature <= 0.0:
        raise ValueError("embedding_temperature must be positive")
    if (
        output.context_structural_embeddings is None
        and output.context_structural_points is None
        and output.context_structural_product_points is None
    ):
        raise ValueError("model output does not contain structural context representations")

    losses: list[Tensor] = []
    for sample_index in range(batch.context_mask.shape[0]):
        valid_contexts = torch.nonzero(batch.context_mask[sample_index], as_tuple=False).flatten()
        context_count = int(valid_contexts.numel())
        if context_count <= 1:
            continue

        embedding_distances = _context_embedding_distance_matrix(output, sample_index, valid_contexts)
        ast_distances = _context_ast_distance_matrix(batch, sample_index, valid_contexts).to(
            dtype=embedding_distances.dtype,
            device=embedding_distances.device,
        )
        diagonal = torch.eye(context_count, dtype=torch.bool, device=embedding_distances.device)

        target_logits = (-ast_distances / tree_temperature).masked_fill(diagonal, float("-inf"))
        model_logits = (-embedding_distances / embedding_temperature).masked_fill(diagonal, float("-inf"))
        target_probs = F.softmax(target_logits, dim=-1)
        model_log_probs = F.log_softmax(model_logits, dim=-1)

        safe_target_log = torch.log(torch.clamp(target_probs, min=eps)).masked_fill(diagonal, 0.0)
        safe_model_log_probs = model_log_probs.masked_fill(diagonal, 0.0)
        safe_target_probs = target_probs.masked_fill(diagonal, 0.0)
        losses.append(torch.sum(safe_target_probs * (safe_target_log - safe_model_log_probs), dim=-1))

    if not losses:
        return output.logits.new_tensor(0.0)
    return torch.cat(losses).mean()


def batch_structural_distance_regularizer(output: Code2HypTorchOutput, batch: Code2HypBatch) -> Tensor:
    """Distance-preservation regularizer over AST path contexts within each method."""
    embedding_distances, ast_distances = _collect_structural_pair_distances(output, batch)
    if embedding_distances.numel() == 0:
        return output.logits.new_tensor(0.0)
    return structural_distance_loss(embedding_distances, ast_distances)


def batch_structural_multi_metric_distance_regularizer(
    output: Code2HypTorchOutput,
    batch: Code2HypBatch,
    target_distances: tuple[StructuralTargetDistance, ...] = ("prefix_tree", "edit", "jaccard_bigrams"),
) -> Tensor:
    """Average distance-preservation loss over several AST-path proxy metrics.

    This objective is intentionally stricter than the single prefix-trie loss:
    the same learned geometry must align with more than one structural relation
    between serialized AST paths. It is useful as a cross-metric control because
    evaluation on edit/Jaccard distances is no longer purely out-of-objective.
    """
    if not target_distances:
        raise ValueError("target_distances must not be empty")
    losses: list[Tensor] = []
    for target_distance in target_distances:
        embedding_distances, ast_distances = _collect_structural_pair_distances(output, batch, target_distance)
        if embedding_distances.numel() == 0:
            continue
        losses.append(structural_distance_loss(embedding_distances, ast_distances))
    if not losses:
        return output.logits.new_tensor(0.0)
    return torch.stack(losses).mean()


def batch_structural_normalized_stress(
    output: Code2HypTorchOutput,
    batch: Code2HypBatch,
    target_distance: StructuralTargetDistance = "prefix_tree",
) -> Tensor:
    """Normalized metric stress over AST path contexts within each method."""
    embedding_distances, ast_distances = _collect_structural_pair_distances(output, batch, target_distance)
    if embedding_distances.numel() == 0:
        return output.logits.new_tensor(0.0)
    return structural_normalized_stress(embedding_distances, ast_distances)


def batch_structural_rank_regularizer(
    output: Code2HypTorchOutput,
    batch: Code2HypBatch,
    margin: float = 0.05,
) -> Tensor:
    """Rank-preservation regularizer over AST path contexts within each method."""
    embedding_distances, ast_distances = _collect_structural_pair_distances(output, batch)
    if embedding_distances.numel() == 0:
        return output.logits.new_tensor(0.0)
    return structural_rank_loss(embedding_distances, ast_distances, margin=margin)


def batch_structural_spearman_correlation(
    output: Code2HypTorchOutput,
    batch: Code2HypBatch,
    target_distance: StructuralTargetDistance = "prefix_tree",
) -> Tensor:
    """Diagnostic Spearman correlation over context-pair AST and embedding distances."""
    embedding_distances, ast_distances = _collect_structural_pair_distances(output, batch, target_distance)
    if embedding_distances.numel() == 0:
        return output.logits.new_tensor(0.0)
    return structural_spearman_correlation(embedding_distances, ast_distances)


class Code2HypTorchModel(nn.Module):
    """Trainable forward core for controlled Code2Hyp experiments."""

    def __init__(self, config: Code2HypTorchConfig, variant: Variant) -> None:
        super().__init__()
        if variant not in (
            "euclidean",
            "euclidean_metric",
            "bounded_euclidean_metric",
            "euclidean_tree",
            "product",
            "hyperbolic",
            "hyperbolic_attention",
            "hyperbolic_frechet",
            "hyperbolic_path_message_passing",
            "hyperbolic_path_attention_message_passing",
            "hyperbolic_path_depth_attention_message_passing",
            "hyperbolic_path_dual_attention_message_passing",
            "lorentz_path_dual_attention_message_passing",
            "lorentz",
            "lorentz_product",
            "factorized_product",
            "factorized_product_learned_metric",
            "factorized_product_three_metric",
            "factorized_product_channel_mixer",
            "code2hyp_product_frechet",
            "code2hyp_code2vec_attention_frechet",
            "code2vec_context_transform",
            "code2vec_context_transform_l1",
            "code2hyp_context_transform_frechet",
            "code2hyp_product_context_transform_frechet",
            "code2hyp_context_transform_product_bias_frechet",
            "code2hyp_branch_product_context_transform_frechet",
            "code2hyp_context_transform_branch_product_bias_frechet",
            "code2hyp_context_transform_latent_lca_branch_product_bias_frechet",
            "code2hyp_context_transform_latent_lca_prior_branch_product_bias_frechet",
            "code2hyp_context_transform_branch_sequence_product_bias_frechet",
        ):
            raise ValueError(f"unknown Code2Hyp variant: {variant}")
        self.config = config
        self.variant = variant
        self.branch_left_dim = (config.structural_dim + 1) // 2
        self.branch_right_dim = config.structural_dim - self.branch_left_dim
        self.token_embeddings = nn.Embedding(config.token_vocab_size, config.token_dim)
        self.ast_node_embeddings = nn.Embedding(config.ast_node_vocab_size, config.structural_dim)
        if config.path_encoder == "gru":
            self.ast_path_encoder = nn.GRU(
                input_size=config.structural_dim,
                hidden_size=config.structural_dim,
                batch_first=True,
            )
        elif config.path_encoder == "mean":
            self.ast_path_encoder = None
        else:
            raise ValueError(f"unknown path_encoder: {config.path_encoder}")
        if config.representation_transform == "tanh":
            self.representation_transform_layer = nn.Linear(config.representation_dim, config.representation_dim)
        elif config.representation_transform == "identity":
            self.representation_transform_layer = None
        else:
            raise ValueError(f"unknown representation_transform: {config.representation_transform}")
        if variant in (
            "code2vec_context_transform",
            "code2vec_context_transform_l1",
            "code2hyp_context_transform_frechet",
            "code2hyp_product_context_transform_frechet",
            "code2hyp_context_transform_product_bias_frechet",
            "code2hyp_branch_product_context_transform_frechet",
            "code2hyp_context_transform_branch_product_bias_frechet",
            "code2hyp_context_transform_latent_lca_branch_product_bias_frechet",
            "code2hyp_context_transform_latent_lca_prior_branch_product_bias_frechet",
            "code2hyp_context_transform_branch_sequence_product_bias_frechet",
        ):
            self.context_transform_layer = nn.Linear(config.representation_dim, config.representation_dim)
        else:
            self.context_transform_layer = None
        if variant in (
            "code2hyp_branch_product_context_transform_frechet",
            "code2hyp_context_transform_branch_product_bias_frechet",
            "code2hyp_context_transform_latent_lca_branch_product_bias_frechet",
            "code2hyp_context_transform_latent_lca_prior_branch_product_bias_frechet",
        ):
            if self.branch_right_dim <= 0:
                raise ValueError("structural_dim must be at least 2 for branch-product encoding")
            self.branch_left_projection = nn.Linear(config.structural_dim, self.branch_left_dim, bias=False)
            self.branch_right_projection = nn.Linear(config.structural_dim, self.branch_right_dim, bias=False)
        else:
            self.branch_left_projection = None
            self.branch_right_projection = None
        if variant == "code2hyp_context_transform_branch_sequence_product_bias_frechet":
            if self.branch_right_dim <= 0:
                raise ValueError("structural_dim must be at least 2 for branch-sequence-product encoding")
            self.branch_left_sequence_encoder = nn.GRU(
                input_size=config.structural_dim,
                hidden_size=self.branch_left_dim,
                batch_first=True,
            )
            self.branch_right_sequence_encoder = nn.GRU(
                input_size=config.structural_dim,
                hidden_size=self.branch_right_dim,
                batch_first=True,
            )
        else:
            self.branch_left_sequence_encoder = None
            self.branch_right_sequence_encoder = None
        if variant in (
            "code2hyp_context_transform_latent_lca_branch_product_bias_frechet",
            "code2hyp_context_transform_latent_lca_prior_branch_product_bias_frechet",
        ):
            self.branch_pivot_query = nn.Parameter(torch.empty(config.structural_dim))
        else:
            self.register_parameter("branch_pivot_query", None)
        if variant == "code2hyp_context_transform_latent_lca_prior_branch_product_bias_frechet":
            initial_prior_weight = 1.0
            raw_prior_weight = math.log(math.expm1(initial_prior_weight))
            self.raw_branch_pivot_center_prior_weight = nn.Parameter(
                torch.tensor(raw_prior_weight, dtype=torch.float32),
            )
        else:
            self.register_parameter("raw_branch_pivot_center_prior_weight", None)
        self.attention_query = nn.Parameter(torch.empty(config.representation_dim))
        self.tree_feature_projection = nn.Linear(4, 1, bias=False) if variant == "euclidean_tree" else None
        if variant in (
            "hyperbolic_path_message_passing",
            "hyperbolic_path_attention_message_passing",
            "hyperbolic_path_depth_attention_message_passing",
            "hyperbolic_path_dual_attention_message_passing",
            "lorentz_path_dual_attention_message_passing",
        ):
            if config.path_message_passing_steps <= 0:
                raise ValueError("path_message_passing_steps must be positive")
            self.path_message_linear = nn.Linear(config.structural_dim, config.structural_dim, bias=False)
            self.path_update_linear = nn.Linear(2 * config.structural_dim, config.structural_dim)
        else:
            self.path_message_linear = None
            self.path_update_linear = None
        if variant in (
            "hyperbolic_path_attention_message_passing",
            "hyperbolic_path_depth_attention_message_passing",
        ):
            self.path_node_attention_query = nn.Parameter(torch.empty(config.structural_dim))
        else:
            self.register_parameter("path_node_attention_query", None)
        if variant in (
            "hyperbolic_path_dual_attention_message_passing",
            "lorentz_path_dual_attention_message_passing",
        ):
            self.path_root_attention_query = nn.Parameter(torch.empty(config.structural_dim))
            self.path_leaf_attention_query = nn.Parameter(torch.empty(config.structural_dim))
            self.path_dual_attention_projection = nn.Linear(2 * config.structural_dim, config.structural_dim)
        else:
            self.register_parameter("path_root_attention_query", None)
            self.register_parameter("path_leaf_attention_query", None)
            self.path_dual_attention_projection = None
        if variant == "hyperbolic_path_depth_attention_message_passing":
            self.raw_path_depth_attention_bias = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
        else:
            self.register_parameter("raw_path_depth_attention_bias", None)
        if variant == "factorized_product_channel_mixer":
            if config.factorized_mixer_rank <= 0:
                raise ValueError("factorized_mixer_rank must be positive")
            self.factorized_channel_down = nn.Linear(
                config.representation_dim,
                config.factorized_mixer_rank,
                bias=False,
            )
            self.factorized_channel_up = nn.Linear(
                config.factorized_mixer_rank,
                config.representation_dim,
                bias=False,
            )
        else:
            self.factorized_channel_down = None
            self.factorized_channel_up = None
        self.decoder = nn.Linear(config.representation_dim, config.label_vocab_size)

        if (
            variant
            in (
                "product",
                "hyperbolic",
                "hyperbolic_attention",
                "hyperbolic_frechet",
                "hyperbolic_path_message_passing",
                "hyperbolic_path_attention_message_passing",
                "hyperbolic_path_depth_attention_message_passing",
                "hyperbolic_path_dual_attention_message_passing",
                "lorentz_path_dual_attention_message_passing",
                "lorentz",
                "lorentz_product",
                "factorized_product",
                "factorized_product_learned_metric",
                "factorized_product_three_metric",
                "factorized_product_channel_mixer",
                "code2hyp_product_frechet",
                "code2hyp_code2vec_attention_frechet",
                "code2hyp_context_transform_frechet",
                "code2hyp_product_context_transform_frechet",
                "code2hyp_context_transform_product_bias_frechet",
                "code2hyp_branch_product_context_transform_frechet",
                "code2hyp_context_transform_branch_product_bias_frechet",
                "code2hyp_context_transform_latent_lca_branch_product_bias_frechet",
                "code2hyp_context_transform_latent_lca_prior_branch_product_bias_frechet",
                "code2hyp_context_transform_branch_sequence_product_bias_frechet",
            )
            and config.trainable_curvature
        ):
            initial = max(config.curvature - config.eps, config.eps)
            raw = math.log(math.expm1(initial))
            self.raw_curvature = nn.Parameter(torch.tensor(raw, dtype=torch.float32))
        else:
            self.register_buffer("raw_curvature", torch.tensor(math.nan), persistent=False)
        if variant in (
            "factorized_product_learned_metric",
            "factorized_product_three_metric",
            "code2hyp_product_frechet",
            "code2hyp_product_context_transform_frechet",
            "code2hyp_context_transform_product_bias_frechet",
            "code2hyp_branch_product_context_transform_frechet",
            "code2hyp_context_transform_branch_product_bias_frechet",
            "code2hyp_context_transform_latent_lca_branch_product_bias_frechet",
            "code2hyp_context_transform_latent_lca_prior_branch_product_bias_frechet",
            "code2hyp_context_transform_branch_sequence_product_bias_frechet",
        ):
            initial_weight = max(1.0 - config.eps, config.eps)
            raw_weight = math.log(math.expm1(initial_weight))
            metric_count = (
                4
                if variant
                in (
                    "code2hyp_branch_product_context_transform_frechet",
                    "code2hyp_context_transform_branch_product_bias_frechet",
                    "code2hyp_context_transform_latent_lca_branch_product_bias_frechet",
                    "code2hyp_context_transform_latent_lca_prior_branch_product_bias_frechet",
                    "code2hyp_context_transform_branch_sequence_product_bias_frechet",
                )
                else
                3
                if variant
                in (
                    "factorized_product_three_metric",
                    "code2hyp_product_frechet",
                    "code2hyp_product_context_transform_frechet",
                    "code2hyp_context_transform_product_bias_frechet",
                )
                else 2
            )
            self.raw_factorized_metric_weights = nn.Parameter(
                torch.full((metric_count,), raw_weight, dtype=torch.float32),
            )
        else:
            self.register_buffer("raw_factorized_metric_weights", torch.full((2,), math.nan), persistent=False)
        if variant in (
            "code2hyp_context_transform_product_bias_frechet",
            "code2hyp_branch_product_context_transform_frechet",
            "code2hyp_context_transform_branch_product_bias_frechet",
            "code2hyp_context_transform_latent_lca_branch_product_bias_frechet",
            "code2hyp_context_transform_latent_lca_prior_branch_product_bias_frechet",
            "code2hyp_context_transform_branch_sequence_product_bias_frechet",
        ):
            initial_bias_weight = 0.1
            raw_bias_weight = math.log(math.expm1(initial_bias_weight))
            self.raw_product_attention_bias_weight = nn.Parameter(torch.tensor(raw_bias_weight, dtype=torch.float32))
        else:
            self.register_buffer("raw_product_attention_bias_weight", torch.tensor(math.nan), persistent=False)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.token_embeddings.weight, mean=0.0, std=0.05)
        nn.init.normal_(self.ast_node_embeddings.weight, mean=0.0, std=0.05)
        if self.ast_path_encoder is not None:
            for name, parameter in self.ast_path_encoder.named_parameters():
                if "weight" in name:
                    nn.init.xavier_uniform_(parameter)
                elif "bias" in name:
                    nn.init.zeros_(parameter)
        if self.representation_transform_layer is not None:
            nn.init.xavier_uniform_(self.representation_transform_layer.weight)
            nn.init.zeros_(self.representation_transform_layer.bias)
        if self.context_transform_layer is not None:
            nn.init.xavier_uniform_(self.context_transform_layer.weight)
            nn.init.zeros_(self.context_transform_layer.bias)
        if self.branch_left_projection is not None and self.branch_right_projection is not None:
            nn.init.xavier_uniform_(self.branch_left_projection.weight)
            nn.init.xavier_uniform_(self.branch_right_projection.weight)
        if self.branch_left_sequence_encoder is not None and self.branch_right_sequence_encoder is not None:
            for encoder in (self.branch_left_sequence_encoder, self.branch_right_sequence_encoder):
                for name, parameter in encoder.named_parameters():
                    if "weight" in name:
                        nn.init.xavier_uniform_(parameter)
                    elif "bias" in name:
                        nn.init.zeros_(parameter)
        if self.branch_pivot_query is not None:
            nn.init.normal_(self.branch_pivot_query, mean=0.0, std=0.05)
        if self.tree_feature_projection is not None:
            nn.init.zeros_(self.tree_feature_projection.weight)
        if self.path_message_linear is not None and self.path_update_linear is not None:
            nn.init.xavier_uniform_(self.path_message_linear.weight)
            nn.init.xavier_uniform_(self.path_update_linear.weight)
            nn.init.zeros_(self.path_update_linear.bias)
        if self.path_node_attention_query is not None:
            nn.init.normal_(self.path_node_attention_query, mean=0.0, std=0.05)
        if self.path_root_attention_query is not None:
            nn.init.normal_(self.path_root_attention_query, mean=0.0, std=0.05)
        if self.path_leaf_attention_query is not None:
            nn.init.normal_(self.path_leaf_attention_query, mean=0.0, std=0.05)
        if self.path_dual_attention_projection is not None:
            nn.init.xavier_uniform_(self.path_dual_attention_projection.weight)
            nn.init.zeros_(self.path_dual_attention_projection.bias)
        if self.factorized_channel_down is not None and self.factorized_channel_up is not None:
            nn.init.xavier_uniform_(self.factorized_channel_down.weight)
            nn.init.normal_(self.factorized_channel_up.weight, mean=0.0, std=1e-3)
        nn.init.normal_(self.attention_query, mean=0.0, std=0.05)
        nn.init.xavier_uniform_(self.decoder.weight)
        nn.init.zeros_(self.decoder.bias)

    def curvature(self) -> Tensor:
        if (
            self.variant
            in (
                "product",
                "hyperbolic",
                "hyperbolic_attention",
                "hyperbolic_frechet",
                "hyperbolic_path_message_passing",
                "hyperbolic_path_attention_message_passing",
                "hyperbolic_path_depth_attention_message_passing",
                "hyperbolic_path_dual_attention_message_passing",
                "lorentz_path_dual_attention_message_passing",
                "lorentz",
                "lorentz_product",
                "factorized_product",
                "factorized_product_learned_metric",
                "factorized_product_three_metric",
                "factorized_product_channel_mixer",
                "code2hyp_product_frechet",
                "code2hyp_code2vec_attention_frechet",
                "code2hyp_context_transform_frechet",
                "code2hyp_product_context_transform_frechet",
                "code2hyp_context_transform_product_bias_frechet",
                "code2hyp_branch_product_context_transform_frechet",
                "code2hyp_context_transform_branch_product_bias_frechet",
                "code2hyp_context_transform_latent_lca_branch_product_bias_frechet",
                "code2hyp_context_transform_latent_lca_prior_branch_product_bias_frechet",
                "code2hyp_context_transform_branch_sequence_product_bias_frechet",
            )
            and self.config.trainable_curvature
        ):
            return F.softplus(self.raw_curvature) + self.config.eps
        return self.attention_query.new_tensor(self.config.curvature)

    def factorized_metric_weights(self) -> Tensor:
        if self.variant in (
            "factorized_product_three_metric",
            "code2hyp_product_frechet",
            "code2hyp_product_context_transform_frechet",
            "code2hyp_context_transform_product_bias_frechet",
            "code2hyp_branch_product_context_transform_frechet",
            "code2hyp_context_transform_branch_product_bias_frechet",
            "code2hyp_context_transform_latent_lca_branch_product_bias_frechet",
            "code2hyp_context_transform_latent_lca_prior_branch_product_bias_frechet",
            "code2hyp_context_transform_branch_sequence_product_bias_frechet",
        ):
            return F.softplus(self.raw_factorized_metric_weights) + self.config.eps
        if self.variant == "factorized_product_learned_metric":
            return F.softplus(self.raw_factorized_metric_weights) + self.config.eps
        if self.variant == "factorized_product":
            return self.attention_query.new_ones(2)
        if self.variant == "factorized_product_channel_mixer":
            return self.attention_query.new_ones(2)
        return self.attention_query.new_ones(2)

    def product_attention_bias_weight(self) -> Tensor:
        if self.variant in (
            "code2hyp_context_transform_product_bias_frechet",
            "code2hyp_branch_product_context_transform_frechet",
            "code2hyp_context_transform_branch_product_bias_frechet",
            "code2hyp_context_transform_latent_lca_branch_product_bias_frechet",
            "code2hyp_context_transform_latent_lca_prior_branch_product_bias_frechet",
            "code2hyp_context_transform_branch_sequence_product_bias_frechet",
        ):
            return F.softplus(self.raw_product_attention_bias_weight) + self.config.eps
        return self.attention_query.new_tensor(0.0)

    def branch_pivot_center_prior_weight(self) -> Tensor:
        if self.raw_branch_pivot_center_prior_weight is None:
            return self.attention_query.new_tensor(0.0)
        return F.softplus(self.raw_branch_pivot_center_prior_weight) + self.config.eps

    def factorized_channel_mixer_rank(self) -> int:
        if self.variant != "factorized_product_channel_mixer":
            return 0
        return self.config.factorized_mixer_rank

    def forward(self, batch: Code2HypBatch) -> Code2HypTorchOutput:
        start_vectors = self.token_embeddings(batch.start_tokens)
        end_vectors = self.token_embeddings(batch.end_tokens)
        curvature = self.curvature()
        if self.variant in (
            "hyperbolic_path_message_passing",
            "hyperbolic_path_attention_message_passing",
            "hyperbolic_path_depth_attention_message_passing",
            "hyperbolic_path_dual_attention_message_passing",
            "lorentz_path_dual_attention_message_passing",
        ):
            (
                path_tangents,
                path_node_attention,
                path_node_attention_monotonicity,
                path_node_attention_pair,
            ) = self._hyperbolic_path_message_tangents(
                batch.ast_paths,
                batch.ast_path_mask,
                curvature,
            )
        else:
            path_tangents = self._path_tangents(batch.ast_paths, batch.ast_path_mask)
            path_node_attention = None
            path_node_attention_monotonicity = None
            path_node_attention_pair = None

        context_representations = torch.cat([start_vectors, path_tangents, end_vectors], dim=-1)
        structural_embedding_metric: StructuralEmbeddingMetric = "l2"
        structural_product_points: tuple[Tensor, ...] | None = None
        context_structural_product_points: tuple[Tensor, ...] | None = None
        structural_product_distance_metric: ProductDistanceMetric = "riemannian_l2"

        if self.variant == "euclidean":
            attention = self._attention(context_representations, batch.context_mask)
            representation = torch.sum(context_representations * attention.unsqueeze(-1), dim=1)
            structural_points = None
            context_structural_embeddings = path_tangents
            context_structural_points = None
            context_tree_features = None
            structural_geometry = None
        elif self.variant in ("code2vec_context_transform", "code2vec_context_transform_l1"):
            transformed_contexts = self._code2vec_context_transform(context_representations)
            attention = self._attention(transformed_contexts, batch.context_mask)
            representation = torch.sum(transformed_contexts * attention.unsqueeze(-1), dim=1)
            structural_points = None
            context_structural_embeddings = path_tangents
            context_structural_points = None
            context_tree_features = None
            structural_geometry = None
            structural_embedding_metric = "l1" if self.variant == "code2vec_context_transform_l1" else "l2"
        elif self.variant == "euclidean_metric":
            attention = self._euclidean_metric_attention(context_representations, batch.context_mask)
            representation = torch.sum(context_representations * attention.unsqueeze(-1), dim=1)
            structural_points = None
            context_structural_embeddings = context_representations
            context_structural_points = None
            context_tree_features = None
            structural_geometry = None
        elif self.variant == "bounded_euclidean_metric":
            bounded_contexts = torch_project_to_ball(context_representations, curvature=curvature, eps=self.config.eps)
            attention = self._bounded_euclidean_metric_attention(bounded_contexts, batch.context_mask, curvature)
            representation = torch_project_to_ball(
                torch.sum(bounded_contexts * attention.unsqueeze(-1), dim=1),
                curvature=curvature,
                eps=self.config.eps,
            )
            structural_points = None
            context_structural_embeddings = bounded_contexts
            context_structural_points = None
            context_tree_features = None
            structural_geometry = None
        elif self.variant == "euclidean_tree":
            attention, context_tree_features = self._tree_metric_attention(context_representations, batch)
            representation = torch.sum(context_representations * attention.unsqueeze(-1), dim=1)
            structural_points = None
            context_structural_embeddings = context_representations
            context_structural_points = None
            structural_geometry = None
        elif self.variant == "product":
            path_points = torch_expmap0(path_tangents, curvature=curvature, eps=self.config.eps)
            path_logs = torch_logmap0(path_points, curvature=curvature)
            context_representations = torch.cat([start_vectors, path_logs, end_vectors], dim=-1)
            attention = self._attention(context_representations, batch.context_mask)

            start_centroid = torch.sum(start_vectors * attention.unsqueeze(-1), dim=1)
            end_centroid = torch.sum(end_vectors * attention.unsqueeze(-1), dim=1)
            structural_points = torch_poincare_weighted_midpoint(
                path_points,
                attention,
                curvature=curvature,
                eps=self.config.eps,
            )
            structural_log = torch_logmap0(structural_points, curvature=curvature)
            representation = torch.cat([start_centroid, structural_log, end_centroid], dim=-1)
            context_structural_embeddings = path_logs
            context_structural_points = path_points
            context_tree_features = None
            structural_geometry = "poincare"
        elif self.variant in (
            "factorized_product",
            "factorized_product_learned_metric",
            "factorized_product_three_metric",
            "factorized_product_channel_mixer",
            "code2hyp_product_frechet",
        ):
            path_points = torch_expmap0(path_tangents, curvature=curvature, eps=self.config.eps)
            path_logs = torch_logmap0(path_points, curvature=curvature)
            attention = self._factorized_product_attention(
                start_vectors,
                path_points,
                end_vectors,
                batch.context_mask,
                curvature,
            )

            start_centroid = torch.sum(start_vectors * attention.unsqueeze(-1), dim=1)
            end_centroid = torch.sum(end_vectors * attention.unsqueeze(-1), dim=1)
            if self.variant == "code2hyp_product_frechet":
                structural_points = torch_poincare_frechet_mean(
                    path_points,
                    attention,
                    curvature=curvature,
                    steps=self.config.frechet_steps,
                    step_size=self.config.frechet_step_size,
                    eps=self.config.eps,
                )
            else:
                structural_points = torch_poincare_weighted_midpoint(
                    path_points,
                    attention,
                    curvature=curvature,
                    eps=self.config.eps,
                )
            structural_log = torch_logmap0(structural_points, curvature=curvature)
            representation = torch.cat([start_centroid, structural_log, end_centroid], dim=-1)
            representation = self._apply_factorized_channel_mixer(representation)
            context_structural_embeddings = path_logs
            context_structural_points = path_points
            context_tree_features = None
            structural_geometry = "poincare"
        elif self.variant == "code2hyp_code2vec_attention_frechet":
            path_points = torch_expmap0(path_tangents, curvature=curvature, eps=self.config.eps)
            path_logs = torch_logmap0(path_points, curvature=curvature)
            code2vec_contexts = torch.cat([start_vectors, path_logs, end_vectors], dim=-1)
            attention = self._attention(code2vec_contexts, batch.context_mask)

            start_centroid = torch.sum(start_vectors * attention.unsqueeze(-1), dim=1)
            end_centroid = torch.sum(end_vectors * attention.unsqueeze(-1), dim=1)
            structural_points = torch_poincare_frechet_mean(
                path_points,
                attention,
                curvature=curvature,
                steps=self.config.frechet_steps,
                step_size=self.config.frechet_step_size,
                eps=self.config.eps,
            )
            structural_log = torch_logmap0(structural_points, curvature=curvature)
            representation = torch.cat([start_centroid, structural_log, end_centroid], dim=-1)
            context_structural_embeddings = path_logs
            context_structural_points = path_points
            context_tree_features = None
            structural_geometry = "poincare"
        elif self.variant == "code2hyp_context_transform_frechet":
            path_points = torch_expmap0(path_tangents, curvature=curvature, eps=self.config.eps)
            path_logs = torch_logmap0(path_points, curvature=curvature)
            raw_contexts = torch.cat([start_vectors, path_logs, end_vectors], dim=-1)
            transformed_contexts = self._code2vec_context_transform(raw_contexts)
            attention = self._attention(transformed_contexts, batch.context_mask)

            start_centroid = torch.sum(start_vectors * attention.unsqueeze(-1), dim=1)
            end_centroid = torch.sum(end_vectors * attention.unsqueeze(-1), dim=1)
            structural_points = torch_poincare_frechet_mean(
                path_points,
                attention,
                curvature=curvature,
                steps=self.config.frechet_steps,
                step_size=self.config.frechet_step_size,
                eps=self.config.eps,
            )
            structural_log = torch_logmap0(structural_points, curvature=curvature)
            representation = torch.cat([start_centroid, structural_log, end_centroid], dim=-1)
            context_structural_embeddings = path_logs
            context_structural_points = path_points
            context_tree_features = None
            structural_geometry = "poincare"
        elif self.variant == "code2hyp_product_context_transform_frechet":
            path_points = torch_expmap0(path_tangents, curvature=curvature, eps=self.config.eps)
            path_logs = torch_logmap0(path_points, curvature=curvature)
            raw_contexts = torch.cat([start_vectors, path_logs, end_vectors], dim=-1)
            transformed_contexts = self._code2vec_context_transform(raw_contexts)
            attention = self._factorized_product_attention(
                start_vectors,
                path_points,
                end_vectors,
                batch.context_mask,
                curvature,
            )

            structural_points = torch_poincare_frechet_mean(
                path_points,
                attention,
                curvature=curvature,
                steps=self.config.frechet_steps,
                step_size=self.config.frechet_step_size,
                eps=self.config.eps,
            )
            representation = torch.sum(transformed_contexts * attention.unsqueeze(-1), dim=1)
            context_structural_embeddings = path_logs
            context_structural_points = path_points
            context_tree_features = None
            structural_geometry = "poincare"
        elif self.variant == "code2hyp_context_transform_product_bias_frechet":
            path_points = torch_expmap0(path_tangents, curvature=curvature, eps=self.config.eps)
            path_logs = torch_logmap0(path_points, curvature=curvature)
            raw_contexts = torch.cat([start_vectors, path_logs, end_vectors], dim=-1)
            transformed_contexts = self._code2vec_context_transform(raw_contexts)
            attention = self._code2hyp_context_product_bias_attention(
                transformed_contexts,
                start_vectors,
                path_points,
                end_vectors,
                batch.context_mask,
                curvature,
            )

            structural_points = torch_poincare_frechet_mean(
                path_points,
                attention,
                curvature=curvature,
                steps=self.config.frechet_steps,
                step_size=self.config.frechet_step_size,
                eps=self.config.eps,
            )
            representation = torch.sum(transformed_contexts * attention.unsqueeze(-1), dim=1)
            context_structural_embeddings = path_logs
            context_structural_points = path_points
            context_tree_features = None
            structural_geometry = "poincare"
        elif self.variant == "code2hyp_context_transform_branch_product_bias_frechet":
            path_points = torch_expmap0(path_tangents, curvature=curvature, eps=self.config.eps)
            path_logs = torch_logmap0(path_points, curvature=curvature)
            left_tangents, right_tangents = self._branch_path_tangents(batch.ast_paths, batch.ast_path_mask)
            left_points = torch_expmap0(left_tangents, curvature=curvature, eps=self.config.eps)
            right_points = torch_expmap0(right_tangents, curvature=curvature, eps=self.config.eps)
            raw_contexts = torch.cat([start_vectors, path_logs, end_vectors], dim=-1)
            transformed_contexts = self._code2vec_context_transform(raw_contexts)
            attention = self._code2hyp_branch_product_bias_attention(
                transformed_contexts,
                start_vectors,
                left_points,
                right_points,
                end_vectors,
                batch.context_mask,
                curvature,
            )

            left_structural_points = torch_poincare_frechet_mean(
                left_points,
                attention,
                curvature=curvature,
                steps=self.config.frechet_steps,
                step_size=self.config.frechet_step_size,
                eps=self.config.eps,
            )
            right_structural_points = torch_poincare_frechet_mean(
                right_points,
                attention,
                curvature=curvature,
                steps=self.config.frechet_steps,
                step_size=self.config.frechet_step_size,
                eps=self.config.eps,
            )
            representation = torch.sum(transformed_contexts * attention.unsqueeze(-1), dim=1)
            structural_points = None
            context_structural_embeddings = path_logs
            context_structural_points = None
            structural_product_points = (left_structural_points, right_structural_points)
            context_structural_product_points = (left_points, right_points)
            context_tree_features = None
            structural_geometry = "poincare_product"
        elif self.variant == "code2hyp_context_transform_branch_sequence_product_bias_frechet":
            path_points = torch_expmap0(path_tangents, curvature=curvature, eps=self.config.eps)
            path_logs = torch_logmap0(path_points, curvature=curvature)
            left_tangents, right_tangents = self._branch_sequence_path_tangents(
                batch.ast_paths,
                batch.ast_path_mask,
            )
            left_points = torch_expmap0(left_tangents, curvature=curvature, eps=self.config.eps)
            right_points = torch_expmap0(right_tangents, curvature=curvature, eps=self.config.eps)
            raw_contexts = torch.cat([start_vectors, path_logs, end_vectors], dim=-1)
            transformed_contexts = self._code2vec_context_transform(raw_contexts)
            attention = self._code2hyp_branch_product_bias_attention(
                transformed_contexts,
                start_vectors,
                left_points,
                right_points,
                end_vectors,
                batch.context_mask,
                curvature,
            )

            left_structural_points = torch_poincare_frechet_mean(
                left_points,
                attention,
                curvature=curvature,
                steps=self.config.frechet_steps,
                step_size=self.config.frechet_step_size,
                eps=self.config.eps,
            )
            right_structural_points = torch_poincare_frechet_mean(
                right_points,
                attention,
                curvature=curvature,
                steps=self.config.frechet_steps,
                step_size=self.config.frechet_step_size,
                eps=self.config.eps,
            )
            representation = torch.sum(transformed_contexts * attention.unsqueeze(-1), dim=1)
            structural_points = None
            context_structural_embeddings = path_logs
            context_structural_points = None
            structural_product_points = (left_structural_points, right_structural_points)
            context_structural_product_points = (left_points, right_points)
            context_tree_features = None
            structural_geometry = "poincare_product"
        elif self.variant in (
            "code2hyp_context_transform_latent_lca_branch_product_bias_frechet",
            "code2hyp_context_transform_latent_lca_prior_branch_product_bias_frechet",
        ):
            path_points = torch_expmap0(path_tangents, curvature=curvature, eps=self.config.eps)
            path_logs = torch_logmap0(path_points, curvature=curvature)
            left_tangents, right_tangents, pivot_attention = self._latent_lca_branch_path_tangents(
                batch.ast_paths,
                batch.ast_path_mask,
            )
            left_points = torch_expmap0(left_tangents, curvature=curvature, eps=self.config.eps)
            right_points = torch_expmap0(right_tangents, curvature=curvature, eps=self.config.eps)
            raw_contexts = torch.cat([start_vectors, path_logs, end_vectors], dim=-1)
            transformed_contexts = self._code2vec_context_transform(raw_contexts)
            attention = self._code2hyp_branch_product_bias_attention(
                transformed_contexts,
                start_vectors,
                left_points,
                right_points,
                end_vectors,
                batch.context_mask,
                curvature,
            )

            left_structural_points = torch_poincare_frechet_mean(
                left_points,
                attention,
                curvature=curvature,
                steps=self.config.frechet_steps,
                step_size=self.config.frechet_step_size,
                eps=self.config.eps,
            )
            right_structural_points = torch_poincare_frechet_mean(
                right_points,
                attention,
                curvature=curvature,
                steps=self.config.frechet_steps,
                step_size=self.config.frechet_step_size,
                eps=self.config.eps,
            )
            representation = torch.sum(transformed_contexts * attention.unsqueeze(-1), dim=1)
            structural_points = None
            context_structural_embeddings = path_logs
            context_structural_points = None
            structural_product_points = (left_structural_points, right_structural_points)
            context_structural_product_points = (left_points, right_points)
            context_tree_features = None
            structural_geometry = "poincare_product"
            path_node_attention = pivot_attention
        elif self.variant == "code2hyp_branch_product_context_transform_frechet":
            left_tangents, right_tangents = self._branch_path_tangents(batch.ast_paths, batch.ast_path_mask)
            left_points = torch_expmap0(left_tangents, curvature=curvature, eps=self.config.eps)
            right_points = torch_expmap0(right_tangents, curvature=curvature, eps=self.config.eps)
            left_logs = torch_logmap0(left_points, curvature=curvature)
            right_logs = torch_logmap0(right_points, curvature=curvature)
            path_logs = torch.cat([left_logs, right_logs], dim=-1)
            raw_contexts = torch.cat([start_vectors, path_logs, end_vectors], dim=-1)
            transformed_contexts = self._code2vec_context_transform(raw_contexts)
            attention = self._code2hyp_branch_product_bias_attention(
                transformed_contexts,
                start_vectors,
                left_points,
                right_points,
                end_vectors,
                batch.context_mask,
                curvature,
            )

            left_structural_points = torch_poincare_frechet_mean(
                left_points,
                attention,
                curvature=curvature,
                steps=self.config.frechet_steps,
                step_size=self.config.frechet_step_size,
                eps=self.config.eps,
            )
            right_structural_points = torch_poincare_frechet_mean(
                right_points,
                attention,
                curvature=curvature,
                steps=self.config.frechet_steps,
                step_size=self.config.frechet_step_size,
                eps=self.config.eps,
            )
            representation = torch.sum(transformed_contexts * attention.unsqueeze(-1), dim=1)
            structural_points = None
            context_structural_embeddings = path_logs
            context_structural_points = None
            structural_product_points = (left_structural_points, right_structural_points)
            context_structural_product_points = (left_points, right_points)
            context_tree_features = None
            structural_geometry = "poincare_product"
        elif self.variant == "lorentz_product":
            path_points = torch_lorentz_expmap0(path_tangents, curvature=curvature)
            path_logs = torch_lorentz_logmap0(path_points, curvature=curvature)
            attention = self._factorized_lorentz_product_attention(
                start_vectors,
                path_points,
                end_vectors,
                batch.context_mask,
                curvature,
            )

            start_centroid = torch.sum(start_vectors * attention.unsqueeze(-1), dim=1)
            end_centroid = torch.sum(end_vectors * attention.unsqueeze(-1), dim=1)
            structural_points = torch_lorentz_weighted_centroid(path_points, attention, curvature=curvature)
            structural_log = torch_lorentz_logmap0(structural_points, curvature=curvature)
            representation = torch.cat([start_centroid, structural_log, end_centroid], dim=-1)
            context_structural_embeddings = path_logs
            context_structural_points = path_points
            context_tree_features = None
            structural_geometry = "lorentz"
        elif self.variant in (
            "hyperbolic",
            "hyperbolic_path_message_passing",
            "hyperbolic_path_attention_message_passing",
            "hyperbolic_path_depth_attention_message_passing",
            "hyperbolic_path_dual_attention_message_passing",
        ):
            context_points = torch_expmap0(context_representations, curvature=curvature, eps=self.config.eps)
            attention = self._hyperbolic_attention(context_points, batch.context_mask, curvature)
            structural_points = torch_poincare_weighted_midpoint(
                context_points,
                attention,
                curvature=curvature,
                eps=self.config.eps,
            )
            representation = torch_logmap0(structural_points, curvature=curvature)
            context_structural_embeddings = torch_logmap0(context_points, curvature=curvature)
            context_structural_points = context_points
            context_tree_features = None
            structural_geometry = "poincare"
        elif self.variant == "lorentz_path_dual_attention_message_passing":
            context_points = torch_lorentz_expmap0(context_representations, curvature=curvature)
            attention = self._lorentz_attention(context_points, batch.context_mask, curvature)
            structural_points = torch_lorentz_weighted_centroid(context_points, attention, curvature=curvature)
            representation = torch_lorentz_logmap0(structural_points, curvature=curvature)
            context_structural_embeddings = torch_lorentz_logmap0(context_points, curvature=curvature)
            context_structural_points = context_points
            context_tree_features = None
            structural_geometry = "lorentz"
        elif self.variant == "hyperbolic_frechet":
            context_points = torch_expmap0(context_representations, curvature=curvature, eps=self.config.eps)
            attention = self._hyperbolic_attention(context_points, batch.context_mask, curvature)
            structural_points = torch_poincare_frechet_mean(
                context_points,
                attention,
                curvature=curvature,
                steps=self.config.frechet_steps,
                step_size=self.config.frechet_step_size,
                eps=self.config.eps,
            )
            representation = torch_logmap0(structural_points, curvature=curvature)
            context_structural_embeddings = torch_logmap0(context_points, curvature=curvature)
            context_structural_points = context_points
            context_tree_features = None
            structural_geometry = "poincare"
        elif self.variant == "hyperbolic_attention":
            context_points = torch_expmap0(context_representations, curvature=curvature, eps=self.config.eps)
            attention = self._hyperbolic_attention(context_points, batch.context_mask, curvature)
            context_logs = torch_logmap0(context_points, curvature=curvature)
            representation = torch.sum(context_logs * attention.unsqueeze(-1), dim=1)
            structural_points = None
            context_structural_embeddings = context_logs
            context_structural_points = context_points
            context_tree_features = None
            structural_geometry = "poincare"
        elif self.variant == "lorentz":
            context_points = torch_lorentz_expmap0(context_representations, curvature=curvature)
            attention = self._lorentz_attention(context_points, batch.context_mask, curvature)
            structural_points = torch_lorentz_weighted_centroid(context_points, attention, curvature=curvature)
            representation = torch_lorentz_logmap0(structural_points, curvature=curvature)
            context_structural_embeddings = torch_lorentz_logmap0(context_points, curvature=curvature)
            context_structural_points = context_points
            context_tree_features = None
            structural_geometry = "lorentz"
        else:
            raise RuntimeError(f"unhandled Code2Hyp variant: {self.variant}")

        representation = self._transform_representation(representation)
        logits = self.decoder(representation)
        return Code2HypTorchOutput(
            logits=logits,
            representation=representation,
            attention=attention,
            curvature=curvature,
            structural_points=structural_points,
            context_structural_embeddings=context_structural_embeddings,
            context_structural_points=context_structural_points,
            structural_product_points=structural_product_points,
            context_structural_product_points=context_structural_product_points,
            structural_product_distance_metric=structural_product_distance_metric,
            context_tree_features=context_tree_features,
            structural_geometry=structural_geometry,
            structural_embedding_metric=structural_embedding_metric,
            path_node_attention=path_node_attention,
            path_node_attention_pair=path_node_attention_pair,
            path_node_attention_monotonicity_loss=path_node_attention_monotonicity,
        )

    def _apply_factorized_channel_mixer(self, representation: Tensor) -> Tensor:
        if self.variant != "factorized_product_channel_mixer":
            return representation
        if self.factorized_channel_down is None or self.factorized_channel_up is None:
            raise RuntimeError("factorized channel mixer layers are not initialized")
        return representation + self.factorized_channel_up(torch.tanh(self.factorized_channel_down(representation)))

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)

    def relative_parameter_overhead(self, baseline: "Code2HypTorchModel") -> float:
        baseline_count = baseline.parameter_count()
        if baseline_count == 0:
            return 0.0
        return (self.parameter_count() - baseline_count) / baseline_count

    def _path_tangents(self, ast_paths: Tensor, ast_path_mask: Tensor) -> Tensor:
        embeddings = self.ast_node_embeddings(ast_paths)
        mask = ast_path_mask.unsqueeze(-1).to(dtype=embeddings.dtype)
        if self.ast_path_encoder is not None:
            batch_size, context_count, path_length, structural_dim = embeddings.shape
            flat_embeddings = (embeddings * mask).reshape(batch_size * context_count, path_length, structural_dim)
            encoded, _ = self.ast_path_encoder(flat_embeddings)
            lengths = torch.clamp(ast_path_mask.long().sum(dim=2), min=1)
            flat_lengths = lengths.reshape(batch_size * context_count)
            gather_index = (flat_lengths - 1).view(-1, 1, 1).expand(-1, 1, self.config.structural_dim)
            flat_output = torch.gather(encoded, dim=1, index=gather_index).squeeze(1)
            return flat_output.reshape(batch_size, context_count, self.config.structural_dim)
        summed = torch.sum(embeddings * mask, dim=2)
        counts = torch.clamp(mask.sum(dim=2), min=1.0)
        return summed / counts

    def _branch_path_tangents(self, ast_paths: Tensor, ast_path_mask: Tensor) -> tuple[Tensor, Tensor]:
        if self.branch_left_projection is None or self.branch_right_projection is None:
            raise RuntimeError("branch projections are only available for branch-product variants")
        embeddings = self.ast_node_embeddings(ast_paths)
        left_mask, right_mask = ast_path_midpoint_branch_masks(ast_path_mask)

        def pooled(mask: Tensor) -> Tensor:
            weights = mask.unsqueeze(-1).to(dtype=embeddings.dtype)
            summed = torch.sum(embeddings * weights, dim=2)
            counts = torch.clamp(weights.sum(dim=2), min=1.0)
            return summed / counts

        return self.branch_left_projection(pooled(left_mask)), self.branch_right_projection(pooled(right_mask))

    def _branch_sequence_path_tangents(self, ast_paths: Tensor, ast_path_mask: Tensor) -> tuple[Tensor, Tensor]:
        if self.branch_left_sequence_encoder is None or self.branch_right_sequence_encoder is None:
            raise RuntimeError("branch sequence encoders are only available for branch-sequence-product variants")
        embeddings = self.ast_node_embeddings(ast_paths)
        left_mask, right_mask = ast_path_midpoint_branch_masks(ast_path_mask)
        left_embeddings = torch.flip(embeddings, dims=(2,))
        left_mask_reversed = torch.flip(left_mask, dims=(2,))
        left_tangents = self._masked_branch_sequence_encoding(
            self.branch_left_sequence_encoder,
            left_embeddings,
            left_mask_reversed,
            self.branch_left_dim,
        )
        right_tangents = self._masked_branch_sequence_encoding(
            self.branch_right_sequence_encoder,
            embeddings,
            right_mask,
            self.branch_right_dim,
        )
        return left_tangents, right_tangents

    def _masked_branch_sequence_encoding(
        self,
        encoder: nn.GRU,
        embeddings: Tensor,
        mask: Tensor,
        output_dim: int,
    ) -> Tensor:
        weights = mask.unsqueeze(-1).to(dtype=embeddings.dtype)
        batch_size, context_count, path_length, structural_dim = embeddings.shape
        flat_embeddings = (embeddings * weights).reshape(batch_size * context_count, path_length, structural_dim)
        encoded, _ = encoder(flat_embeddings)
        positions = torch.arange(path_length, device=mask.device, dtype=torch.long).view(1, 1, path_length)
        last_indices = torch.where(mask, positions, torch.zeros_like(positions)).max(dim=2).values
        gather_index = last_indices.reshape(-1, 1, 1).expand(-1, 1, output_dim)
        flat_output = torch.gather(encoded, dim=1, index=gather_index).squeeze(1)
        valid = mask.any(dim=2).reshape(-1, 1).to(dtype=flat_output.dtype)
        flat_output = flat_output * valid
        return flat_output.reshape(batch_size, context_count, output_dim)

    def _latent_lca_branch_path_tangents(
        self,
        ast_paths: Tensor,
        ast_path_mask: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        if (
            self.branch_left_projection is None
            or self.branch_right_projection is None
            or self.branch_pivot_query is None
        ):
            raise RuntimeError("latent LCA branch encoding is only available for latent branch-product variants")
        embeddings = self.ast_node_embeddings(ast_paths)
        valid = ast_path_mask.to(dtype=embeddings.dtype)
        pivot_scores = torch.sum(embeddings * self.branch_pivot_query.view(1, 1, 1, -1), dim=-1)
        if self.raw_branch_pivot_center_prior_weight is not None:
            path_length = ast_path_mask.shape[-1]
            positions = torch.arange(path_length, dtype=embeddings.dtype, device=embeddings.device).view(1, 1, -1)
            lengths = torch.clamp(ast_path_mask.sum(dim=-1).to(dtype=embeddings.dtype), min=1.0)
            midpoint = (lengths - 1.0).unsqueeze(-1) / 2.0
            scale = torch.clamp(lengths.unsqueeze(-1), min=1.0)
            center_prior = -torch.abs(positions - midpoint) / scale
            pivot_scores = pivot_scores + self.branch_pivot_center_prior_weight() * center_prior
        pivot_scores = pivot_scores.masked_fill(~ast_path_mask, -1e9)
        pivot_attention = torch.softmax(pivot_scores, dim=-1) * valid
        pivot_attention = pivot_attention / torch.clamp(pivot_attention.sum(dim=-1, keepdim=True), min=1e-12)

        left_weights = torch.flip(
            torch.cumsum(torch.flip(pivot_attention, dims=(-1,)), dim=-1),
            dims=(-1,),
        )
        right_weights = torch.cumsum(pivot_attention, dim=-1)
        left_weights = left_weights * valid
        right_weights = right_weights * valid
        left_weights = left_weights / torch.clamp(left_weights.sum(dim=-1, keepdim=True), min=1e-12)
        right_weights = right_weights / torch.clamp(right_weights.sum(dim=-1, keepdim=True), min=1e-12)

        left_pooled = torch.sum(embeddings * left_weights.unsqueeze(-1), dim=2)
        right_pooled = torch.sum(embeddings * right_weights.unsqueeze(-1), dim=2)
        return (
            self.branch_left_projection(left_pooled),
            self.branch_right_projection(right_pooled),
            pivot_attention,
        )

    def _hyperbolic_path_message_tangents(
        self,
        ast_paths: Tensor,
        ast_path_mask: Tensor,
        curvature: Tensor,
    ) -> tuple[Tensor, Tensor | None, Tensor | None, Tensor | None]:
        if self.path_message_linear is None or self.path_update_linear is None:
            raise RuntimeError("path message passing layers are only available for hyperbolic_path_message_passing")
        embeddings = self.ast_node_embeddings(ast_paths)
        mask = ast_path_mask.unsqueeze(-1).to(dtype=embeddings.dtype)
        points = torch_expmap0(embeddings, curvature=curvature, eps=self.config.eps)

        for _ in range(self.config.path_message_passing_steps):
            logs = torch_logmap0(points, curvature=curvature)
            previous_logs = torch.zeros_like(logs)
            next_logs = torch.zeros_like(logs)
            previous_logs[:, :, 1:, :] = logs[:, :, :-1, :]
            next_logs[:, :, :-1, :] = logs[:, :, 1:, :]

            previous_mask = torch.zeros_like(ast_path_mask)
            next_mask = torch.zeros_like(ast_path_mask)
            previous_mask[:, :, 1:] = ast_path_mask[:, :, :-1]
            next_mask[:, :, :-1] = ast_path_mask[:, :, 1:]
            previous_weight = previous_mask.unsqueeze(-1).to(dtype=logs.dtype)
            next_weight = next_mask.unsqueeze(-1).to(dtype=logs.dtype)
            neighbor_count = torch.clamp(previous_weight + next_weight, min=1.0)
            neighbor_mean = (previous_logs * previous_weight + next_logs * next_weight) / neighbor_count

            message = self.path_message_linear(neighbor_mean)
            updated_logs = torch.tanh(self.path_update_linear(torch.cat([logs, message], dim=-1)))
            updated_logs = updated_logs * mask
            points = torch_expmap0(updated_logs, curvature=curvature, eps=self.config.eps)

        updated_node_logs = torch_logmap0(points, curvature=curvature) * mask
        if self.variant in (
            "hyperbolic_path_attention_message_passing",
            "hyperbolic_path_depth_attention_message_passing",
        ):
            pooled, attention_weights = self._path_node_attention_pool(updated_node_logs, ast_path_mask)
            monotonicity = path_node_attention_monotonicity_loss(attention_weights, ast_path_mask)
            return pooled, attention_weights, monotonicity, None

        if self.variant in (
            "hyperbolic_path_dual_attention_message_passing",
            "lorentz_path_dual_attention_message_passing",
        ):
            pooled, attention_pair = self._path_node_dual_attention_pool(updated_node_logs, ast_path_mask)
            averaged_attention = attention_pair.mean(dim=-2)
            return pooled, averaged_attention, None, attention_pair

        if self.ast_path_encoder is not None:
            batch_size, context_count, path_length, structural_dim = updated_node_logs.shape
            flat_embeddings = updated_node_logs.reshape(batch_size * context_count, path_length, structural_dim)
            encoded, _ = self.ast_path_encoder(flat_embeddings)
            lengths = torch.clamp(ast_path_mask.long().sum(dim=2), min=1)
            flat_lengths = lengths.reshape(batch_size * context_count)
            gather_index = (flat_lengths - 1).view(-1, 1, 1).expand(-1, 1, self.config.structural_dim)
            flat_output = torch.gather(encoded, dim=1, index=gather_index).squeeze(1)
            return flat_output.reshape(batch_size, context_count, self.config.structural_dim), None, None, None

        summed = torch.sum(updated_node_logs, dim=2)
        counts = torch.clamp(mask.sum(dim=2), min=1.0)
        return summed / counts, None, None, None

    def _path_node_attention_pool(self, node_logs: Tensor, ast_path_mask: Tensor) -> tuple[Tensor, Tensor]:
        if self.path_node_attention_query is None:
            raise RuntimeError("path-node attention query is only available for path-attention message passing")
        scores = torch.sum(node_logs * self.path_node_attention_query.view(1, 1, 1, -1), dim=-1)
        if self.variant == "hyperbolic_path_depth_attention_message_passing":
            if self.raw_path_depth_attention_bias is None:
                raise RuntimeError("depth attention bias is not initialized")
            path_length = ast_path_mask.shape[-1]
            positions = torch.arange(path_length, dtype=node_logs.dtype, device=node_logs.device).view(1, 1, -1)
            lengths = ast_path_mask.to(dtype=node_logs.dtype).sum(dim=-1, keepdim=True)
            denominator = torch.clamp(lengths - 1.0, min=1.0)
            centered_depth = 2.0 * positions / denominator - 1.0
            centered_depth = torch.where(lengths > 1.0, centered_depth, torch.zeros_like(centered_depth))
            scores = scores + self.raw_path_depth_attention_bias * centered_depth
        scores = scores.masked_fill(~ast_path_mask, -1e9)
        weights = torch.softmax(scores, dim=-1) * ast_path_mask.to(dtype=node_logs.dtype)
        weights = weights / torch.clamp(weights.sum(dim=-1, keepdim=True), min=1e-12)
        return torch.sum(node_logs * weights.unsqueeze(-1), dim=2), weights

    def _path_node_dual_attention_pool(self, node_logs: Tensor, ast_path_mask: Tensor) -> tuple[Tensor, Tensor]:
        if (
            self.path_root_attention_query is None
            or self.path_leaf_attention_query is None
            or self.path_dual_attention_projection is None
        ):
            raise RuntimeError("dual path-node attention layers are not initialized")

        path_length = ast_path_mask.shape[-1]
        positions = torch.arange(path_length, dtype=node_logs.dtype, device=node_logs.device).view(1, 1, -1)
        lengths = ast_path_mask.to(dtype=node_logs.dtype).sum(dim=-1, keepdim=True)
        denominator = torch.clamp(lengths - 1.0, min=1.0)
        centered_depth = 2.0 * positions / denominator - 1.0
        centered_depth = torch.where(lengths > 1.0, centered_depth, torch.zeros_like(centered_depth))

        root_scores = torch.sum(node_logs * self.path_root_attention_query.view(1, 1, 1, -1), dim=-1)
        leaf_scores = torch.sum(node_logs * self.path_leaf_attention_query.view(1, 1, 1, -1), dim=-1)
        root_scores = root_scores - centered_depth
        leaf_scores = leaf_scores + centered_depth
        scores = torch.stack([root_scores, leaf_scores], dim=-2)
        scores = scores.masked_fill(~ast_path_mask.unsqueeze(-2), -1e9)
        weights = torch.softmax(scores, dim=-1) * ast_path_mask.unsqueeze(-2).to(dtype=node_logs.dtype)
        weights = weights / torch.clamp(weights.sum(dim=-1, keepdim=True), min=1e-12)
        pooled = torch.sum(node_logs.unsqueeze(-3) * weights.unsqueeze(-1), dim=-2)
        merged = torch.tanh(self.path_dual_attention_projection(torch.cat([pooled[..., 0, :], pooled[..., 1, :]], dim=-1)))
        return merged, weights

    def _attention(self, context_representations: Tensor, context_mask: Tensor) -> Tensor:
        scores = torch.sum(context_representations * self.attention_query, dim=-1)
        scores = scores.masked_fill(~context_mask, -torch.inf)
        return torch.softmax(scores, dim=1)

    def _code2vec_context_transform(self, context_representations: Tensor) -> Tensor:
        if self.context_transform_layer is None:
            raise RuntimeError("code2vec context transform layer is not initialized")
        return torch.tanh(self.context_transform_layer(context_representations))

    def _euclidean_metric_attention(self, context_representations: Tensor, context_mask: Tensor) -> Tensor:
        scores = self._euclidean_metric_scores(context_representations)
        scores = scores.masked_fill(~context_mask, -torch.inf)
        return torch.softmax(scores, dim=1)

    def _euclidean_metric_scores(self, context_representations: Tensor) -> Tensor:
        query = self.attention_query.view(1, 1, -1)
        distances2 = torch.sum((context_representations - query) ** 2, dim=-1)
        return -distances2

    def _bounded_euclidean_metric_attention(
        self,
        bounded_contexts: Tensor,
        context_mask: Tensor,
        curvature: Tensor,
    ) -> Tensor:
        bounded_query = torch_project_to_ball(
            self.attention_query.view(1, -1),
            curvature=curvature,
            eps=self.config.eps,
        )[0]
        distances2 = torch.sum((bounded_contexts - bounded_query.view(1, 1, -1)) ** 2, dim=-1)
        scores = -distances2
        scores = scores.masked_fill(~context_mask, -torch.inf)
        return torch.softmax(scores, dim=1)

    def _factorized_product_attention(
        self,
        start_vectors: Tensor,
        path_points: Tensor,
        end_vectors: Tensor,
        context_mask: Tensor,
        curvature: Tensor,
    ) -> Tensor:
        scores = self._factorized_product_scores(start_vectors, path_points, end_vectors, curvature)
        scores = scores.masked_fill(~context_mask, -torch.inf)
        return torch.softmax(scores, dim=1)

    def _factorized_product_scores(
        self,
        start_vectors: Tensor,
        path_points: Tensor,
        end_vectors: Tensor,
        curvature: Tensor,
    ) -> Tensor:
        token_dim = self.config.token_dim
        structural_dim = self.config.structural_dim
        query_start = self.attention_query[:token_dim].view(1, 1, token_dim)
        query_path_tangent = self.attention_query[token_dim : token_dim + structural_dim].view(1, structural_dim)
        query_end = self.attention_query[token_dim + structural_dim :].view(1, 1, token_dim)
        query_path = torch_expmap0(query_path_tangent, curvature=curvature, eps=self.config.eps)[0]
        query_paths = query_path.view(1, 1, structural_dim).expand_as(path_points)

        start_distances2 = torch.sum((start_vectors - query_start) ** 2, dim=-1)
        end_distances2 = torch.sum((end_vectors - query_end) ** 2, dim=-1)
        path_distances = torch_poincare_distance(path_points, query_paths, curvature=curvature)
        lexical_distances2 = start_distances2 + end_distances2
        path_distances2 = path_distances.square()
        if self.variant in (
            "factorized_product_three_metric",
            "code2hyp_product_frechet",
            "code2hyp_product_context_transform_frechet",
            "code2hyp_context_transform_product_bias_frechet",
        ):
            metric_weights = self.factorized_metric_weights()
            scores = -(
                metric_weights[0] * start_distances2
                + metric_weights[1] * path_distances2
                + metric_weights[2] * end_distances2
            )
        elif self.variant == "factorized_product_learned_metric":
            metric_weights = self.factorized_metric_weights()
            scores = -(metric_weights[0] * lexical_distances2 + metric_weights[1] * path_distances2)
        else:
            scores = -(lexical_distances2 + path_distances2)
        return scores

    def _code2hyp_context_product_bias_attention(
        self,
        transformed_contexts: Tensor,
        start_vectors: Tensor,
        path_points: Tensor,
        end_vectors: Tensor,
        context_mask: Tensor,
        curvature: Tensor,
    ) -> Tensor:
        semantic_scores = torch.sum(transformed_contexts * self.attention_query, dim=-1)
        product_scores = self._factorized_product_scores(start_vectors, path_points, end_vectors, curvature)
        scores = semantic_scores + self.product_attention_bias_weight() * product_scores
        scores = scores.masked_fill(~context_mask, -torch.inf)
        return torch.softmax(scores, dim=1)

    def _code2hyp_branch_product_bias_attention(
        self,
        transformed_contexts: Tensor,
        start_vectors: Tensor,
        left_path_points: Tensor,
        right_path_points: Tensor,
        end_vectors: Tensor,
        context_mask: Tensor,
        curvature: Tensor,
    ) -> Tensor:
        semantic_scores = torch.sum(transformed_contexts * self.attention_query, dim=-1)
        product_scores = self._branch_product_scores(
            start_vectors,
            left_path_points,
            right_path_points,
            end_vectors,
            curvature,
        )
        scores = semantic_scores + self.product_attention_bias_weight() * product_scores
        scores = scores.masked_fill(~context_mask, -torch.inf)
        return torch.softmax(scores, dim=1)

    def _branch_product_scores(
        self,
        start_vectors: Tensor,
        left_path_points: Tensor,
        right_path_points: Tensor,
        end_vectors: Tensor,
        curvature: Tensor,
    ) -> Tensor:
        token_dim = self.config.token_dim
        structural_start = token_dim
        structural_end = token_dim + self.config.structural_dim
        query_start = self.attention_query[:token_dim].view(1, 1, token_dim)
        query_structural = self.attention_query[structural_start:structural_end]
        query_left_tangent = query_structural[: self.branch_left_dim].view(1, self.branch_left_dim)
        query_right_tangent = query_structural[self.branch_left_dim :].view(1, self.branch_right_dim)
        query_end = self.attention_query[structural_end:].view(1, 1, token_dim)

        query_left = torch_expmap0(query_left_tangent, curvature=curvature, eps=self.config.eps)[0]
        query_right = torch_expmap0(query_right_tangent, curvature=curvature, eps=self.config.eps)[0]
        query_left_points = query_left.view(1, 1, self.branch_left_dim).expand_as(left_path_points)
        query_right_points = query_right.view(1, 1, self.branch_right_dim).expand_as(right_path_points)

        start_distances2 = torch.sum((start_vectors - query_start) ** 2, dim=-1)
        end_distances2 = torch.sum((end_vectors - query_end) ** 2, dim=-1)
        left_distances2 = torch_poincare_distance(left_path_points, query_left_points, curvature=curvature).square()
        right_distances2 = torch_poincare_distance(right_path_points, query_right_points, curvature=curvature).square()
        metric_weights = self.factorized_metric_weights()
        if metric_weights.numel() != 4:
            raise RuntimeError("branch-product metric requires four component weights")
        return -(
            metric_weights[0] * start_distances2
            + metric_weights[1] * left_distances2
            + metric_weights[2] * right_distances2
            + metric_weights[3] * end_distances2
        )

    def _factorized_lorentz_product_attention(
        self,
        start_vectors: Tensor,
        path_points: Tensor,
        end_vectors: Tensor,
        context_mask: Tensor,
        curvature: Tensor,
    ) -> Tensor:
        token_dim = self.config.token_dim
        structural_dim = self.config.structural_dim
        query_start = self.attention_query[:token_dim].view(1, 1, token_dim)
        query_path_tangent = self.attention_query[token_dim : token_dim + structural_dim].view(1, structural_dim)
        query_end = self.attention_query[token_dim + structural_dim :].view(1, 1, token_dim)
        query_path = torch_lorentz_expmap0(query_path_tangent, curvature=curvature)[0]
        query_paths = query_path.view(1, 1, structural_dim + 1).expand_as(path_points)

        start_distances2 = torch.sum((start_vectors - query_start) ** 2, dim=-1)
        end_distances2 = torch.sum((end_vectors - query_end) ** 2, dim=-1)
        path_distances = torch_lorentz_distance(path_points, query_paths, curvature=curvature)
        scores = -(start_distances2 + path_distances.square() + end_distances2)
        scores = scores.masked_fill(~context_mask, -torch.inf)
        return torch.softmax(scores, dim=1)

    def _tree_metric_attention(self, context_representations: Tensor, batch: Code2HypBatch) -> tuple[Tensor, Tensor]:
        if self.tree_feature_projection is None:
            raise RuntimeError("tree_feature_projection is only available for euclidean_tree variant")
        features = tree_context_features(batch).to(dtype=context_representations.dtype, device=context_representations.device)
        scores = self._euclidean_metric_scores(context_representations)
        scores = scores + self.tree_feature_projection(features).squeeze(-1)
        context_mask = batch.context_mask
        scores = scores.masked_fill(~context_mask, -torch.inf)
        return torch.softmax(scores, dim=1), features

    def _hyperbolic_attention(self, context_points: Tensor, context_mask: Tensor, curvature: Tensor) -> Tensor:
        query_point = torch_expmap0(self.attention_query.view(1, -1), curvature=curvature, eps=self.config.eps)[0]
        query_points = query_point.view(1, 1, -1).expand_as(context_points)
        distances = torch_poincare_distance(context_points, query_points, curvature=curvature)
        scores = -(distances * distances)
        scores = scores.masked_fill(~context_mask, -torch.inf)
        return torch.softmax(scores, dim=1)

    def _lorentz_attention(self, context_points: Tensor, context_mask: Tensor, curvature: Tensor) -> Tensor:
        query_point = torch_lorentz_expmap0(self.attention_query.view(1, -1), curvature=curvature)[0]
        query_points = query_point.view(1, 1, -1).expand_as(context_points)
        distances = torch_lorentz_distance(context_points, query_points, curvature=curvature)
        scores = -(distances * distances)
        scores = scores.masked_fill(~context_mask, -torch.inf)
        return torch.softmax(scores, dim=1)

    def _transform_representation(self, representation: Tensor) -> Tensor:
        if self.representation_transform_layer is None:
            return representation
        return torch.tanh(self.representation_transform_layer(representation))
