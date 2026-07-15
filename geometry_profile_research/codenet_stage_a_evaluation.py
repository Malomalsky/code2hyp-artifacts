from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor

from geometry_profile_research.batched_transport import (
    batched_role_product_cost,
    batched_sinkhorn_transport_objective,
    debiased_objective_from_self_terms,
)
from geometry_profile_research.constant_curvature import ProductMeasure, RoleProductGeometry


@dataclass(frozen=True)
class CalibrationSummary:
    """Train-only robust scale with equal weight per sampled program pair."""

    cost_scale: float
    pair_count: int
    positive_pair_count: int
    same_cluster_pair_count: int
    cross_cluster_pair_count: int
    aggregation: str = "median_of_pairwise_positive_cost_medians"


@dataclass(frozen=True)
class RoleWeightCalibration:
    euclidean_weights: tuple[float, float, float]
    canonical_weights: tuple[float, float, float]
    lca_scale: float
    start_scale: float
    end_scale: float
    pooled_endpoint_scale: float
    pair_count: int


@dataclass(frozen=True)
class RetrievalSummary:
    problem_macro_map_at_r: float
    query_macro_map_at_r: float
    mrr: float
    recall_at_1: float
    recall_at_5: float
    recall_at_10: float
    mean_first_relevant_rank: float
    query_count: int
    problem_count: int
    task_scores: dict[str, float]
    query_scores: dict[str, float]


def calibration_cost_scale(
    measures_by_id: Mapping[str, ProductMeasure],
    calibration_pairs: Sequence[Mapping[str, Any]],
    geometry: RoleProductGeometry,
    *,
    batch_size: int = 64,
) -> CalibrationSummary:
    """Estimate one robust cost scale from frozen training pairs only.

    Every sampled program pair first contributes the median of its positive
    path-to-path costs. The final scale is the median across program pairs, so
    large ASTs do not receive greater weight merely because they contain more
    sampled paths.
    """

    if not calibration_pairs:
        raise ValueError("calibration_pairs must not be empty")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    pair_medians: list[Tensor] = []
    same_count = 0
    cross_count = 0
    for start in range(0, len(calibration_pairs), batch_size):
        rows = calibration_pairs[start : start + batch_size]
        left = []
        right = []
        for row in rows:
            if row.get("pair_type") == "same_cluster":
                same_count += 1
            elif row.get("pair_type") == "cross_cluster":
                cross_count += 1
            else:
                raise ValueError("calibration pair has an unknown pair_type")
            left_id = str(row["left_source_relpath"])
            right_id = str(row["right_source_relpath"])
            try:
                left.append(measures_by_id[left_id])
                right.append(measures_by_id[right_id])
            except KeyError as error:
                raise ValueError(f"calibration measure is missing: {error.args[0]}") from error
        problem = batched_role_product_cost(geometry, left, right)
        for index, (left_size, right_size) in enumerate(zip(problem.left_sizes, problem.right_sizes)):
            values = problem.cost[index, :left_size, :right_size].detach().reshape(-1)
            positive = values[torch.isfinite(values) & (values > 0.0)]
            if positive.numel():
                pair_medians.append(torch.quantile(positive, 0.5).to(dtype=torch.float64, device="cpu"))
    if not pair_medians:
        raise ValueError("calibration pairs contain no positive finite costs")
    scale = float(torch.quantile(torch.stack(pair_medians), 0.5))
    if not torch.isfinite(torch.tensor(scale)) or scale <= 0.0:
        raise ValueError("calibration produced a non-positive cost scale")
    return CalibrationSummary(
        cost_scale=scale,
        pair_count=len(calibration_pairs),
        positive_pair_count=len(pair_medians),
        same_cluster_pair_count=same_count,
        cross_cluster_pair_count=cross_count,
    )


def calibrate_euclidean_role_weights(
    measures_by_id: Mapping[str, ProductMeasure],
    calibration_pairs: Sequence[Mapping[str, Any]],
    *,
    batch_size: int = 64,
) -> RoleWeightCalibration:
    """Calibrate equal-budget LCA/start/end roles on frozen training pairs."""

    scales = []
    for factor_index in range(3):
        geometry = RoleProductGeometry(
            factor_curvatures=(0.0, 0.0, 0.0),
            factor_weights=tuple(1.0 if index == factor_index else 0.0 for index in range(3)),
            side_weight=0.0,
            unoriented=False,
        )
        scales.append(
            calibration_cost_scale(
                measures_by_id,
                calibration_pairs,
                geometry,
                batch_size=batch_size,
            ).cost_scale
        )
    endpoint_scale = float(torch.quantile(torch.tensor(scales[1:], dtype=torch.float64), 0.5))
    if scales[0] <= 0.0 or endpoint_scale <= 0.0:
        raise ValueError("role calibration requires positive LCA and endpoint scales")
    euclidean = (
        1.0 / (3.0 * scales[0]),
        1.0 / (3.0 * endpoint_scale),
        1.0 / (3.0 * endpoint_scale),
    )
    return RoleWeightCalibration(
        euclidean_weights=euclidean,
        canonical_weights=tuple(weight / 4.0 for weight in euclidean),
        lca_scale=scales[0],
        start_scale=scales[1],
        end_scale=scales[2],
        pooled_endpoint_scale=endpoint_scale,
        pair_count=len(calibration_pairs),
    )


def precompute_self_objectives(
    measures: Sequence[ProductMeasure],
    geometry: RoleProductGeometry,
    *,
    epsilon: float,
    batch_size: int = 64,
    sinkhorn_iterations: int = 128,
    projection_iterations: int = 2048,
    marginal_tolerance: float = 1e-7,
) -> Tensor:
    """Compute one reusable regularized self-transport objective per measure."""

    if not measures:
        raise ValueError("measures must not be empty")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    values = []
    for start in range(0, len(measures), batch_size):
        chunk = tuple(measures[start : start + batch_size])
        problems = batched_role_product_cost(geometry, chunk, chunk)
        values.append(
            batched_sinkhorn_transport_objective(
                problems.cost,
                problems.left_mass,
                problems.right_mass,
                epsilon=epsilon,
                iterations=sinkhorn_iterations,
                projection_iterations=projection_iterations,
                marginal_tolerance=marginal_tolerance,
            )
        )
    return torch.cat(values)


def full_gallery_sinkhorn_divergence(
    queries: Sequence[ProductMeasure],
    gallery: Sequence[ProductMeasure],
    geometry: RoleProductGeometry,
    *,
    epsilon: float,
    query_batch_size: int = 4,
    gallery_batch_size: int = 32,
    sinkhorn_iterations: int = 128,
    projection_iterations: int = 2048,
    marginal_tolerance: float = 1e-7,
    query_self_objectives: Tensor | None = None,
    gallery_self_objectives: Tensor | None = None,
) -> Tensor:
    """Evaluate exact Sinkhorn divergences for every query-gallery pair."""

    if not queries or not gallery:
        raise ValueError("queries and gallery must not be empty")
    if query_batch_size <= 0 or gallery_batch_size <= 0:
        raise ValueError("query_batch_size and gallery_batch_size must be positive")
    query_self = (
        precompute_self_objectives(
            queries,
            geometry,
            epsilon=epsilon,
            batch_size=max(query_batch_size, 1),
            sinkhorn_iterations=sinkhorn_iterations,
            projection_iterations=projection_iterations,
            marginal_tolerance=marginal_tolerance,
        )
        if query_self_objectives is None
        else _validated_self_values(query_self_objectives, expected=len(queries), name="query_self_objectives")
    )
    gallery_self = (
        precompute_self_objectives(
            gallery,
            geometry,
            epsilon=epsilon,
            batch_size=max(gallery_batch_size, 1),
            sinkhorn_iterations=sinkhorn_iterations,
            projection_iterations=projection_iterations,
            marginal_tolerance=marginal_tolerance,
        )
        if gallery_self_objectives is None
        else _validated_self_values(gallery_self_objectives, expected=len(gallery), name="gallery_self_objectives")
    )
    device = queries[0].points.device
    scores = torch.empty((len(queries), len(gallery)), dtype=torch.float64, device=device)
    query_self = query_self.to(dtype=torch.float64, device=device)
    gallery_self = gallery_self.to(dtype=torch.float64, device=device)

    for query_start in range(0, len(queries), query_batch_size):
        query_chunk = tuple(queries[query_start : query_start + query_batch_size])
        for gallery_start in range(0, len(gallery), gallery_batch_size):
            gallery_chunk = tuple(gallery[gallery_start : gallery_start + gallery_batch_size])
            left_pairs = tuple(query for query in query_chunk for _ in gallery_chunk)
            right_pairs = gallery_chunk * len(query_chunk)
            problems = batched_role_product_cost(geometry, left_pairs, right_pairs)
            cross = batched_sinkhorn_transport_objective(
                problems.cost,
                problems.left_mass,
                problems.right_mass,
                epsilon=epsilon,
                iterations=sinkhorn_iterations,
                projection_iterations=projection_iterations,
                marginal_tolerance=marginal_tolerance,
            )
            left_self = query_self[query_start : query_start + len(query_chunk)].repeat_interleave(len(gallery_chunk))
            right_self = gallery_self[gallery_start : gallery_start + len(gallery_chunk)].repeat(len(query_chunk))
            block = debiased_objective_from_self_terms(cross, left_self, right_self)
            scores[
                query_start : query_start + len(query_chunk),
                gallery_start : gallery_start + len(gallery_chunk),
            ] = block.reshape(len(query_chunk), len(gallery_chunk))
    return scores


def summarize_problem_macro_retrieval(
    distances: Tensor,
    *,
    query_ids: Sequence[str],
    query_cluster_ids: Sequence[str],
    gallery_ids: Sequence[str],
    gallery_cluster_ids: Sequence[str],
    r: int = 8,
    require_exact_relevant_count: bool = True,
) -> RetrievalSummary:
    """Summarize distances with equal weight per duplicate-closed problem cluster."""

    values = torch.as_tensor(distances, dtype=torch.float64, device="cpu")
    if values.shape != (len(query_ids), len(gallery_ids)):
        raise ValueError("distance matrix shape does not match query/gallery metadata")
    if len(query_ids) != len(query_cluster_ids) or len(gallery_ids) != len(gallery_cluster_ids):
        raise ValueError("each item must have one duplicate-closed problem-cluster ID")
    if len(query_ids) != len(set(query_ids)) or len(gallery_ids) != len(set(gallery_ids)):
        raise ValueError("query and gallery IDs must be unique")
    if not torch.isfinite(values).all():
        raise ValueError("distance matrix must contain only finite values")
    if r <= 0:
        raise ValueError("r must be positive")

    task_ap: dict[str, list[float]] = {}
    query_scores: dict[str, float] = {}
    first_ranks: list[int] = []
    for query_index, (query_id, cluster_id) in enumerate(zip(query_ids, query_cluster_ids)):
        relevant = [
            index
            for index, candidate_cluster in enumerate(gallery_cluster_ids)
            if candidate_cluster == cluster_id
        ]
        if not relevant:
            raise ValueError(f"query {query_id!r} has no relevant gallery item")
        if require_exact_relevant_count and len(relevant) != r:
            raise ValueError(
                f"query {query_id!r} has {len(relevant)} relevant items; the frozen protocol requires exactly {r}"
            )
        relevant_set = set(relevant)
        order = sorted(
            range(len(gallery_ids)),
            key=lambda index: (float(values[query_index, index]), gallery_ids[index].encode("utf-8")),
        )
        ranked_relevance = tuple(index in relevant_set for index in order)
        average_precision = average_precision_at_r(
            ranked_relevance,
            total_positives=len(relevant),
            r=r,
        )
        first_rank = next(rank for rank, is_relevant in enumerate(ranked_relevance, start=1) if is_relevant)
        task_ap.setdefault(cluster_id, []).append(average_precision)
        query_scores[query_id] = average_precision
        first_ranks.append(first_rank)

    task_scores = {
        cluster_id: sum(scores) / len(scores)
        for cluster_id, scores in sorted(task_ap.items())
    }
    return RetrievalSummary(
        problem_macro_map_at_r=sum(task_scores.values()) / len(task_scores),
        query_macro_map_at_r=sum(query_scores.values()) / len(query_scores),
        mrr=sum(1.0 / rank for rank in first_ranks) / len(first_ranks),
        recall_at_1=sum(rank <= 1 for rank in first_ranks) / len(first_ranks),
        recall_at_5=sum(rank <= 5 for rank in first_ranks) / len(first_ranks),
        recall_at_10=sum(rank <= 10 for rank in first_ranks) / len(first_ranks),
        mean_first_relevant_rank=sum(first_ranks) / len(first_ranks),
        query_count=len(query_ids),
        problem_count=len(task_scores),
        task_scores=task_scores,
        query_scores=query_scores,
    )


def average_precision_at_r(
    ranked_relevance: Sequence[bool],
    *,
    total_positives: int,
    r: int,
) -> float:
    """Average precision truncated at R with a fixed relevant denominator."""

    denominator = min(int(r), int(total_positives))
    if denominator <= 0:
        return 0.0
    hits = 0
    precision_sum = 0.0
    for rank, relevant in enumerate(ranked_relevance[:r], start=1):
        if relevant:
            hits += 1
            precision_sum += hits / rank
    return precision_sum / denominator


def _validated_self_values(values: Tensor, *, expected: int, name: str) -> Tensor:
    result = torch.as_tensor(values)
    if result.ndim != 1 or result.numel() != expected:
        raise ValueError(f"{name} must contain {expected} values")
    if not torch.isfinite(result).all():
        raise ValueError(f"{name} must contain only finite values")
    return result
