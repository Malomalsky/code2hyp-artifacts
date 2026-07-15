from __future__ import annotations

import argparse
import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Sequence

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from geometry_profile_research.constant_curvature import (
    ConcatenatedEuclideanGeometry,
    ProductMeasure,
    RoleProductGeometry,
    median_positive_cost_scale,
    scaled_sinkhorn_epsilon,
)
from geometry_profile_research.raw_ast import terminal_to_terminal_paths
from geometry_profile_research.raw_ast_code2hyp import AnchorMode, RawASTCode2Hyp
from geometry_profile_research.raw_ast_retrieval import ItemScope, PositiveMode
from scripts.run_dta_factor_matrix import _encode_split, _set_reproducible_seeds, _train_shared_encoder
from scripts.run_dta_level_b_c_retrieval import (
    BenchmarkLevel,
    LabeledItem,
    SplitItems,
    TaskSource,
    _candidate_indices,
    _collect_labeled_items,
    _split_items,
)


Treatment = Literal[
    "true_lca",
    "zero_anchor",
    "root_anchor",
    "depth_matched_shuffled",
    "program_shuffled_lca",
    "full_path_no_explicit_lca",
    "endpoint_only",
]
GeometryName = Literal["EEE", "HEE_near_zero", "HEE", "HHH"]
StudyStage = Literal["pilot", "confirmatory"]

DEFAULT_TREATMENTS: tuple[Treatment, ...] = (
    "true_lca",
    "zero_anchor",
    "root_anchor",
    "depth_matched_shuffled",
    "program_shuffled_lca",
    "full_path_no_explicit_lca",
    "endpoint_only",
)
DEFAULT_GEOMETRIES: tuple[GeometryName, ...] = ("EEE", "HEE_near_zero", "HEE", "HHH")


@dataclass(frozen=True)
class GeometrySpec:
    name: GeometryName
    factor_curvatures: tuple[float, float, float]


GEOMETRY_SPECS: dict[GeometryName, GeometrySpec] = {
    "EEE": GeometrySpec("EEE", (0.0, 0.0, 0.0)),
    "HEE_near_zero": GeometrySpec("HEE_near_zero", (1e-4, 0.0, 0.0)),
    "HEE": GeometrySpec("HEE", (1.0, 0.0, 0.0)),
    "HHH": GeometrySpec("HHH", (1.0, 1.0, 1.0)),
}


def run_lca_causal_matrix(
    *,
    tasks: Sequence[TaskSource],
    output_path: Path,
    benchmark_level: BenchmarkLevel = "B_independent_solution",
    treatments: Sequence[Treatment] = DEFAULT_TREATMENTS,
    geometries: Sequence[GeometryName] = DEFAULT_GEOMETRIES,
    language: Literal["auto", "java", "python"] = "python",
    dim: int = 4,
    epochs: int = 2,
    learning_rate: float = 1e-2,
    max_files_per_task: int | None = 128,
    max_methods_per_task: int | None = 48,
    train_per_task: int = 16,
    query_per_task: int = 8,
    gallery_per_task: int = 8,
    max_paths: int = 64,
    seed: int = 20260625,
    min_structural_gap: float = 0.05,
    positive_mode: PositiveMode = "alpha_structural_noop",
    item_scope: ItemScope = "callable",
    sinkhorn_iterations: int = 8,
    sinkhorn_projection_iterations: int = 512,
    kappa: float = 0.05,
    max_ball_fraction: float = 0.35,
    hard_negatives_per_query: int = 16,
    side_block_weight: float = 0.0,
    ap_at_r: int = 8,
    path_selection_policy: str = "lca_depth_stratified",
    study_stage: StudyStage = "pilot",
) -> dict[str, Any]:
    """Run Gate A and Gate C with one neutral encoder and frozen cost scales.

    Gate A varies only the LCA treatment in Euclidean geometry. Gate C fixes the
    true-LCA measure and varies the role-specific geometry. Cost normalization,
    coordinate scaling, the split, and the encoder are estimated once from the
    training split and then frozen across every cell.
    """

    _validate_design(treatments, geometries, side_block_weight=side_block_weight, ap_at_r=ap_at_r)
    if study_stage == "confirmatory" and (max_paths != 64 or ap_at_r != 8 or gallery_per_task < 8):
        raise ValueError("confirmatory stage requires max_paths=64, ap_at_r=8, and gallery_per_task>=8")
    reproducibility = _set_reproducible_seeds(seed)
    labeled = _collect_labeled_items(
        tasks,
        language=language,
        max_files_per_task=max_files_per_task,
        max_methods_per_task=max_methods_per_task,
        max_paths=max_paths,
        item_scope=item_scope,
    )
    split = _split_items(
        labeled,
        train_per_task=train_per_task,
        query_per_task=query_per_task,
        gallery_per_task=gallery_per_task,
        seed=seed,
    )
    model, training_history = _train_shared_encoder(
        split.train,
        dim=dim,
        max_paths=max_paths,
        epochs=epochs,
        learning_rate=learning_rate,
        min_structural_gap=min_structural_gap,
        positive_mode=positive_mode,
        sinkhorn_iterations=sinkhorn_iterations,
        manifold="euclidean",
        path_object_mode="lca_product",
        method_aggregation="measure",
        path_selection_policy=path_selection_policy,
        lambda_retrieval=0.0,
        training_seed=seed,
    )

    encoded: dict[Treatment, dict[str, list[ProductMeasure]]] = {}
    treatment_diagnostics: dict[str, Any] = {}
    for treatment in treatments:
        treatment_payload, diagnostics = _encode_treatment(
            model,
            split,
            treatment=treatment,
            max_paths=max_paths,
            seed=seed,
        )
        encoded[treatment] = treatment_payload
        if diagnostics:
            treatment_diagnostics[treatment] = diagnostics
    true_lca = encoded["true_lca"]
    specs = tuple(GEOMETRY_SPECS[name] for name in geometries)
    role_scales = _role_point_scales(
        true_lca["train"],
        max_curvatures=tuple(max(spec.factor_curvatures[index] for spec in specs) for index in range(3)),
        max_ball_fraction=max_ball_fraction,
    )
    encoded = {
        treatment: {
            split_name: [_scale_measure_by_role(measure, role_scales=role_scales) for measure in measures]
            for split_name, measures in payload.items()
        }
        for treatment, payload in encoded.items()
    }
    true_lca = encoded["true_lca"]

    euclidean_weights, normalization = _frozen_role_weights(
        true_lca["train"],
        side_block_weight=side_block_weight,
    )
    canonical_weights = tuple(weight / 4.0 for weight in euclidean_weights)
    base_geometry = RoleProductGeometry(
        factor_curvatures=(0.0, 0.0, 0.0),
        factor_weights=euclidean_weights,
        side_weight=float(normalization["normalized_side_weight"]),
        unoriented=True,
    )
    train_costs = [
        base_geometry.path_cost_matrix(left, right)
        for left in true_lca["train"]
        for right in true_lca["train"]
    ]
    train_cost_scale = median_positive_cost_scale(train_costs)
    epsilon = scaled_sinkhorn_epsilon(train_cost_scale, kappa=kappa)

    rows: list[dict[str, Any]] = []
    for treatment in treatments:
        weights = (0.0, euclidean_weights[1], euclidean_weights[2]) if treatment == "endpoint_only" else euclidean_weights
        rows.append(
            _evaluate_cell(
                split=split,
                query_measures=encoded[treatment]["query"],
                gallery_measures=encoded[treatment]["gallery"],
                geometry=RoleProductGeometry(
                    factor_curvatures=(0.0, 0.0, 0.0),
                    factor_weights=weights,
                    side_weight=float(normalization["normalized_side_weight"]),
                    unoriented=True,
                ),
                benchmark_level=benchmark_level,
                hard_negatives_per_query=hard_negatives_per_query,
                epsilon=epsilon,
                sinkhorn_iterations=sinkhorn_iterations,
                sinkhorn_projection_iterations=sinkhorn_projection_iterations,
                ap_at_r=ap_at_r,
                metadata={
                    "gate": "A",
                    "cell_id": f"EEE__{treatment}__measure",
                    "geometry": "EEE",
                    "treatment": treatment,
                    "factor_curvatures": [0.0, 0.0, 0.0],
                    "factor_weights": list(weights),
                },
            )
        )
        _write_payload(output_path, _payload(tasks, split, rows, treatments, specs, status="partial", config={}))

    rows.append(
        _evaluate_cell(
            split=split,
            query_measures=true_lca["query"],
            gallery_measures=true_lca["gallery"],
            geometry=ConcatenatedEuclideanGeometry(
                factor_weights=euclidean_weights,
                side_weight=float(normalization["normalized_side_weight"]),
                unoriented=True,
            ),
            benchmark_level=benchmark_level,
            hard_negatives_per_query=hard_negatives_per_query,
            epsilon=epsilon,
            sinkhorn_iterations=sinkhorn_iterations,
            sinkhorn_projection_iterations=sinkhorn_projection_iterations,
            ap_at_r=ap_at_r,
            metadata={
                "gate": "A_sanity_control",
                "cell_id": "EEE_concat__true_lca__measure",
                "geometry": "EEE_concat",
                "treatment": "true_lca",
                "factor_curvatures": [0.0, 0.0, 0.0],
                "factor_weights": list(euclidean_weights),
                "expected_identity_with": "EEE__true_lca__measure",
            },
        )
    )
    _write_payload(output_path, _payload(tasks, split, rows, treatments, specs, status="partial", config={}))

    for spec in specs:
        if spec.name == "EEE":
            continue
        implementation_weights = _matched_role_weights(
            canonical_weights,
            factor_curvatures=spec.factor_curvatures,
        )
        rows.append(
            _evaluate_cell(
                split=split,
                query_measures=true_lca["query"],
                gallery_measures=true_lca["gallery"],
                geometry=RoleProductGeometry(
                    factor_curvatures=spec.factor_curvatures,
                    factor_weights=implementation_weights,
                    side_weight=float(normalization["normalized_side_weight"]),
                    unoriented=True,
                ),
                benchmark_level=benchmark_level,
                hard_negatives_per_query=hard_negatives_per_query,
                epsilon=epsilon,
                sinkhorn_iterations=sinkhorn_iterations,
                sinkhorn_projection_iterations=sinkhorn_projection_iterations,
                ap_at_r=ap_at_r,
                metadata={
                    "gate": "C",
                    "cell_id": f"{spec.name}__true_lca__measure",
                    "geometry": spec.name,
                    "treatment": "true_lca",
                    "factor_curvatures": list(spec.factor_curvatures),
                    "factor_weights": list(implementation_weights),
                    "canonical_factor_weights": list(canonical_weights),
                },
            )
        )
        _write_payload(output_path, _payload(tasks, split, rows, treatments, specs, status="partial", config={}))

    config = {
        "benchmark_level": benchmark_level,
        "study_stage": study_stage,
        "language": language,
        "dim": dim,
        "epochs": epochs,
        "seed": seed,
        "reproducibility": reproducibility,
        "max_paths": max_paths,
        "path_selection_policy": path_selection_policy,
        "item_scope": item_scope,
        "ap_at_r": ap_at_r,
        "neutral_encoder": True,
        "retrieval_loss_weight": 0.0,
        "representation": "uniform measure over unoriented (LCA, endpoint, endpoint) path objects",
        "role_point_scales": list(role_scales),
        "normalization": normalization,
        "canonical_factor_weights": list(canonical_weights),
        "euclidean_limit_matching": (
            "implementation weight is 4*w for c=0 and w for c>0 because "
            "d_H,c converges to 2*Euclidean distance as c approaches zero"
        ),
        "train_cost_scale": train_cost_scale,
        "sinkhorn_kappa": kappa,
        "sinkhorn_epsilon": epsilon,
        "training_history": training_history,
        "treatment_diagnostics": treatment_diagnostics,
        "frozen_across_cells": [
            "encoder",
            "split",
            "path sampling",
            "role coordinate scales",
            "factor weights",
            "side weight",
            "train cost scale",
            "Sinkhorn epsilon",
        ],
    }
    payload = _payload(tasks, split, rows, treatments, specs, status="complete", config=config)
    payload["contrasts"] = _planned_contrasts(rows)
    _write_payload(output_path, payload)
    return payload


def _validate_design(
    treatments: Sequence[Treatment],
    geometries: Sequence[GeometryName],
    *,
    side_block_weight: float,
    ap_at_r: int,
) -> None:
    if "true_lca" not in treatments:
        raise ValueError("true_lca is required to estimate frozen training scales")
    if "EEE" not in geometries:
        raise ValueError("EEE is required as the Gate C reference")
    if not 0.0 <= side_block_weight < 1.0:
        raise ValueError("side_block_weight must be in [0, 1)")
    if ap_at_r <= 0:
        raise ValueError("ap_at_r must be positive")
    invalid_treatments = sorted(set(treatments) - set(DEFAULT_TREATMENTS))
    invalid_geometries = sorted(set(geometries) - set(DEFAULT_GEOMETRIES))
    if invalid_treatments:
        raise ValueError(f"unknown treatments: {invalid_treatments}")
    if invalid_geometries:
        raise ValueError(f"unknown geometries: {invalid_geometries}")


def _encode_treatment(
    model: RawASTCode2Hyp,
    split: SplitItems,
    *,
    treatment: Treatment,
    max_paths: int,
    seed: int,
) -> tuple[dict[str, list[ProductMeasure]], dict[str, Any]]:
    derived_controls = {
        "endpoint_only",
        "program_shuffled_lca",
        "depth_matched_shuffled",
        "full_path_no_explicit_lca",
    }
    anchor_mode: AnchorMode = "true_lca" if treatment in derived_controls else treatment
    original_anchor_mode = model.anchor_mode
    model.anchor_mode = anchor_mode
    try:
        encoded = {
            "train": _encode_split(model, split.train, path_object_mode="lca_product", max_paths=max_paths),
            "query": _encode_split(model, split.query, path_object_mode="lca_product", max_paths=max_paths),
            "gallery": _encode_split(model, split.gallery, path_object_mode="lca_product", max_paths=max_paths),
        }
        if treatment == "program_shuffled_lca":
            return (
                {
                    split_name: _permute_anchors_between_measures(measures, seed=_derived_seed(seed, split_name))
                    for split_name, measures in encoded.items()
                },
                {
                    split_name: {
                        "path_count": sum(measure.points.shape[0] for measure in measures),
                        "same_program_assignments": 0,
                        "anchor_marginal_preserved": True,
                    }
                    for split_name, measures in encoded.items()
                },
            )
        if treatment == "depth_matched_shuffled":
            result: dict[str, list[ProductMeasure]] = {}
            diagnostics: dict[str, Any] = {}
            for split_name, measures, items in (
                ("train", encoded["train"], split.train),
                ("query", encoded["query"], split.query),
                ("gallery", encoded["gallery"], split.gallery),
            ):
                result[split_name], diagnostics[split_name] = _permute_anchors_within_lca_depth(
                    measures,
                    items,
                    seed=_derived_seed(seed, split_name),
                    max_paths=max_paths,
                )
            return result, diagnostics
        if treatment == "full_path_no_explicit_lca":
            result = {
                split_name: _replace_anchor_with_full_path_pool(
                    model,
                    measures,
                    items,
                    max_paths=max_paths,
                )
                for split_name, measures, items in (
                    ("train", encoded["train"], split.train),
                    ("query", encoded["query"], split.query),
                    ("gallery", encoded["gallery"], split.gallery),
                )
            }
            return result, {
                "definition": "factor_0 = arithmetic mean of shared-encoder node points along the complete AST path",
                "preserved": ["path set", "endpoints", "masses", "dimension 3d", "shared node encoder"],
                "explicit_lca_factor": False,
            }
        return encoded, {}
    finally:
        model.anchor_mode = original_anchor_mode


def _permute_anchors_between_measures(
    measures: Sequence[ProductMeasure],
    *,
    seed: int,
) -> list[ProductMeasure]:
    """Permute LCA factors across programs while preserving their marginal.

    The constrained permutation never assigns an anchor back to a path object
    from the same program. Endpoints, path masses, and side features remain
    fixed. Thus the control destroys LCA-endpoint association without changing
    the multiset of LCA vectors in the evaluated split.
    """

    locations = [
        (measure_index, path_index)
        for measure_index, measure in enumerate(measures)
        for path_index in range(measure.points.shape[0])
    ]
    if len({measure_index for measure_index, _ in locations}) < 2:
        raise ValueError("program-level LCA permutation requires at least two measures")
    sources = _cross_measure_permutation(locations, seed=seed)
    if sources is None:
        raise ValueError("could not construct a cross-program LCA permutation")

    points = [measure.points.clone() for measure in measures]
    for (target_measure, target_path), (source_measure, source_path) in zip(locations, sources):
        points[target_measure][target_path, 0] = measures[source_measure].points[source_path, 0]
    return [
        ProductMeasure(
            points=point_tensor,
            mass=measure.mass,
            side_features=measure.side_features,
            reversed_side_features=measure.reversed_side_features,
        )
        for measure, point_tensor in zip(measures, points)
    ]


def _replace_anchor_with_full_path_pool(
    model: RawASTCode2Hyp,
    measures: Sequence[ProductMeasure],
    items: Sequence[LabeledItem],
    *,
    max_paths: int,
) -> list[ProductMeasure]:
    if len(measures) != len(items):
        raise ValueError("measures and labeled items must be aligned")
    result = []
    for measure, item in zip(measures, items):
        tree = item.item.tree
        paths = terminal_to_terminal_paths(
            tree,
            max_paths=max_paths,
            selection_policy=model.path_selection_policy,
        )
        if len(paths) != measure.points.shape[0]:
            raise ValueError("selected paths must align with encoded path objects")
        node_points = model.encode_nodes(tree).detach()
        points = measure.points.clone()
        for path_index, path in enumerate(paths):
            points[path_index, 0] = node_points[list(path.nodes)].mean(dim=0)
        result.append(
            ProductMeasure(
                points=points,
                mass=measure.mass,
                side_features=measure.side_features,
                reversed_side_features=measure.reversed_side_features,
            )
        )
    return result


def _permute_anchors_within_lca_depth(
    measures: Sequence[ProductMeasure],
    items: Sequence[LabeledItem],
    *,
    seed: int,
    max_paths: int,
) -> tuple[list[ProductMeasure], dict[str, Any]]:
    """Permute LCA vectors across programs inside exact LCA-depth strata."""

    if len(measures) != len(items):
        raise ValueError("measures and labeled items must be aligned")
    locations_by_depth: dict[int, list[tuple[int, int]]] = {}
    for measure_index, (measure, item) in enumerate(zip(measures, items)):
        paths = terminal_to_terminal_paths(
            item.item.tree,
            max_paths=max_paths,
            selection_policy="lca_depth_stratified",
        )
        depths = tuple(item.item.tree.depth(path.lca(item.item.tree)) for path in paths)
        if len(depths) != measure.points.shape[0]:
            raise ValueError("selected LCA depths must align with encoded path objects")
        for path_index, depth in enumerate(depths):
            locations_by_depth.setdefault(int(depth), []).append((measure_index, path_index))

    points = [measure.points.clone() for measure in measures]
    shuffled_count = 0
    fallback_count = 0
    stratum_diagnostics = []
    for depth, locations in sorted(locations_by_depth.items()):
        sources = _cross_measure_permutation(locations, seed=_derived_seed(seed, f"depth-{depth}"))
        if sources is None:
            fallback_count += len(locations)
            stratum_diagnostics.append(
                {
                    "depth": depth,
                    "path_count": len(locations),
                    "program_count": len({measure_index for measure_index, _ in locations}),
                    "status": "unchanged_no_cross_program_derangement",
                }
            )
            continue
        for (target_measure, target_path), (source_measure, source_path) in zip(locations, sources):
            points[target_measure][target_path, 0] = measures[source_measure].points[source_path, 0]
        shuffled_count += len(locations)
        stratum_diagnostics.append(
            {
                "depth": depth,
                "path_count": len(locations),
                "program_count": len({measure_index for measure_index, _ in locations}),
                "status": "cross_program_permuted",
            }
        )
    result = [
        ProductMeasure(
            points=point_tensor,
            mass=measure.mass,
            side_features=measure.side_features,
            reversed_side_features=measure.reversed_side_features,
        )
        for measure, point_tensor in zip(measures, points)
    ]
    path_count = shuffled_count + fallback_count
    return result, {
        "path_count": path_count,
        "shuffled_count": shuffled_count,
        "fallback_count": fallback_count,
        "fallback_fraction": fallback_count / path_count if path_count else 0.0,
        "anchor_marginal_preserved_within_depth": True,
        "same_program_assignments_in_shuffled_strata": 0,
        "strata": stratum_diagnostics,
    }


def _cross_measure_permutation(
    locations: Sequence[tuple[int, int]],
    *,
    seed: int,
) -> list[tuple[int, int]] | None:
    if len({measure_index for measure_index, _ in locations}) < 2:
        return None
    rng = random.Random(seed)
    shifts = list(range(1, len(locations)))
    rng.shuffle(shifts)
    for shift in shifts:
        sources = list(locations[shift:]) + list(locations[:shift])
        if all(target[0] != source[0] for target, source in zip(locations, sources)):
            return sources
    return None


def _role_point_scales(
    measures: Sequence[ProductMeasure],
    *,
    max_curvatures: Sequence[float],
    max_ball_fraction: float,
) -> tuple[float, float, float]:
    if len(max_curvatures) != 3:
        raise ValueError("three role curvatures are required")
    if not 0.0 < max_ball_fraction < 1.0:
        raise ValueError("max_ball_fraction must be in (0, 1)")
    scales = []
    for factor_index, curvature in enumerate(max_curvatures):
        if curvature <= 0.0:
            scales.append(1.0)
            continue
        max_norm = max(
            float(torch.linalg.vector_norm(measure.points[:, factor_index], dim=-1).max())
            for measure in measures
        )
        allowed = max_ball_fraction / math.sqrt(curvature)
        scales.append(1.0 if max_norm <= 0.0 else min(1.0, allowed / max_norm))
    return tuple(scales)  # type: ignore[return-value]


def _scale_measure_by_role(
    measure: ProductMeasure,
    *,
    role_scales: Sequence[float],
) -> ProductMeasure:
    scale = measure.points.new_tensor(tuple(role_scales)).view(1, -1, 1)
    return ProductMeasure(
        points=measure.points * scale,
        mass=measure.mass,
        side_features=measure.side_features,
        reversed_side_features=measure.reversed_side_features,
    )


def _frozen_role_weights(
    train_measures: Sequence[ProductMeasure],
    *,
    side_block_weight: float,
) -> tuple[tuple[float, float, float], dict[str, Any]]:
    factor_scales = []
    for factor_index in range(3):
        factor_geometry = RoleProductGeometry(
            factor_curvatures=(0.0, 0.0, 0.0),
            factor_weights=tuple(1.0 if index == factor_index else 0.0 for index in range(3)),
            side_weight=0.0,
            unoriented=False,
        )
        factor_scales.append(
            _median_positive_or_zero(
                factor_geometry.path_cost_matrix(left, right)
                for left in train_measures
                for right in train_measures
            )
        )
    positive_endpoint_scales = [value for value in factor_scales[1:] if value > 1e-12]
    endpoint_scale = _median(positive_endpoint_scales) if positive_endpoint_scales else 0.0
    point_budget = 1.0 - side_block_weight
    lca_weight = point_budget / (3.0 * factor_scales[0]) if factor_scales[0] > 1e-12 else 0.0
    endpoint_weight = point_budget / (3.0 * endpoint_scale) if endpoint_scale > 1e-12 else 0.0

    side_scale = 0.0
    normalized_side_weight = 0.0
    if side_block_weight > 0.0:
        side_geometry = RoleProductGeometry(
            factor_curvatures=(0.0, 0.0, 0.0),
            factor_weights=(0.0, 0.0, 0.0),
            side_weight=1.0,
            unoriented=True,
        )
        side_scale = _median_positive_or_zero(
            side_geometry.path_cost_matrix(left, right)
            for left in train_measures
            for right in train_measures
        )
        if side_scale > 1e-12:
            normalized_side_weight = side_block_weight / side_scale

    euclidean_weights = (lca_weight, endpoint_weight, endpoint_weight)
    return euclidean_weights, {
        "source": "true_lca_train_split_only",
        "factor_cost_scales": factor_scales,
        "pooled_endpoint_cost_scale": endpoint_scale,
        "euclidean_implementation_factor_weights": list(euclidean_weights),
        "canonical_factor_weights": [weight / 4.0 for weight in euclidean_weights],
        "point_block_weight": point_budget,
        "side_block_weight": side_block_weight,
        "side_cost_scale": side_scale,
        "normalized_side_weight": normalized_side_weight,
        "endpoint_weight_constraint": "w_start = w_end",
    }


def _matched_role_weights(
    canonical_weights: Sequence[float],
    *,
    factor_curvatures: Sequence[float],
) -> tuple[float, float, float]:
    """Map canonical product weights to the standard Poincare convention.

    Standard Poincare distance tends to twice Euclidean distance as curvature
    approaches zero. A Euclidean factor therefore needs four times the
    canonical squared-distance coefficient to represent the same limiting
    metric contribution.
    """

    if len(canonical_weights) != 3 or len(factor_curvatures) != 3:
        raise ValueError("three canonical weights and curvatures are required")
    result = tuple(
        float(weight) if float(curvature) > 0.0 else 4.0 * float(weight)
        for weight, curvature in zip(canonical_weights, factor_curvatures)
    )
    return result  # type: ignore[return-value]


def _evaluate_cell(
    *,
    split: SplitItems,
    query_measures: Sequence[ProductMeasure],
    gallery_measures: Sequence[ProductMeasure],
    geometry: RoleProductGeometry | ConcatenatedEuclideanGeometry,
    benchmark_level: BenchmarkLevel,
    hard_negatives_per_query: int,
    epsilon: float,
    sinkhorn_iterations: int,
    sinkhorn_projection_iterations: int,
    ap_at_r: int,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    query_records: list[dict[str, Any]] = []
    task_ap: dict[str, list[float]] = {}
    for query_item, query_measure in zip(split.query, query_measures):
        candidates = _candidate_indices(
            query_item,
            split.gallery,
            benchmark_level=benchmark_level,
            hard_negatives_per_query=hard_negatives_per_query,
        )
        positives = {position for position, index in enumerate(candidates) if split.gallery[index].task == query_item.task}
        if not candidates or not positives:
            continue
        distances = [
            float(
                _sinkhorn_with_retry(
                    geometry,
                    query_measure,
                    gallery_measures[index],
                    epsilon=epsilon,
                    iterations=sinkhorn_iterations,
                    projection_iterations=sinkhorn_projection_iterations,
                ).detach()
            )
            for index in candidates
        ]
        order = sorted(range(len(candidates)), key=lambda position: (distances[position], candidates[position]))
        ranked_relevance = [position in positives for position in order]
        positive_ranks = [rank for rank, relevant in enumerate(ranked_relevance, start=1) if relevant]
        best_rank = min(positive_ranks)
        average_precision = _average_precision_at_r(ranked_relevance, total_positives=len(positives), r=ap_at_r)
        task_ap.setdefault(query_item.task, []).append(average_precision)
        query_records.append(
            {
                "query_id": query_item.item.item_id,
                "query_task": query_item.task,
                "ap_at_r": average_precision,
                "best_positive_rank": best_rank,
                "candidate_count": len(candidates),
                "positive_count": len(positives),
                "ranked_gallery_ids": [split.gallery[candidates[position]].item.item_id for position in order],
                "ranked_gallery_tasks": [split.gallery[candidates[position]].task for position in order],
                "ranked_distances": [distances[position] for position in order],
            }
        )
    if not query_records:
        raise ValueError("evaluation produced no valid queries")
    task_scores = {task: sum(values) / len(values) for task, values in sorted(task_ap.items())}
    best_ranks = [int(record["best_positive_rank"]) for record in query_records]
    factor_curvatures = getattr(geometry, "factor_curvatures", (0.0, 0.0, 0.0))
    return {
        **metadata,
        "effective_lca_sectional_curvature": (
            0.0
            if factor_curvatures[0] == 0.0 or geometry.factor_weights[0] == 0.0
            else -factor_curvatures[0] / geometry.factor_weights[0]
        ),
        "query_count": len(query_records),
        "task_count": len(task_scores),
        "map_at_r": sum(task_scores.values()) / len(task_scores),
        "mrr": sum(1.0 / rank for rank in best_ranks) / len(best_ranks),
        "recall_at_1": sum(rank <= 1 for rank in best_ranks) / len(best_ranks),
        "recall_at_5": sum(rank <= 5 for rank in best_ranks) / len(best_ranks),
        "mean_rank": sum(best_ranks) / len(best_ranks),
        "task_scores": task_scores,
        "query_records": query_records,
    }


def _average_precision_at_r(ranked_relevance: Sequence[bool], *, total_positives: int, r: int) -> float:
    denominator = min(r, total_positives)
    if denominator <= 0:
        return 0.0
    hits = 0
    precision_sum = 0.0
    for rank, relevant in enumerate(ranked_relevance[:r], start=1):
        if relevant:
            hits += 1
            precision_sum += hits / rank
    return precision_sum / denominator


def _sinkhorn_with_retry(
    geometry: RoleProductGeometry | ConcatenatedEuclideanGeometry,
    left: ProductMeasure,
    right: ProductMeasure,
    *,
    epsilon: float,
    iterations: int,
    projection_iterations: int,
) -> torch.Tensor:
    attempts = tuple(
        dict.fromkeys(
            (
                projection_iterations,
                max(2048, projection_iterations * 2),
                max(4096, projection_iterations * 4),
            )
        )
    )
    last_error: ValueError | None = None
    for attempt in attempts:
        try:
            return geometry.sinkhorn_divergence(
                left,
                right,
                epsilon=epsilon,
                iterations=iterations,
                projection_iterations=attempt,
            )
        except ValueError as error:
            if "coupling row sums" not in str(error) and "coupling column sums" not in str(error):
                raise
            last_error = error
    assert last_error is not None
    raise last_error


def _planned_contrasts(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    by_id = {row["cell_id"]: row for row in rows}
    planned = (
        ("H1_true_lca_vs_zero_anchor", "EEE__true_lca__measure", "EEE__zero_anchor__measure"),
        ("true_lca_vs_endpoint_only", "EEE__true_lca__measure", "EEE__endpoint_only__measure"),
        ("true_lca_vs_program_shuffled_lca", "EEE__true_lca__measure", "EEE__program_shuffled_lca__measure"),
        ("true_lca_vs_full_path_pool", "EEE__true_lca__measure", "EEE__full_path_no_explicit_lca__measure"),
        ("product_vs_equal_capacity_concat_identity", "EEE__true_lca__measure", "EEE_concat__true_lca__measure"),
        ("HEE_vs_EEE", "HEE__true_lca__measure", "EEE__true_lca__measure"),
        ("HEE_vs_near_zero_HEE", "HEE__true_lca__measure", "HEE_near_zero__true_lca__measure"),
        ("HEE_vs_HHH", "HEE__true_lca__measure", "HHH__true_lca__measure"),
    )
    contrasts = []
    for name, treatment_id, control_id in planned:
        if treatment_id not in by_id or control_id not in by_id:
            continue
        treatment = by_id[treatment_id]
        control = by_id[control_id]
        shared_tasks = sorted(set(treatment["task_scores"]) & set(control["task_scores"]))
        task_deltas = {
            task: treatment["task_scores"][task] - control["task_scores"][task]
            for task in shared_tasks
        }
        contrasts.append(
            {
                "name": name,
                "treatment_cell": treatment_id,
                "control_cell": control_id,
                "map_at_r_difference": treatment["map_at_r"] - control["map_at_r"],
                "mrr_difference": treatment["mrr"] - control["mrr"],
                "task_differences": task_deltas,
            }
        )
    return contrasts


def _derived_seed(seed: int, label: str) -> int:
    value = int(seed)
    for character in label:
        value = (value * 131 + ord(character)) % (2**32)
    return value


def _median_positive_or_zero(matrices: Sequence[torch.Tensor] | Any) -> float:
    values = []
    for matrix in matrices:
        flattened = matrix.detach().reshape(-1)
        positive = flattened[torch.isfinite(flattened) & (flattened > 0.0)]
        if positive.numel():
            values.extend(float(value) for value in positive.cpu())
    return _median(values) if values else 0.0


def _median(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return 0.5 * (ordered[middle - 1] + ordered[middle])


def _payload(
    tasks: Sequence[TaskSource],
    split: SplitItems,
    rows: Sequence[dict[str, Any]],
    treatments: Sequence[Treatment],
    geometries: Sequence[GeometrySpec],
    *,
    status: str,
    config: dict[str, Any],
) -> dict[str, Any]:
    expected = len(treatments) + 1 + sum(spec.name != "EEE" for spec in geometries)
    return {
        "experiment": "code2hyp_lca_causal_and_role_geometry_matrix",
        "status": status if len(rows) == expected else "partial",
        "completed_runs": len(rows),
        "expected_runs": expected,
        "config": config,
        "tasks": [{"label": task.label, "source": str(task.source)} for task in tasks],
        "split": {
            "train_ids": [item.item.item_id for item in split.train],
            "query_ids": [item.item.item_id for item in split.query],
            "gallery_ids": [item.item.item_id for item in split.gallery],
        },
        "runs": list(rows),
    }


def _write_payload(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _parse_tasks(values: Sequence[Sequence[str]]) -> tuple[TaskSource, ...]:
    return tuple(TaskSource(label=value[0], source=Path(value[1])) for value in values)


def _parse_csv(value: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in value.split(",") if part.strip())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the frozen Code2Hyp LCA causal and role-geometry matrix.")
    parser.add_argument("--manifest", type=Path, default=None, help="Materialized external-corpus manifest.")
    parser.add_argument("--task", nargs=2, action="append", metavar=("LABEL", "SOURCE"), default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--benchmark-level", choices=("B_independent_solution", "C_structural_hard_negative"), default="B_independent_solution")
    parser.add_argument("--treatments", default=",".join(DEFAULT_TREATMENTS))
    parser.add_argument("--geometries", default=",".join(DEFAULT_GEOMETRIES))
    parser.add_argument("--language", choices=("auto", "java", "python"), default=None)
    parser.add_argument("--dim", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-2)
    parser.add_argument("--max-files-per-task", type=int, default=128)
    parser.add_argument("--max-methods-per-task", type=int, default=48)
    parser.add_argument("--train-per-task", type=int, default=16)
    parser.add_argument("--query-per-task", type=int, default=8)
    parser.add_argument("--gallery-per-task", type=int, default=8)
    parser.add_argument("--max-paths", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260625)
    parser.add_argument("--sinkhorn-iterations", type=int, default=8)
    parser.add_argument("--sinkhorn-projection-iterations", type=int, default=512)
    parser.add_argument("--kappa", type=float, default=0.05)
    parser.add_argument("--max-ball-fraction", type=float, default=0.35)
    parser.add_argument("--hard-negatives-per-query", type=int, default=16)
    parser.add_argument("--side-block-weight", type=float, default=0.0)
    parser.add_argument("--ap-at-r", type=int, default=8)
    parser.add_argument("--item-scope", choices=("callable", "module", "callable_or_module"), default=None)
    parser.add_argument("--study-stage", choices=("pilot", "confirmatory"), default="pilot")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if bool(args.manifest) == bool(args.task):
        raise SystemExit("provide exactly one of --manifest or --task")
    if args.manifest:
        from scripts.run_factor_matrix_from_manifest import (
            item_scope_from_manifest,
            language_from_manifest,
            task_sources_from_manifest,
        )

        tasks = task_sources_from_manifest(args.manifest)
        language = args.language or language_from_manifest(args.manifest)
        item_scope = args.item_scope or item_scope_from_manifest(args.manifest)
    else:
        tasks = _parse_tasks(args.task)
        language = args.language or "python"
        item_scope = args.item_scope or "callable"
    payload = run_lca_causal_matrix(
        tasks=tasks,
        output_path=args.output,
        benchmark_level=args.benchmark_level,
        treatments=_parse_csv(args.treatments),  # type: ignore[arg-type]
        geometries=_parse_csv(args.geometries),  # type: ignore[arg-type]
        language=language,
        dim=args.dim,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        max_files_per_task=args.max_files_per_task,
        max_methods_per_task=args.max_methods_per_task,
        train_per_task=args.train_per_task,
        query_per_task=args.query_per_task,
        gallery_per_task=args.gallery_per_task,
        max_paths=args.max_paths,
        seed=args.seed,
        sinkhorn_iterations=args.sinkhorn_iterations,
        sinkhorn_projection_iterations=args.sinkhorn_projection_iterations,
        kappa=args.kappa,
        max_ball_fraction=args.max_ball_fraction,
        hard_negatives_per_query=args.hard_negatives_per_query,
        side_block_weight=args.side_block_weight,
        ap_at_r=args.ap_at_r,
        item_scope=item_scope,
        study_stage=args.study_stage,
    )
    print(f"status={payload['status']} completed={payload['completed_runs']}/{payload['expected_runs']} output={args.output}")


if __name__ == "__main__":
    main()
