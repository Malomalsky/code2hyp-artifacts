from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Sequence

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from geometry_profile_research.code2hyp_torch import torch_expmap0, torch_logmap0
from geometry_profile_research.constant_curvature import (
    ConstantCurvatureProduct,
    ProductMeasure,
    median_positive_cost_scale,
    scaled_sinkhorn_epsilon,
)
from geometry_profile_research.raw_ast import terminal_to_terminal_paths
from geometry_profile_research.raw_ast_code2hyp import (
    MethodAggregation,
    PathObjectMode,
    RawASTCode2Hyp,
    build_raw_ast_token_vocab,
)
from geometry_profile_research.raw_ast_retrieval import ItemScope, PositiveMode, RawASTRetrievalItem, build_retrieval_triples
from scripts.run_dta_level_b_c_retrieval import (
    BenchmarkLevel,
    LabeledItem,
    SplitItems,
    TaskSource,
    _candidate_indices,
    _collect_labeled_items,
    _point_scale,
    _scale_measure,
)


GeometryCell = Literal["E", "H_1e-4", "H_1"]
EncoderPolicy = Literal["shared_euclidean", "geometry_aware"]
CostMode = Literal[
    "point_only",
    "side_only",
    "unnormalized_combined",
    "train_normalized_combined",
    "train_weighted_combined",
    "validation_selected_combined",
]
DEFAULT_GEOMETRIES: tuple[GeometryCell, ...] = ("E", "H_1e-4", "H_1")
DEFAULT_PATH_OBJECTS: tuple[PathObjectMode, ...] = ("single_point", "lca_product")
DEFAULT_METHOD_AGGREGATIONS: tuple[MethodAggregation, ...] = ("centroid", "measure")
DEFAULT_COST_MODES: tuple[CostMode, ...] = ("unnormalized_combined",)
VALIDATION_POINT_WEIGHT_GRID: tuple[float, ...] = (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0)


@dataclass(frozen=True)
class FactorCell:
    geometry: GeometryCell
    curvature: float
    path_object_mode: PathObjectMode
    method_aggregation: MethodAggregation

    @property
    def cell_id(self) -> str:
        return f"{self.geometry}__{self.path_object_mode}__{self.method_aggregation}"


@dataclass(frozen=True)
class CostSpec:
    mode: CostMode
    side_weight: float
    point_weight: float | None = None

    @property
    def label(self) -> str:
        if self.mode == "train_weighted_combined":
            if self.point_weight is None:
                raise ValueError("train_weighted_combined requires point_weight")
            return f"{self.mode}_p{_format_weight_token(self.point_weight)}"
        return self.mode


def run_dta_factor_matrix(
    *,
    tasks: Sequence[TaskSource],
    output_path: Path,
    benchmark_level: BenchmarkLevel = "B_independent_solution",
    geometries: Sequence[GeometryCell] = DEFAULT_GEOMETRIES,
    path_object_modes: Sequence[PathObjectMode] = DEFAULT_PATH_OBJECTS,
    method_aggregations: Sequence[MethodAggregation] = DEFAULT_METHOD_AGGREGATIONS,
    language: Literal["auto", "java", "python"] = "python",
    dim: int = 4,
    epochs: int = 1,
    learning_rate: float = 1e-2,
    max_files_per_task: int | None = 128,
    max_methods_per_task: int | None = 48,
    train_per_task: int = 16,
    query_per_task: int = 8,
    gallery_per_task: int = 1,
    max_paths: int = 16,
    seed: int = 20260625,
    min_structural_gap: float = 0.05,
    positive_mode: PositiveMode = "alpha_structural_noop",
    item_scope: ItemScope = "callable",
    sinkhorn_iterations: int = 6,
    sinkhorn_projection_iterations: int = 512,
    kappa: float = 0.05,
    side_weight: float = 1.0,
    side_weights: Sequence[float] | None = None,
    point_weights: Sequence[float] | None = None,
    cost_modes: Sequence[CostMode] | None = None,
    max_ball_fraction: float = 0.35,
    hard_negatives_per_query: int = 6,
    encoder_policy: EncoderPolicy = "shared_euclidean",
    path_selection_policy: str = "lca_depth_stratified",
) -> dict[str, Any]:
    """Run the reviewer-directed 3 x 2 x 2 frozen DTA factor matrix.

    A single Euclidean raw-AST encoder is trained on the disjoint training split.
    The factor cells then vary only the mathematical representation used for
    retrieval: node geometry, path object, and method aggregation. This avoids
    conflating curvature with different optimization paths.
    """

    if encoder_policy not in {"shared_euclidean", "geometry_aware"}:
        raise ValueError(f"unknown encoder_policy: {encoder_policy!r}")
    reproducibility = _set_reproducible_seeds(seed)
    cells = _factor_cells(geometries, path_object_modes, method_aggregations)
    side_weight_values = _normalize_side_weights(side_weight=side_weight, side_weights=side_weights)
    point_weight_values = _normalize_point_weights(point_weights)
    cost_specs = _cost_specs(side_weight_values=side_weight_values, point_weight_values=point_weight_values, cost_modes=cost_modes)
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
    rows: list[dict[str, Any]] = []
    if encoder_policy == "shared_euclidean":
        model, training_history = _train_shared_encoder(
            split.train,
            dim=dim,
            max_paths=max_paths,
            epochs=epochs,
            learning_rate=learning_rate,
            min_structural_gap=min_structural_gap,
            positive_mode=positive_mode,
            sinkhorn_iterations=sinkhorn_iterations,
            path_selection_policy=path_selection_policy,
        )
        encoded_by_mode = {
            mode: {
                "train": _encode_split(model, split.train, path_object_mode=mode, max_paths=max_paths),
                "query": _encode_split(model, split.query, path_object_mode=mode, max_paths=max_paths),
                "gallery": _encode_split(model, split.gallery, path_object_mode=mode, max_paths=max_paths),
            }
            for mode in path_object_modes
        }
        max_curvature = max(cell.curvature for cell in cells)
        point_scale = _point_scale(
            [measure for mode_payload in encoded_by_mode.values() for measure in mode_payload["train"]],
            max_curvature=max_curvature,
            max_ball_fraction=max_ball_fraction,
        )
        embedding_norm_diagnostics = {
            mode: _embedding_norm_diagnostics(
                [measure for split_measures in mode_payload.values() for measure in split_measures],
                point_scale=point_scale,
                curvatures=sorted({cell.curvature for cell in cells}),
            )
            for mode, mode_payload in encoded_by_mode.items()
        }
        encoded_by_mode = {
            mode: {split_name: [_scale_measure(measure, point_scale=point_scale) for measure in measures] for split_name, measures in mode_payload.items()}
            for mode, mode_payload in encoded_by_mode.items()
        }
        for cell in cells:
            for cost_spec in cost_specs:
                rows.append(
                    _run_factor_cell(
                        split=split,
                        cell=cell,
                        mode_payload=encoded_by_mode[cell.path_object_mode],
                        benchmark_level=benchmark_level,
                        sinkhorn_iterations=sinkhorn_iterations,
                        sinkhorn_projection_iterations=sinkhorn_projection_iterations,
                        hard_negatives_per_query=hard_negatives_per_query,
                        kappa=kappa,
                        cost_spec=cost_spec,
                        metadata={
                            "encoder_policy": encoder_policy,
                            "trained_manifold": model.manifold,
                            "trained_curvature": model.curvature,
                            "point_scale": point_scale,
                            "embedding_norm_diagnostics": embedding_norm_diagnostics[cell.path_object_mode],
                            "training_history": training_history,
                        },
                    )
                )
                _write_payload(
                    output_path,
                    _payload(
                        tasks,
                        split,
                        rows,
                        cells,
                        side_weights=side_weight_values,
                        point_weights=point_weight_values,
                        cost_modes=tuple(spec.label for spec in cost_specs),
                        benchmark_level=benchmark_level,
                        language=language,
                        dim=dim,
                        epochs=epochs,
                        seed=seed,
                        reproducibility=reproducibility,
                        item_scope=item_scope,
                        encoder_policy=encoder_policy,
                        path_selection_policy=path_selection_policy,
                    ),
                )
    else:
        for cell in cells:
            model, training_history = _train_shared_encoder(
                split.train,
                dim=dim,
                max_paths=max_paths,
                epochs=epochs,
                learning_rate=learning_rate,
                min_structural_gap=min_structural_gap,
                positive_mode=positive_mode,
                sinkhorn_iterations=sinkhorn_iterations,
                manifold=_cell_manifold(cell),
                curvature=_cell_model_curvature(cell),
                path_object_mode=cell.path_object_mode,
                method_aggregation=cell.method_aggregation,
                path_selection_policy=path_selection_policy,
            )
            mode_payload = {
                "train": _encode_split(model, split.train, path_object_mode=cell.path_object_mode, max_paths=max_paths),
                "query": _encode_split(model, split.query, path_object_mode=cell.path_object_mode, max_paths=max_paths),
                "gallery": _encode_split(model, split.gallery, path_object_mode=cell.path_object_mode, max_paths=max_paths),
            }
            embedding_norm_diagnostics = _embedding_norm_diagnostics(
                [measure for measures in mode_payload.values() for measure in measures],
                point_scale=1.0,
                curvatures=(cell.curvature,),
            )
            for cost_spec in cost_specs:
                rows.append(
                    _run_factor_cell(
                        split=split,
                        cell=cell,
                        mode_payload=mode_payload,
                        benchmark_level=benchmark_level,
                        sinkhorn_iterations=sinkhorn_iterations,
                        sinkhorn_projection_iterations=sinkhorn_projection_iterations,
                        hard_negatives_per_query=hard_negatives_per_query,
                        kappa=kappa,
                        cost_spec=cost_spec,
                        metadata={
                            "encoder_policy": encoder_policy,
                            "trained_manifold": model.manifold,
                            "trained_curvature": model.curvature,
                            "point_scale": 1.0,
                            "embedding_norm_diagnostics": embedding_norm_diagnostics,
                            "training_history": training_history,
                        },
                    )
                )
                _write_payload(
                    output_path,
                    _payload(
                        tasks,
                        split,
                        rows,
                        cells,
                        side_weights=side_weight_values,
                        point_weights=point_weight_values,
                        cost_modes=tuple(spec.label for spec in cost_specs),
                        benchmark_level=benchmark_level,
                        language=language,
                        dim=dim,
                        epochs=epochs,
                        seed=seed,
                        reproducibility=reproducibility,
                        item_scope=item_scope,
                        encoder_policy=encoder_policy,
                        path_selection_policy=path_selection_policy,
                    ),
                )
    payload = _payload(
        tasks,
        split,
        rows,
        cells,
        side_weights=side_weight_values,
        point_weights=point_weight_values,
        cost_modes=tuple(spec.label for spec in cost_specs),
        benchmark_level=benchmark_level,
        language=language,
        dim=dim,
        epochs=epochs,
        seed=seed,
        reproducibility=reproducibility,
        item_scope=item_scope,
        encoder_policy=encoder_policy,
        path_selection_policy=path_selection_policy,
    )
    _write_payload(output_path, payload)
    return payload


def _factor_cells(
    geometries: Sequence[GeometryCell],
    path_object_modes: Sequence[PathObjectMode],
    method_aggregations: Sequence[MethodAggregation],
) -> tuple[FactorCell, ...]:
    if not geometries:
        raise ValueError("at least one geometry is required")
    if not path_object_modes:
        raise ValueError("at least one path_object_mode is required")
    if not method_aggregations:
        raise ValueError("at least one method_aggregation is required")
    invalid_geometries = [value for value in geometries if value not in DEFAULT_GEOMETRIES]
    if invalid_geometries:
        raise ValueError(f"unknown geometry cells: {invalid_geometries!r}")
    invalid_path_modes = [value for value in path_object_modes if value not in DEFAULT_PATH_OBJECTS]
    if invalid_path_modes:
        raise ValueError(f"unknown path_object_mode values: {invalid_path_modes!r}")
    invalid_aggs = [value for value in method_aggregations if value not in DEFAULT_METHOD_AGGREGATIONS]
    if invalid_aggs:
        raise ValueError(f"unknown method_aggregation values: {invalid_aggs!r}")
    return tuple(
        FactorCell(
            geometry=geometry,
            curvature=_geometry_curvature(geometry),
            path_object_mode=path_object_mode,
            method_aggregation=method_aggregation,
        )
        for geometry in geometries
        for path_object_mode in path_object_modes
        for method_aggregation in method_aggregations
    )


def _geometry_curvature(geometry: GeometryCell) -> float:
    if geometry == "E":
        return 0.0
    if geometry == "H_1e-4":
        return 1e-4
    if geometry == "H_1":
        return 1.0
    raise ValueError(f"unknown geometry: {geometry!r}")


def _set_reproducible_seeds(seed: int) -> dict[str, Any]:
    random.seed(seed)
    torch.manual_seed(seed)
    deterministic_algorithms = True
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except TypeError:
        torch.use_deterministic_algorithms(True)
    except Exception:
        deterministic_algorithms = False
    return {
        "python_random_seed": seed,
        "torch_manual_seed": seed,
        "torch_deterministic_algorithms": deterministic_algorithms,
    }


def _normalize_side_weights(*, side_weight: float, side_weights: Sequence[float] | None) -> tuple[float, ...]:
    values = (float(side_weight),) if side_weights is None else tuple(float(value) for value in side_weights)
    if not values:
        raise ValueError("at least one side weight is required")
    if any(value < 0.0 for value in values):
        raise ValueError(f"side weights must be non-negative: {values!r}")
    return values


def _normalize_point_weights(point_weights: Sequence[float] | None) -> tuple[float, ...]:
    values = (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0) if point_weights is None else tuple(float(value) for value in point_weights)
    if not values:
        raise ValueError("at least one point weight is required")
    if any(value < 0.0 or value > 1.0 for value in values):
        raise ValueError(f"point weights must be in [0, 1]: {values!r}")
    return values


def _format_weight_token(value: float) -> str:
    return f"{float(value):.2f}".replace(".", "p")


def _cost_specs(
    *,
    side_weight_values: Sequence[float],
    point_weight_values: Sequence[float],
    cost_modes: Sequence[CostMode] | None,
) -> tuple[CostSpec, ...]:
    if cost_modes is None:
        return tuple(CostSpec(mode="unnormalized_combined", side_weight=value) for value in side_weight_values)
    if not cost_modes:
        raise ValueError("at least one cost mode is required")
    valid = {
        "point_only",
        "side_only",
        "unnormalized_combined",
        "train_normalized_combined",
        "train_weighted_combined",
        "validation_selected_combined",
    }
    invalid = [value for value in cost_modes if value not in valid]
    if invalid:
        raise ValueError(f"unknown cost modes: {invalid!r}")
    base_side_weight = float(side_weight_values[0])
    specs: list[CostSpec] = []
    for mode in cost_modes:
        if mode == "train_weighted_combined":
            specs.extend(CostSpec(mode=mode, side_weight=base_side_weight, point_weight=value) for value in point_weight_values)
        else:
            specs.append(CostSpec(mode=mode, side_weight=base_side_weight))
    return tuple(specs)


def _cell_manifold(cell: FactorCell) -> Literal["euclidean", "poincare"]:
    return "euclidean" if cell.curvature == 0.0 else "poincare"


def _cell_model_curvature(cell: FactorCell) -> float:
    return cell.curvature if cell.curvature > 0.0 else 1.0


def _split_items(
    labeled: Sequence[LabeledItem],
    *,
    train_per_task: int,
    query_per_task: int,
    gallery_per_task: int,
    seed: int,
) -> SplitItems:
    rng = random.Random(seed)
    by_task: dict[str, list[LabeledItem]] = {}
    for item in labeled:
        by_task.setdefault(item.task, []).append(item)
    train: list[LabeledItem] = []
    query: list[LabeledItem] = []
    gallery: list[LabeledItem] = []
    for task, items in sorted(by_task.items()):
        ordered = sorted(items, key=lambda value: value.item.item_id)
        rng.shuffle(ordered)
        required = train_per_task + query_per_task + gallery_per_task
        if len(ordered) < required:
            raise ValueError(f"task {task!r} has {len(ordered)} items; {required} required")
        train.extend(ordered[:train_per_task])
        query.extend(ordered[train_per_task : train_per_task + query_per_task])
        gallery.extend(ordered[train_per_task + query_per_task : required])
    _assert_disjoint(train, query, gallery)
    return SplitItems(train=tuple(train), query=tuple(query), gallery=tuple(gallery))


def _assert_disjoint(train: Sequence[LabeledItem], query: Sequence[LabeledItem], gallery: Sequence[LabeledItem]) -> None:
    groups = {"train": train, "query": query, "gallery": gallery}
    ids_by_group = {name: {item.item.item_id for item in values} for name, values in groups.items()}
    for left_name, left_ids in ids_by_group.items():
        for right_name, right_ids in ids_by_group.items():
            if left_name >= right_name:
                continue
            overlap = left_ids & right_ids
            if overlap:
                raise ValueError(f"{left_name}/{right_name} split overlap: {sorted(overlap)[:3]}")


def _train_shared_encoder(
    train_items: Sequence[LabeledItem],
    *,
    dim: int,
    max_paths: int,
    epochs: int,
    learning_rate: float,
    min_structural_gap: float,
    positive_mode: PositiveMode,
    sinkhorn_iterations: int,
    manifold: Literal["euclidean", "poincare"] = "euclidean",
    curvature: float = 1.0,
    path_object_mode: PathObjectMode = "lca_product",
    method_aggregation: MethodAggregation = "measure",
    path_selection_policy: str = "lca_depth_stratified",
    lambda_retrieval: float = 1.0,
) -> tuple[RawASTCode2Hyp, list[dict[str, float]]]:
    raw_items = tuple(item.item for item in train_items)
    vocab = build_raw_ast_token_vocab(tuple(item.tree for item in raw_items), terminal_policy="class", node_input_mode="label_only")
    model = RawASTCode2Hyp(
        vocab,
        dim=dim,
        manifold=manifold,
        max_paths=max_paths,
        terminal_policy="class",
        node_input_mode="label_only",
        path_object_mode=path_object_mode,
        method_aggregation=method_aggregation,
        path_cost_orientation="directed",
        curvature=curvature,
        path_selection_policy=path_selection_policy,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    triples = build_retrieval_triples(raw_items, min_structural_gap=min_structural_gap, positive_mode=positive_mode)
    history: list[dict[str, float]] = []
    for epoch in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        loss = model.training_loss(
            triples,
            sinkhorn_epsilon=0.05,
            sinkhorn_iterations=sinkhorn_iterations,
            lambda_retrieval=lambda_retrieval,
        )
        loss["loss"].backward()
        optimizer.step()
        history.append(
            {
                "epoch": float(epoch + 1),
                "loss": float(loss["loss"].detach()),
                "retrieval": float(loss["retrieval"].detach()),
                "edge": float(loss["edge"].detach()),
                "gromov_lca": float(loss["gromov_lca"].detach()),
                "gromov_lca_mean_abs_residual": float(loss["gromov_lca_mean_abs_residual"].detach()),
                "retrieval_weight": float(loss["retrieval_weight"].detach()),
            }
        )
    return model, history


def _encode_split(
    model: RawASTCode2Hyp,
    items: Sequence[LabeledItem],
    *,
    path_object_mode: PathObjectMode,
    max_paths: int,
) -> list[ProductMeasure]:
    original_mode = model.path_object_mode
    model.path_object_mode = path_object_mode
    try:
        return [_encode_labeled_item(model, item, max_paths=max_paths) for item in items]
    finally:
        model.path_object_mode = original_mode


def _encode_labeled_item(model: RawASTCode2Hyp, item: LabeledItem, *, max_paths: int) -> ProductMeasure:
    raw = model.encode_method(
        item.item.tree,
        paths=terminal_to_terminal_paths(
            item.item.tree,
            max_paths=max_paths,
            selection_policy=model.path_selection_policy,
        ),
    )
    side = torch.cat((raw.left_branch, raw.right_branch), dim=-1)
    reversed_side = torch.cat((raw.right_branch, raw.left_branch), dim=-1)
    return ProductMeasure(
        points=raw.points.detach(),
        mass=raw.mass.detach(),
        side_features=side.detach(),
        reversed_side_features=reversed_side.detach(),
    )


def _aggregate_measures(
    measures: Sequence[ProductMeasure],
    *,
    method_aggregation: MethodAggregation,
    curvature: float,
) -> list[ProductMeasure]:
    if method_aggregation == "measure":
        return list(measures)
    if method_aggregation != "centroid":
        raise ValueError(f"unknown method_aggregation: {method_aggregation!r}")
    return [_centroid_measure(measure, curvature=curvature) for measure in measures]


def _centroid_measure(measure: ProductMeasure, *, curvature: float) -> ProductMeasure:
    weights = measure.mass.view(-1, 1, 1)
    if curvature > 0.0:
        tangent = torch_logmap0(measure.points, curvature)
        point_centroid = torch_expmap0(torch.sum(weights * tangent, dim=0), curvature)
    else:
        point_centroid = torch.sum(weights * measure.points, dim=0)
    side_centroid = None
    reversed_side_centroid = None
    if measure.side_features is not None:
        side_centroid = torch.sum(measure.mass.view(-1, 1) * measure.side_features, dim=0, keepdim=True)
    if measure.reversed_side_features is not None:
        reversed_side_centroid = torch.sum(
            measure.mass.view(-1, 1) * measure.reversed_side_features,
            dim=0,
            keepdim=True,
        )
    return ProductMeasure(
        points=point_centroid.unsqueeze(0),
        mass=measure.mass.new_ones((1,)),
        side_features=side_centroid,
        reversed_side_features=reversed_side_centroid,
    )


def _embedding_norm_diagnostics(
    measures: Sequence[ProductMeasure],
    *,
    point_scale: float,
    curvatures: Sequence[float],
) -> dict[str, Any]:
    """Summarize whether encoded points live in a curvature-active region."""

    if not measures:
        return {
            "point_scale": float(point_scale),
            "unscaled_norm_min": 0.0,
            "unscaled_norm_median": 0.0,
            "unscaled_norm_mean": 0.0,
            "unscaled_norm_max": 0.0,
            "scaled_norm_min": 0.0,
            "scaled_norm_median": 0.0,
            "scaled_norm_mean": 0.0,
            "scaled_norm_max": 0.0,
            "curvature_radius_fractions": {},
        }
    norms = torch.cat(
        [
            torch.linalg.vector_norm(measure.points.reshape(-1, measure.points.shape[-1]), dim=-1).detach().cpu()
            for measure in measures
        ]
    )
    scaled = norms * float(point_scale)
    curvature_fractions: dict[str, dict[str, float]] = {}
    for curvature in curvatures:
        if curvature <= 0.0:
            continue
        sqrt_c = math.sqrt(float(curvature))
        fractions = scaled * sqrt_c
        curvature_fractions[str(curvature)] = {
            "ball_radius": 1.0 / sqrt_c,
            "scaled_radius_fraction_min": float(torch.min(fractions)),
            "scaled_radius_fraction_median": float(torch.quantile(fractions, 0.5)),
            "scaled_radius_fraction_mean": float(torch.mean(fractions)),
            "scaled_radius_fraction_max": float(torch.max(fractions)),
            "near_boundary_fraction_gt_0_8": float(torch.mean((fractions > 0.8).to(torch.float32))),
        }
    return {
        "point_scale": float(point_scale),
        "unscaled_norm_min": float(torch.min(norms)),
        "unscaled_norm_median": float(torch.quantile(norms, 0.5)),
        "unscaled_norm_mean": float(torch.mean(norms)),
        "unscaled_norm_max": float(torch.max(norms)),
        "scaled_norm_min": float(torch.min(scaled)),
        "scaled_norm_median": float(torch.quantile(scaled, 0.5)),
        "scaled_norm_mean": float(torch.mean(scaled)),
        "scaled_norm_max": float(torch.max(scaled)),
        "curvature_radius_fractions": curvature_fractions,
    }


def _cost_component_diagnostics(
    measures: Sequence[ProductMeasure],
    *,
    geometry: ConstantCurvatureProduct,
    total_costs: Sequence[torch.Tensor],
) -> dict[str, float]:
    """Separate geometric path-object cost from Euclidean side-feature cost."""

    point_geometry = ConstantCurvatureProduct(
        curvature=geometry.curvature,
        factor_weights=geometry.factor_weights,
        side_weight=0.0,
        unoriented=geometry.unoriented,
    )
    point_costs = [point_geometry.path_cost_matrix(left, right) for left in measures for right in measures]
    side_costs = [torch.clamp(total - point, min=0.0) for total, point in zip(total_costs, point_costs)]
    total_scale = median_positive_cost_scale(total_costs)
    point_scale = _median_positive_or_zero(point_costs)
    side_scale = _median_positive_or_zero(side_costs)
    return {
        "total_cost_scale": float(total_scale),
        "point_cost_scale": float(point_scale),
        "side_cost_scale": float(side_scale),
        "point_cost_share": float(point_scale / total_scale) if total_scale > 0.0 else 0.0,
        "side_cost_share": float(side_scale / total_scale) if total_scale > 0.0 else 0.0,
    }


def _geometry_for_cost_mode(
    train_measures: Sequence[ProductMeasure],
    train_items: Sequence[LabeledItem],
    *,
    curvature: float,
    cost_spec: CostSpec,
) -> tuple[ConstantCurvatureProduct, dict[str, Any]]:
    factor_count = train_measures[0].points.shape[1]
    if cost_spec.mode == "point_only":
        return (
            ConstantCurvatureProduct(curvature=curvature, factor_weights=(1.0,) * factor_count, side_weight=0.0),
            {"source": "prespecified", "component": "point_only"},
        )
    if cost_spec.mode == "side_only":
        return (
            ConstantCurvatureProduct(curvature=curvature, factor_weights=(0.0,) * factor_count, side_weight=cost_spec.side_weight),
            {"source": "prespecified", "component": "side_only"},
        )
    if cost_spec.mode == "unnormalized_combined":
        return (
            ConstantCurvatureProduct(curvature=curvature, factor_weights=(1.0,) * factor_count, side_weight=cost_spec.side_weight),
            {"source": "none"},
        )
    if cost_spec.mode not in {"train_normalized_combined", "train_weighted_combined", "validation_selected_combined"}:
        raise ValueError(f"unknown cost mode: {cost_spec.mode!r}")

    factor_scales = _train_factor_cost_scales(train_measures, curvature=curvature)
    unit_factor_weights, factor_diagnostics = _normalized_factor_weights(factor_scales)
    side_scale = _train_side_cost_scale(train_measures, curvature=curvature)
    if cost_spec.mode == "train_weighted_combined":
        if cost_spec.point_weight is None:
            raise ValueError("train_weighted_combined requires point_weight")
        point_weight = float(cost_spec.point_weight)
        side_block_weight = 1.0 - point_weight
        factor_weights = tuple(point_weight * weight for weight in unit_factor_weights)
        normalized_side_weight = side_block_weight * cost_spec.side_weight / side_scale
        return (
            ConstantCurvatureProduct(curvature=curvature, factor_weights=factor_weights, side_weight=normalized_side_weight),
            {
                "source": "train_split_fixed_weight",
                "factor_scales": list(factor_scales),
                **factor_diagnostics,
                "side_scale": side_scale,
                "base_side_weight": cost_spec.side_weight,
                "point_weight": point_weight,
                "side_weight": side_block_weight,
            },
        )

    if cost_spec.mode == "validation_selected_combined":
        selection = _select_validation_product_weight(
            train_items=train_items,
            train_measures=train_measures,
            curvature=curvature,
            unit_factor_weights=unit_factor_weights,
            side_scale=side_scale,
            base_side_weight=cost_spec.side_weight,
            point_weight_grid=VALIDATION_POINT_WEIGHT_GRID,
        )
        point_weight = float(selection["selected_point_weight"])
        side_block_weight = float(selection["selected_side_weight"])
        factor_weights = tuple(point_weight * weight for weight in unit_factor_weights)
        normalized_side_weight = side_block_weight * cost_spec.side_weight / side_scale
        return (
            ConstantCurvatureProduct(curvature=curvature, factor_weights=factor_weights, side_weight=normalized_side_weight),
            {
                "source": "train_split_internal_validation",
                "factor_scales": list(factor_scales),
                **factor_diagnostics,
                "side_scale": side_scale,
                "base_side_weight": cost_spec.side_weight,
                **selection,
            },
        )

    factor_weights = unit_factor_weights
    normalized_side_weight = cost_spec.side_weight / side_scale
    return (
        ConstantCurvatureProduct(curvature=curvature, factor_weights=factor_weights, side_weight=normalized_side_weight),
        {
            "source": "train_split",
            "factor_scales": list(factor_scales),
            **factor_diagnostics,
            "side_scale": side_scale,
            "base_side_weight": cost_spec.side_weight,
        },
    )


def _train_factor_cost_scales(measures: Sequence[ProductMeasure], *, curvature: float) -> tuple[float, ...]:
    factor_count = measures[0].points.shape[1]
    scales = []
    for factor_index in range(factor_count):
        weights = tuple(1.0 if index == factor_index else 0.0 for index in range(factor_count))
        geometry = ConstantCurvatureProduct(curvature=curvature, factor_weights=weights, side_weight=0.0)
        costs = [geometry.path_cost_matrix(left, right) for left in measures for right in measures]
        scales.append(_median_positive_or_zero(costs))
    return tuple(scales)


def _normalized_factor_weights(
    factor_scales: Sequence[float],
    *,
    relative_floor: float = 1e-6,
    absolute_floor: float = 1e-12,
) -> tuple[tuple[float, ...], dict[str, Any]]:
    """Normalize the whole point block while suppressing degenerate factors.

    Every active factor receives an equal share of the point-block budget after
    scale normalization. This keeps single-point and product-path cells matched
    despite their different factor counts. Near-constant factors are assigned
    zero weight instead of amplifying floating-point noise.
    """

    if not factor_scales:
        raise ValueError("factor_scales must not be empty")
    if relative_floor < 0.0 or absolute_floor <= 0.0:
        raise ValueError("factor scale floors must be non-negative and positive, respectively")
    scales = tuple(float(scale) for scale in factor_scales)
    if any(not math.isfinite(scale) or scale < 0.0 for scale in scales):
        raise ValueError("factor_scales must be finite and non-negative")
    threshold = max(absolute_floor, max(scales) * relative_floor)
    active = tuple(index for index, scale in enumerate(scales) if scale > threshold)
    if not active:
        return (
            tuple(0.0 for _ in scales),
            {
                "factor_scale_floor": threshold,
                "active_factor_indices": [],
                "degenerate_factor_indices": list(range(len(scales))),
                "active_factor_count": 0,
            },
        )
    active_count = len(active)
    weights = tuple((1.0 / (active_count * scale)) if index in active else 0.0 for index, scale in enumerate(scales))
    return (
        weights,
        {
            "factor_scale_floor": threshold,
            "active_factor_indices": list(active),
            "degenerate_factor_indices": [index for index in range(len(scales)) if index not in active],
            "active_factor_count": active_count,
        },
    )


def _train_side_cost_scale(measures: Sequence[ProductMeasure], *, curvature: float) -> float:
    factor_count = measures[0].points.shape[1]
    geometry = ConstantCurvatureProduct(curvature=curvature, factor_weights=(0.0,) * factor_count, side_weight=1.0)
    costs = [geometry.path_cost_matrix(left, right) for left in measures for right in measures]
    return median_positive_cost_scale(costs)


def _select_validation_product_weight(
    *,
    train_items: Sequence[LabeledItem],
    train_measures: Sequence[ProductMeasure],
    curvature: float,
    unit_factor_weights: Sequence[float],
    side_scale: float,
    base_side_weight: float,
    point_weight_grid: Sequence[float],
) -> dict[str, Any]:
    """Choose a point/side product-cost balance using only the training split.

    The selector uses a fast expected-ground-cost proxy instead of Sinkhorn
    retrieval. This keeps the validation step cheap and avoids touching the
    held-out query/gallery evaluation split.
    """

    factor_count = train_measures[0].points.shape[1]
    normalized_point = ConstantCurvatureProduct(
        curvature=curvature,
        factor_weights=tuple(unit_factor_weights),
        side_weight=0.0,
    )
    normalized_side = ConstantCurvatureProduct(
        curvature=curvature,
        factor_weights=(0.0,) * factor_count,
        side_weight=base_side_weight / side_scale,
    )
    internal_queries, internal_gallery = _train_internal_retrieval_split(train_items)
    by_query: dict[int, list[tuple[bool, float, float]]] = {}
    for query_index in internal_queries:
        query_item = train_items[query_index]
        for candidate_index in internal_gallery:
            candidate_item = train_items[candidate_index]
            is_positive = query_item.task == candidate_item.task
            point_cost = _expected_total_ground_cost(
                normalized_point,
                train_measures[query_index],
                train_measures[candidate_index],
            )
            side_cost = _expected_total_ground_cost(
                normalized_side,
                train_measures[query_index],
                train_measures[candidate_index],
            )
            by_query.setdefault(query_index, []).append((is_positive, point_cost, side_cost))

    grid_scores = []
    for raw_point_weight in point_weight_grid:
        point_weight = float(raw_point_weight)
        side_weight = 1.0 - point_weight
        reciprocal_ranks = []
        for rows in by_query.values():
            if not any(is_positive for is_positive, _, _ in rows):
                continue
            scored = [
                (point_weight * point_cost + side_weight * side_cost, position, is_positive)
                for position, (is_positive, point_cost, side_cost) in enumerate(rows)
            ]
            ordered = sorted(scored, key=lambda value: (value[0], value[1]))
            positive_ranks = [rank for rank, (_, _, is_positive) in enumerate(ordered, start=1) if is_positive]
            if positive_ranks:
                reciprocal_ranks.append(1.0 / min(positive_ranks))
        mean_mrr = sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0.0
        grid_scores.append(
            {
                "point_weight": point_weight,
                "side_weight": side_weight,
                "proxy_mrr": mean_mrr,
                "query_count": len(reciprocal_ranks),
            }
        )

    selected = max(grid_scores, key=lambda row: (row["proxy_mrr"], row["point_weight"]))
    return {
        "selected_point_weight": selected["point_weight"],
        "selected_side_weight": selected["side_weight"],
        "selected_proxy_mrr": selected["proxy_mrr"],
        "selection_query_count": selected["query_count"],
        "selection_strategy": "one_train_gallery_item_per_task",
        "selection_grid": grid_scores,
    }


def _train_internal_retrieval_split(train_items: Sequence[LabeledItem]) -> tuple[tuple[int, ...], tuple[int, ...]]:
    by_task: dict[str, list[int]] = {}
    for index, item in enumerate(train_items):
        by_task.setdefault(item.task, []).append(index)
    internal_queries: list[int] = []
    internal_gallery: list[int] = []
    for _, indices in sorted(by_task.items()):
        if len(indices) < 2:
            continue
        internal_gallery.append(indices[-1])
        internal_queries.extend(indices[:-1])
    return tuple(internal_queries), tuple(internal_gallery)


def _expected_total_ground_cost(
    geometry: ConstantCurvatureProduct,
    left: ProductMeasure,
    right: ProductMeasure,
) -> float:
    matrix = geometry.path_cost_matrix(left, right)
    weights = left.mass.view(-1, 1) * right.mass.view(1, -1)
    return float(torch.sum(weights * matrix).detach())


def _median_positive_or_zero(cost_matrices: Sequence[torch.Tensor]) -> float:
    positives = []
    for matrix in cost_matrices:
        values = torch.as_tensor(matrix).detach().reshape(-1)
        values = values[torch.isfinite(values) & (values > 0.0)]
        if values.numel():
            positives.append(values.cpu())
    if not positives:
        return 0.0
    return float(torch.quantile(torch.cat(positives), 0.5))


def _run_factor_cell(
    *,
    split: SplitItems,
    cell: FactorCell,
    mode_payload: dict[str, list[ProductMeasure]],
    benchmark_level: BenchmarkLevel,
    sinkhorn_iterations: int,
    sinkhorn_projection_iterations: int,
    hard_negatives_per_query: int,
    kappa: float,
    cost_spec: CostSpec,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    train_measures = _aggregate_measures(
        mode_payload["train"],
        method_aggregation=cell.method_aggregation,
        curvature=cell.curvature,
    )
    query_measures = _aggregate_measures(
        mode_payload["query"],
        method_aggregation=cell.method_aggregation,
        curvature=cell.curvature,
    )
    gallery_measures = _aggregate_measures(
        mode_payload["gallery"],
        method_aggregation=cell.method_aggregation,
        curvature=cell.curvature,
    )
    geometry, cost_normalization = _geometry_for_cost_mode(
        train_measures,
        split.train,
        curvature=cell.curvature,
        cost_spec=cost_spec,
    )
    train_costs = [geometry.path_cost_matrix(left, right) for left in train_measures for right in train_measures]
    cost_scale = median_positive_cost_scale(train_costs)
    epsilon = scaled_sinkhorn_epsilon(cost_scale, kappa=kappa)
    cost_component_diagnostics = _cost_component_diagnostics(
        train_measures,
        geometry=geometry,
        total_costs=train_costs,
    )
    return _evaluate_cell(
        split=split,
        query_measures=query_measures,
        gallery_measures=gallery_measures,
        geometry=geometry,
        benchmark_level=benchmark_level,
        epsilon=epsilon,
        sinkhorn_iterations=sinkhorn_iterations,
        sinkhorn_projection_iterations=sinkhorn_projection_iterations,
        hard_negatives_per_query=hard_negatives_per_query,
        cell=cell,
        metadata={
            **metadata,
            "cost_mode": cost_spec.label,
            "cost_mode_family": cost_spec.mode,
            "side_weight": geometry.side_weight,
            "factor_weights": list(geometry.factor_weights),
            "cost_normalization": cost_normalization,
            "cost_scale": cost_scale,
            "epsilon": epsilon,
            "cost_component_diagnostics": cost_component_diagnostics,
        },
    )


def _evaluate_cell(
    *,
    split: SplitItems,
    query_measures: Sequence[ProductMeasure],
    gallery_measures: Sequence[ProductMeasure],
    geometry: ConstantCurvatureProduct,
    benchmark_level: BenchmarkLevel,
    epsilon: float,
    sinkhorn_iterations: int,
    sinkhorn_projection_iterations: int,
    hard_negatives_per_query: int,
    cell: FactorCell,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    query_records = []
    ranks = []
    recalls_at_1 = []
    recalls_at_5 = []
    candidate_counts = []
    positive_counts = []
    total_distances_for_diagnostics = []
    point_expected_costs = []
    side_expected_costs = []
    total_expected_costs = []
    transport_entropies = []
    for query_item, query_measure in zip(split.query, query_measures):
        candidate_indices = _candidate_indices(
            query_item,
            split.gallery,
            benchmark_level=benchmark_level,
            hard_negatives_per_query=hard_negatives_per_query,
        )
        if not candidate_indices:
            continue
        positive_positions = [position for position, index in enumerate(candidate_indices) if split.gallery[index].task == query_item.task]
        if not positive_positions:
            continue
        distances = []
        pair_payloads = []
        for index in candidate_indices:
            gallery_measure = gallery_measures[index]
            value = _sinkhorn_divergence_with_retry(
                geometry,
                query_measure,
                gallery_measure,
                epsilon=epsilon,
                iterations=sinkhorn_iterations,
                projection_iterations=sinkhorn_projection_iterations,
            )
            distance = float(value.detach())
            distances.append(distance)
            point_expected, side_expected, total_expected = _expected_component_costs(
                geometry,
                query_measure,
                gallery_measure,
            )
            pair_payloads.append((point_expected, side_expected, total_expected))
            total_distances_for_diagnostics.append(distance)
            point_expected_costs.append(point_expected)
            side_expected_costs.append(side_expected)
            total_expected_costs.append(total_expected)
        ordered = sorted(range(len(distances)), key=lambda position: (distances[position], position))
        best_positive_rank = min(ordered.index(position) + 1 for position in positive_positions)
        ranks.append(best_positive_rank)
        recalls_at_1.append(float(best_positive_rank <= 1))
        recalls_at_5.append(float(best_positive_rank <= 5))
        candidate_counts.append(len(candidate_indices))
        positive_counts.append(len(positive_positions))
        nearest_negative_position = next((position for position in ordered if position not in positive_positions), None)
        nearest_positive_position = min(positive_positions, key=lambda position: (distances[position], position))
        entropy_positions = [nearest_positive_position]
        if nearest_negative_position is not None:
            entropy_positions.append(nearest_negative_position)
        for position in entropy_positions:
            entropy = _transport_entropy(
                geometry,
                query_measure,
                gallery_measures[candidate_indices[position]],
                epsilon=epsilon,
                iterations=sinkhorn_iterations,
                projection_iterations=sinkhorn_projection_iterations,
            )
            transport_entropies.append(entropy)
        query_records.append(
            {
                "query_id": query_item.item.item_id,
                "query_task": query_item.task,
                "rank": best_positive_rank,
                "candidate_count": len(candidate_indices),
                "positive_count": len(positive_positions),
                "nearest_positive_id": split.gallery[candidate_indices[nearest_positive_position]].item.item_id,
                "nearest_positive_distance": distances[nearest_positive_position],
                "nearest_negative_id": split.gallery[candidate_indices[nearest_negative_position]].item.item_id if nearest_negative_position is not None else "",
                "nearest_negative_task": split.gallery[candidate_indices[nearest_negative_position]].task if nearest_negative_position is not None else "",
                "nearest_negative_distance": distances[nearest_negative_position] if nearest_negative_position is not None else float("inf"),
                "nearest_positive_point_expected_cost": pair_payloads[nearest_positive_position][0],
                "nearest_positive_side_expected_cost": pair_payloads[nearest_positive_position][1],
                "nearest_negative_point_expected_cost": pair_payloads[nearest_negative_position][0] if nearest_negative_position is not None else float("inf"),
                "nearest_negative_side_expected_cost": pair_payloads[nearest_negative_position][1] if nearest_negative_position is not None else float("inf"),
            }
        )
    if not ranks:
        raise ValueError("evaluation produced no valid queries")
    return {
        **metadata,
        "cell_id": cell.cell_id,
        "geometry": cell.geometry,
        "curvature": cell.curvature,
        "path_object_mode": cell.path_object_mode,
        "method_aggregation": cell.method_aggregation,
        "benchmark_level": benchmark_level,
        "query_count": len(ranks),
        "candidate_count_mean": sum(candidate_counts) / len(candidate_counts),
        "positive_count_mean": sum(positive_counts) / len(positive_counts),
        "recall_at_1": sum(recalls_at_1) / len(recalls_at_1),
        "recall_at_5": sum(recalls_at_5) / len(recalls_at_5),
        "mrr": sum(1.0 / rank for rank in ranks) / len(ranks),
        "mean_rank": sum(float(rank) for rank in ranks) / len(ranks),
        "retrieval_diagnostics": _retrieval_diagnostics(
            total_distances=total_distances_for_diagnostics,
            point_expected_costs=point_expected_costs,
            side_expected_costs=side_expected_costs,
            total_expected_costs=total_expected_costs,
            transport_entropies=transport_entropies,
        ),
        "query_records": query_records,
    }


def _expected_component_costs(
    geometry: ConstantCurvatureProduct,
    left: ProductMeasure,
    right: ProductMeasure,
) -> tuple[float, float, float]:
    """Mass-weighted expected point/side ground-cost components for one pair."""

    point_geometry = ConstantCurvatureProduct(
        curvature=geometry.curvature,
        factor_weights=geometry.factor_weights,
        side_weight=0.0,
        unoriented=geometry.unoriented,
    )
    total_matrix = geometry.path_cost_matrix(left, right)
    point_matrix = point_geometry.path_cost_matrix(left, right)
    side_matrix = torch.clamp(total_matrix - point_matrix, min=0.0)
    weights = left.mass.view(-1, 1) * right.mass.view(1, -1)
    point_expected = torch.sum(weights * point_matrix)
    side_expected = torch.sum(weights * side_matrix)
    total_expected = torch.sum(weights * total_matrix)
    return float(point_expected.detach()), float(side_expected.detach()), float(total_expected.detach())


def _transport_entropy(
    geometry: ConstantCurvatureProduct,
    left: ProductMeasure,
    right: ProductMeasure,
    *,
    epsilon: float,
    iterations: int,
    projection_iterations: int,
) -> float:
    plan = geometry.transport_plan(
        left,
        right,
        epsilon=epsilon,
        iterations=iterations,
        projection_iterations=projection_iterations,
    )
    values = plan.detach().reshape(-1)
    values = values[torch.isfinite(values) & (values > 0.0)]
    if values.numel() == 0:
        return 0.0
    mass = values / torch.clamp(values.sum(), min=1e-30)
    return float((-mass * torch.log(torch.clamp(mass, min=1e-30))).sum().detach())


def _retrieval_diagnostics(
    *,
    total_distances: Sequence[float],
    point_expected_costs: Sequence[float],
    side_expected_costs: Sequence[float],
    total_expected_costs: Sequence[float],
    transport_entropies: Sequence[float],
) -> dict[str, float]:
    return {
        "candidate_pair_count": float(len(total_distances)),
        "total_distance_point_expected_cost_spearman": _spearman_correlation(total_distances, point_expected_costs),
        "total_distance_side_expected_cost_spearman": _spearman_correlation(total_distances, side_expected_costs),
        "total_expected_cost_mean": _mean_float(total_expected_costs),
        "total_expected_cost_median": _median_float(total_expected_costs),
        "point_expected_cost_mean": _mean_float(point_expected_costs),
        "point_expected_cost_median": _median_float(point_expected_costs),
        "side_expected_cost_mean": _mean_float(side_expected_costs),
        "side_expected_cost_median": _median_float(side_expected_costs),
        "transport_entropy_mean": _mean_float(transport_entropies),
        "transport_entropy_median": _median_float(transport_entropies),
        "transport_entropy_pair_count": float(len(transport_entropies)),
    }


def _spearman_correlation(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right) or len(left) < 2:
        return 0.0
    left_ranks = _average_ranks(left)
    right_ranks = _average_ranks(right)
    return _pearson_correlation(left_ranks, right_ranks)


def _average_ranks(values: Sequence[float]) -> list[float]:
    indexed = sorted(enumerate(float(value) for value in values), key=lambda item: item[1])
    ranks = [0.0 for _ in indexed]
    index = 0
    while index < len(indexed):
        end = index + 1
        while end < len(indexed) and indexed[end][1] == indexed[index][1]:
            end += 1
        average_rank = 0.5 * (index + 1 + end)
        for position in range(index, end):
            ranks[indexed[position][0]] = average_rank
        index = end
    return ranks


def _pearson_correlation(left: Sequence[float], right: Sequence[float]) -> float:
    left_mean = _mean_float(left)
    right_mean = _mean_float(right)
    left_centered = [float(value) - left_mean for value in left]
    right_centered = [float(value) - right_mean for value in right]
    numerator = sum(a * b for a, b in zip(left_centered, right_centered))
    left_norm = math.sqrt(sum(value * value for value in left_centered))
    right_norm = math.sqrt(sum(value * value for value in right_centered))
    denominator = left_norm * right_norm
    if denominator <= 0.0:
        return 0.0
    return float(numerator / denominator)


def _mean_float(values: Sequence[float]) -> float:
    return float(sum(float(value) for value in values) / len(values)) if values else 0.0


def _median_float(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return 0.5 * (ordered[middle - 1] + ordered[middle])


def _sinkhorn_divergence_with_retry(
    geometry: ConstantCurvatureProduct,
    left: ProductMeasure,
    right: ProductMeasure,
    *,
    epsilon: float,
    iterations: int,
    projection_iterations: int,
) -> torch.Tensor:
    """Compute Sinkhorn divergence, increasing balancing precision if needed.

    Small relative epsilons can require more Sinkhorn-Knopp projection steps to
    pass the coupling marginal checks. Retrying only those cases preserves the
    requested regularization while avoiding a globally expensive 4096-step run.
    """

    attempts = tuple(dict.fromkeys((projection_iterations, max(2048, projection_iterations * 2), max(4096, projection_iterations * 4))))
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
            message = str(error)
            if "coupling row sums" not in message and "coupling column sums" not in message:
                raise
            last_error = error
    assert last_error is not None
    raise last_error


def _payload(
    tasks: Sequence[TaskSource],
    split: SplitItems,
    rows: Sequence[dict[str, Any]],
    cells: Sequence[FactorCell],
    side_weights: Sequence[float],
    point_weights: Sequence[float],
    cost_modes: Sequence[str],
    *,
    benchmark_level: BenchmarkLevel,
    language: str,
    dim: int,
    epochs: int,
    seed: int,
    reproducibility: dict[str, Any],
    item_scope: ItemScope,
    encoder_policy: EncoderPolicy,
    path_selection_policy: str,
) -> dict[str, Any]:
    expected_runs = len(cells) * len(cost_modes)
    return {
        "experiment": "dta_code2hyp_factor_matrix",
        "benchmark_level": benchmark_level,
        "status": "complete" if len(rows) == expected_runs else "partial",
        "completed_runs": len(rows),
        "expected_runs": expected_runs,
        "config": {
            "tasks": [{"label": task.label, "source": str(task.source)} for task in tasks],
            "language": language,
            "geometries": sorted({cell.geometry for cell in cells}),
            "path_object_modes": sorted({cell.path_object_mode for cell in cells}),
            "method_aggregations": sorted({cell.method_aggregation for cell in cells}),
            "side_weights": list(side_weights),
            "point_weights": list(point_weights),
            "cost_modes": list(cost_modes),
            "dim": dim,
            "epochs": epochs,
            "seed": seed,
            "reproducibility": reproducibility,
            "item_scope": item_scope,
            "encoder_policy": encoder_policy,
            "path_selection_policy": path_selection_policy,
            "split_policy": "disjoint train/query/gallery methods within every task",
            "representation_policy": (
                "one shared frozen Euclidean encoder; cells vary only geometry, path object, and method aggregation"
                if encoder_policy == "shared_euclidean"
                else "matched geometry-aware encoder trained separately for each factor cell under the same retrieval objective"
            ),
            "positive_definition": "accepted gallery solution from the same DTA task, excluding the query method",
            "negative_definition": (
                "all accepted gallery solutions from other tasks"
                if benchmark_level == "B_independent_solution"
                else "structurally similar accepted gallery solutions from other tasks"
            ),
            "primary_contrasts": [
                "LCA-product - single-point at H_1 + measure",
                "measure - centroid at H_1 + LCA-product",
                "H_1e-4 - E at LCA-product + measure",
                "H_1 - H_1e-4 at LCA-product + measure",
                "validation-selected combined product cost - fixed combined product costs",
            ],
        },
        "split": {
            "train_count": len(split.train),
            "query_count": len(split.query),
            "gallery_count": len(split.gallery),
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
    return tuple(TaskSource(_safe_label(label), Path(source)) for label, source in values)


def _safe_label(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip()).strip("._") or "task"


def _parse_csv(value: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in value.split(",") if part.strip())


def _parse_float_csv(value: str | None) -> tuple[float, ...] | None:
    if value is None:
        return None
    return tuple(float(part.strip()) for part in value.split(",") if part.strip())


def _parse_cost_modes(value: str | None) -> tuple[CostMode, ...] | None:
    if value is None:
        return None
    return tuple(part.strip() for part in value.split(",") if part.strip())  # type: ignore[return-value]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the DTA Code2Hyp 3x2x2 frozen factor matrix.")
    parser.add_argument("--task", action="append", nargs=2, metavar=("LABEL", "PATH"), required=True)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "outputs/dta_code2hyp_factor_matrix.json")
    parser.add_argument("--benchmark-level", choices=("B_independent_solution", "C_structural_hard_negative"), default="B_independent_solution")
    parser.add_argument("--geometries", default="E,H_1e-4,H_1")
    parser.add_argument("--path-object-modes", default="single_point,lca_product")
    parser.add_argument("--method-aggregations", default="centroid,measure")
    parser.add_argument("--language", choices=("auto", "java", "python"), default="python")
    parser.add_argument("--dim", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-2)
    parser.add_argument("--max-files-per-task", type=int, default=128)
    parser.add_argument("--max-methods-per-task", type=int, default=48)
    parser.add_argument("--train-per-task", type=int, default=16)
    parser.add_argument("--query-per-task", type=int, default=8)
    parser.add_argument("--gallery-per-task", type=int, default=1)
    parser.add_argument("--max-paths", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260625)
    parser.add_argument("--min-structural-gap", type=float, default=0.05)
    parser.add_argument("--positive-mode", choices=("alpha_rename", "structural_noop", "alpha_structural_noop"), default="alpha_structural_noop")
    parser.add_argument(
        "--item-scope",
        choices=("callable", "module", "callable_or_module"),
        default="callable",
        help="Retrieval unit: functions/methods, whole program/module, or callables with module fallback.",
    )
    parser.add_argument("--sinkhorn-iterations", type=int, default=6)
    parser.add_argument("--sinkhorn-projection-iterations", type=int, default=512)
    parser.add_argument("--kappa", type=float, default=0.05)
    parser.add_argument("--side-weight", type=float, default=1.0)
    parser.add_argument(
        "--side-weights",
        default=None,
        help="Comma-separated side-feature weights for a sweep. Overrides --side-weight when provided.",
    )
    parser.add_argument(
        "--point-weights",
        default=None,
        help="Comma-separated point-channel weights for train_weighted_combined; values must be in [0,1].",
    )
    parser.add_argument(
        "--cost-modes",
        default=None,
        help=(
            "Comma-separated cost modes: point_only,side_only,unnormalized_combined,"
            "train_normalized_combined,train_weighted_combined,validation_selected_combined."
        ),
    )
    parser.add_argument("--max-ball-fraction", type=float, default=0.35)
    parser.add_argument("--hard-negatives-per-query", type=int, default=6)
    parser.add_argument(
        "--encoder-policy",
        choices=("shared_euclidean", "geometry_aware"),
        default="shared_euclidean",
        help="Use the frozen shared Euclidean encoder or train a matched geometry-aware encoder per factor cell.",
    )
    parser.add_argument(
        "--path-selection-policy",
        choices=("preorder_first", "hash_sorted", "lca_depth_stratified", "lca_depth_affine_sampled"),
        default="lca_depth_stratified",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    payload = run_dta_factor_matrix(
        tasks=_parse_tasks(args.task),
        output_path=args.output,
        benchmark_level=args.benchmark_level,
        geometries=_parse_csv(args.geometries),  # type: ignore[arg-type]
        path_object_modes=_parse_csv(args.path_object_modes),  # type: ignore[arg-type]
        method_aggregations=_parse_csv(args.method_aggregations),  # type: ignore[arg-type]
        language=args.language,
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
        min_structural_gap=args.min_structural_gap,
        positive_mode=args.positive_mode,
        item_scope=args.item_scope,
        sinkhorn_iterations=args.sinkhorn_iterations,
        sinkhorn_projection_iterations=args.sinkhorn_projection_iterations,
        kappa=args.kappa,
        side_weight=args.side_weight,
        side_weights=_parse_float_csv(args.side_weights),
        point_weights=_parse_float_csv(args.point_weights),
        cost_modes=_parse_cost_modes(args.cost_modes),
        max_ball_fraction=args.max_ball_fraction,
        hard_negatives_per_query=args.hard_negatives_per_query,
        encoder_policy=args.encoder_policy,
        path_selection_policy=args.path_selection_policy,
    )
    print(f"status={payload['status']} completed={payload['completed_runs']}/{payload['expected_runs']} output={args.output}")


if __name__ == "__main__":
    main()
