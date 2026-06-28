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

from geometry_profile_research.constant_curvature import (
    ConstantCurvatureProduct,
    ProductMeasure,
    median_positive_cost_scale,
    scaled_sinkhorn_epsilon,
)
from geometry_profile_research.raw_ast import terminal_to_terminal_paths
from geometry_profile_research.raw_ast_code2hyp import RawASTCode2Hyp, build_raw_ast_token_vocab
from geometry_profile_research.raw_ast_retrieval import (
    ItemScope,
    PositiveMode,
    RawASTRetrievalItem,
    build_retrieval_triples,
    structural_gap,
    terminal_jaccard_similarity,
)
from scripts.run_raw_ast_code2hyp_retrieval import Language, collect_retrieval_items


BenchmarkLevel = Literal["B_independent_solution", "C_structural_hard_negative"]
DEFAULT_CURVATURES = (0.0, 1e-4, 1.0)


@dataclass(frozen=True)
class TaskSource:
    label: str
    source: Path


@dataclass(frozen=True)
class LabeledItem:
    task: str
    item: RawASTRetrievalItem


@dataclass(frozen=True)
class SplitItems:
    train: tuple[LabeledItem, ...]
    query: tuple[LabeledItem, ...]
    gallery: tuple[LabeledItem, ...]


def run_dta_level_b_c_retrieval(
    *,
    tasks: Sequence[TaskSource],
    output_path: Path,
    benchmark_level: BenchmarkLevel = "B_independent_solution",
    language: Language = "python",
    curvatures: Sequence[float] = DEFAULT_CURVATURES,
    dim: int = 4,
    epochs: int = 2,
    learning_rate: float = 1e-2,
    max_files_per_task: int | None = 128,
    max_methods_per_task: int | None = 64,
    train_per_task: int = 16,
    query_per_task: int = 8,
    gallery_per_task: int = 16,
    max_paths: int = 16,
    seed: int = 20260625,
    min_structural_gap: float = 0.05,
    positive_mode: PositiveMode = "alpha_structural_noop",
    item_scope: ItemScope = "callable",
    sinkhorn_iterations: int = 8,
    sinkhorn_projection_iterations: int = 512,
    kappa: float = 0.05,
    side_weight: float = 1.0,
    max_ball_fraction: float = 0.35,
    hard_negatives_per_query: int = 16,
) -> dict[str, Any]:
    """Run Level B/C retrieval with disjoint train/query/gallery methods.

    Level B evaluates whether another independently written accepted solution of
    the same task is retrieved from a gallery containing solutions to multiple
    tasks. Level C restricts negatives to structurally similar solutions from
    other tasks, giving a harder discrimination setting.
    """

    if len(tasks) < 2:
        raise ValueError("at least two task sources are required")
    if benchmark_level not in {"B_independent_solution", "C_structural_hard_negative"}:
        raise ValueError(f"unknown benchmark_level: {benchmark_level!r}")
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
    model, training_history = _train_encoder(
        split.train,
        dim=dim,
        max_paths=max_paths,
        epochs=epochs,
        learning_rate=learning_rate,
        min_structural_gap=min_structural_gap,
        positive_mode=positive_mode,
        sinkhorn_iterations=sinkhorn_iterations,
    )
    train_measures = [_encode_labeled_item(model, labeled_item, max_paths=max_paths) for labeled_item in split.train]
    query_measures = [_encode_labeled_item(model, labeled_item, max_paths=max_paths) for labeled_item in split.query]
    gallery_measures = [_encode_labeled_item(model, labeled_item, max_paths=max_paths) for labeled_item in split.gallery]
    point_scale = _point_scale(train_measures, max_curvature=max(curvatures), max_ball_fraction=max_ball_fraction)
    train_measures = [_scale_measure(measure, point_scale=point_scale) for measure in train_measures]
    query_measures = [_scale_measure(measure, point_scale=point_scale) for measure in query_measures]
    gallery_measures = [_scale_measure(measure, point_scale=point_scale) for measure in gallery_measures]

    rows = []
    for curvature in curvatures:
        geometry = ConstantCurvatureProduct(curvature=float(curvature), side_weight=side_weight)
        train_costs = [geometry.path_cost_matrix(left, right) for left in train_measures for right in train_measures]
        cost_scale = median_positive_cost_scale(train_costs)
        epsilon = scaled_sinkhorn_epsilon(cost_scale, kappa=kappa)
        rows.append(
            _evaluate_level(
                split=split,
                query_measures=query_measures,
                gallery_measures=gallery_measures,
                geometry=geometry,
                benchmark_level=benchmark_level,
                epsilon=epsilon,
                sinkhorn_iterations=sinkhorn_iterations,
                sinkhorn_projection_iterations=sinkhorn_projection_iterations,
                hard_negatives_per_query=hard_negatives_per_query,
                metadata={
                    "curvature": float(curvature),
                    "cost_scale": cost_scale,
                    "epsilon": epsilon,
                    "point_scale": point_scale,
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
                benchmark_level=benchmark_level,
                language=language,
                curvatures=curvatures,
                dim=dim,
                epochs=epochs,
                seed=seed,
                item_scope=item_scope,
            ),
        )
    payload = _payload(
        tasks,
        split,
        rows,
        benchmark_level=benchmark_level,
        language=language,
        curvatures=curvatures,
        dim=dim,
        epochs=epochs,
        seed=seed,
        item_scope=item_scope,
    )
    _write_payload(output_path, payload)
    return payload


def _collect_labeled_items(
    tasks: Sequence[TaskSource],
    *,
    language: Language,
    max_files_per_task: int | None,
    max_methods_per_task: int | None,
    max_paths: int,
    item_scope: ItemScope = "callable",
) -> tuple[LabeledItem, ...]:
    collected = []
    for task in tasks:
        items = collect_retrieval_items(
            (task.source,),
            language=language,
            max_files=max_files_per_task,
            max_methods=max_methods_per_task,
            min_paths=2,
            max_paths=max_paths,
            item_scope=item_scope,
        )
        if len(items) < 3:
            raise ValueError(f"task {task.label!r} must contain at least three method-level items")
        collected.extend(LabeledItem(task=task.label, item=item) for item in items)
    return tuple(collected)


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
            raise ValueError(f"task {task!r} has {len(ordered)} items; {required} required for split")
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


def _train_encoder(
    train_items: Sequence[LabeledItem],
    *,
    dim: int,
    max_paths: int,
    epochs: int,
    learning_rate: float,
    min_structural_gap: float,
    positive_mode: PositiveMode,
    sinkhorn_iterations: int,
) -> tuple[RawASTCode2Hyp, list[dict[str, float]]]:
    raw_items = tuple(item.item for item in train_items)
    vocab = build_raw_ast_token_vocab(tuple(item.tree for item in raw_items), terminal_policy="class", node_input_mode="label_only")
    model = RawASTCode2Hyp(
        vocab,
        dim=dim,
        manifold="euclidean",
        max_paths=max_paths,
        terminal_policy="class",
        node_input_mode="label_only",
        path_object_mode="lca_product",
        method_aggregation="measure",
        path_cost_orientation="directed",
        curvature=1.0,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    triples = build_retrieval_triples(raw_items, min_structural_gap=min_structural_gap, positive_mode=positive_mode)
    history = []
    for epoch in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        loss = model.training_loss(
            triples,
            sinkhorn_epsilon=0.05,
            sinkhorn_iterations=sinkhorn_iterations,
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
            }
        )
    return model, history


def _encode_labeled_item(model: RawASTCode2Hyp, item: LabeledItem, *, max_paths: int) -> ProductMeasure:
    raw = model.encode_method(item.item.tree, paths=terminal_to_terminal_paths(item.item.tree, max_paths=max_paths))
    side = torch.cat((raw.left_branch, raw.right_branch), dim=-1)
    return ProductMeasure(points=raw.points.detach(), mass=raw.mass.detach(), side_features=side.detach())


def _scale_measure(measure: ProductMeasure, *, point_scale: float) -> ProductMeasure:
    return ProductMeasure(points=measure.points * point_scale, mass=measure.mass, side_features=measure.side_features)


def _point_scale(measures: Sequence[ProductMeasure], *, max_curvature: float, max_ball_fraction: float) -> float:
    if max_curvature <= 0.0:
        return 1.0
    max_norm = max(float(torch.linalg.vector_norm(measure.points.reshape(-1, measure.points.shape[-1]), dim=-1).max()) for measure in measures)
    if max_norm <= 0.0:
        return 1.0
    allowed = max_ball_fraction / math.sqrt(max_curvature)
    return min(1.0, allowed / max_norm)


def _evaluate_level(
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
    metadata: dict[str, Any],
) -> dict[str, Any]:
    query_records = []
    ranks = []
    recalls_at_1 = []
    recalls_at_5 = []
    candidate_counts = []
    positive_counts = []
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
        for index in candidate_indices:
            value = geometry.sinkhorn_divergence(
                query_measure,
                gallery_measures[index],
                epsilon=epsilon,
                iterations=sinkhorn_iterations,
                projection_iterations=sinkhorn_projection_iterations,
            )
            distances.append(float(value.detach()))
        ordered = sorted(range(len(distances)), key=lambda position: (distances[position], position))
        best_positive_rank = min(ordered.index(position) + 1 for position in positive_positions)
        ranks.append(best_positive_rank)
        recalls_at_1.append(float(best_positive_rank <= 1))
        recalls_at_5.append(float(best_positive_rank <= 5))
        candidate_counts.append(len(candidate_indices))
        positive_counts.append(len(positive_positions))
        nearest_negative_position = next((position for position in ordered if position not in positive_positions), None)
        nearest_positive_position = min(positive_positions, key=lambda position: (distances[position], position))
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
            }
        )
    if not ranks:
        raise ValueError("evaluation produced no valid queries")
    return {
        **metadata,
        "benchmark_level": benchmark_level,
        "query_count": len(ranks),
        "candidate_count_mean": sum(candidate_counts) / len(candidate_counts),
        "positive_count_mean": sum(positive_counts) / len(positive_counts),
        "recall_at_1": sum(recalls_at_1) / len(recalls_at_1),
        "recall_at_5": sum(recalls_at_5) / len(recalls_at_5),
        "mrr": sum(1.0 / rank for rank in ranks) / len(ranks),
        "mean_rank": sum(float(rank) for rank in ranks) / len(ranks),
        "query_records": query_records,
    }


def _candidate_indices(
    query_item: LabeledItem,
    gallery: Sequence[LabeledItem],
    *,
    benchmark_level: BenchmarkLevel,
    hard_negatives_per_query: int,
) -> list[int]:
    positive_indices = [index for index, candidate in enumerate(gallery) if candidate.task == query_item.task]
    negative_indices = [index for index, candidate in enumerate(gallery) if candidate.task != query_item.task]
    if benchmark_level == "B_independent_solution":
        return positive_indices + negative_indices
    scored_negatives = []
    for index in negative_indices:
        candidate = gallery[index]
        similarity = terminal_jaccard_similarity(query_item.item.tree, candidate.item.tree)
        gap = structural_gap(query_item.item.tree, candidate.item.tree)
        structural_similarity = 1.0 - gap
        scored_negatives.append((structural_similarity, similarity, candidate.item.item_id, index))
    scored_negatives.sort(reverse=True)
    selected_negatives = [index for _, _, _, index in scored_negatives[:hard_negatives_per_query]]
    return positive_indices + selected_negatives


def _payload(
    tasks: Sequence[TaskSource],
    split: SplitItems,
    rows: Sequence[dict[str, Any]],
    *,
    benchmark_level: BenchmarkLevel,
    language: Language,
    curvatures: Sequence[float],
    dim: int,
    epochs: int,
    seed: int,
    item_scope: ItemScope,
) -> dict[str, Any]:
    return {
        "experiment": "dta_level_b_c_retrieval",
        "benchmark_level": benchmark_level,
        "status": "complete" if len(rows) == len(curvatures) else "partial",
        "completed_runs": len(rows),
        "expected_runs": len(curvatures),
        "config": {
            "tasks": [{"label": task.label, "source": str(task.source)} for task in tasks],
            "language": language,
            "curvatures": [float(value) for value in curvatures],
            "dim": dim,
            "epochs": epochs,
            "seed": seed,
            "item_scope": item_scope,
            "split_policy": "disjoint train/query/gallery methods within every task",
            "positive_definition": "accepted gallery solution from the same DTA task, excluding the query method",
            "negative_definition": (
                "all accepted gallery solutions from other tasks"
                if benchmark_level == "B_independent_solution"
                else "structurally similar accepted gallery solutions from other tasks"
            ),
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


def _parse_csv_floats(value: str) -> tuple[float, ...]:
    return tuple(float(part.strip()) for part in value.split(",") if part.strip())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run DTA Level B/C independent-solution retrieval.")
    parser.add_argument("--task", action="append", nargs=2, metavar=("LABEL", "PATH"), required=True)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "outputs/dta_level_b_c_retrieval.json")
    parser.add_argument("--benchmark-level", choices=("B_independent_solution", "C_structural_hard_negative"), default="B_independent_solution")
    parser.add_argument("--language", choices=("auto", "java", "python"), default="python")
    parser.add_argument("--curvatures", default="0,1e-4,1")
    parser.add_argument("--dim", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-2)
    parser.add_argument("--max-files-per-task", type=int, default=128)
    parser.add_argument("--max-methods-per-task", type=int, default=64)
    parser.add_argument("--train-per-task", type=int, default=16)
    parser.add_argument("--query-per-task", type=int, default=8)
    parser.add_argument("--gallery-per-task", type=int, default=16)
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
    parser.add_argument("--sinkhorn-iterations", type=int, default=8)
    parser.add_argument("--sinkhorn-projection-iterations", type=int, default=512)
    parser.add_argument("--kappa", type=float, default=0.05)
    parser.add_argument("--side-weight", type=float, default=1.0)
    parser.add_argument("--max-ball-fraction", type=float, default=0.35)
    parser.add_argument("--hard-negatives-per-query", type=int, default=16)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    payload = run_dta_level_b_c_retrieval(
        tasks=_parse_tasks(args.task),
        output_path=args.output,
        benchmark_level=args.benchmark_level,
        language=args.language,
        curvatures=_parse_csv_floats(args.curvatures),
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
        max_ball_fraction=args.max_ball_fraction,
        hard_negatives_per_query=args.hard_negatives_per_query,
    )
    print(f"status={payload['status']} completed={payload['completed_runs']}/{payload['expected_runs']} output={args.output}")


if __name__ == "__main__":
    main()
