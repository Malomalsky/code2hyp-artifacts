from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Iterable, Literal, Sequence

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from geometry_profile_research.java_raw_ast import parse_java_ast_tree
from geometry_profile_research.code2hyp_torch import torch_project_to_ball
from geometry_profile_research.raw_ast import RawAstTree, terminal_to_terminal_paths
from geometry_profile_research.raw_ast_code2hyp import (
    MethodAggregation,
    NodeInputMode,
    PathCostOrientation,
    PathObjectMode,
    RawASTCode2Hyp,
    TerminalPolicy,
    build_raw_ast_token_vocab,
)
from geometry_profile_research.raw_ast_retrieval import (
    ItemScope,
    PositiveMode,
    RawASTRetrievalItem,
    build_retrieval_triples,
    make_positive_tree,
    retrieval_item_trees,
    select_hard_negative,
    structural_gap,
    terminal_jaccard_similarity,
)
from geometry_profile_research.python_raw_ast import parse_python_ast_tree


Geometry = Literal["poincare", "euclidean"]
Language = Literal["auto", "java", "python"]


def run_retrieval_experiment(
    *,
    sources: Sequence[Path],
    output_path: Path,
    language: Language = "auto",
    geometry: Geometry = "poincare",
    dim: int = 8,
    epochs: int = 10,
    learning_rate: float = 1e-2,
    max_files: int | None = None,
    max_methods: int | None = None,
    max_paths: int = 16,
    seed: int = 20260623,
    min_structural_gap: float = 0.05,
    sinkhorn_iterations: int = 30,
    sinkhorn_epsilon: float = 0.05,
    terminal_policy: TerminalPolicy = "type",
    node_input_mode: NodeInputMode = "label_depth_prefix",
    path_object_mode: PathObjectMode = "lca_product",
    method_aggregation: MethodAggregation = "measure",
    path_cost_orientation: PathCostOrientation = "directed",
    curvature: float = 1.0,
    positive_mode: PositiveMode = "alpha_rename",
    item_scope: ItemScope = "callable",
    lambda_edge: float = 0.1,
    lambda_gromov: float = 0.1,
    lambda_branch: float = 0.1,
    lambda_reversal: float = 0.1,
) -> dict[str, Any]:
    """Train/evaluate canonical Code2Hyp on Java method structural retrieval."""

    torch.manual_seed(seed)
    items = collect_retrieval_items(
        sources,
        language=language,
        max_files=max_files,
        max_methods=max_methods,
        min_paths=2,
        max_paths=max_paths,
        item_scope=item_scope,
    )
    if len(items) < 2:
        raise ValueError("retrieval experiment requires at least two method-level items")

    vocab = build_raw_ast_token_vocab(
        tuple(item.tree for item in items),
        terminal_policy=terminal_policy,
        node_input_mode=node_input_mode,
    )
    model = RawASTCode2Hyp(
        vocab,
        dim=dim,
        manifold=geometry,
        max_paths=max_paths,
        terminal_policy=terminal_policy,
        node_input_mode=node_input_mode,
        path_object_mode=path_object_mode,
        method_aggregation=method_aggregation,
        path_cost_orientation=path_cost_orientation,
        curvature=curvature,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    triples = build_retrieval_triples(
        items,
        min_structural_gap=min_structural_gap,
        positive_mode=positive_mode,
    )
    hard_negatives = _hard_negative_diagnostics(items, min_structural_gap=min_structural_gap)

    history = []
    for epoch in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        loss = model.training_loss(
            triples,
            lambda_edge=lambda_edge,
            lambda_gromov=lambda_gromov,
            lambda_branch=lambda_branch,
            lambda_reversal=lambda_reversal,
            sinkhorn_epsilon=sinkhorn_epsilon,
            sinkhorn_iterations=sinkhorn_iterations,
        )
        loss["loss"].backward()
        optimizer.step()
        history.append(
            {
                "epoch": epoch + 1,
                "loss": float(loss["loss"].detach()),
                "retrieval": float(loss["retrieval"].detach()),
                "edge": float(loss["edge"].detach()),
                "gromov_lca": float(loss["gromov_lca"].detach()),
                "branch_length": float(loss["branch_length"].detach()),
                "reversal": float(loss["reversal"].detach()),
            }
        )

    evaluation = evaluate_positive_retrieval(
        model,
        items,
        max_paths=max_paths,
        sinkhorn_iterations=sinkhorn_iterations,
        sinkhorn_epsilon=sinkhorn_epsilon,
        positive_mode=positive_mode,
        include_query_records=True,
    )
    query_records = evaluation.pop("query_records")
    metrics = evaluation
    geometry_diagnostics = _geometry_diagnostics(model, items, max_paths=max_paths)
    transport_type = "sinkhorn_divergence" if method_aggregation == "measure" else "weighted_centroid_distance"
    payload: dict[str, Any] = {
        "config": {
            "geometry": geometry,
            "language": language,
            "dim": dim,
            "epochs": epochs,
            "learning_rate": learning_rate,
            "max_files": max_files,
            "max_methods": max_methods,
            "max_paths": max_paths,
            "seed": seed,
            "min_structural_gap": min_structural_gap,
            "sinkhorn_iterations": sinkhorn_iterations,
            "sinkhorn_epsilon": sinkhorn_epsilon,
            "terminal_policy": terminal_policy,
            "node_input_mode": node_input_mode,
            "path_object_mode": path_object_mode,
            "method_aggregation": method_aggregation,
            "path_cost_orientation": path_cost_orientation,
            "curvature": curvature,
            "positive_mode": positive_mode,
            "item_scope": item_scope,
            "lambda_edge": lambda_edge,
            "lambda_gromov": lambda_gromov,
            "lambda_branch": lambda_branch,
            "lambda_reversal": lambda_reversal,
        },
        "transport": {
            "type": transport_type,
            "epsilon": sinkhorn_epsilon,
            "iterations": sinkhorn_iterations,
            "debiased": method_aggregation == "measure",
            "normalize_cost": True,
            "mass": "uniform over sampled AST terminal-to-terminal paths",
            "method_aggregation": method_aggregation,
            "path_object_mode": path_object_mode,
            "path_cost_orientation": path_cost_orientation,
            "ground_cost": (
                "squared selected-geometry path-object distances plus squared "
                "Euclidean distances between left/right branch GRU codes"
            ),
        },
        "geometry_diagnostics": geometry_diagnostics,
        "item_count": len(items),
        "vocab_size": len(vocab),
        "items": [_item_summary(item) for item in items],
        "hard_negatives": hard_negatives,
        "training_history": history,
        "metrics": metrics,
        "query_records": query_records,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return payload


def collect_retrieval_items(
    sources: Sequence[Path],
    *,
    language: Language = "auto",
    max_files: int | None = None,
    max_methods: int | None = None,
    min_paths: int = 2,
    max_paths: int = 16,
    item_scope: ItemScope = "callable",
) -> tuple[RawASTRetrievalItem, ...]:
    """Parse supported source files and return explicit-scope retrieval items."""

    items: list[RawASTRetrievalItem] = []
    for source_path in _iter_source_files(sources, language=language, max_files=max_files):
        try:
            source_language = _language_for_path(source_path, language=language)
            tree = _parse_source_tree(source_path, language=source_language)
        except Exception:
            continue
        for item_index, subtree in enumerate(retrieval_item_trees(tree, item_scope=item_scope)):
            if len(terminal_to_terminal_paths(subtree, max_paths=max_paths)) < min_paths:
                continue
            name = subtree.attributes.get(subtree.root_id, {}).get("name", "")
            scope_label = "callable" if subtree.root_id != tree.root_id else "module"
            item_id = f"{source_language}:{source_path}:{scope_label}:{item_index}:{name}"
            items.append(RawASTRetrievalItem(item_id=item_id, tree=subtree, language=source_language))
            if max_methods is not None and len(items) >= max_methods:
                return tuple(items)
    return tuple(items)


def collect_java_retrieval_items(
    sources: Sequence[Path],
    *,
    max_files: int | None = None,
    max_methods: int | None = None,
    min_paths: int = 2,
    max_paths: int = 16,
) -> tuple[RawASTRetrievalItem, ...]:
    """Backward-compatible Java-only wrapper."""

    return collect_retrieval_items(
        sources,
        language="java",
        max_files=max_files,
        max_methods=max_methods,
        min_paths=min_paths,
        max_paths=max_paths,
        item_scope="callable",
    )


def evaluate_positive_retrieval(
    model: RawASTCode2Hyp,
    items: Sequence[RawASTRetrievalItem],
    *,
    max_paths: int,
    sinkhorn_iterations: int = 30,
    sinkhorn_epsilon: float = 0.05,
    positive_mode: PositiveMode = "alpha_rename",
    include_query_records: bool = False,
) -> dict[str, Any]:
    """Rank each positive control against method-level negatives."""

    ranks = []
    positive_distances = []
    nearest_negative_distances = []
    query_records: list[dict[str, Any]] = []
    with torch.no_grad():
        for anchor in items:
            anchor_measure = model.encode_method(anchor.tree, paths=terminal_to_terminal_paths(anchor.tree, max_paths=max_paths))
            positive_tree = make_positive_tree(anchor.tree, mode=positive_mode)
            candidate_records = [("__positive__", positive_tree)] + [
                (item.item_id, item.tree) for item in items if item.item_id != anchor.item_id
            ]
            distances = []
            for _, candidate in candidate_records:
                candidate_measure = model.encode_method(candidate, paths=terminal_to_terminal_paths(candidate, max_paths=max_paths))
                distance = model.method_distance(
                    anchor_measure,
                    candidate_measure,
                    epsilon=sinkhorn_epsilon,
                    sinkhorn_iterations=sinkhorn_iterations,
                )
                distances.append(float(distance.detach()))
            positive_distances.append(distances[0])
            ordered = sorted(range(len(distances)), key=lambda index: (distances[index], index))
            rank = ordered.index(0) + 1
            ranks.append(rank)
            if len(distances) > 1:
                nearest_negative_index = min(range(1, len(distances)), key=lambda index: (distances[index], index))
                nearest_negative_distance = distances[nearest_negative_index]
                nearest_negative_id = candidate_records[nearest_negative_index][0]
            else:
                nearest_negative_distance = float("inf")
                nearest_negative_id = ""
            nearest_negative_distances.append(nearest_negative_distance)
            if include_query_records:
                top_candidates = [
                    {
                        "candidate_id": candidate_records[index][0],
                        "distance": distances[index],
                        "is_positive": index == 0,
                    }
                    for index in ordered[: min(5, len(ordered))]
                ]
                query_records.append(
                    {
                        "anchor_id": anchor.item_id,
                        "rank": rank,
                        "candidate_count": len(candidate_records),
                        "positive_distance": distances[0],
                        "nearest_negative_id": nearest_negative_id,
                        "nearest_negative_distance": nearest_negative_distance,
                        "margin": nearest_negative_distance - distances[0],
                        "top_candidates": top_candidates,
                    }
                )
    total = len(ranks)
    margins = [negative - positive for positive, negative in zip(positive_distances, nearest_negative_distances)]
    metrics: dict[str, Any] = {
        "recall_at_1": sum(rank <= 1 for rank in ranks) / total,
        "recall_at_3": sum(rank <= 3 for rank in ranks) / total,
        "recall_at_5": sum(rank <= 5 for rank in ranks) / total,
        "ndcg_at_1": sum(_single_positive_ndcg(rank, 1) for rank in ranks) / total,
        "ndcg_at_3": sum(_single_positive_ndcg(rank, 3) for rank in ranks) / total,
        "ndcg_at_5": sum(_single_positive_ndcg(rank, 5) for rank in ranks) / total,
        "mrr": sum(1.0 / rank for rank in ranks) / total,
        "mean_rank": sum(float(rank) for rank in ranks) / total,
        "positive_distance_mean": sum(positive_distances) / total,
        "nearest_negative_distance_mean": sum(nearest_negative_distances) / total,
        "margin_mean": sum(margins) / total,
        "margin_min": min(margins),
    }
    if include_query_records:
        metrics["query_records"] = query_records
    return metrics


def _single_positive_ndcg(rank: int, cutoff: int) -> float:
    if rank > cutoff:
        return 0.0
    return 1.0 / math.log2(rank + 1.0)


def evaluate_alpha_retrieval(
    model: RawASTCode2Hyp,
    items: Sequence[RawASTRetrievalItem],
    *,
    max_paths: int,
    sinkhorn_iterations: int = 30,
) -> dict[str, float]:
    """Backward-compatible wrapper for the original alpha-renaming control."""

    return evaluate_positive_retrieval(
        model,
        items,
        max_paths=max_paths,
        sinkhorn_iterations=sinkhorn_iterations,
        sinkhorn_epsilon=0.05,
        positive_mode="alpha_rename",
    )


def _geometry_diagnostics(
    model: RawASTCode2Hyp,
    items: Sequence[RawASTRetrievalItem],
    *,
    max_paths: int,
) -> dict[str, Any]:
    if model.manifold == "euclidean":
        return {"manifold": "euclidean"}
    points = []
    with torch.no_grad():
        for item in items:
            measure = model.encode_method(item.tree, paths=terminal_to_terminal_paths(item.tree, max_paths=max_paths))
            points.append(measure.points.reshape(-1, model.dim))
    if not points:
        return {
            "manifold": "poincare",
            "curvature": model.curvature,
            "sqrt_curvature_norm_mean": 0.0,
            "sqrt_curvature_norm_max": 0.0,
            "near_boundary_fraction": 0.0,
            "projection_active_fraction": 0.0,
        }
    all_points = torch.cat(points, dim=0)
    curvature = float(model.curvature)
    sqrt_curvature_norm = torch.sqrt(torch.tensor(curvature, dtype=all_points.dtype, device=all_points.device)) * torch.linalg.vector_norm(
        all_points,
        dim=-1,
    )
    projected_again = torch_project_to_ball(all_points, curvature)
    projection_delta = torch.linalg.vector_norm(projected_again - all_points, dim=-1)
    return {
        "manifold": "poincare",
        "curvature": curvature,
        "fixed_curvature": True,
        "sqrt_curvature_norm_mean": float(sqrt_curvature_norm.mean().detach()),
        "sqrt_curvature_norm_max": float(sqrt_curvature_norm.max().detach()),
        "near_boundary_fraction": float((sqrt_curvature_norm >= 0.99).float().mean().detach()),
        "projection_active_fraction": float((projection_delta > 1e-7).float().mean().detach()),
    }


def _iter_source_files(sources: Sequence[Path], *, language: Language, max_files: int | None) -> tuple[Path, ...]:
    files: list[Path] = []
    for source in sources:
        if source.is_dir():
            files.extend(sorted(path for path in source.rglob("*") if path.is_file() and _path_matches_language(path, language)))
        elif source.is_file() and _path_matches_language(source, language):
            files.append(source)
    return tuple(files[:max_files] if max_files is not None else files)


def _path_matches_language(path: Path, language: Language) -> bool:
    if language == "java":
        return path.suffix == ".java"
    if language == "python":
        return path.suffix == ".py"
    return path.suffix in {".java", ".py"}


def _language_for_path(path: Path, *, language: Language) -> Literal["java", "python"]:
    if language in {"java", "python"}:
        return language
    if path.suffix == ".java":
        return "java"
    if path.suffix == ".py":
        return "python"
    raise ValueError(f"cannot infer source language from suffix: {path}")


def _parse_source_tree(path: Path, *, language: Literal["java", "python"]) -> RawAstTree:
    source = path.read_text(encoding="utf-8", errors="replace")
    if language == "java":
        return parse_java_ast_tree(source)
    return parse_python_ast_tree(source)


def _item_summary(item: RawASTRetrievalItem) -> dict[str, Any]:
    return {
        "item_id": item.item_id,
        "language": item.language,
        "node_count": len(item.tree.preorder()),
        "path_count": len(terminal_to_terminal_paths(item.tree)),
        "root_label": item.tree.labels.get(item.tree.root_id, ""),
        "name": item.tree.attributes.get(item.tree.root_id, {}).get("name", ""),
    }


def _hard_negative_diagnostics(
    items: Sequence[RawASTRetrievalItem],
    *,
    min_structural_gap: float,
) -> list[dict[str, Any]]:
    rows = []
    for item in items:
        candidates = tuple(candidate for candidate in items if candidate.item_id != item.item_id)
        negative = select_hard_negative(item, candidates, min_structural_gap=min_structural_gap)
        rows.append(
            {
                "anchor_id": item.item_id,
                "negative_id": negative.item_id,
                "lexical_similarity": terminal_jaccard_similarity(item.tree, negative.tree),
                "structural_gap": structural_gap(item.tree, negative.tree),
            }
        )
    return rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run canonical raw-AST Code2Hyp structural retrieval.")
    parser.add_argument("--source", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, default=Path("outputs/raw_ast_code2hyp_retrieval.json"))
    parser.add_argument("--language", choices=("auto", "java", "python"), default="auto")
    parser.add_argument("--geometry", choices=("poincare", "euclidean"), default="poincare")
    parser.add_argument("--dim", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=1e-2)
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--max-methods", type=int, default=None)
    parser.add_argument("--max-paths", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260623)
    parser.add_argument("--min-structural-gap", type=float, default=0.05)
    parser.add_argument("--sinkhorn-iterations", type=int, default=30)
    parser.add_argument("--sinkhorn-epsilon", type=float, default=0.05)
    parser.add_argument("--terminal-policy", choices=("type", "class", "value"), default="type")
    parser.add_argument(
        "--node-input-mode",
        choices=("label_only", "label_depth", "label_depth_prefix"),
        default="label_depth_prefix",
    )
    parser.add_argument("--path-object-mode", choices=("single_point", "lca_product"), default="lca_product")
    parser.add_argument("--method-aggregation", choices=("centroid", "measure"), default="measure")
    parser.add_argument("--path-cost-orientation", choices=("directed", "unoriented"), default="directed")
    parser.add_argument("--curvature", type=float, default=1.0)
    parser.add_argument(
        "--item-scope",
        choices=("callable", "module", "callable_or_module"),
        default="callable",
        help="Retrieval unit: functions/methods, whole program/module, or callables with module fallback.",
    )
    parser.add_argument(
        "--positive-mode",
        choices=("alpha_rename", "structural_noop", "alpha_structural_noop"),
        default="alpha_rename",
    )
    parser.add_argument("--lambda-edge", type=float, default=0.1)
    parser.add_argument("--lambda-gromov", type=float, default=0.1)
    parser.add_argument("--lambda-branch", type=float, default=0.1)
    parser.add_argument("--lambda-reversal", type=float, default=0.1)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    payload = run_retrieval_experiment(
        sources=tuple(args.source),
        output_path=args.output,
        language=args.language,
        geometry=args.geometry,
        dim=args.dim,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        max_files=args.max_files,
        max_methods=args.max_methods,
        max_paths=args.max_paths,
        seed=args.seed,
        min_structural_gap=args.min_structural_gap,
        sinkhorn_iterations=args.sinkhorn_iterations,
        sinkhorn_epsilon=args.sinkhorn_epsilon,
        terminal_policy=args.terminal_policy,
        node_input_mode=args.node_input_mode,
        path_object_mode=args.path_object_mode,
        method_aggregation=args.method_aggregation,
        path_cost_orientation=args.path_cost_orientation,
        curvature=args.curvature,
        positive_mode=args.positive_mode,
        item_scope=args.item_scope,
        lambda_edge=args.lambda_edge,
        lambda_gromov=args.lambda_gromov,
        lambda_branch=args.lambda_branch,
        lambda_reversal=args.lambda_reversal,
    )
    metrics = payload["metrics"]
    print(
        f"wrote {args.output} | items={payload['item_count']} "
        f"MRR={metrics['mrr']:.4f} R@1={metrics['recall_at_1']:.4f}"
    )


if __name__ == "__main__":
    main()
