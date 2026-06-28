from __future__ import annotations

import argparse
import json
import math
import random
import sys
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from statistics import mean, median
from typing import Any, Sequence

import torch
from torch import Tensor

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from geometry_profile_research.gromov_wasserstein import (
    MetricMeasureSpace,
    entropic_fused_gromov_wasserstein,
    entropic_gromov_wasserstein,
    metric_measure_space_from_raw_ast_paths,
    sinkhorn_divergence,
)
from geometry_profile_research.java_raw_ast import parse_java_ast_tree
from geometry_profile_research.raw_ast import RawAstPath, RawAstTree, leaf_node_ids, terminal_to_terminal_paths


@dataclass(frozen=True)
class RawAstMethodSpace:
    source_path: str
    scope_node: int
    scope_name: str
    scope_label: str
    tree: RawAstTree
    paths: tuple[RawAstPath, ...]
    node_count: int
    leaf_count: int
    path_count: int
    structural_relation: str
    structural_space: MetricMeasureSpace
    endpoint_space: MetricMeasureSpace
    edge_space: MetricMeasureSpace
    path_tokens: tuple[frozenset[str], ...]
    feature_self_cost: Tensor
    centroid_features: Tensor


def _iter_java_files(sources: Sequence[Path]) -> tuple[Path, ...]:
    files: list[Path] = []
    for source in sources:
        if source.is_dir():
            files.extend(sorted(path for path in source.rglob("*.java") if path.is_file()))
        elif source.is_file() and source.suffix == ".java":
            files.append(source)
    return tuple(files)


def _callable_nodes(tree: RawAstTree) -> tuple[int, ...]:
    return tuple(
        node
        for node in tree.preorder()
        if tree.labels.get(node) in {"MethodDeclaration", "ConstructorDeclaration"}
    )


def _subtree_size(tree: RawAstTree, root_id: int) -> int:
    count = 0
    stack = [root_id]
    while stack:
        node = stack.pop()
        count += 1
        stack.extend(tree.children_by_node.get(node, ()))
    return count


def _path_tokens(tree: RawAstTree, path: RawAstPath) -> frozenset[str]:
    tokens: set[str] = set()
    for index, node in enumerate(path.nodes):
        label = tree.labels.get(node, "")
        tokens.add(f"node:{index}:{label}")
        tokens.add(f"node:any:{label}")
    for left, right in zip(path.nodes, path.nodes[1:]):
        left_label = tree.labels.get(left, "")
        right_label = tree.labels.get(right, "")
        if tree.parent_by_node.get(left) == right:
            direction = "up"
        elif tree.parent_by_node.get(right) == left:
            direction = "down"
        else:
            direction = "cross"
        tokens.add(f"edge:{direction}:{left_label}>{right_label}")
    return frozenset(tokens)


def _jaccard_distance(left: frozenset[str], right: frozenset[str]) -> float:
    union = left | right
    if not union:
        return 0.0
    return 1.0 - len(left & right) / len(union)


def _feature_cost(left_tokens: Sequence[frozenset[str]], right_tokens: Sequence[frozenset[str]]) -> Tensor:
    rows = [
        [_jaccard_distance(left, right) for right in right_tokens]
        for left in left_tokens
    ]
    return torch.tensor(rows, dtype=torch.float32)


def _diameter_normalized_space(space: MetricMeasureSpace) -> MetricMeasureSpace:
    diameter = torch.max(space.distance)
    if float(diameter) <= 1e-12:
        return space
    return MetricMeasureSpace(distance=space.distance / diameter, mass=space.mass)


def _composite_structural_space(tree: RawAstTree, paths: Sequence[RawAstPath], relation_names: Sequence[str]) -> MetricMeasureSpace:
    if not relation_names:
        raise ValueError("composite structural space requires at least one relation")
    spaces = [_structural_space_for_relation(tree, paths, relation_name) for relation_name in relation_names]
    distance = torch.stack([space.distance for space in spaces]).mean(dim=0)
    return MetricMeasureSpace(distance=distance, mass=spaces[0].mass)


def _structural_space_for_relation(tree: RawAstTree, paths: Sequence[RawAstPath], relation: str) -> MetricMeasureSpace:
    if relation == "endpoint":
        raw_space = metric_measure_space_from_raw_ast_paths(tree, paths, relation="unoriented_endpoint")
    elif relation == "lca_depth":
        raw_space = metric_measure_space_from_raw_ast_paths(tree, paths, relation="lca_depth_difference")
    elif relation == "lca_anchored_product":
        raw_space = metric_measure_space_from_raw_ast_paths(tree, paths, relation="lca_anchored_product")
    elif relation == "edge_jaccard":
        raw_space = metric_measure_space_from_raw_ast_paths(tree, paths, relation="edge_jaccard")
    elif relation == "path_length":
        raw_space = metric_measure_space_from_raw_ast_paths(tree, paths, relation="path_length_difference")
    elif relation == "multi_endpoint_lca_edge":
        return _composite_structural_space(tree, paths, ("endpoint", "lca_anchored_product", "edge_jaccard"))
    elif relation == "multi_endpoint_lca_edge_length":
        return _composite_structural_space(
            tree,
            paths,
            ("endpoint", "lca_anchored_product", "edge_jaccard", "path_length"),
        )
    else:
        raise ValueError(f"unknown structural relation: {relation!r}")
    return _diameter_normalized_space(raw_space)


def _upper_mean(matrix: Tensor) -> float:
    if matrix.shape[0] < 2:
        return 0.0
    values = matrix[torch.triu_indices(matrix.shape[0], matrix.shape[1], offset=1).unbind()]
    return float(values.mean())


def _upper_std(matrix: Tensor) -> float:
    if matrix.shape[0] < 3:
        return 0.0
    values = matrix[torch.triu_indices(matrix.shape[0], matrix.shape[1], offset=1).unbind()]
    return float(values.std(unbiased=False))


def _centroid_features(
    tree: RawAstTree,
    root_id: int,
    paths: Sequence[RawAstPath],
    endpoint_space: MetricMeasureSpace,
    edge_space: MetricMeasureSpace,
) -> Tensor:
    lengths = torch.tensor([path.length for path in paths], dtype=torch.float32)
    lca_depths = torch.tensor([tree.depth(path.lca(tree)) for path in paths], dtype=torch.float32)
    return torch.tensor(
        [
            float(lengths.mean()),
            float(lengths.std(unbiased=False)) if lengths.numel() > 1 else 0.0,
            float(lca_depths.mean()),
            float(lca_depths.std(unbiased=False)) if lca_depths.numel() > 1 else 0.0,
            _upper_mean(endpoint_space.distance),
            _upper_std(endpoint_space.distance),
            _upper_mean(edge_space.distance),
            _upper_std(edge_space.distance),
            float(_subtree_size(tree, root_id)),
            float(len(leaf_node_ids(tree, root_id=root_id))),
        ],
        dtype=torch.float32,
    )


def _method_space_from_tree(
    tree: RawAstTree,
    source_path: Path,
    scope_node: int,
    *,
    max_paths_per_method: int,
    min_paths_per_method: int,
    structural_relation: str,
) -> RawAstMethodSpace | None:
    paths = terminal_to_terminal_paths(tree, max_paths=max_paths_per_method, root_id=scope_node)
    if len(paths) < min_paths_per_method:
        return None
    endpoint_space = _diameter_normalized_space(
        metric_measure_space_from_raw_ast_paths(tree, paths, relation="unoriented_endpoint")
    )
    edge_space = metric_measure_space_from_raw_ast_paths(tree, paths, relation="edge_jaccard")
    structural_space = _structural_space_for_relation(tree, paths, structural_relation)
    path_tokens = tuple(_path_tokens(tree, path) for path in paths)
    return RawAstMethodSpace(
        source_path=str(source_path),
        scope_node=scope_node,
        scope_name=tree.attributes.get(scope_node, {}).get("name", ""),
        scope_label=tree.labels.get(scope_node, ""),
        tree=tree,
        paths=tuple(paths),
        node_count=_subtree_size(tree, scope_node),
        leaf_count=len(leaf_node_ids(tree, root_id=scope_node)),
        path_count=len(paths),
        structural_relation=structural_relation,
        structural_space=structural_space,
        endpoint_space=endpoint_space,
        edge_space=edge_space,
        path_tokens=path_tokens,
        feature_self_cost=_feature_cost(path_tokens, path_tokens),
        centroid_features=_centroid_features(tree, scope_node, paths, endpoint_space, edge_space),
    )


def collect_raw_ast_method_spaces(
    sources: Sequence[Path],
    *,
    max_files: int | None = None,
    max_methods: int | None = None,
    max_paths_per_method: int = 16,
    min_paths_per_method: int = 4,
    structural_relation: str = "endpoint",
    sample_seed: int | None = None,
) -> tuple[RawAstMethodSpace, ...]:
    """Collect method-level raw-AST metric-measure spaces from Java sources."""

    files = _iter_java_files(sources)
    if max_files is not None:
        files = files[:max_files]
    spaces: list[RawAstMethodSpace] = []
    for source_path in files:
        try:
            tree = parse_java_ast_tree(source_path.read_text(encoding="utf-8", errors="replace"))
        except Exception:
            continue
        for scope_node in _callable_nodes(tree):
            method_space = _method_space_from_tree(
                tree,
                source_path,
                scope_node,
                max_paths_per_method=max_paths_per_method,
                min_paths_per_method=min_paths_per_method,
                structural_relation=structural_relation,
            )
            if method_space is None:
                continue
            spaces.append(method_space)
            if sample_seed is None and max_methods is not None and len(spaces) >= max_methods:
                return tuple(spaces)
    if sample_seed is not None and max_methods is not None and len(spaces) > max_methods:
        rng = random.Random(sample_seed)
        return tuple(rng.sample(spaces, max_methods))
    return tuple(spaces)


def _summarize(values: Sequence[float]) -> dict[str, float]:
    if not values:
        return {"count": 0.0, "mean": 0.0, "median": 0.0, "p90": 0.0, "max": 0.0}
    ordered = sorted(values)
    p90_index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * 0.9)))
    return {
        "count": float(len(values)),
        "mean": float(mean(values)),
        "median": float(median(values)),
        "p90": float(ordered[p90_index]),
        "max": float(max(values)),
    }


def _average_ranks(values: Sequence[float]) -> list[float]:
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(indexed)
    i = 0
    while i < len(indexed):
        j = i + 1
        while j < len(indexed) and indexed[j][1] == indexed[i][1]:
            j += 1
        rank = 0.5 * (i + j - 1) + 1.0
        for k in range(i, j):
            ranks[indexed[k][0]] = rank
        i = j
    return ranks


def _pearson(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right) or len(left) < 2:
        return 0.0
    left_mean = mean(left)
    right_mean = mean(right)
    numerator = sum((x - left_mean) * (y - right_mean) for x, y in zip(left, right))
    left_norm = math.sqrt(sum((x - left_mean) ** 2 for x in left))
    right_norm = math.sqrt(sum((y - right_mean) ** 2 for y in right))
    if left_norm <= 1e-12 or right_norm <= 1e-12:
        return 0.0
    return numerator / (left_norm * right_norm)


def _spearman(left: Sequence[float], right: Sequence[float]) -> float:
    return _pearson(_average_ranks(left), _average_ranks(right))


def _zscore_rows(matrix: Tensor) -> Tensor:
    mean_value = matrix.mean(dim=0, keepdim=True)
    std_value = matrix.std(dim=0, keepdim=True, unbiased=False)
    return (matrix - mean_value) / torch.clamp(std_value, min=1e-6)


def _empty_distance_matrix(n_items: int) -> dict[str, list[list[float]]]:
    return {
        "centroid": [[0.0 for _ in range(n_items)] for _ in range(n_items)],
        "ot_feature": [[0.0 for _ in range(n_items)] for _ in range(n_items)],
        "gw_structure": [[0.0 for _ in range(n_items)] for _ in range(n_items)],
        "fgw": [[0.0 for _ in range(n_items)] for _ in range(n_items)],
    }


def _retrieval_overlap(distance_matrices: dict[str, list[list[float]]], *, k: int) -> dict[str, float]:
    gold = distance_matrices["fgw"]
    n_items = len(gold)
    if n_items <= 1:
        return {name: 0.0 for name in distance_matrices}
    k = min(k, n_items - 1)
    scores: dict[str, float] = {}
    for name, matrix in distance_matrices.items():
        overlaps: list[float] = []
        for row_index in range(n_items):
            gold_neighbors = [
                index for index, _ in sorted(
                    ((index, value) for index, value in enumerate(gold[row_index]) if index != row_index),
                    key=lambda item: item[1],
                )[:k]
            ]
            predicted_neighbors = [
                index for index, _ in sorted(
                    ((index, value) for index, value in enumerate(matrix[row_index]) if index != row_index),
                    key=lambda item: item[1],
                )[:k]
            ]
            overlaps.append(len(set(gold_neighbors) & set(predicted_neighbors)) / k)
        scores[name] = float(mean(overlaps))
    return scores


def _method_summary(space: RawAstMethodSpace) -> dict[str, Any]:
    return {
        "source_path": space.source_path,
        "scope_node": space.scope_node,
        "scope_name": space.scope_name,
        "scope_label": space.scope_label,
        "node_count": space.node_count,
        "leaf_count": space.leaf_count,
        "path_count": space.path_count,
        "structural_relation": space.structural_relation,
    }


def run_raw_ast_fgw_benchmark(
    sources: Sequence[Path],
    *,
    max_files: int | None = 20,
    max_methods: int | None = 32,
    max_paths_per_method: int = 16,
    min_paths_per_method: int = 4,
    structural_relation: str = "endpoint",
    sample_seed: int | None = None,
    pair_limit: int | None = None,
    alpha: float = 0.5,
    epsilon: float = 0.05,
    gw_iterations: int = 8,
    sinkhorn_iterations: int = 80,
) -> dict[str, Any]:
    spaces = collect_raw_ast_method_spaces(
        sources,
        max_files=max_files,
        max_methods=max_methods,
        max_paths_per_method=max_paths_per_method,
        min_paths_per_method=min_paths_per_method,
        structural_relation=structural_relation,
        sample_seed=sample_seed,
    )
    if len(spaces) < 2:
        raise ValueError("raw-AST FGW benchmark requires at least two eligible methods")

    centroid = _zscore_rows(torch.stack([space.centroid_features for space in spaces]))
    matrices = _empty_distance_matrix(len(spaces))
    pair_records: list[dict[str, Any]] = []
    plan_entropies: list[float] = []
    marginal_residuals: list[float] = []

    pair_indices = list(combinations(range(len(spaces)), 2))
    total_pair_count = len(pair_indices)
    if pair_limit is not None:
        pair_indices = pair_indices[:pair_limit]
    for left_index, right_index in pair_indices:
        left = spaces[left_index]
        right = spaces[right_index]
        feature_cost = _feature_cost(left.path_tokens, right.path_tokens)
        centroid_distance = float(torch.linalg.norm(centroid[left_index] - centroid[right_index]))
        ot_feature = float(
            torch.clamp(
                sinkhorn_divergence(
                    feature_cost,
                    left.endpoint_space.mass,
                    right.endpoint_space.mass,
                    left_self_cost=left.feature_self_cost,
                    right_self_cost=right.feature_self_cost,
                    epsilon=epsilon,
                    iterations=sinkhorn_iterations,
                ),
                min=0.0,
            )
        )
        gw_result = entropic_gromov_wasserstein(
            left.structural_space,
            right.structural_space,
            epsilon=epsilon,
            iterations=gw_iterations,
            sinkhorn_iterations=sinkhorn_iterations,
        )
        fgw_result = entropic_fused_gromov_wasserstein(
            left.structural_space,
            right.structural_space,
            feature_cost,
            alpha=alpha,
            epsilon=epsilon,
            iterations=gw_iterations,
            sinkhorn_iterations=sinkhorn_iterations,
        )
        distances = {
            "centroid": centroid_distance,
            "ot_feature": ot_feature,
            "gw_structure": float(gw_result.objective),
            "fgw": float(fgw_result.objective),
        }
        for name, value in distances.items():
            matrices[name][left_index][right_index] = value
            matrices[name][right_index][left_index] = value
        plan_entropies.append(fgw_result.plan_entropy)
        marginal_residuals.append(fgw_result.max_marginal_residual)
        pair_records.append(
            {
                "left": left_index,
                "right": right_index,
                **distances,
                "fgw_feature_term": float(fgw_result.feature_term) if fgw_result.feature_term is not None else 0.0,
                "fgw_structure_term": float(fgw_result.structure_term),
                "fgw_plan_entropy": fgw_result.plan_entropy,
                "fgw_max_marginal_residual": fgw_result.max_marginal_residual,
            }
        )

    metric_values = {
        name: [record[name] for record in pair_records]
        for name in matrices
    }
    complete_pair_matrix = len(pair_records) == total_pair_count
    return {
        "config": {
            "sources": [str(source) for source in sources],
            "max_files": max_files,
            "max_methods": max_methods,
            "max_paths_per_method": max_paths_per_method,
            "min_paths_per_method": min_paths_per_method,
            "structural_relation": structural_relation,
            "sample_seed": sample_seed,
            "pair_limit": pair_limit,
            "alpha": alpha,
            "epsilon": epsilon,
            "gw_iterations": gw_iterations,
            "sinkhorn_iterations": sinkhorn_iterations,
        },
        "method_count": len(spaces),
        "pair_count": len(pair_records),
        "total_pair_count": total_pair_count,
        "complete_pair_matrix": complete_pair_matrix,
        "methods": [_method_summary(space) for space in spaces],
        "distance_summary": {name: _summarize(values) for name, values in metric_values.items()},
        "spearman_against_fgw": {
            name: _spearman(values, metric_values["fgw"])
            for name, values in metric_values.items()
            if name != "fgw"
        },
        "retrieval_overlap_at_1": _retrieval_overlap(matrices, k=1) if complete_pair_matrix else None,
        "retrieval_overlap_at_3": _retrieval_overlap(matrices, k=3) if complete_pair_matrix else None,
        "mean_plan_entropy": float(mean(plan_entropies)) if plan_entropies else 0.0,
        "max_marginal_residual": float(max(marginal_residuals)) if marginal_residuals else 0.0,
        "pairs": pair_records,
    }


def write_markdown_report(result: dict[str, Any], output_path: Path) -> None:
    lines: list[str] = [
        "# Raw-AST FGW benchmark",
        "",
        "This experiment treats each Java method as a metric-measure space of terminal-to-terminal raw-AST paths.",
        "",
        f"- Methods: `{result['method_count']}`",
        f"- Pairs: `{result['pair_count']}` of `{result['total_pair_count']}`",
        f"- Complete pair matrix: `{result['complete_pair_matrix']}`",
        f"- Mean FGW plan entropy: `{result['mean_plan_entropy']:.6f}`",
        f"- Max marginal residual: `{result['max_marginal_residual']:.6g}`",
        "",
        "## Distance summaries",
        "",
        "| Distance | Mean | Median | P90 | Max |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, summary in result["distance_summary"].items():
        lines.append(
            f"| {name} | {summary['mean']:.6f} | {summary['median']:.6f} | "
            f"{summary['p90']:.6f} | {summary['max']:.6f} |"
        )
    lines.extend(["", "## Agreement with FGW", "", "| Approximation | Spearman vs FGW | Top-1 overlap | Top-3 overlap |", "|---|---:|---:|---:|"])
    top1_scores = result["retrieval_overlap_at_1"] or {}
    top3_scores = result["retrieval_overlap_at_3"] or {}
    for name, value in result["spearman_against_fgw"].items():
        top1 = top1_scores.get(name)
        top3 = top3_scores.get(name)
        top1_text = f"{top1:.6f}" if top1 is not None else "n/a"
        top3_text = f"{top3:.6f}" if top3 is not None else "n/a"
        lines.append(f"| {name} | {value:.6f} | {top1_text} | {top3_text} |")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The FGW distance is used here as the raw-AST structural target, not as a downstream quality metric.",
            "Centroid, feature-only OT and structure-only GW are evaluated as approximations of this target.",
            "The result should therefore be read as a geometry diagnostic for the reviewer-proposed formulation.",
            "",
        ]
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a raw-AST FGW benchmark over Java methods.")
    parser.add_argument("--source", type=Path, action="append", required=True)
    parser.add_argument("--max-files", type=int, default=20)
    parser.add_argument("--max-methods", type=int, default=32)
    parser.add_argument("--max-paths-per-method", type=int, default=16)
    parser.add_argument("--min-paths-per-method", type=int, default=4)
    parser.add_argument("--sample-seed", type=int, default=None)
    parser.add_argument(
        "--structural-relation",
        choices=(
            "endpoint",
            "lca_depth",
            "lca_anchored_product",
            "edge_jaccard",
            "path_length",
            "multi_endpoint_lca_edge",
            "multi_endpoint_lca_edge_length",
        ),
        default="endpoint",
        help="Internal raw-AST relation matrix used as the GW/FGW structural term.",
    )
    parser.add_argument("--pair-limit", type=int, default=None)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--epsilon", type=float, default=0.05)
    parser.add_argument("--gw-iterations", type=int, default=8)
    parser.add_argument("--sinkhorn-iterations", type=int, default=80)
    parser.add_argument("--output", type=Path, default=Path("outputs/raw_ast_fgw_benchmark.json"))
    parser.add_argument("--report", type=Path, default=Path("reports/raw_ast_fgw_benchmark.md"))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = run_raw_ast_fgw_benchmark(
        tuple(args.source),
        max_files=args.max_files,
        max_methods=args.max_methods,
        max_paths_per_method=args.max_paths_per_method,
        min_paths_per_method=args.min_paths_per_method,
        structural_relation=args.structural_relation,
        sample_seed=args.sample_seed,
        pair_limit=args.pair_limit,
        alpha=args.alpha,
        epsilon=args.epsilon,
        gw_iterations=args.gw_iterations,
        sinkhorn_iterations=args.sinkhorn_iterations,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    write_markdown_report(result, args.report)
    print(json.dumps({key: result[key] for key in ("method_count", "pair_count", "spearman_against_fgw", "retrieval_overlap_at_1")}, indent=2, sort_keys=True))
    print(f"wrote {args.output}")
    print(f"wrote {args.report}")


if __name__ == "__main__":
    main()
