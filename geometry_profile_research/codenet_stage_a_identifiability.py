from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

from geometry_profile_research.raw_ast import RawAstTree, terminal_to_terminal_paths
from geometry_profile_research.raw_ast_code2hyp import _root_to_node_tokens


COUNT_FIELDS = (
    "node_count",
    "distinct_node_signature_count",
    "colliding_node_count",
    "distinct_true_lca_node_count",
    "distinct_lca_signature_count",
    "colliding_true_lca_node_count",
    "selected_path_object_count",
    "distinct_path_signature_count",
    "colliding_path_object_count",
)
RATE_FIELDS = (
    "node_collision_rate",
    "lca_anchor_collision_rate",
    "path_object_collision_rate",
)


def program_identifiability_diagnostics(
    tree: RawAstTree,
    *,
    terminal_policy: str = "class",
    node_input_mode: str = "label_only",
    path_selection_policy: str = "lca_depth_affine_sampled",
    max_paths: int = 64,
) -> dict[str, float | int]:
    """Measure encoder-input collisions for one raw AST program."""

    nodes = tuple(tree.preorder())
    node_signatures = {
        node: _root_to_node_tokens(
            tree,
            node,
            terminal_policy=terminal_policy,
            input_mode=node_input_mode,
        )
        for node in nodes
    }
    paths = terminal_to_terminal_paths(
        tree,
        max_paths=max_paths,
        selection_policy=path_selection_policy,
    )
    if not paths:
        raise ValueError("identifiability audit requires at least one selected AST path")

    lca_nodes = {path.lca(tree) for path in paths}
    lca_signatures = {node_signatures[node] for node in lca_nodes}
    path_objects = {
        (path.lca(tree), *sorted((path.start, path.end)))
        for path in paths
    }
    path_signatures = set()
    for path in paths:
        start_signature = node_signatures[path.start]
        end_signature = node_signatures[path.end]
        endpoints = tuple(sorted((start_signature, end_signature)))
        path_signatures.add((node_signatures[path.lca(tree)], *endpoints))

    counts = {
        "node_count": len(nodes),
        "distinct_node_signature_count": len(set(node_signatures.values())),
        "distinct_true_lca_node_count": len(lca_nodes),
        "distinct_lca_signature_count": len(lca_signatures),
        "selected_path_object_count": len(path_objects),
        "distinct_path_signature_count": len(path_signatures),
    }
    counts["colliding_node_count"] = counts["node_count"] - counts["distinct_node_signature_count"]
    counts["colliding_true_lca_node_count"] = (
        counts["distinct_true_lca_node_count"] - counts["distinct_lca_signature_count"]
    )
    counts["colliding_path_object_count"] = (
        counts["selected_path_object_count"] - counts["distinct_path_signature_count"]
    )
    return {
        **counts,
        "node_collision_rate": _safe_rate(counts["colliding_node_count"], counts["node_count"]),
        "lca_anchor_collision_rate": _safe_rate(
            counts["colliding_true_lca_node_count"],
            counts["distinct_true_lca_node_count"],
        ),
        "path_object_collision_rate": _safe_rate(
            counts["colliding_path_object_count"],
            counts["selected_path_object_count"],
        ),
    }


def summarize_identifiability_diagnostics(
    rows: Sequence[Mapping[str, float | int]],
    *,
    quantiles: Sequence[float] = (0.0, 0.25, 0.5, 0.75, 1.0),
) -> dict[str, Any]:
    """Aggregate fixed per-program diagnostics without task labels."""

    if not rows:
        raise ValueError("identifiability summary requires at least one program")
    if any(value < 0.0 or value > 1.0 for value in quantiles):
        raise ValueError("quantiles must lie in [0, 1]")
    totals = {
        field: sum(int(row[field]) for row in rows)
        for field in COUNT_FIELDS
    }
    micro_rates = {
        "node_collision_rate": _safe_rate(totals["colliding_node_count"], totals["node_count"]),
        "lca_anchor_collision_rate": _safe_rate(
            totals["colliding_true_lca_node_count"],
            totals["distinct_true_lca_node_count"],
        ),
        "path_object_collision_rate": _safe_rate(
            totals["colliding_path_object_count"],
            totals["selected_path_object_count"],
        ),
    }
    program_macro_rates = {
        field: sum(float(row[field]) for row in rows) / len(rows)
        for field in RATE_FIELDS
    }
    distributions = {
        field: {
            _quantile_key(probability): _linear_quantile(
                sorted(float(row[field]) for row in rows),
                probability,
            )
            for probability in quantiles
        }
        for field in RATE_FIELDS
    }
    return {
        "program_count": len(rows),
        "micro_counts": totals,
        "micro_rates": micro_rates,
        "program_macro_rates": program_macro_rates,
        "program_rate_quantiles": distributions,
    }


def _safe_rate(numerator: int, denominator: int) -> float:
    return 0.0 if denominator == 0 else numerator / denominator


def _linear_quantile(sorted_values: Sequence[float], probability: float) -> float:
    if not sorted_values:
        raise ValueError("quantile requires at least one value")
    position = (len(sorted_values) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(sorted_values[lower])
    weight = position - lower
    return float(sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight)


def _quantile_key(probability: float) -> str:
    return f"q{int(round(100 * probability)):03d}"
