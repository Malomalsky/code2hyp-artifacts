#!/usr/bin/env python3
"""Evaluate LCA-path Code2Hyp retrieval variants.

This script is intentionally separate from the confirmatory structural-only
benchmark. The structural-only benchmark answers whether LCA anchoring helps
when lexical values are suppressed. This script evaluates the practical
multiview step: keep the LCA-anchored path object as a clean structural view,
combine it with lexical and AST-count evidence, and select the weights on the
training split before task-level solution retrieval.
"""

from __future__ import annotations

import argparse
import io
import json
import math
import sys
import tokenize
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from geometry_profile_research.python_raw_ast import parse_python_ast_tree
from geometry_profile_research.raw_ast import RawAstPath, RawAstTree, terminal_to_terminal_paths


@dataclass(frozen=True)
class HybridInput:
    dataset: str
    path: Path


VARIANTS = (
    "code2hyp_path_signature_kernel",
    "code2hyp_path_signature_plus_tokens",
    "code2hyp_path_signature_shape_only",
    "code2hyp_multiview_selected",
    "code2hyp_multiview_no_lca_selected",
    "token_ast_selected",
)

VIEWS = ("path_signature_plus_tokens", "token_bag", "ast_node_bag", "token_count_bag", "ast_node_count_bag")
FULL_WEIGHT_GRID = (
    {"path_signature_plus_tokens": 1.0, "token_bag": 0.0, "ast_node_bag": 0.0, "token_count_bag": 0.0, "ast_node_count_bag": 0.0},
    {"path_signature_plus_tokens": 0.0, "token_bag": 1.0, "ast_node_bag": 0.0, "token_count_bag": 0.0, "ast_node_count_bag": 0.0},
    {"path_signature_plus_tokens": 0.0, "token_bag": 0.0, "ast_node_bag": 1.0, "token_count_bag": 0.0, "ast_node_count_bag": 0.0},
    {"path_signature_plus_tokens": 0.0, "token_bag": 0.0, "ast_node_bag": 0.0, "token_count_bag": 1.0, "ast_node_count_bag": 0.0},
    {"path_signature_plus_tokens": 0.0, "token_bag": 0.0, "ast_node_bag": 0.0, "token_count_bag": 0.0, "ast_node_count_bag": 1.0},
    {"path_signature_plus_tokens": 0.2, "token_bag": 0.0, "ast_node_bag": 0.0, "token_count_bag": 0.6, "ast_node_count_bag": 0.2},
    {"path_signature_plus_tokens": 0.2, "token_bag": 0.0, "ast_node_bag": 0.0, "token_count_bag": 0.2, "ast_node_count_bag": 0.6},
    {"path_signature_plus_tokens": 0.2, "token_bag": 0.0, "ast_node_bag": 0.0, "token_count_bag": 0.4, "ast_node_count_bag": 0.4},
    {"path_signature_plus_tokens": 0.1, "token_bag": 0.0, "ast_node_bag": 0.0, "token_count_bag": 0.8, "ast_node_count_bag": 0.1},
    {"path_signature_plus_tokens": 0.1, "token_bag": 0.0, "ast_node_bag": 0.0, "token_count_bag": 0.1, "ast_node_count_bag": 0.8},
    {"path_signature_plus_tokens": 0.34, "token_bag": 0.0, "ast_node_bag": 0.0, "token_count_bag": 0.33, "ast_node_count_bag": 0.33},
    {"path_signature_plus_tokens": 0.0, "token_bag": 0.0, "ast_node_bag": 0.0, "token_count_bag": 0.5, "ast_node_count_bag": 0.5},
    {"path_signature_plus_tokens": 0.0, "token_bag": 0.0, "ast_node_bag": 0.0, "token_count_bag": 0.8, "ast_node_count_bag": 0.2},
    {"path_signature_plus_tokens": 0.0, "token_bag": 0.0, "ast_node_bag": 0.0, "token_count_bag": 0.2, "ast_node_count_bag": 0.8},
)
NO_LCA_VIEWS = ("token_bag", "ast_node_bag", "token_count_bag", "ast_node_count_bag")
NO_LCA_WEIGHT_GRID = (
    {"token_bag": 1.0, "ast_node_bag": 0.0, "token_count_bag": 0.0, "ast_node_count_bag": 0.0},
    {"token_bag": 0.75, "ast_node_bag": 0.25, "token_count_bag": 0.0, "ast_node_count_bag": 0.0},
    {"token_bag": 0.5, "ast_node_bag": 0.5, "token_count_bag": 0.0, "ast_node_count_bag": 0.0},
    {"token_bag": 0.25, "ast_node_bag": 0.75, "token_count_bag": 0.0, "ast_node_count_bag": 0.0},
    {"token_bag": 0.0, "ast_node_bag": 1.0, "token_count_bag": 0.0, "ast_node_count_bag": 0.0},
    {"token_bag": 0.0, "ast_node_bag": 0.0, "token_count_bag": 1.0, "ast_node_count_bag": 0.0},
    {"token_bag": 0.0, "ast_node_bag": 0.0, "token_count_bag": 0.0, "ast_node_count_bag": 1.0},
    {"token_bag": 0.0, "ast_node_bag": 0.0, "token_count_bag": 0.5, "ast_node_count_bag": 0.5},
    {"token_bag": 0.0, "ast_node_bag": 0.0, "token_count_bag": 0.8, "ast_node_count_bag": 0.2},
    {"token_bag": 0.0, "ast_node_bag": 0.0, "token_count_bag": 0.2, "ast_node_count_bag": 0.8},
    {"token_bag": 0.2, "ast_node_bag": 0.0, "token_count_bag": 0.6, "ast_node_count_bag": 0.2},
    {"token_bag": 0.2, "ast_node_bag": 0.0, "token_count_bag": 0.2, "ast_node_count_bag": 0.6},
    {"token_bag": 0.2, "ast_node_bag": 0.0, "token_count_bag": 0.4, "ast_node_count_bag": 0.4},
)
TOKEN_AST_VIEWS = ("token_bag", "ast_node_bag")
TOKEN_AST_WEIGHT_GRID = (
    {"token_bag": 1.0, "ast_node_bag": 0.0},
    {"token_bag": 0.75, "ast_node_bag": 0.25},
    {"token_bag": 0.5, "ast_node_bag": 0.5},
    {"token_bag": 0.25, "ast_node_bag": 0.75},
    {"token_bag": 0.0, "ast_node_bag": 1.0},
)
MULTIVIEW_VARIANTS = {
    "code2hyp_multiview_selected": (VIEWS, FULL_WEIGHT_GRID),
    "code2hyp_multiview_no_lca_selected": (NO_LCA_VIEWS, NO_LCA_WEIGHT_GRID),
    "token_ast_selected": (TOKEN_AST_VIEWS, TOKEN_AST_WEIGHT_GRID),
}
LCA_VIEW_CHOICES = (
    "path_signature_plus_tokens",
    "code2hyp_path_signature_kernel",
    "code2hyp_path_signature_shape_only",
)


def run_hybrid(
    inputs: Sequence[HybridInput],
    *,
    max_paths: int = 128,
    path_selection_policy: str = "preorder_first",
    lca_view: str = "path_signature_plus_tokens",
    lca_selection_margin: float | None = None,
    weight_grid_mode: str = "compact",
) -> dict[str, Any]:
    if lca_view not in LCA_VIEW_CHOICES:
        raise ValueError(f"unknown LCA view: {lca_view!r}")
    if weight_grid_mode not in {"compact", "expanded"}:
        raise ValueError(f"unknown weight grid mode: {weight_grid_mode!r}")
    multiview_variants = _multiview_variants(lca_view, weight_grid_mode=weight_grid_mode)
    multiview_view_names = {
        view
        for views, _weight_grid in multiview_variants.values()
        for view in views
    }
    rows: list[dict[str, Any]] = []
    for item in inputs:
        payload = json.loads(item.path.read_text(encoding="utf-8"))
        split = payload["split"]
        seed = int(payload["config"]["seed"])
        train_ids = [str(value) for value in split["train_ids"]]
        query_ids = [str(value) for value in split["query_ids"]]
        gallery_ids = [str(value) for value in split["gallery_ids"]]
        idf_by_variant = {
            variant: _idf(
                [
                    _raw_features_for_id(
                        item_id,
                        variant,
                        max_paths=max_paths,
                        path_selection_policy=path_selection_policy,
                    )
                    for item_id in train_ids
                ]
            )
            for variant in (set(VARIANTS) | multiview_view_names) - set(multiview_variants)
        }
        raw_cache: dict[tuple[str, str], Counter[str]] = {}
        vector_cache: dict[tuple[str, str], Counter[str]] = {}
        for variant in VARIANTS:
            if variant in multiview_variants:
                views, weight_grid = multiview_variants[variant]
                selected_weights, validation_mrr = _select_multiview_weights(
                    train_ids,
                    idf_by_variant,
                    raw_cache,
                    vector_cache,
                    views=views,
                    weight_grid=weight_grid,
                    max_paths=max_paths,
                    path_selection_policy=path_selection_policy,
                    guard_view=lca_view if variant == "code2hyp_multiview_selected" else None,
                    selection_margin=lca_selection_margin if variant == "code2hyp_multiview_selected" else None,
                )
                gallery_view_vectors = {
                    gallery_id: {
                        view: _vector_for_id(
                            gallery_id,
                            view,
                            idf_by_variant[view],
                            raw_cache,
                            vector_cache,
                            max_paths=max_paths,
                            path_selection_policy=path_selection_policy,
                        )
                        for view in views
                    }
                    for gallery_id in gallery_ids
                }
                for query_id in query_ids:
                    query_task = _task_from_id(query_id)
                    query_views = {
                        view: _vector_for_id(
                            query_id,
                            view,
                            idf_by_variant[view],
                            raw_cache,
                            vector_cache,
                            max_paths=max_paths,
                            path_selection_policy=path_selection_policy,
                        )
                        for view in views
                    }
                    scored = [
                        (
                            _multiview_score(query_views, gallery_views, selected_weights, views=views),
                            gallery_id,
                        )
                        for gallery_id, gallery_views in gallery_view_vectors.items()
                    ]
                    scored.sort(key=lambda value: (-value[0], value[1]))
                    rank = _first_positive_rank(scored, query_task)
                    rows.append(
                        {
                            "dataset": item.dataset,
                            "source_file": str(item.path),
                            "seed": seed,
                            "variant": variant,
                            "selected_weights": selected_weights,
                            "train_validation_mrr": validation_mrr,
                            "query_id": query_id,
                            "query_task": query_task,
                            "rank": rank,
                            "mrr": 1.0 / rank,
                            "recall_at_1": 1.0 if rank <= 1 else 0.0,
                            "recall_at_5": 1.0 if rank <= 5 else 0.0,
                            "mean_rank": float(rank),
                        }
                    )
                continue
            gallery_vectors = {
                gallery_id: _vector_for_id(
                    gallery_id,
                    variant,
                    idf_by_variant[variant],
                    raw_cache,
                    vector_cache,
                    max_paths=max_paths,
                    path_selection_policy=path_selection_policy,
                )
                for gallery_id in gallery_ids
            }
            for query_id in query_ids:
                query_task = _task_from_id(query_id)
                query_vector = _vector_for_id(
                    query_id,
                    variant,
                    idf_by_variant[variant],
                    raw_cache,
                    vector_cache,
                    max_paths=max_paths,
                    path_selection_policy=path_selection_policy,
                )
                scored = [
                    (_cosine(query_vector, gallery_vector), gallery_id)
                    for gallery_id, gallery_vector in gallery_vectors.items()
                ]
                scored.sort(key=lambda value: (-value[0], value[1]))
                rank = _first_positive_rank(scored, query_task)
                rows.append(
                    {
                        "dataset": item.dataset,
                        "source_file": str(item.path),
                        "seed": seed,
                        "variant": variant,
                        "query_id": query_id,
                        "query_task": query_task,
                        "rank": rank,
                        "mrr": 1.0 / rank,
                        "recall_at_1": 1.0 if rank <= 1 else 0.0,
                        "recall_at_5": 1.0 if rank <= 5 else 0.0,
                        "mean_rank": float(rank),
                    }
                )
    return {
        "inputs": [{"dataset": item.dataset, "path": str(item.path)} for item in inputs],
        "max_paths": max_paths,
        "path_selection_policy": path_selection_policy,
        "lca_view": lca_view,
        "lca_selection_margin": lca_selection_margin,
        "weight_grid_mode": weight_grid_mode,
        "weight_grid": {
            variant: grid
            for variant, (_views, grid) in multiview_variants.items()
        },
        "query_rows": rows,
        "cell_summaries": _summaries(rows),
    }


def _multiview_variants(
    lca_view: str,
    *,
    weight_grid_mode: str = "compact",
) -> dict[str, tuple[tuple[str, ...], tuple[Mapping[str, float], ...]]]:
    return {
        "code2hyp_multiview_selected": (
            (lca_view, "token_bag", "ast_node_bag", "token_count_bag", "ast_node_count_bag"),
            _full_weight_grid(lca_view, mode=weight_grid_mode),
        ),
        "code2hyp_multiview_no_lca_selected": (NO_LCA_VIEWS, NO_LCA_WEIGHT_GRID),
        "token_ast_selected": (TOKEN_AST_VIEWS, TOKEN_AST_WEIGHT_GRID),
    }


def _full_weight_grid(lca_view: str, *, mode: str = "compact") -> tuple[Mapping[str, float], ...]:
    zero_lca_grid = _lift_no_lca_grid(lca_view)
    base_grid: Sequence[Mapping[str, float]]
    base_grid = FULL_WEIGHT_GRID if mode == "compact" else (*FULL_WEIGHT_GRID, *_expanded_lca_weight_grid())
    if lca_view == "path_signature_plus_tokens":
        return _deduplicate_weight_grid((*base_grid, *zero_lca_grid))
    renamed = []
    for weights in base_grid:
        updated = {
            (lca_view if view == "path_signature_plus_tokens" else view): value
            for view, value in weights.items()
        }
        renamed.append(updated)
    return _deduplicate_weight_grid((*renamed, *zero_lca_grid))


def _expanded_lca_weight_grid() -> tuple[Mapping[str, float], ...]:
    grid: list[Mapping[str, float]] = []
    lca_weights = (0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5)
    count_splits = (
        (1.0, 0.0),
        (0.8, 0.2),
        (0.6, 0.4),
        (0.5, 0.5),
        (0.4, 0.6),
        (0.2, 0.8),
        (0.0, 1.0),
    )
    token_bag_shares = (0.0, 0.1, 0.2, 0.3)
    for lca_weight in lca_weights:
        residual = 1.0 - lca_weight
        for token_bag_share in token_bag_shares:
            remaining = residual - token_bag_share
            if remaining < 0.0:
                continue
            for token_count_share, ast_count_share in count_splits:
                grid.append(
                    {
                        "path_signature_plus_tokens": lca_weight,
                        "token_bag": token_bag_share,
                        "ast_node_bag": 0.0,
                        "token_count_bag": remaining * token_count_share,
                        "ast_node_count_bag": remaining * ast_count_share,
                    }
                )
    return tuple(grid)


def _lift_no_lca_grid(lca_view: str) -> tuple[Mapping[str, float], ...]:
    return tuple({lca_view: 0.0, **weights} for weights in NO_LCA_WEIGHT_GRID)


def _deduplicate_weight_grid(weight_grid: Sequence[Mapping[str, float]]) -> tuple[Mapping[str, float], ...]:
    seen: set[tuple[tuple[str, float], ...]] = set()
    deduplicated: list[Mapping[str, float]] = []
    for weights in weight_grid:
        key = tuple(sorted((view, float(value)) for view, value in weights.items()))
        if key in seen:
            continue
        seen.add(key)
        deduplicated.append(dict(weights))
    return tuple(deduplicated)


def _raw_features_for_id(
    item_id: str,
    variant: str,
    *,
    max_paths: int,
    path_selection_policy: str = "preorder_first",
) -> Counter[str]:
    path = _path_from_id(item_id)
    tree = parse_python_ast_tree(path.read_text(encoding="utf-8"))
    if variant == "code2hyp_path_signature_shape_only":
        return _path_signature_features(
            tree,
            max_paths=max_paths,
            path_selection_policy=path_selection_policy,
            include_tokens=False,
            include_token_bag=False,
        )
    if variant == "code2hyp_path_signature_kernel":
        return _path_signature_features(
            tree,
            max_paths=max_paths,
            path_selection_policy=path_selection_policy,
            include_tokens=True,
            include_token_bag=False,
        )
    if variant in {"code2hyp_path_signature_plus_tokens", "path_signature_plus_tokens"}:
        features = _path_signature_features(
            tree,
            max_paths=max_paths,
            path_selection_policy=path_selection_policy,
            include_tokens=True,
            include_token_bag=True,
        )
        features.update({f"rawtok:{key}": value for key, value in _token_bag(path, strip_values=False).items()})
        return features
    if variant == "token_bag":
        return Counter({f"rawtok:{key}": value for key, value in _token_bag(path, strip_values=False).items()})
    if variant == "ast_node_bag":
        return Counter({f"ast_node:{key}": value for key, value in Counter(tree.labels.values()).items()})
    if variant == "token_count_bag":
        return Counter({f"rawtok:{key}": value for key, value in _token_bag(path, strip_values=False).items()})
    if variant == "ast_node_count_bag":
        return Counter({f"ast_node:{key}": value for key, value in Counter(tree.labels.values()).items()})
    raise ValueError(f"unknown hybrid variant: {variant}")


def _vector_for_id(
    item_id: str,
    variant: str,
    idf: Mapping[str, float],
    raw_cache: dict[tuple[str, str], Counter[str]],
    vector_cache: dict[tuple[str, str], Counter[str]],
    *,
    max_paths: int,
    path_selection_policy: str = "preorder_first",
) -> Counter[str]:
    key = (item_id, variant)
    if key not in vector_cache:
        raw = raw_cache.setdefault(
            key,
            _raw_features_for_id(
                item_id,
                variant,
                max_paths=max_paths,
                path_selection_policy=path_selection_policy,
            ),
        )
        if variant.endswith("_count_bag"):
            vector_cache[key] = Counter(raw)
        else:
            default_idf = _default_idf(idf)
            vector_cache[key] = Counter({feature: value * idf.get(feature, default_idf) for feature, value in raw.items()})
    return vector_cache[key]


def _path_signature_features(
    tree: RawAstTree,
    *,
    max_paths: int,
    path_selection_policy: str = "preorder_first",
    include_tokens: bool,
    include_token_bag: bool,
) -> Counter[str]:
    paths = terminal_to_terminal_paths(tree, max_paths=max_paths, selection_policy=path_selection_policy)
    if not paths:
        paths = (tree.path_between(tree.root_id, tree.root_id),)
    features: Counter[str] = Counter()
    normalizer = max(len(paths), 1)
    max_depth = max((tree.depth(node) for node in tree.preorder()), default=1) or 1
    for path_object in paths:
        weight = 1.0 / normalizer
        lca = path_object.lca(tree)
        start = path_object.start
        end = path_object.end
        lca_label = _node_label(tree, lca, include_tokens=include_tokens)
        start_label = _node_label(tree, start, include_tokens=include_tokens)
        end_label = _node_label(tree, end, include_tokens=include_tokens)
        length_bin = _bucket(path_object.length, bins=(2, 4, 8, 16, 32))
        depth_bin = _bucket(tree.depth(lca) / max_depth, bins=(0.1, 0.25, 0.5, 0.75, 1.0))
        for feature in (
            f"lca:{lca_label}",
            f"start:{start_label}",
            f"end:{end_label}",
            f"triple:{lca_label}|{start_label}|{end_label}",
            f"len:{length_bin}",
            f"lca_depth:{depth_bin}",
        ):
            features[feature] += weight
        path_labels = [_node_label(tree, node, include_tokens=include_tokens) for node in path_object.nodes]
        for left, right in zip(path_labels, path_labels[1:]):
            features[f"path_bigram:{left}>{right}"] += weight
        for label in path_labels:
            features[f"path_node:{label}"] += 0.25 * weight
    if include_token_bag:
        for node in tree.preorder():
            token = tree.attributes.get(node, {}).get("terminal_token") or tree.attributes.get(node, {}).get("name")
            if token:
                features[f"ast_token:{_normalize_token(token)}"] += 1.0
    return features


def _node_label(tree: RawAstTree, node: int, *, include_tokens: bool) -> str:
    label = tree.labels.get(node, "")
    if not include_tokens:
        return label
    attributes = tree.attributes.get(node, {})
    token = attributes.get("terminal_token")
    if token:
        return f"{label}:{_normalize_token(token)}"
    name = attributes.get("name")
    if name:
        return f"{label}:{_normalize_token(name)}"
    return label


def _normalize_token(token: str) -> str:
    token = " ".join(str(token).split())
    if token.isidentifier():
        return token
    if token.replace(".", "", 1).isdigit():
        return "NUMBER"
    if len(token) > 32:
        return token[:31] + "..."
    return token


def _token_bag(path: Path, *, strip_values: bool) -> Counter[str]:
    source = path.read_text(encoding="utf-8")
    counts: Counter[str] = Counter()
    ignored = {
        tokenize.ENCODING,
        tokenize.ENDMARKER,
        tokenize.INDENT,
        tokenize.DEDENT,
        tokenize.NEWLINE,
        tokenize.NL,
        tokenize.COMMENT,
    }
    for token in tokenize.generate_tokens(io.StringIO(source).readline):
        if token.type in ignored:
            continue
        if strip_values and token.type == tokenize.NAME:
            value = "NAME"
        elif strip_values and token.type == tokenize.NUMBER:
            value = "NUMBER"
        elif strip_values and token.type == tokenize.STRING:
            value = "STRING"
        else:
            value = token.string
        counts[value] += 1
    return counts


def _idf(documents: Sequence[Counter[str]]) -> dict[str, float]:
    document_frequency: Counter[str] = Counter()
    for document in documents:
        document_frequency.update(document.keys())
    n_documents = max(len(documents), 1)
    return {
        feature: math.log((n_documents + 1.0) / (frequency + 1.0)) + 1.0
        for feature, frequency in document_frequency.items()
    }


def _default_idf(idf: Mapping[str, float]) -> float:
    if not idf:
        return 1.0
    return max(idf.values())


def _cosine(left: Counter[str], right: Counter[str]) -> float:
    if not left or not right:
        return 0.0
    dot = sum(value * right.get(feature, 0.0) for feature, value in left.items())
    left_norm = math.sqrt(sum(value * value for value in left.values()))
    right_norm = math.sqrt(sum(value * value for value in right.values()))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return dot / (left_norm * right_norm)


def _select_multiview_weights(
    train_ids: Sequence[str],
    idf_by_variant: Mapping[str, Mapping[str, float]],
    raw_cache: dict[tuple[str, str], Counter[str]],
    vector_cache: dict[tuple[str, str], Counter[str]],
    *,
    views: Sequence[str],
    weight_grid: Sequence[Mapping[str, float]],
    max_paths: int,
    path_selection_policy: str = "preorder_first",
    guard_view: str | None = None,
    selection_margin: float | None = None,
) -> tuple[dict[str, float], float]:
    grouped: dict[str, list[str]] = defaultdict(list)
    for item_id in train_ids:
        grouped[_task_from_id(item_id)].append(item_id)
    grouped = {task: sorted(ids) for task, ids in grouped.items() if len(ids) >= 2}
    if not grouped:
        return dict(weight_grid[0]), 0.0

    all_train_ids = sorted({item_id for ids in grouped.values() for item_id in ids})
    all_views = {
        item_id: {
            view: _vector_for_id(
                item_id,
                view,
                idf_by_variant[view],
                raw_cache,
                vector_cache,
                max_paths=max_paths,
                path_selection_policy=path_selection_policy,
            )
            for view in views
        }
        for item_id in all_train_ids
    }
    max_folds = min(4, min(len(ids) for ids in grouped.values()))
    folds: list[tuple[list[str], list[str]]] = []
    for fold_index in range(max_folds):
        gallery_ids: list[str] = []
        query_ids: list[str] = []
        for _, ids in sorted(grouped.items()):
            gallery_id = ids[fold_index % len(ids)]
            gallery_ids.append(gallery_id)
            query_ids.extend(item_id for item_id in ids if item_id != gallery_id)
        folds.append((query_ids, gallery_ids))

    pair_view_scores: dict[tuple[str, str], dict[str, float]] = {}
    for query_ids, gallery_ids in folds:
        for query_id in query_ids:
            for gallery_id in gallery_ids:
                key = (query_id, gallery_id)
                if key in pair_view_scores:
                    continue
                pair_view_scores[key] = {
                    view: _cosine(all_views[query_id][view], all_views[gallery_id][view])
                    for view in views
                }

    best_weights = dict(weight_grid[0])
    best_mrr = -1.0
    best_zero_weights: dict[str, float] | None = None
    best_zero_mrr = -1.0
    for weights in weight_grid:
        ranks = []
        for query_ids, gallery_ids in folds:
            for query_id in query_ids:
                task = _task_from_id(query_id)
                scored = [
                    (_weighted_cached_score(pair_view_scores[(query_id, gallery_id)], weights, views=views), gallery_id)
                    for gallery_id in gallery_ids
                ]
                scored.sort(key=lambda value: (-value[0], value[1]))
                ranks.append(_first_positive_rank(scored, task))
        mrr = _mean(1.0 / rank for rank in ranks)
        path_weight = weights.get(guard_view or "path_signature_plus_tokens", 0.0)
        best_path_weight = best_weights.get(guard_view or "path_signature_plus_tokens", 0.0)
        if guard_view is not None and float(weights.get(guard_view, 0.0)) == 0.0 and mrr > best_zero_mrr:
            best_zero_mrr = mrr
            best_zero_weights = dict(weights)
        if mrr > best_mrr or (mrr == best_mrr and path_weight > best_path_weight):
            best_mrr = mrr
            best_weights = dict(weights)
    if (
        selection_margin is not None
        and guard_view is not None
        and best_zero_weights is not None
        and float(best_weights.get(guard_view, 0.0)) > 0.0
        and best_mrr <= best_zero_mrr + selection_margin
    ):
        return best_zero_weights, best_zero_mrr
    return best_weights, best_mrr


def _weighted_cached_score(
    view_scores: Mapping[str, float],
    weights: Mapping[str, float],
    *,
    views: Sequence[str],
) -> float:
    return sum(float(weights.get(view, 0.0)) * view_scores[view] for view in views)


def _multiview_score(
    left_views: Mapping[str, Counter[str]],
    right_views: Mapping[str, Counter[str]],
    weights: Mapping[str, float],
    *,
    views: Sequence[str],
) -> float:
    return sum(
        float(weights.get(view, 0.0)) * _cosine(left_views[view], right_views[view])
        for view in views
    )


def _bucket(value: float, *, bins: Sequence[float]) -> str:
    for border in bins:
        if value <= border:
            return f"<= {border:g}"
    return f"> {bins[-1]:g}"


def _first_positive_rank(scored: Sequence[tuple[float, str]], query_task: str) -> int:
    for index, (_, gallery_id) in enumerate(scored, start=1):
        if _task_from_id(gallery_id) == query_task:
            return index
    raise ValueError(f"no positive gallery item for task {query_task!r}")


def _path_from_id(item_id: str) -> Path:
    return PROJECT_ROOT / item_id.split(":", 1)[1].rsplit(":module:", 1)[0]


def _task_from_id(item_id: str) -> str:
    return Path(item_id.split(":", 1)[1].rsplit(":module:", 1)[0]).parent.name


def _summaries(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["dataset"], row["variant"])].append(row)
    summaries = []
    for (dataset, variant), block in sorted(grouped.items()):
        summaries.append(
            {
                "dataset": dataset,
                "variant": variant,
                "query_count": len(block),
                "seed_count": len({row["seed"] for row in block}),
                "task_count": len({row["query_task"] for row in block}),
                "selected_weights_by_seed": _selected_weights_by_seed(block),
                "train_validation_mrr_by_seed": _validation_mrr_by_seed(block),
                "mrr": _mean(row["mrr"] for row in block),
                "recall_at_1": _mean(row["recall_at_1"] for row in block),
                "recall_at_5": _mean(row["recall_at_5"] for row in block),
                "mean_rank": _mean(row["mean_rank"] for row in block),
            }
        )
    return summaries


def _selected_weights_by_seed(rows: Sequence[dict[str, Any]]) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for row in rows:
        if "selected_weights" in row:
            result[str(row["seed"])] = row["selected_weights"]
    return result


def _validation_mrr_by_seed(rows: Sequence[dict[str, Any]]) -> dict[str, float]:
    result: dict[str, float] = {}
    for row in rows:
        if "train_validation_mrr" in row:
            result[str(row["seed"])] = row["train_validation_mrr"]
    return result


def format_markdown(result: dict[str, Any]) -> str:
    lines = [
        "# Code2Hyp hybrid task-level retrieval benchmark",
        "",
        "The variants evaluate LCA-anchored AST path objects as one view in a train-selected multiview retrieval kernel.",
        "",
        "| Dataset | Variant | Queries | Tasks | Seeds | MRR | Recall@1 | Recall@5 | Mean rank |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in result["cell_summaries"]:
        lines.append(
            f"| {row['dataset']} | {row['variant']} | {row['query_count']} | {row['task_count']} | {row['seed_count']} | "
            f"{row['mrr']:.4f} | {row['recall_at_1']:.4f} | {row['recall_at_5']:.4f} | {row['mean_rank']:.4f} |"
        )
    lines.append("")
    return "\n".join(lines)


def _mean(values: Iterable[float]) -> float:
    values_list = list(values)
    if not values_list:
        raise ValueError("cannot average empty values")
    return sum(values_list) / len(values_list)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", nargs=2, action="append", metavar=("DATASET", "PATH"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--json-output", type=Path, required=True)
    parser.add_argument("--max-paths", type=int, default=128)
    parser.add_argument(
        "--path-selection-policy",
        choices=("preorder_first", "hash_sorted", "lca_depth_stratified", "lca_depth_affine_sampled"),
        default="preorder_first",
        help="How terminal leaf pairs are selected before max-path truncation.",
    )
    parser.add_argument(
        "--lca-view",
        choices=LCA_VIEW_CHOICES,
        default="path_signature_plus_tokens",
        help="Which LCA-path feature view is used inside Code2Hyp multiview.",
    )
    parser.add_argument(
        "--lca-selection-margin",
        type=float,
        default=None,
        help="Use the LCA view only if its train-fold MRR exceeds the best zero-LCA candidate by this margin.",
    )
    parser.add_argument(
        "--weight-grid-mode",
        choices=("compact", "expanded"),
        default="compact",
        help="Use the compact manuscript grid or a finer train-selected convex weight grid.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = run_hybrid(
        [HybridInput(dataset, Path(path)) for dataset, path in args.input],
        max_paths=args.max_paths,
        path_selection_policy=args.path_selection_policy,
        lca_view=args.lca_view,
        lca_selection_margin=args.lca_selection_margin,
        weight_grid_mode=args.weight_grid_mode,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(format_markdown(result), encoding="utf-8")
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
