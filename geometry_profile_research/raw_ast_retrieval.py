from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from statistics import mean
from typing import Literal, Sequence

from geometry_profile_research.raw_ast import RawAstTree, leaf_node_ids


_IDENTIFIER_RE = re.compile(r"^[A-Za-z_$][A-Za-z0-9_$]*$")
_JAVA_NON_RENAMABLE = frozenset(
    {
        "abstract",
        "assert",
        "boolean",
        "break",
        "byte",
        "case",
        "catch",
        "char",
        "class",
        "const",
        "continue",
        "default",
        "do",
        "double",
        "else",
        "enum",
        "extends",
        "false",
        "final",
        "finally",
        "float",
        "for",
        "goto",
        "if",
        "implements",
        "import",
        "instanceof",
        "int",
        "interface",
        "long",
        "native",
        "new",
        "null",
        "package",
        "private",
        "protected",
        "public",
        "return",
        "short",
        "static",
        "strictfp",
        "super",
        "switch",
        "synchronized",
        "this",
        "throw",
        "throws",
        "transient",
        "true",
        "try",
        "void",
        "volatile",
        "while",
    }
)
CALLABLE_LABELS = frozenset({"MethodDeclaration", "ConstructorDeclaration", "FunctionDef", "AsyncFunctionDef"})
PositiveMode = Literal["alpha_rename", "structural_noop", "alpha_structural_noop"]
ItemScope = Literal["callable", "module", "callable_or_module"]


@dataclass(frozen=True)
class RawASTRetrievalItem:
    """One method-level item for structural-retrieval training."""

    item_id: str
    tree: RawAstTree
    language: str = "unknown"


def alpha_rename_tree(tree: RawAstTree, *, prefix: str = "id") -> RawAstTree:
    """Return a topology-preserving alpha-renamed copy of a raw AST.

    Only identifier-like terminal tokens are renamed. Operators, literals,
    booleans, primitive Java types and empty tokens are preserved. The function
    intentionally keeps raw node ids, parent links, child order and labels
    unchanged, so the result can be used as a positive structural-retrieval pair.
    """

    mapping: dict[str, str] = {}
    renamed_attributes: dict[int, dict[str, str]] = {}
    for node in tree.preorder():
        attributes = dict(tree.attributes.get(node, {}))
        token = attributes.get("terminal_token")
        if token is not None and _is_renamable_identifier(token):
            if token not in mapping:
                mapping[token] = f"{prefix}_{len(mapping)}"
            attributes["terminal_token"] = mapping[token]
        if attributes:
            renamed_attributes[node] = attributes
    return _copy_tree_with_attributes(tree, renamed_attributes)


def structural_noop_tree(tree: RawAstTree, *, label: str = "SyntheticNoOp") -> RawAstTree:
    """Return a semantics-preserving structural augmentation of a raw AST.

    The node is synthetic and deliberately terminal-free. It subdivides an edge
    on a terminal-bearing branch, so terminal-to-terminal paths change while the
    lexical terminal set is preserved. This gives a stronger positive-pair
    control than pure alpha-renaming.
    """

    existing_nodes = tuple(tree.parent_by_node)
    new_id = max(existing_nodes, default=tree.root_id) + 1
    parent_to_split, child_to_split = _edge_to_subdivide(tree)
    edges = []
    for parent, children in tree.children_by_node.items():
        for child in children:
            if (parent, child) == (parent_to_split, child_to_split):
                edges.append((parent, new_id))
                edges.append((new_id, child))
            else:
                edges.append((parent, child))
    labels = dict(tree.labels)
    labels[new_id] = label
    attributes = {node: dict(values) for node, values in tree.attributes.items()}
    split_child_attrs = dict(attributes.get(child_to_split, {}))
    split_child_attrs["child_index"] = "0"
    attributes[child_to_split] = split_child_attrs
    attributes[new_id] = {
        "node_type": label,
        "edge_type": "synthetic_noop",
        "child_index": tree.attributes.get(child_to_split, {}).get("child_index", "0"),
        "synthetic": "true",
    }
    return RawAstTree.from_edges(
        root_id=tree.root_id,
        edges=tuple(edges),
        labels=labels,
        attributes=attributes,
    )


def make_positive_tree(
    tree: RawAstTree,
    *,
    mode: PositiveMode,
    positive_prefix: str = "id",
) -> RawAstTree:
    """Build a positive tree according to the experimental control mode."""

    if mode == "alpha_rename":
        return alpha_rename_tree(tree, prefix=positive_prefix)
    if mode == "structural_noop":
        return structural_noop_tree(tree)
    if mode == "alpha_structural_noop":
        return structural_noop_tree(alpha_rename_tree(tree, prefix=positive_prefix))
    raise ValueError(f"unknown positive mode: {mode}")


def terminal_jaccard_similarity(left: RawAstTree, right: RawAstTree) -> float:
    """Jaccard similarity of non-empty terminal-token sets."""

    left_tokens = set(_terminal_tokens(left))
    right_tokens = set(_terminal_tokens(right))
    union = left_tokens | right_tokens
    if not union:
        return 0.0
    return len(left_tokens & right_tokens) / len(union)


def structural_gap(left: RawAstTree, right: RawAstTree) -> float:
    """Bounded structural dissimilarity used for hard-negative selection."""

    left_features = _shape_features(left)
    right_features = _shape_features(right)
    numeric_gap = sum(_relative_gap(left_features[key], right_features[key]) for key in left_features) / len(left_features)
    label_gap = 1.0 - _counter_jaccard(Counter(left.labels.values()), Counter(right.labels.values()))
    degree_gap = 1.0 - _counter_jaccard(Counter(len(children) for children in left.children_by_node.values()), Counter(len(children) for children in right.children_by_node.values()))
    return min(1.0, 0.5 * numeric_gap + 0.3 * label_gap + 0.2 * degree_gap)


def select_hard_negative(
    anchor: RawASTRetrievalItem,
    candidates: Sequence[RawASTRetrievalItem],
    *,
    min_structural_gap: float = 0.05,
) -> RawASTRetrievalItem:
    """Select a lexically similar but structurally different negative item."""

    if not candidates:
        raise ValueError("select_hard_negative requires at least one candidate")
    scored = []
    fallback = []
    for candidate in candidates:
        if candidate.item_id == anchor.item_id:
            continue
        similarity = terminal_jaccard_similarity(anchor.tree, candidate.tree)
        gap = structural_gap(anchor.tree, candidate.tree)
        score = similarity * gap
        fallback.append((gap, similarity, candidate))
        if gap >= min_structural_gap:
            scored.append((score, similarity, gap, candidate))
    if scored:
        scored.sort(key=lambda item: (item[0], item[1], item[2], item[3].item_id), reverse=True)
        return scored[0][3]
    if fallback:
        fallback.sort(key=lambda item: (item[0], item[1], item[2].item_id), reverse=True)
        return fallback[0][2]
    raise ValueError("select_hard_negative requires a candidate different from the anchor")


def build_retrieval_triples(
    items: Sequence[RawASTRetrievalItem],
    *,
    min_structural_gap: float = 0.05,
    positive_mode: PositiveMode = "alpha_rename",
    positive_prefix: str = "id",
) -> list[tuple[RawAstTree, RawAstTree, RawAstTree]]:
    """Build ``(anchor, positive, hard negative)`` triples."""

    triples: list[tuple[RawAstTree, RawAstTree, RawAstTree]] = []
    for item in items:
        candidates = tuple(candidate for candidate in items if candidate.item_id != item.item_id)
        negative = select_hard_negative(item, candidates, min_structural_gap=min_structural_gap)
        positive = make_positive_tree(item.tree, mode=positive_mode, positive_prefix=positive_prefix)
        triples.append((item.tree, positive, negative.tree))
    return triples


def callable_subtrees(tree: RawAstTree) -> tuple[RawAstTree, ...]:
    """Extract method/constructor subtrees from a parsed Java raw AST."""

    return tuple(
        subtree_as_tree(tree, node)
        for node in tree.preorder()
        if tree.labels.get(node) in CALLABLE_LABELS
    )


def retrieval_item_trees(tree: RawAstTree, *, item_scope: ItemScope = "callable") -> tuple[RawAstTree, ...]:
    """Extract retrieval-unit trees under an explicit corpus-dependent policy.

    DTA-style corpora expose methods/functions as the natural retrieval unit.
    Competitive-programming corpora such as CodeNet/BugNet usually expose a
    complete accepted program as the unit, often without top-level functions.
    Keeping this policy explicit prevents mixing method-level and program-level
    evidence in one experiment.
    """

    callables = callable_subtrees(tree)
    if item_scope == "callable":
        return callables
    if item_scope == "module":
        return (tree,)
    if item_scope == "callable_or_module":
        return callables if callables else (tree,)
    raise ValueError(f"unknown item_scope: {item_scope!r}")


def subtree_as_tree(tree: RawAstTree, root_id: int) -> RawAstTree:
    """Return a rooted raw-AST subtree while preserving original node ids."""

    tree._require_node(root_id)
    nodes: list[int] = []
    stack = [root_id]
    while stack:
        node = stack.pop()
        nodes.append(node)
        stack.extend(reversed(tree.children_by_node.get(node, ())))
    node_set = set(nodes)
    edges = tuple(
        (parent, child)
        for parent in nodes
        for child in tree.children_by_node.get(parent, ())
        if child in node_set
    )
    return RawAstTree.from_edges(
        root_id=root_id,
        edges=edges,
        labels={node: tree.labels.get(node, "") for node in nodes},
        attributes={node: dict(tree.attributes.get(node, {})) for node in nodes if tree.attributes.get(node)},
    )


def _copy_tree_with_attributes(tree: RawAstTree, attributes: dict[int, dict[str, str]]) -> RawAstTree:
    edges = tuple(
        (parent, child)
        for parent, children in tree.children_by_node.items()
        for child in children
    )
    return RawAstTree.from_edges(
        root_id=tree.root_id,
        edges=edges,
        labels=tree.labels,
        attributes=attributes,
    )


def _edge_to_subdivide(tree: RawAstTree) -> tuple[int, int]:
    terminal_nodes = tuple(
        node
        for node in tree.preorder()
        if tree.labels.get(node) == "TerminalToken" and tree.parent_by_node.get(node) is not None
    )
    if terminal_nodes:
        child = terminal_nodes[0]
        parent = tree.parent_by_node[child]
        if parent is not None:
            return parent, child
    for child, parent in tree.parent_by_node.items():
        if parent is not None:
            return parent, child
    raise ValueError("structural_noop_tree requires at least one edge")


def _terminal_tokens(tree: RawAstTree) -> tuple[str, ...]:
    tokens = []
    for node in tree.preorder():
        token = tree.attributes.get(node, {}).get("terminal_token")
        if token:
            tokens.append(token)
    return tuple(tokens)


def _is_renamable_identifier(token: str) -> bool:
    return bool(_IDENTIFIER_RE.match(token)) and token not in _JAVA_NON_RENAMABLE


def _shape_features(tree: RawAstTree) -> dict[str, float]:
    order = tree.preorder()
    depths = [tree.depth(node) for node in order]
    degrees = [len(tree.children_by_node.get(node, ())) for node in order]
    return {
        "node_count": float(len(order)),
        "leaf_count": float(len(leaf_node_ids(tree))),
        "max_depth": float(max(depths, default=0)),
        "mean_depth": float(mean(depths)) if depths else 0.0,
        "max_degree": float(max(degrees, default=0)),
        "mean_degree": float(mean(degrees)) if degrees else 0.0,
    }


def _relative_gap(left: float, right: float) -> float:
    scale = max(abs(left), abs(right), 1.0)
    return abs(left - right) / scale


def _counter_jaccard(left: Counter, right: Counter) -> float:
    union_keys = set(left) | set(right)
    if not union_keys:
        return 1.0
    intersection = sum(min(left[key], right[key]) for key in union_keys)
    union = sum(max(left[key], right[key]) for key in union_keys)
    return intersection / union if union else 1.0
