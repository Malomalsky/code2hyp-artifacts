from __future__ import annotations

import ast
from collections import Counter, defaultdict
from typing import Iterable

from .graphs import SimpleGraph


DEFAULT_IGNORED_AST_TYPES = frozenset(
    {
        "Load",
        "Store",
        "Del",
        "Param",
        "alias",
        "arguments",
        "arg",
    }
)


def parse_python_ast(code: str) -> ast.AST:
    """Parse Python source code into the standard-library AST."""
    return ast.parse(code)


def _node_label(index: int, node: ast.AST) -> str:
    return f"{index}:{type(node).__name__}"


def build_ast_graph(code: str) -> SimpleGraph:
    """Build an undirected AST graph with stable node ids and depth labels."""
    tree = parse_python_ast(code)
    graph = SimpleGraph()
    counter = 0

    def visit(node: ast.AST, depth: int, parent_label: str | None) -> None:
        nonlocal counter
        label = _node_label(counter, node)
        counter += 1
        graph.add_node(label, depth=depth)
        if parent_label is not None:
            graph.add_edge(parent_label, label)

        for child in ast.iter_child_nodes(node):
            visit(child, depth + 1, label)

    visit(tree, depth=0, parent_label=None)
    return graph


def ast_root_paths(code: str) -> list[str]:
    """Represent every AST node as a deterministic root-to-node path.

    Components include sibling order and node type. This keeps repeated AST node
    types distinguishable while preserving the hierarchy needed for tree-based
    geometric diagnostics.
    """
    tree = parse_python_ast(code)
    paths: list[str] = []

    def visit(node: ast.AST, current_path: str) -> None:
        paths.append(current_path)
        for child_index, child in enumerate(ast.iter_child_nodes(node)):
            child_component = f"{child_index:03d}-{type(child).__name__}"
            visit(child, f"{current_path}/{child_component}")

    visit(tree, type(tree).__name__)
    return paths


def ast_node_histogram(
    code: str,
    *,
    ignored_node_types: Iterable[str] = DEFAULT_IGNORED_AST_TYPES,
) -> dict[str, int]:
    """Count AST node types after removing syntactic context-only nodes."""
    ignored = set(ignored_node_types)
    tree = parse_python_ast(code)
    histogram: Counter[str] = Counter()
    for node in ast.walk(tree):
        node_type = type(node).__name__
        if node_type not in ignored:
            histogram[node_type] += 1
    return dict(sorted(histogram.items()))


def ast_transition_counts(
    code: str,
    *,
    ignored_node_types: Iterable[str] = DEFAULT_IGNORED_AST_TYPES,
) -> dict[tuple[str, str], int]:
    """Count parent-child transitions between AST node types.

    This is the structural core of the DTA Markov-chain baseline: each AST node
    type is a state, and a directed transition is observed from a parent type to
    each child type.
    """
    ignored = set(ignored_node_types)
    tree = parse_python_ast(code)
    counts: Counter[tuple[str, str]] = Counter()

    for parent in ast.walk(tree):
        parent_type = type(parent).__name__
        if parent_type in ignored:
            continue
        for child in ast.iter_child_nodes(parent):
            child_type = type(child).__name__
            if child_type in ignored:
                continue
            counts[(parent_type, child_type)] += 1

    return dict(sorted(counts.items()))


def ast_markov_probabilities(
    code: str,
    *,
    ignored_node_types: Iterable[str] = DEFAULT_IGNORED_AST_TYPES,
) -> dict[tuple[str, str], float]:
    """Return row-normalized AST transition probabilities.

    For every parent AST type `u`, outgoing probabilities over child AST types
    `v` sum to one. These probabilities are the object later compared via
    Jensen-Shannon divergence in the DTA baseline.
    """
    counts = ast_transition_counts(code, ignored_node_types=ignored_node_types)
    row_totals: defaultdict[str, int] = defaultdict(int)
    for (parent, _child), count in counts.items():
        row_totals[parent] += count

    return {
        (parent, child): count / row_totals[parent]
        for (parent, child), count in counts.items()
    }


def ast_markov_rows(
    code: str,
    *,
    ignored_node_types: Iterable[str] = DEFAULT_IGNORED_AST_TYPES,
) -> dict[str, dict[str, float]]:
    """Return Markov transition probabilities grouped by parent AST type."""
    probabilities = ast_markov_probabilities(
        code,
        ignored_node_types=ignored_node_types,
    )
    rows: defaultdict[str, dict[str, float]] = defaultdict(dict)
    for (parent, child), probability in probabilities.items():
        rows[parent][child] = probability
    return {parent: dict(sorted(children.items())) for parent, children in sorted(rows.items())}
