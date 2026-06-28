from __future__ import annotations

import ast
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from geometry_profile_research.raw_ast import NodeId, RawAstTree


@dataclass
class _AstBuilder:
    labels: dict[NodeId, str]
    attributes: dict[NodeId, dict[str, str]]
    edges: list[tuple[NodeId, NodeId]]
    next_id: int = 0

    def add_node(self, label: str, attributes: dict[str, str] | None = None) -> NodeId:
        node_id = self.next_id
        self.next_id += 1
        self.labels[node_id] = label
        if attributes:
            self.attributes[node_id] = attributes
        return node_id

    def add_edge(self, parent: NodeId, child: NodeId) -> None:
        self.edges.append((parent, child))


def parse_python_ast_tree(source: str) -> RawAstTree:
    """Parse Python source into the shared raw-AST representation."""

    root = ast.parse(source)
    builder = _AstBuilder(labels={}, attributes={}, edges=[])

    def visit(
        value: Any,
        parent_id: NodeId | None = None,
        *,
        edge_type: str = "root",
        child_index: int = 0,
    ) -> NodeId | None:
        if _is_terminal_value(value):
            if parent_id is None:
                return None
            node_id = builder.add_node(
                "TerminalToken",
                {
                    "node_type": "TerminalToken",
                    "edge_type": edge_type,
                    "child_index": str(child_index),
                    "terminal_token": str(value),
                },
            )
            builder.add_edge(parent_id, node_id)
            return node_id
        if not isinstance(value, ast.AST):
            return None
        attributes = _node_attributes(value)
        if parent_id is not None:
            attributes["edge_type"] = edge_type
            attributes["child_index"] = str(child_index)
        node_id = builder.add_node(type(value).__name__, attributes)
        if parent_id is not None:
            builder.add_edge(parent_id, node_id)
        for child_edge_type, child_index_value, child in _iter_node_children(value):
            visit(child, node_id, edge_type=child_edge_type, child_index=child_index_value)
        return node_id

    root_id = visit(root)
    if root_id is None:
        raise ValueError("Python parser did not return an AST root")
    return RawAstTree.from_edges(
        root_id=root_id,
        edges=builder.edges,
        labels=builder.labels,
        attributes=builder.attributes,
    )


def _node_attributes(node: ast.AST) -> dict[str, str]:
    node_type = type(node).__name__
    attributes: dict[str, str] = {"node_type": node_type}
    name = getattr(node, "name", None)
    if isinstance(name, str) and name:
        attributes["name"] = name
    if hasattr(node, "lineno") and hasattr(node, "col_offset"):
        attributes["source_span"] = f"{getattr(node, 'lineno')}:{getattr(node, 'col_offset')}"
    return attributes


def _iter_node_children(node: ast.AST) -> Iterable[tuple[str, int, Any]]:
    for attr_name, value in ast.iter_fields(node):
        child_index = 0
        for child in _iter_nested(value):
            if _is_supported_child(child):
                yield attr_name, child_index, child
                child_index += 1


def _iter_nested(value: Any) -> Iterable[Any]:
    if isinstance(value, (list, tuple)):
        for item in value:
            yield from _iter_nested(item)
    else:
        yield value


def _is_terminal_value(value: Any) -> bool:
    return value is None or isinstance(value, (str, int, float, bool))


def _is_supported_child(value: Any) -> bool:
    if _is_terminal_value(value):
        return True
    return isinstance(value, ast.AST)
