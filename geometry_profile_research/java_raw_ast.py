from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from .raw_ast import NodeId, RawAstTree, leaf_node_ids


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


def parse_java_ast_tree(source: str) -> RawAstTree:
    """Parse Java source into a rooted raw AST.

    The parser dependency is intentionally optional. Install the `raw-ast`
    extra to use this function in corpus-level experiments.
    """

    try:
        import javalang
        from javalang.ast import Node
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("parse_java_ast_tree requires the optional dependency: javalang") from exc

    root = javalang.parse.parse(source)
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
        if not isinstance(value, Node):
            return None
        node_attributes = _node_attributes(value)
        if parent_id is not None:
            node_attributes["edge_type"] = edge_type
            node_attributes["child_index"] = str(child_index)
        node_id = builder.add_node(type(value).__name__, node_attributes)
        if parent_id is not None:
            builder.add_edge(parent_id, node_id)
        for child_edge_type, child_index_value, child in _iter_node_children(value):
            visit(child, node_id, edge_type=child_edge_type, child_index=child_index_value)
        return node_id

    root_id = visit(root)
    if root_id is None:
        raise ValueError("Java parser did not return an AST root")
    return RawAstTree.from_edges(
        root_id=root_id,
        edges=builder.edges,
        labels=builder.labels,
        attributes=builder.attributes,
    )


def _node_attributes(node: Any) -> dict[str, str]:
    node_type = type(node).__name__
    attributes: dict[str, str] = {"node_type": node_type}
    name = getattr(node, "name", None)
    if isinstance(name, str) and name:
        attributes["name"] = name
    position = getattr(node, "position", None)
    if position is not None:
        line = getattr(position, "line", None)
        column = getattr(position, "column", None)
        if line is not None and column is not None:
            attributes["source_span"] = f"{line}:{column}"
    return attributes


def _iter_node_children(node: Any) -> Iterable[tuple[str, int, Any]]:
    for attr_name in getattr(node, "attrs", ()):
        value = getattr(node, attr_name)
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
    return isinstance(value, (str, int, float, bool))


def _is_supported_child(value: Any) -> bool:
    if value is None:
        return False
    if _is_terminal_value(value):
        return True
    return hasattr(value, "attrs")
