from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import PurePosixPath
from typing import Iterable


@dataclass
class SimpleGraph:
    """Small undirected graph with enough operations for geometry diagnostics."""

    _adjacency: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))
    _depths: dict[str, int] = field(default_factory=dict)

    def add_node(self, node: str, depth: int | None = None) -> None:
        self._adjacency[node]
        if depth is not None:
            self._depths[node] = depth

    def add_edge(self, left: str, right: str) -> None:
        self._adjacency[left].add(right)
        self._adjacency[right].add(left)

    @property
    def nodes(self) -> set[str]:
        return set(self._adjacency)

    def neighbors(self, node: str) -> set[str]:
        return set(self._adjacency.get(node, set()))

    def depth(self, node: str) -> int:
        if node in self._depths:
            return self._depths[node]
        if node == "":
            return 0
        return len([part for part in node.split("/") if part])


def normalize_path(path: str) -> str:
    """Normalize a source-code path to stable POSIX form without leading slash."""
    raw = str(path).replace("\\", "/").strip()
    if not raw:
        return ""
    normalized = str(PurePosixPath(raw))
    if normalized == ".":
        return ""
    return normalized.lstrip("/")


def path_prefixes(path: str) -> list[str]:
    normalized = normalize_path(path)
    if not normalized:
        return [""]

    parts = [part for part in normalized.split("/") if part]
    prefixes = [""]
    for index in range(1, len(parts) + 1):
        prefixes.append("/".join(parts[:index]))
    return prefixes


def build_file_tree_graph(paths: Iterable[str]) -> SimpleGraph:
    """Build an undirected rooted file-tree graph from POSIX-like paths."""
    graph = SimpleGraph()
    graph.add_node("", depth=0)

    for path in paths:
        prefixes = path_prefixes(path)
        for depth, node in enumerate(prefixes):
            graph.add_node(node, depth=depth)
            if depth > 0:
                graph.add_edge(prefixes[depth - 1], node)

    return graph
