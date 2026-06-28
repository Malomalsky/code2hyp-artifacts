from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from math import sqrt
from typing import Mapping, Sequence


NodeId = int
UndirectedEdge = tuple[NodeId, NodeId]


@dataclass(frozen=True)
class RawAstPath:
    """Terminal-to-terminal AST path with stable raw node identifiers."""

    start: NodeId
    end: NodeId
    nodes: tuple[NodeId, ...]

    def __post_init__(self) -> None:
        if not self.nodes:
            raise ValueError("raw AST path must contain at least one node")
        if self.nodes[0] != self.start:
            raise ValueError("path start must match the first node")
        if self.nodes[-1] != self.end:
            raise ValueError("path end must match the last node")

    @property
    def length(self) -> int:
        return len(self.nodes) - 1

    @property
    def undirected_edges(self) -> frozenset[UndirectedEdge]:
        return frozenset(_ordered_edge(left, right) for left, right in zip(self.nodes, self.nodes[1:]))

    def reversed(self) -> RawAstPath:
        reversed_nodes = tuple(reversed(self.nodes))
        return RawAstPath(start=self.end, end=self.start, nodes=reversed_nodes)

    def lca(self, tree: RawAstTree) -> NodeId:
        return tree.lca(self.start, self.end)


@dataclass(frozen=True)
class RawAstPathRelationRecord:
    """Pairwise raw-AST relation target for two terminal-to-terminal paths."""

    left_index: int
    right_index: int
    oriented_endpoint_distance: int
    unoriented_endpoint_distance: int
    edge_symmetric_difference: int
    edge_jaccard_distance: float
    left_lca_depth: int
    right_lca_depth: int
    lca_depth_difference: int
    lca_anchored_product_distance: float
    left_path_length: int
    right_path_length: int
    path_length_difference: int

    def as_dict(self) -> dict[str, int | float]:
        return {
            "left_index": self.left_index,
            "right_index": self.right_index,
            "oriented_endpoint_distance": self.oriented_endpoint_distance,
            "unoriented_endpoint_distance": self.unoriented_endpoint_distance,
            "edge_symmetric_difference": self.edge_symmetric_difference,
            "edge_jaccard_distance": self.edge_jaccard_distance,
            "left_lca_depth": self.left_lca_depth,
            "right_lca_depth": self.right_lca_depth,
            "lca_depth_difference": self.lca_depth_difference,
            "lca_anchored_product_distance": self.lca_anchored_product_distance,
            "left_path_length": self.left_path_length,
            "right_path_length": self.right_path_length,
            "path_length_difference": self.path_length_difference,
        }


@dataclass(frozen=True)
class RawAstPathRelationMatrices:
    """Dense pairwise raw-AST relation targets for a method-level path set."""

    oriented_endpoint_distance: tuple[tuple[int, ...], ...]
    unoriented_endpoint_distance: tuple[tuple[int, ...], ...]
    edge_symmetric_difference: tuple[tuple[int, ...], ...]
    edge_jaccard_distance: tuple[tuple[float, ...], ...]
    lca_depth: tuple[int, ...]
    lca_depth_difference: tuple[tuple[int, ...], ...]
    lca_anchored_product_distance: tuple[tuple[float, ...], ...]
    path_length: tuple[int, ...]
    path_length_difference: tuple[tuple[int, ...], ...]


@dataclass(frozen=True)
class RawAstOrderRelationRecord:
    """Candidate AST partial-order relation for entailment/order objectives.

    ``label=1`` means that ``ancestor`` is a true ancestor of ``descendant`` in
    the rooted AST. ``label=0`` is a deterministic incomparable control pair
    with the same JSON shape, used for matched non-order supervision.
    """

    ancestor: NodeId
    descendant: NodeId
    label: int
    ancestor_depth: int
    descendant_depth: int
    tree_distance: int
    is_direct_edge: bool
    ancestor_label: str
    descendant_label: str
    ancestor_name: str = ""
    descendant_name: str = ""

    def as_dict(self) -> dict[str, int | str | bool]:
        return {
            "ancestor": self.ancestor,
            "descendant": self.descendant,
            "label": self.label,
            "ancestor_depth": self.ancestor_depth,
            "descendant_depth": self.descendant_depth,
            "tree_distance": self.tree_distance,
            "is_direct_edge": self.is_direct_edge,
            "ancestor_label": self.ancestor_label,
            "descendant_label": self.descendant_label,
            "ancestor_name": self.ancestor_name,
            "descendant_name": self.descendant_name,
        }


@dataclass(frozen=True)
class RawAstTree:
    """Rooted raw AST with stable node ids and parent links."""

    root_id: NodeId
    parent_by_node: Mapping[NodeId, NodeId | None]
    children_by_node: Mapping[NodeId, tuple[NodeId, ...]]
    labels: Mapping[NodeId, str]
    attributes: Mapping[NodeId, Mapping[str, str]] = field(default_factory=dict)

    @classmethod
    def from_edges(
        cls,
        root_id: NodeId,
        edges: Sequence[tuple[NodeId, NodeId]],
        labels: Mapping[NodeId, str] | None = None,
        attributes: Mapping[NodeId, Mapping[str, str]] | None = None,
    ) -> RawAstTree:
        parent_by_node: dict[NodeId, NodeId | None] = {root_id: None}
        children_by_node: dict[NodeId, list[NodeId]] = {root_id: []}
        for parent, child in edges:
            if child in parent_by_node:
                raise ValueError(f"node {child!r} has more than one parent")
            parent_by_node.setdefault(parent, None)
            children_by_node.setdefault(parent, [])
            children_by_node.setdefault(child, [])
            parent_by_node[child] = parent
            children_by_node[parent].append(child)

        if root_id not in parent_by_node:
            raise ValueError("root_id must be present in the tree")
        if parent_by_node[root_id] is not None:
            raise ValueError("root node must not have a parent")
        frozen_children = {node: tuple(children) for node, children in children_by_node.items()}
        tree = cls(
            root_id=root_id,
            parent_by_node=dict(parent_by_node),
            children_by_node=frozen_children,
            labels=dict(labels or {}),
            attributes={node: dict(values) for node, values in (attributes or {}).items()},
        )
        tree._validate_connected()
        return tree

    def _validate_connected(self) -> None:
        nodes = set(self.parent_by_node)
        reachable = set(self.preorder())
        if reachable != nodes:
            missing = sorted(nodes - reachable)
            raise ValueError(f"tree contains nodes unreachable from root: {missing!r}")

    def preorder(self) -> tuple[NodeId, ...]:
        order: list[NodeId] = []
        stack = [self.root_id]
        while stack:
            node = stack.pop()
            order.append(node)
            stack.extend(reversed(self.children_by_node.get(node, ())))
        return tuple(order)

    def parent(self, node: NodeId) -> NodeId | None:
        self._require_node(node)
        return self.parent_by_node[node]

    def depth(self, node: NodeId) -> int:
        self._require_node(node)
        depth = 0
        current = node
        while current != self.root_id:
            parent = self.parent_by_node[current]
            if parent is None:
                raise ValueError(f"node {node!r} is disconnected from root")
            current = parent
            depth += 1
        return depth

    def ancestors(self, node: NodeId) -> tuple[NodeId, ...]:
        self._require_node(node)
        result: list[NodeId] = []
        current: NodeId | None = node
        while current is not None:
            result.append(current)
            current = self.parent_by_node[current]
        return tuple(result)

    def lca(self, left: NodeId, right: NodeId) -> NodeId:
        left_ancestors = set(self.ancestors(left))
        current: NodeId | None = right
        while current is not None:
            if current in left_ancestors:
                return current
            current = self.parent_by_node[current]
        raise ValueError("tree is disconnected")

    def tree_distance(self, left: NodeId, right: NodeId) -> int:
        ancestor = self.lca(left, right)
        return self.depth(left) + self.depth(right) - 2 * self.depth(ancestor)

    def path_between(self, start: NodeId, end: NodeId) -> RawAstPath:
        ancestor = self.lca(start, end)
        upward: list[NodeId] = []
        current = start
        while current != ancestor:
            upward.append(current)
            parent = self.parent_by_node[current]
            if parent is None:
                raise ValueError("tree is disconnected")
            current = parent
        upward.append(ancestor)

        downward_reversed: list[NodeId] = []
        current = end
        while current != ancestor:
            downward_reversed.append(current)
            parent = self.parent_by_node[current]
            if parent is None:
                raise ValueError("tree is disconnected")
            current = parent
        nodes = tuple(upward + list(reversed(downward_reversed)))
        return RawAstPath(start=start, end=end, nodes=nodes)

    def _require_node(self, node: NodeId) -> None:
        if node not in self.parent_by_node:
            raise KeyError(f"unknown AST node id: {node!r}")


def _ordered_edge(left: NodeId, right: NodeId) -> UndirectedEdge:
    return (left, right) if left <= right else (right, left)


def gromov_lca_depth(tree: RawAstTree, left: NodeId, right: NodeId) -> float:
    """Gromov product (left|right)_root, equal to LCA depth in a rooted tree."""

    root = tree.root_id
    return 0.5 * (
        tree.tree_distance(root, left)
        + tree.tree_distance(root, right)
        - tree.tree_distance(left, right)
    )


def edge_symmetric_difference_distance(left: RawAstPath, right: RawAstPath) -> int:
    return len(left.undirected_edges ^ right.undirected_edges)


def edge_jaccard_distance(left: RawAstPath, right: RawAstPath) -> float:
    union = left.undirected_edges | right.undirected_edges
    if not union:
        return 0.0
    intersection = left.undirected_edges & right.undirected_edges
    return 1.0 - len(intersection) / len(union)


def oriented_endpoint_distance(tree: RawAstTree, left: RawAstPath, right: RawAstPath) -> int:
    return tree.tree_distance(left.start, right.start) + tree.tree_distance(left.end, right.end)


def unoriented_endpoint_distance(tree: RawAstTree, left: RawAstPath, right: RawAstPath) -> int:
    return min(
        oriented_endpoint_distance(tree, left, right),
        oriented_endpoint_distance(tree, left, right.reversed()),
    )


def lca_anchored_product_distance(tree: RawAstTree, left: RawAstPath, right: RawAstPath) -> float:
    """Unoriented LCA-anchored product distance between two AST paths.

    The path object is represented by its LCA node and two terminal endpoints.
    Endpoint reversal is quotiented out because ``terminal_to_terminal_paths``
    emits unordered leaf pairs in a deterministic traversal order.
    """

    left_lca = left.lca(tree)
    right_lca = right.lca(tree)
    lca_term = tree.tree_distance(left_lca, right_lca) ** 2
    direct_endpoint_term = tree.tree_distance(left.start, right.start) ** 2 + tree.tree_distance(left.end, right.end) ** 2
    reversed_endpoint_term = tree.tree_distance(left.start, right.end) ** 2 + tree.tree_distance(left.end, right.start) ** 2
    return sqrt(float(lca_term + min(direct_endpoint_term, reversed_endpoint_term)))


def find_nodes_by_label(tree: RawAstTree, label: str) -> tuple[NodeId, ...]:
    return tuple(node for node in tree.preorder() if tree.labels.get(node) == label)


def leaf_node_ids(tree: RawAstTree, root_id: NodeId | None = None) -> tuple[NodeId, ...]:
    root = tree.root_id if root_id is None else root_id
    tree._require_node(root)
    order: list[NodeId] = []
    stack = [root]
    while stack:
        node = stack.pop()
        order.append(node)
        stack.extend(reversed(tree.children_by_node.get(node, ())))
    return tuple(node for node in order if not tree.children_by_node.get(node))


def _subtree_preorder(tree: RawAstTree, root_id: NodeId | None = None) -> tuple[NodeId, ...]:
    root = tree.root_id if root_id is None else root_id
    tree._require_node(root)
    order: list[NodeId] = []
    stack = [root]
    while stack:
        node = stack.pop()
        order.append(node)
        stack.extend(reversed(tree.children_by_node.get(node, ())))
    return tuple(order)


def terminal_to_terminal_paths(
    tree: RawAstTree,
    max_paths: int | None = None,
    root_id: NodeId | None = None,
    selection_policy: str = "preorder_first",
) -> tuple[RawAstPath, ...]:
    """Deterministic terminal-to-terminal raw AST paths over leaf nodes.

    ``preorder_first`` keeps the historical behavior and returns the first
    leaf pairs in preorder. ``hash_sorted`` ranks all leaf pairs by a stable
    digest before truncation, which is useful as a sensitivity check against
    first-K preorder bias. ``lca_depth_stratified`` groups pairs by LCA depth
    and selects them round-robin across depth strata, with stable hash order
    inside each stratum.
    """

    if max_paths is not None and max_paths < 0:
        raise ValueError("max_paths must be non-negative or None")
    if selection_policy not in {"preorder_first", "hash_sorted", "lca_depth_stratified"}:
        raise ValueError(f"unknown path selection policy: {selection_policy!r}")
    leaves = leaf_node_ids(tree, root_id=root_id)
    pairs = [(left, right) for left_index, left in enumerate(leaves) for right in leaves[left_index + 1 :]]
    if selection_policy == "hash_sorted":
        pairs.sort(key=lambda pair: _stable_leaf_pair_digest(tree, pair[0], pair[1]))
    elif selection_policy == "lca_depth_stratified":
        pairs = _lca_depth_stratified_pairs(tree, pairs, max_paths=max_paths)
    paths: list[RawAstPath] = []
    if max_paths == 0:
        return tuple(paths)
    for left, right in pairs:
        paths.append(tree.path_between(left, right))
        if max_paths is not None and len(paths) >= max_paths:
            return tuple(paths)
    return tuple(paths)


def _lca_depth_stratified_pairs(
    tree: RawAstTree,
    pairs: Sequence[tuple[NodeId, NodeId]],
    *,
    max_paths: int | None,
) -> list[tuple[NodeId, NodeId]]:
    strata: dict[int, list[tuple[NodeId, NodeId]]] = {}
    for left, right in pairs:
        depth = tree.depth(tree.lca(left, right))
        strata.setdefault(depth, []).append((left, right))
    for depth_pairs in strata.values():
        depth_pairs.sort(key=lambda pair: _stable_leaf_pair_digest(tree, pair[0], pair[1]))
    if max_paths is None:
        return [pair for depth in sorted(strata) for pair in strata[depth]]
    selected: list[tuple[NodeId, NodeId]] = []
    depth_order = sorted(strata)
    offsets = {depth: 0 for depth in depth_order}
    while len(selected) < max_paths:
        added = False
        for depth in depth_order:
            offset = offsets[depth]
            depth_pairs = strata[depth]
            if offset >= len(depth_pairs):
                continue
            selected.append(depth_pairs[offset])
            offsets[depth] = offset + 1
            added = True
            if len(selected) >= max_paths:
                break
        if not added:
            break
    return selected


def _stable_leaf_pair_digest(tree: RawAstTree, left: NodeId, right: NodeId) -> bytes:
    payload = "|".join(
        (
            str(left),
            tree.labels.get(left, ""),
            tree.attributes.get(left, {}).get("terminal_token", ""),
            str(right),
            tree.labels.get(right, ""),
            tree.attributes.get(right, {}).get("terminal_token", ""),
        )
    )
    return hashlib.blake2b(payload.encode("utf-8"), digest_size=16).digest()


def _is_strict_ancestor(tree: RawAstTree, candidate_ancestor: NodeId, node: NodeId) -> bool:
    current = tree.parent_by_node[node]
    while current is not None:
        if current == candidate_ancestor:
            return True
        current = tree.parent_by_node[current]
    return False


def _order_record(
    tree: RawAstTree,
    ancestor: NodeId,
    descendant: NodeId,
    *,
    label: int,
) -> RawAstOrderRelationRecord:
    return RawAstOrderRelationRecord(
        ancestor=ancestor,
        descendant=descendant,
        label=label,
        ancestor_depth=tree.depth(ancestor),
        descendant_depth=tree.depth(descendant),
        tree_distance=tree.tree_distance(ancestor, descendant),
        is_direct_edge=tree.parent_by_node[descendant] == ancestor if label == 1 else False,
        ancestor_label=tree.labels.get(ancestor, ""),
        descendant_label=tree.labels.get(descendant, ""),
        ancestor_name=tree.attributes.get(ancestor, {}).get("name", ""),
        descendant_name=tree.attributes.get(descendant, {}).get("name", ""),
    )


def raw_ast_order_relation_records(
    tree: RawAstTree,
    *,
    root_id: NodeId | None = None,
    include_incomparable: bool = False,
    max_records: int | None = None,
    direct_only: bool = False,
) -> tuple[RawAstOrderRelationRecord, ...]:
    """Return raw-AST partial-order targets inside a tree or callable subtree.

    Positive records are emitted first in deterministic preorder as
    ``ancestor -> descendant`` pairs. Optional incomparable controls are added
    afterward in preorder-pair order. This keeps the target suitable for
    hyperbolic entailment/order experiments while preserving a matched
    non-order control set.
    """

    if max_records is not None and max_records < 0:
        raise ValueError("max_records must be non-negative or None")

    nodes = _subtree_preorder(tree, root_id=root_id)
    node_set = set(nodes)
    records: list[RawAstOrderRelationRecord] = []

    def append_if_room(record: RawAstOrderRelationRecord) -> bool:
        if max_records is not None and len(records) >= max_records:
            return False
        records.append(record)
        return True

    for descendant in nodes:
        if descendant == nodes[0]:
            continue
        ancestors_from_root: list[NodeId] = []
        current = tree.parent_by_node[descendant]
        while current is not None and current in node_set:
            ancestors_from_root.append(current)
            current = tree.parent_by_node[current]
        ancestors_from_root.reverse()
        for ancestor in ancestors_from_root:
            if direct_only and tree.parent_by_node[descendant] != ancestor:
                continue
            if not append_if_room(_order_record(tree, ancestor, descendant, label=1)):
                return tuple(records)

    if not include_incomparable:
        return tuple(records)

    for left_index, left in enumerate(nodes):
        for right in nodes[left_index + 1 :]:
            if _is_strict_ancestor(tree, left, right) or _is_strict_ancestor(tree, right, left):
                continue
            if not append_if_room(_order_record(tree, left, right, label=0)):
                return tuple(records)
    return tuple(records)


def raw_ast_path_relation_matrices(
    tree: RawAstTree,
    paths: Sequence[RawAstPath],
) -> RawAstPathRelationMatrices:
    """Build exact raw-AST relation matrices for a set of path contexts."""

    frozen_paths = tuple(paths)
    lca_depth = tuple(tree.depth(path.lca(tree)) for path in frozen_paths)
    path_length = tuple(path.length for path in frozen_paths)
    oriented_rows: list[tuple[int, ...]] = []
    unoriented_rows: list[tuple[int, ...]] = []
    edge_symmetric_rows: list[tuple[int, ...]] = []
    edge_jaccard_rows: list[tuple[float, ...]] = []
    lca_depth_difference_rows: list[tuple[int, ...]] = []
    lca_anchored_product_rows: list[tuple[float, ...]] = []
    path_length_difference_rows: list[tuple[int, ...]] = []

    for left_index, left in enumerate(frozen_paths):
        oriented_row: list[int] = []
        unoriented_row: list[int] = []
        edge_symmetric_row: list[int] = []
        edge_jaccard_row: list[float] = []
        lca_depth_difference_row: list[int] = []
        lca_anchored_product_row: list[float] = []
        path_length_difference_row: list[int] = []
        for right_index, right in enumerate(frozen_paths):
            oriented_row.append(oriented_endpoint_distance(tree, left, right))
            unoriented_row.append(unoriented_endpoint_distance(tree, left, right))
            edge_symmetric_row.append(edge_symmetric_difference_distance(left, right))
            edge_jaccard_row.append(edge_jaccard_distance(left, right))
            lca_depth_difference_row.append(abs(lca_depth[left_index] - lca_depth[right_index]))
            lca_anchored_product_row.append(lca_anchored_product_distance(tree, left, right))
            path_length_difference_row.append(abs(path_length[left_index] - path_length[right_index]))
        oriented_rows.append(tuple(oriented_row))
        unoriented_rows.append(tuple(unoriented_row))
        edge_symmetric_rows.append(tuple(edge_symmetric_row))
        edge_jaccard_rows.append(tuple(edge_jaccard_row))
        lca_depth_difference_rows.append(tuple(lca_depth_difference_row))
        lca_anchored_product_rows.append(tuple(lca_anchored_product_row))
        path_length_difference_rows.append(tuple(path_length_difference_row))

    return RawAstPathRelationMatrices(
        oriented_endpoint_distance=tuple(oriented_rows),
        unoriented_endpoint_distance=tuple(unoriented_rows),
        edge_symmetric_difference=tuple(edge_symmetric_rows),
        edge_jaccard_distance=tuple(edge_jaccard_rows),
        lca_depth=lca_depth,
        lca_depth_difference=tuple(lca_depth_difference_rows),
        lca_anchored_product_distance=tuple(lca_anchored_product_rows),
        path_length=path_length,
        path_length_difference=tuple(path_length_difference_rows),
    )


def raw_ast_path_relation_records(
    tree: RawAstTree,
    paths: Sequence[RawAstPath],
) -> tuple[RawAstPathRelationRecord, ...]:
    """Return JSON-friendly upper-triangle raw-AST relation targets."""

    frozen_paths = tuple(paths)
    lca_depth = tuple(tree.depth(path.lca(tree)) for path in frozen_paths)
    records: list[RawAstPathRelationRecord] = []
    for left_index, left in enumerate(frozen_paths):
        for right_index in range(left_index + 1, len(frozen_paths)):
            right = frozen_paths[right_index]
            records.append(
                RawAstPathRelationRecord(
                    left_index=left_index,
                    right_index=right_index,
                    oriented_endpoint_distance=oriented_endpoint_distance(tree, left, right),
                    unoriented_endpoint_distance=unoriented_endpoint_distance(tree, left, right),
                    edge_symmetric_difference=edge_symmetric_difference_distance(left, right),
                    edge_jaccard_distance=edge_jaccard_distance(left, right),
                    left_lca_depth=lca_depth[left_index],
                    right_lca_depth=lca_depth[right_index],
                    lca_depth_difference=abs(lca_depth[left_index] - lca_depth[right_index]),
                    lca_anchored_product_distance=lca_anchored_product_distance(tree, left, right),
                    left_path_length=left.length,
                    right_path_length=right.length,
                    path_length_difference=abs(left.length - right.length),
                )
            )
    return tuple(records)
