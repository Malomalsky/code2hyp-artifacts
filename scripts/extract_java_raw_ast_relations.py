from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any, Literal

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from geometry_profile_research.java_raw_ast import parse_java_ast_tree
from geometry_profile_research.raw_ast import (
    RawAstPath,
    RawAstTree,
    NodeId,
    leaf_node_ids,
    raw_ast_order_relation_records,
    raw_ast_path_relation_records,
    terminal_to_terminal_paths,
)


Scope = Literal["compilation_unit", "callable"]


def _iter_java_files(sources: Iterable[Path]) -> tuple[Path, ...]:
    files: list[Path] = []
    for source in sources:
        if source.is_dir():
            files.extend(sorted(path for path in source.rglob("*.java") if path.is_file()))
        elif source.is_file() and source.suffix == ".java":
            files.append(source)
    return tuple(files)


def _validate_non_negative(name: str, value: int | None) -> None:
    if value is not None and value < 0:
        raise ValueError(f"{name} must be non-negative or None")


def _path_payload(tree: RawAstTree, path: RawAstPath) -> dict[str, Any]:
    lca_node = path.lca(tree)
    left_branch_node_ids, right_branch_node_ids = _branch_node_ids(tree, path)
    return {
        "start": path.start,
        "end": path.end,
        "length": path.length,
        "lca": lca_node,
        "start_id": path.start,
        "end_id": path.end,
        "lca_id": lca_node,
        "left_branch_node_ids": list(left_branch_node_ids),
        "right_branch_node_ids": list(right_branch_node_ids),
        "directed_edge_types": _directed_edge_types(tree, path),
        "lca_depth": tree.depth(lca_node),
        "start_label": tree.labels.get(path.start, ""),
        "end_label": tree.labels.get(path.end, ""),
        "lca_label": tree.labels.get(lca_node, ""),
    }


def _branch_node_ids(tree: RawAstTree, path: RawAstPath) -> tuple[tuple[NodeId, ...], tuple[NodeId, ...]]:
    lca = path.lca(tree)
    lca_index = path.nodes.index(lca)
    left = tuple(path.nodes[: lca_index + 1])
    right = tuple(reversed(path.nodes[lca_index:]))
    return left, right


def _directed_edge_types(tree: RawAstTree, path: RawAstPath) -> list[str]:
    edge_types: list[str] = []
    for left, right in zip(path.nodes, path.nodes[1:]):
        if tree.parent_by_node.get(left) == right:
            direction = "up"
            child = left
        elif tree.parent_by_node.get(right) == left:
            direction = "down"
            child = right
        else:
            direction = "cross"
            child = right
        attributes = tree.attributes.get(child, {})
        edge_type = attributes.get("edge_type", "")
        child_index = attributes.get("child_index", "")
        edge_types.append(f"{direction}:{edge_type}:{child_index}")
    return edge_types


def _relation_payload(
    tree: RawAstTree,
    source_path: Path,
    *,
    scope: Scope,
    scope_node: NodeId,
    max_paths_per_file: int | None,
    max_records_per_file: int | None,
    include_order_relations: bool,
    max_order_records_per_scope: int | None,
) -> dict[str, Any]:
    leaves = leaf_node_ids(tree, root_id=scope_node)
    paths = terminal_to_terminal_paths(tree, max_paths=max_paths_per_file, root_id=scope_node)
    records = raw_ast_path_relation_records(tree, paths)
    if max_records_per_file is not None:
        records = records[:max_records_per_file]
    scope_label = tree.labels.get(scope_node, "")
    scope_attributes = tree.attributes.get(scope_node, {})
    payload: dict[str, Any] = {
        "status": "ok",
        "source_path": str(source_path),
        "scope": scope,
        "scope_node": scope_node,
        "scope_label": scope_label,
        "scope_name": scope_attributes.get("name", ""),
        "node_count": _subtree_size(tree, scope_node),
        "leaf_count": len(leaves),
        "path_count": len(paths),
        "record_count": len(records),
        "paths": [_path_payload(tree, path) for path in paths],
        "records": [record.as_dict() for record in records],
    }
    if include_order_relations:
        order_records = raw_ast_order_relation_records(
            tree,
            root_id=scope_node,
            include_incomparable=True,
            max_records=max_order_records_per_scope,
        )
        payload["order_record_count"] = len(order_records)
        payload["order_records"] = [record.as_dict() for record in order_records]
    return payload


def _subtree_size(tree: RawAstTree, root_id: NodeId) -> int:
    count = 0
    stack = [root_id]
    while stack:
        node = stack.pop()
        count += 1
        stack.extend(tree.children_by_node.get(node, ()))
    return count


def _callable_nodes(tree: RawAstTree) -> tuple[NodeId, ...]:
    return tuple(
        node
        for node in tree.preorder()
        if tree.labels.get(node) in {"MethodDeclaration", "ConstructorDeclaration"}
    )


def _extract_one_file(
    source_path: Path,
    *,
    scope: Scope,
    max_paths_per_file: int | None,
    max_records_per_file: int | None,
    include_order_relations: bool,
    max_order_records_per_scope: int | None,
) -> list[dict[str, Any]]:
    try:
        source = source_path.read_text(encoding="utf-8", errors="replace")
        tree = parse_java_ast_tree(source)
    except Exception as exc:  # corpus extractor: preserve parse errors as data.
        return [
            {
                "status": "error",
                "source_path": str(source_path),
                "scope": scope,
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        ]

    if scope == "compilation_unit":
        return [
            _relation_payload(
                tree,
                source_path,
                scope=scope,
                scope_node=tree.root_id,
                max_paths_per_file=max_paths_per_file,
                max_records_per_file=max_records_per_file,
                include_order_relations=include_order_relations,
                max_order_records_per_scope=max_order_records_per_scope,
            )
        ]

    callables = _callable_nodes(tree)
    if not callables:
        return [
            {
                "status": "empty",
                "source_path": str(source_path),
                "scope": scope,
                "callable_count": 0,
            }
        ]
    return [
        _relation_payload(
            tree,
            source_path,
            scope=scope,
            scope_node=scope_node,
            max_paths_per_file=max_paths_per_file,
            max_records_per_file=max_records_per_file,
            include_order_relations=include_order_relations,
            max_order_records_per_scope=max_order_records_per_scope,
        )
        for scope_node in callables
    ]


def extract_java_raw_ast_relations(
    sources: Sequence[Path],
    *,
    scope: Scope = "compilation_unit",
    max_files: int | None = None,
    max_paths_per_file: int | None = 128,
    max_records_per_file: int | None = 4096,
    include_order_relations: bool = False,
    max_order_records_per_scope: int | None = 4096,
) -> list[dict[str, Any]]:
    """Extract exact raw-AST path-relation targets from Java source files."""

    _validate_non_negative("max_files", max_files)
    _validate_non_negative("max_paths_per_file", max_paths_per_file)
    _validate_non_negative("max_records_per_file", max_records_per_file)
    _validate_non_negative("max_order_records_per_scope", max_order_records_per_scope)
    java_files = _iter_java_files(sources)
    if max_files is not None:
        java_files = java_files[:max_files]
    payloads: list[dict[str, Any]] = []
    for source_path in java_files:
        payloads.extend(
            _extract_one_file(
                source_path,
                scope=scope,
                max_paths_per_file=max_paths_per_file,
                max_records_per_file=max_records_per_file,
                include_order_relations=include_order_relations,
                max_order_records_per_scope=max_order_records_per_scope,
            )
        )
    return payloads


def write_java_raw_ast_relations_jsonl(
    sources: Sequence[Path],
    output_path: Path,
    *,
    scope: Scope = "compilation_unit",
    max_files: int | None = None,
    max_paths_per_file: int | None = 128,
    max_records_per_file: int | None = 4096,
    include_order_relations: bool = False,
    max_order_records_per_scope: int | None = 4096,
) -> None:
    payloads = extract_java_raw_ast_relations(
        sources,
        scope=scope,
        max_files=max_files,
        max_paths_per_file=max_paths_per_file,
        max_records_per_file=max_records_per_file,
        include_order_relations=include_order_relations,
        max_order_records_per_scope=max_order_records_per_scope,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for payload in payloads:
            handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True))
            handle.write("\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract exact raw-AST path relation targets from Java source files."
    )
    parser.add_argument(
        "--source",
        type=Path,
        action="append",
        required=True,
        help="Java file or directory. Can be passed multiple times.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/java_raw_ast_relations.jsonl"),
        help="Output JSONL path.",
    )
    parser.add_argument(
        "--scope",
        choices=("compilation_unit", "callable"),
        default="compilation_unit",
        help="Extract relations for the whole compilation unit or per method/constructor.",
    )
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--max-paths-per-file", type=int, default=128)
    parser.add_argument("--max-records-per-file", type=int, default=4096)
    parser.add_argument(
        "--include-order-relations",
        action="store_true",
        help="Also emit AST ancestor-descendant and deterministic incomparable-control records.",
    )
    parser.add_argument("--max-order-records-per-scope", type=int, default=4096)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    write_java_raw_ast_relations_jsonl(
        tuple(args.source),
        args.output,
        scope=args.scope,
        max_files=args.max_files,
        max_paths_per_file=args.max_paths_per_file,
        max_records_per_file=args.max_records_per_file,
        include_order_relations=args.include_order_relations,
        max_order_records_per_scope=args.max_order_records_per_scope,
    )
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
