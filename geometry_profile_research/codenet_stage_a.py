from __future__ import annotations

import hashlib
import hmac
import json
from dataclasses import dataclass
from heapq import nsmallest
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from geometry_profile_research.codenet_eligibility import normalize_python_source, stable_sha256
from geometry_profile_research.python_raw_ast import parse_python_ast_tree
from geometry_profile_research.raw_ast import RawAstTree


@dataclass(frozen=True)
class StageAProgram:
    item_id: str
    cluster_id: str
    problem_id: str
    split: str
    role: str
    tree: RawAstTree


@dataclass(frozen=True)
class StageASplit:
    train: tuple[StageAProgram, ...]
    query: tuple[StageAProgram, ...]
    gallery: tuple[StageAProgram, ...]


def load_stage_a_split(
    *,
    source_root: Path,
    train_path: Path,
    validation_path: Path,
    ast_index_path: Path,
) -> StageASplit:
    """Load the frozen whole-program split and verify it against the AST audit."""

    train_rows = list(_iter_jsonl(train_path))
    validation_rows = list(_iter_jsonl(validation_path))
    ast_rows = {str(row["source_relpath"]): row for row in _iter_jsonl(ast_index_path)}
    selected_rows = train_rows + validation_rows
    selected_ids = [str(row["source_relpath"]) for row in selected_rows]
    if len(selected_ids) != len(set(selected_ids)):
        raise ValueError("frozen Stage A program IDs are not unique")
    if set(selected_ids) != set(ast_rows):
        raise ValueError("frozen program manifests and AST audit index do not contain the same sources")
    if any(str(row.get("split")) == "test" for row in selected_rows):
        raise ValueError("test program IDs must remain sealed")

    source_root = source_root.resolve()
    programs = [
        _load_program(source_root=source_root, sample_row=row, ast_row=ast_rows[str(row["source_relpath"])])
        for row in selected_rows
    ]
    train_count = len(train_rows)
    train = tuple(programs[:train_count])
    validation = programs[train_count:]
    query = tuple(program for program in validation if program.role == "query")
    gallery = tuple(program for program in validation if program.role == "gallery")
    if len(train) != 18_560 or len(query) != 776 or len(gallery) != 776:
        raise ValueError(
            "frozen Stage A cardinalities must be train=18560, query=776, gallery=776; "
            f"observed train={len(train)}, query={len(query)}, gallery={len(gallery)}"
        )
    if any(program.role != "train" or program.split != "train" for program in train):
        raise ValueError("training manifest contains a non-training role")
    if any(program.split != "validation" for program in query + gallery):
        raise ValueError("query/gallery programs must belong to validation")
    if {program.item_id for program in query} & {program.item_id for program in gallery}:
        raise ValueError("validation query and gallery programs overlap")
    return StageASplit(train=train, query=query, gallery=gallery)


def select_calibration_pairs(
    train_rows: Sequence[Mapping[str, Any]],
    *,
    beacon_key: bytes,
    dataset_revision: str,
    domain: str,
    same_cluster_count: int,
    cross_cluster_count: int,
) -> tuple[dict[str, Any], ...]:
    """Select fixed train-only calibration pairs by domain-separated HMAC."""

    if len(beacon_key) != 64:
        raise ValueError("beacon_key must contain exactly 64 bytes")
    if same_cluster_count < 0 or cross_cluster_count < 0:
        raise ValueError("calibration pair counts must be non-negative")
    by_cluster: dict[str, list[str]] = {}
    for row in train_rows:
        if str(row.get("split")) != "train" or str(row.get("role")) != "train":
            raise ValueError("calibration pairs may use only frozen training programs")
        cluster_id = str(row["cluster_id"])
        source = str(row["source_relpath"])
        by_cluster.setdefault(cluster_id, []).append(source)
    for sources in by_cluster.values():
        sources.sort()

    same_candidates = (
        (
            _calibration_digest(
                beacon_key,
                domain,
                dataset_revision,
                "same_cluster_rank",
                cluster_id,
                left,
                right,
            ),
            cluster_id,
            left,
            right,
        )
        for cluster_id, sources in sorted(by_cluster.items())
        for left_index, left in enumerate(sources)
        for right in sources[left_index + 1 :]
    )
    selected_same = nsmallest(same_cluster_count, same_candidates)
    if len(selected_same) != same_cluster_count:
        raise ValueError("insufficient same-cluster calibration pairs")

    cluster_ids = sorted(by_cluster)
    cross_candidates = []
    for left_cluster_index, left_cluster in enumerate(cluster_ids):
        for right_cluster in cluster_ids[left_cluster_index + 1 :]:
            left_sources = by_cluster[left_cluster]
            right_sources = by_cluster[right_cluster]
            left_digest = _calibration_digest(
                beacon_key,
                domain,
                dataset_revision,
                "cross_left_program",
                left_cluster,
                right_cluster,
            )
            right_digest = _calibration_digest(
                beacon_key,
                domain,
                dataset_revision,
                "cross_right_program",
                left_cluster,
                right_cluster,
            )
            left_source = left_sources[int.from_bytes(left_digest[:8], "big") % len(left_sources)]
            right_source = right_sources[int.from_bytes(right_digest[:8], "big") % len(right_sources)]
            rank = _calibration_digest(
                beacon_key,
                domain,
                dataset_revision,
                "cross_cluster_rank",
                left_cluster,
                right_cluster,
                left_source,
                right_source,
            )
            cross_candidates.append(
                (rank, left_cluster, right_cluster, left_source, right_source)
            )
    selected_cross = nsmallest(cross_cluster_count, cross_candidates)
    if len(selected_cross) != cross_cluster_count:
        raise ValueError("insufficient cross-cluster calibration pairs")

    rows: list[dict[str, Any]] = []
    for digest, cluster_id, left_source, right_source in selected_same:
        rows.append(
            {
                "pair_type": "same_cluster",
                "left_cluster_id": cluster_id,
                "right_cluster_id": cluster_id,
                "left_source_relpath": left_source,
                "right_source_relpath": right_source,
                "selection_digest": digest.hex(),
            }
        )
    for digest, left_cluster, right_cluster, left_source, right_source in selected_cross:
        rows.append(
            {
                "pair_type": "cross_cluster",
                "left_cluster_id": left_cluster,
                "right_cluster_id": right_cluster,
                "left_source_relpath": left_source,
                "right_source_relpath": right_source,
                "selection_digest": digest.hex(),
            }
        )
    for index, row in enumerate(rows):
        row["pair_index"] = index
    return tuple(rows)


def _load_program(
    *,
    source_root: Path,
    sample_row: Mapping[str, Any],
    ast_row: Mapping[str, Any],
) -> StageAProgram:
    source_relpath = str(sample_row["source_relpath"])
    relative = Path(source_relpath)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"unsafe source path: {source_relpath!r}")
    source_path = (source_root / relative).resolve()
    try:
        source_path.relative_to(source_root)
    except ValueError as error:
        raise ValueError(f"source path escapes source root: {source_relpath!r}") from error
    raw = source_path.read_bytes()
    if stable_sha256(raw) != str(ast_row["raw_source_sha256"]):
        raise ValueError(f"raw source hash mismatch: {source_relpath}")
    canonical = normalize_python_source(raw)
    if not canonical.decode_ok:
        raise ValueError(f"source decode failed after audit: {source_relpath}")
    canonical_bytes = canonical.text.encode("utf-8")
    if stable_sha256(canonical_bytes) != str(ast_row["canonical_source_sha256"]):
        raise ValueError(f"canonical source hash mismatch: {source_relpath}")
    tree = parse_python_ast_tree(canonical.text)
    if len(tree.parent_by_node) != int(ast_row["node_count"]):
        raise ValueError(f"AST node count mismatch: {source_relpath}")
    return StageAProgram(
        item_id=source_relpath,
        cluster_id=str(sample_row["cluster_id"]),
        problem_id=str(sample_row["problem_id"]),
        split=str(sample_row["split"]),
        role=str(sample_row["role"]),
        tree=tree,
    )


def _calibration_digest(key: bytes, *fields: str) -> bytes:
    if any(not field or "\x00" in field for field in fields):
        raise ValueError("HMAC fields must be non-empty and cannot contain NUL")
    return hmac.new(key, "\x00".join(fields).encode("utf-8"), hashlib.sha256).digest()


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSONL at {path}:{line_number}") from error
