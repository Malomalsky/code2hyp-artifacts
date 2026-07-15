from __future__ import annotations

import json
import math
import platform
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from geometry_profile_research.codenet_eligibility import canonical_json_bytes, jsonl_bytes, stable_sha256
from geometry_profile_research.python_raw_ast import parse_python_ast_tree
from geometry_profile_research.raw_ast import leaf_node_ids, terminal_to_terminal_paths


AUDIT_SCHEMA_VERSION = "codenet-python800-stage-a-selected-source-ast-audit-v1"
PROTOCOL_SCHEMA_VERSION = "code2hyp-stage-a-ast-path-protocol-v1"


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSONL at {path}:{line_number}") from error


def audit_source_program(
    source_root: Path,
    sample_row: Mapping[str, Any],
    *,
    max_paths: int,
    selection_policy: str,
) -> dict[str, Any]:
    """Parse and audit one selected CodeNet source without exposing its text."""

    source_relpath = str(sample_row["source_relpath"])
    source_path = _safe_source_path(source_root, source_relpath)
    result: dict[str, Any] = {
        "cluster_id": str(sample_row["cluster_id"]),
        "problem_id": str(sample_row["problem_id"]),
        "role": str(sample_row["role"]),
        "source_relpath": source_relpath,
        "split": str(sample_row["split"]),
        "submission_id": str(sample_row["submission_id"]),
    }
    try:
        raw = source_path.read_bytes()
    except OSError as error:
        return {**result, "audit_ok": False, "failure": f"source_read:{type(error).__name__}"}

    result["raw_source_bytes"] = len(raw)
    result["raw_source_sha256"] = stable_sha256(raw)
    from geometry_profile_research.codenet_eligibility import normalize_python_source

    canonical = normalize_python_source(raw)
    result["detected_encoding"] = canonical.encoding
    if not canonical.decode_ok:
        return {**result, "audit_ok": False, "failure": f"source_decode:{canonical.decode_error}"}
    canonical_bytes = canonical.text.encode("utf-8")
    result["canonical_source_bytes"] = len(canonical_bytes)
    result["canonical_source_sha256"] = stable_sha256(canonical_bytes)
    try:
        tree = parse_python_ast_tree(canonical.text)
    except (SyntaxError, TypeError, ValueError) as error:
        return {**result, "audit_ok": False, "failure": f"ast_parse:{type(error).__name__}"}

    leaves = leaf_node_ids(tree)
    pair_capacity = len(leaves) * (len(leaves) - 1) // 2
    if pair_capacity == 0:
        return {
            **result,
            "audit_ok": False,
            "failure": "fewer_than_two_AST_leaves",
            "node_count": len(tree.parent_by_node),
            "leaf_count": len(leaves),
            "available_terminal_pair_count": pair_capacity,
        }
    paths = terminal_to_terminal_paths(
        tree,
        max_paths=max_paths,
        selection_policy=selection_policy,
    )
    endpoint_pairs = [(path.start, path.end) for path in paths]
    expected_count = min(max_paths, pair_capacity)
    if len(paths) != expected_count or len(set(endpoint_pairs)) != expected_count:
        return {
            **result,
            "audit_ok": False,
            "failure": "selected_path_count_or_uniqueness_mismatch",
            "node_count": len(tree.parent_by_node),
            "leaf_count": len(leaves),
            "available_terminal_pair_count": pair_capacity,
            "selected_path_count": len(paths),
        }
    lca_depths = [tree.depth(path.lca(tree)) for path in paths]
    return {
        **result,
        "audit_ok": True,
        "failure": None,
        "node_count": len(tree.parent_by_node),
        "leaf_count": len(leaves),
        "available_terminal_pair_count": pair_capacity,
        "selected_path_count": len(paths),
        "selected_endpoint_pairs_sha256": stable_sha256(canonical_json_bytes(endpoint_pairs)),
        "selected_lca_depth_histogram": {
            str(depth): count for depth, count in sorted(Counter(lca_depths).items())
        },
    }


def audit_selected_sources(
    *,
    project_root: Path,
    protocol_path: Path,
    train_path: Path,
    validation_path: Path,
    sampling_manifest_path: Path,
    source_root: Path,
    output_dir: Path,
    workers: int = 1,
    progress_every: int = 1_000,
) -> dict[str, Any]:
    """Audit all frozen train/validation sources and write portable artifacts."""

    if workers <= 0:
        raise ValueError("workers must be positive")
    protocol_bytes = protocol_path.read_bytes()
    protocol = json.loads(protocol_bytes)
    _validate_protocol(
        project_root=project_root,
        protocol=protocol,
        train_path=train_path,
        validation_path=validation_path,
        sampling_manifest_path=sampling_manifest_path,
    )
    train_rows = list(iter_jsonl(train_path))
    validation_rows = list(iter_jsonl(validation_path))
    sample_rows = train_rows + validation_rows
    _validate_sample_rows(sample_rows)
    max_paths = int(protocol["path_selection"]["maximum_paths_per_program"])
    selection_policy = str(protocol["implementation"]["policy_name"])
    source_root = source_root.resolve()
    if not source_root.is_dir():
        raise FileNotFoundError(f"CodeNet source root does not exist: {source_root}")

    inputs = [(source_root, row, max_paths, selection_policy) for row in sample_rows]
    results: list[dict[str, Any]] = []
    if workers == 1:
        iterator = (_audit_worker(item) for item in inputs)
        executor = None
    else:
        executor = ProcessPoolExecutor(max_workers=workers)
        iterator = executor.map(_audit_worker, inputs, chunksize=32)
    try:
        for index, result in enumerate(iterator, start=1):
            results.append(result)
            if progress_every > 0 and index % progress_every == 0:
                print(f"audited_sources={index}/{len(sample_rows)}", flush=True)
    finally:
        if executor is not None:
            executor.shutdown()

    summary = _build_summary(results, max_paths=max_paths, selection_policy=selection_policy)
    output_dir.mkdir(parents=True, exist_ok=True)
    index_path = output_dir / "selected_source_ast_index.jsonl"
    summary_path = output_dir / "selected_source_ast_summary.json"
    index_path.write_bytes(jsonl_bytes(results))
    summary_path.write_bytes(canonical_json_bytes(summary))
    manifest = {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "valid_for_stage_a_modeling": bool(summary["valid_for_stage_a_modeling"]),
        "input": {
            "protocol_path": protocol_path.relative_to(project_root).as_posix(),
            "protocol_sha256": stable_sha256(protocol_bytes),
            "sampling_manifest_sha256": stable_sha256(sampling_manifest_path.read_bytes()),
            "train_programs_sha256": stable_sha256(train_path.read_bytes()),
            "validation_programs_sha256": stable_sha256(validation_path.read_bytes()),
            "source_root_name": source_root.name,
        },
        "implementation": {
            "path_selection_policy": selection_policy,
            "maximum_paths_per_program": max_paths,
            "workers_are_non_semantic": True,
            "python_version": sys.version.split()[0],
            "platform": platform.platform(),
        },
        "artifacts": [
            {
                "path": index_path.name,
                "bytes": index_path.stat().st_size,
                "sha256": stable_sha256(index_path.read_bytes()),
            },
            {
                "path": summary_path.name,
                "bytes": summary_path.stat().st_size,
                "sha256": stable_sha256(summary_path.read_bytes()),
            },
        ],
        "summary": summary,
        "protocol_state": {
            "selected_source_AST_audit_generated": True,
            "validation_retrieval_metrics_computed": False,
            "test_program_ids_materialized": False,
            "test_relevance_labels_opened": False,
            "test_retrieval_metrics_computed": False,
        },
    }
    manifest_path = output_dir / "selected_source_ast_manifest.json"
    manifest_path.write_bytes(canonical_json_bytes(manifest))
    sidecar = output_dir / "selected_source_ast_manifest.sha256"
    sidecar.write_text(f"{stable_sha256(manifest_path.read_bytes())}  {manifest_path.name}\n", encoding="ascii")
    return manifest


def _audit_worker(item: tuple[Path, Mapping[str, Any], int, str]) -> dict[str, Any]:
    source_root, row, max_paths, selection_policy = item
    return audit_source_program(
        source_root,
        row,
        max_paths=max_paths,
        selection_policy=selection_policy,
    )


def _validate_protocol(
    *,
    project_root: Path,
    protocol: Mapping[str, Any],
    train_path: Path,
    validation_path: Path,
    sampling_manifest_path: Path,
) -> None:
    if protocol.get("schema_version") != PROTOCOL_SCHEMA_VERSION:
        raise ValueError("unsupported AST path protocol schema")
    if protocol.get("status") != "frozen_before_selected_source_audit_or_validation_metrics":
        raise ValueError("AST path protocol was not frozen before source audit")
    expected_inputs = {
        "train": str(protocol["program_sampling"]["train_sha256"]),
        "validation": str(protocol["program_sampling"]["validation_sha256"]),
        "sampling_manifest": str(protocol["program_sampling"]["manifest_sha256"]),
        "raw_ast": str(protocol["implementation"]["raw_ast_sha256"]),
        "encoder": str(protocol["implementation"]["encoder_sha256"]),
    }
    actual_inputs = {
        "train": stable_sha256(train_path.read_bytes()),
        "validation": stable_sha256(validation_path.read_bytes()),
        "sampling_manifest": stable_sha256(sampling_manifest_path.read_bytes()),
        "raw_ast": stable_sha256((project_root / str(protocol["implementation"]["raw_ast_path"])).read_bytes()),
        "encoder": stable_sha256((project_root / str(protocol["implementation"]["encoder_path"])).read_bytes()),
    }
    mismatches = {
        name: {"actual": actual_inputs[name], "expected": expected}
        for name, expected in expected_inputs.items()
        if actual_inputs[name] != expected
    }
    if mismatches:
        raise ValueError(f"AST path protocol input hash mismatch: {mismatches}")
    if protocol["program_sampling"].get("test_program_ids_materialized") is not False:
        raise ValueError("test program IDs were materialized before the AST source audit")


def _validate_sample_rows(rows: Sequence[Mapping[str, Any]]) -> None:
    sources = [str(row.get("source_relpath", "")) for row in rows]
    if not sources or len(sources) != len(set(sources)) or not all(sources):
        raise ValueError("selected source paths must be non-empty and globally unique")
    if any(str(row.get("split")) == "test" for row in rows):
        raise ValueError("test program IDs must remain sealed")
    if any(any("user" in str(key).casefold() for key in row) for row in rows):
        raise ValueError("selected program rows must not publish user identifiers")


def _safe_source_path(source_root: Path, source_relpath: str) -> Path:
    relative = Path(source_relpath)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"unsafe source_relpath: {source_relpath!r}")
    candidate = (source_root / relative).resolve()
    try:
        candidate.relative_to(source_root.resolve())
    except ValueError as error:
        raise ValueError(f"source path escapes source root: {source_relpath!r}") from error
    return candidate


def _build_summary(
    rows: Sequence[Mapping[str, Any]],
    *,
    max_paths: int,
    selection_policy: str,
) -> dict[str, Any]:
    failures = [dict(row) for row in rows if not bool(row.get("audit_ok"))]
    valid = [row for row in rows if bool(row.get("audit_ok"))]
    below_k = [
        {
            "source_relpath": str(row["source_relpath"]),
            "split": str(row["split"]),
            "role": str(row["role"]),
            "leaf_count": int(row["leaf_count"]),
            "available_terminal_pair_count": int(row["available_terminal_pair_count"]),
            "selected_path_count": int(row["selected_path_count"]),
        }
        for row in valid
        if int(row["available_terminal_pair_count"]) < max_paths
    ]
    return {
        "valid_for_stage_a_modeling": not failures and len(valid) == len(rows),
        "program_count": len(rows),
        "split_counts": dict(sorted(Counter(str(row["split"]) for row in rows).items())),
        "role_counts": dict(sorted(Counter(str(row["role"]) for row in rows).items())),
        "parse_failure_count": len(failures),
        "failures": failures,
        "maximum_paths_per_program": max_paths,
        "path_selection_policy": selection_policy,
        "programs_below_K_count": len(below_k),
        "programs_below_K": below_k,
        "node_count_distribution": _distribution([int(row["node_count"]) for row in valid]),
        "leaf_count_distribution": _distribution([int(row["leaf_count"]) for row in valid]),
        "terminal_pair_capacity_distribution": _distribution(
            [int(row["available_terminal_pair_count"]) for row in valid]
        ),
        "selected_path_count_distribution": _distribution(
            [int(row["selected_path_count"]) for row in valid]
        ),
    }


def _distribution(values: Sequence[int]) -> dict[str, int | float]:
    if not values:
        return {"count": 0}
    ordered = sorted(values)
    return {
        "count": len(ordered),
        "minimum": ordered[0],
        "q25": _linear_quantile(ordered, 0.25),
        "median": _linear_quantile(ordered, 0.5),
        "q75": _linear_quantile(ordered, 0.75),
        "maximum": ordered[-1],
        "mean": sum(ordered) / len(ordered),
    }


def _linear_quantile(ordered: Sequence[int], probability: float) -> int | float:
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight
