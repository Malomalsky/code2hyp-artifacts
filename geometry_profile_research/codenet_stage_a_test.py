from __future__ import annotations

import json
import os
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from geometry_profile_research.codenet_ast_audit import (
    audit_source_program,
    build_source_audit_summary,
)
from geometry_profile_research.codenet_eligibility import (
    canonical_json_bytes,
    jsonl_bytes,
    stable_sha256,
)
from geometry_profile_research.codenet_sampling import (
    representative_program_digest,
    user_order_digest,
)


TEST_EXECUTION_PROTOCOL_SCHEMA = "code2hyp-stage-a-test-execution-protocol-v1"
TEST_OPENING_RECEIPT_SCHEMA = "code2hyp-stage-a-test-opening-receipt-v1"
TEST_MATERIALIZATION_SCHEMA = "code2hyp-stage-a-test-materialization-v1"


def validate_test_execution_protocol(
    *,
    project_root: Path,
    protocol_path: Path,
    input_path_overrides: Mapping[str, Path] | None = None,
) -> dict[str, Any]:
    """Verify every frozen input before the single test-opening transaction."""

    protocol_bytes = protocol_path.read_bytes()
    protocol = json.loads(protocol_bytes)
    if protocol.get("schema_version") != TEST_EXECUTION_PROTOCOL_SCHEMA:
        raise ValueError("unsupported Stage A test-execution protocol")
    if protocol.get("status") != "frozen_during_validation_before_validation_selection_or_test_unseal":
        raise ValueError("test-execution protocol was not frozen at the required stage")
    if protocol.get("state_at_freeze") != {
        "validation_started": True,
        "validation_selection_complete": False,
        "test_program_ids_materialized": False,
        "test_relevance_labels_opened": False,
        "test_retrieval_metrics_computed": False,
    }:
        raise ValueError("test-execution protocol does not record a fail-closed freeze state")

    required_input_names = {
        "registration",
        "sampling_protocol",
        "ast_path_protocol",
        "model_analysis_protocol",
        "test_inference_protocol",
        "split_manifest",
        "cluster_assignments",
        "d5_metadata_manifest",
        "d5_metadata_index",
        "validation_selection_requirement",
    }
    if set(protocol.get("inputs", {})) != required_input_names:
        raise ValueError("test-execution protocol does not pin the exact required input set")

    overrides = dict(input_path_overrides or {})
    checked_inputs: dict[str, dict[str, Any]] = {}
    for name, specification in protocol["inputs"].items():
        if name == "validation_selection_requirement":
            continue
        path = overrides.get(name, project_root / str(specification["path"]))
        if name == "d5_metadata_index":
            checked_inputs[name] = {
                "path": str(specification["path"]),
                "sha256": str(specification["sha256"]),
                "verification": "deferred_until_after_opening_receipt",
            }
            continue
        content = path.read_bytes()
        actual = stable_sha256(content)
        expected = str(specification["sha256"])
        if actual != expected:
            raise ValueError(f"frozen test-execution input hash mismatch: {name}")
        checked_inputs[name] = {
            "path": str(specification["path"]),
            "bytes": len(content),
            "sha256": actual,
        }
    return {
        "protocol": protocol,
        "protocol_sha256": stable_sha256(protocol_bytes),
        "inputs": checked_inputs,
    }


def validate_validation_selection_for_test(
    *,
    selection_path: Path,
    selection_seal_path: Path,
    registered_seeds: Sequence[int],
) -> dict[str, Any]:
    """Fail closed unless validation selection is sealed and test-naive."""

    selection_bytes = selection_path.read_bytes()
    seal_bytes = selection_seal_path.read_bytes()
    selection = json.loads(selection_bytes)
    seal = json.loads(seal_bytes)
    if selection.get("schema_version") != "code2hyp-stage-a-validation-selection-v1":
        raise ValueError("unexpected validation-selection schema")
    if seal.get("schema_version") != "code2hyp-stage-a-validation-selection-seal-v1":
        raise ValueError("unexpected validation-selection seal schema")
    if seal.get("inputs", {}).get("selection", {}).get("sha256") != stable_sha256(selection_bytes):
        raise ValueError("validation selection differs from its seal")
    required_checks = {
        "registered_seed_set_complete",
        "all_seed_results_match_their_seals",
        "selection_recomputed_from_frozen_rule",
        "validation_only",
    }
    checks = seal.get("checks", {})
    if any(checks.get(name) is not True for name in required_checks):
        raise ValueError("validation-selection seal is incomplete")
    if tuple(int(seed) for seed in selection.get("registered_seeds", ())) != tuple(
        int(seed) for seed in registered_seeds
    ):
        raise ValueError("validation selection does not identify the registered seed sequence")
    if len(seal.get("inputs", {}).get("seeds", ())) != len(registered_seeds):
        raise ValueError("validation-selection seal does not bind every registered seed")
    if float(selection["selected_active_curvature"]) != float(seal["selected_active_curvature"]):
        raise ValueError("selected curvature differs between selection and seal")
    if str(selection["selected_cell_id"]) != str(seal["selected_cell_id"]):
        raise ValueError("selected cell differs between selection and seal")
    forbidden_flags = (
        "test_program_ids_materialized",
        "test_relevance_labels_opened",
        "test_retrieval_metrics_computed",
    )
    if any(bool(selection.get(flag)) or bool(seal.get(flag)) for flag in forbidden_flags):
        raise ValueError("validation artifacts indicate prior test access")
    return {
        "selection": selection,
        "selection_sha256": stable_sha256(selection_bytes),
        "seal": seal,
        "seal_sha256": stable_sha256(seal_bytes),
    }


def open_or_resume_test_transaction(
    *,
    output_dir: Path,
    protocol_sha256: str,
    selection_sha256: str,
    selection_seal_sha256: str,
    selected_cell_id: str,
    selected_active_curvature: float,
    implementation: Mapping[str, Any],
    created_utc: str | None = None,
) -> dict[str, Any]:
    """Create one immutable opening receipt or resume that exact transaction."""

    identity = {
        "protocol_sha256": str(protocol_sha256),
        "selection_sha256": str(selection_sha256),
        "selection_seal_sha256": str(selection_seal_sha256),
        "selected_cell_id": str(selected_cell_id),
        "selected_active_curvature": float(selected_active_curvature),
        "implementation": dict(implementation),
    }
    identity_sha256 = stable_sha256(canonical_json_bytes(identity))
    output_dir.mkdir(parents=True, exist_ok=True)
    receipt_path = output_dir / "test_opening_receipt.json"
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if receipt.get("schema_version") != TEST_OPENING_RECEIPT_SCHEMA:
            raise ValueError("existing test-opening receipt has an unexpected schema")
        if receipt.get("opening_ordinal") != 1:
            raise ValueError("existing test-opening receipt is not the single registered opening")
        if receipt.get("transaction_identity_sha256") != identity_sha256:
            raise ValueError("refusing a second or incompatible test-opening transaction")
        if receipt.get("identity") != identity:
            raise ValueError("test-opening receipt identity is inconsistent")
        return {**receipt, "transaction_resumed": True}

    receipt = {
        "schema_version": TEST_OPENING_RECEIPT_SCHEMA,
        "opening_ordinal": 1,
        "created_utc": created_utc or datetime.now(timezone.utc).isoformat(),
        "identity": identity,
        "transaction_identity_sha256": identity_sha256,
        "state_at_receipt": {
            "validation_selection_sealed": True,
            "test_opening_authorized": True,
            "test_program_ids_materialized": False,
            "test_retrieval_metrics_computed": False,
        },
    }
    content = canonical_json_bytes(receipt)
    try:
        descriptor = os.open(receipt_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError:
        return open_or_resume_test_transaction(
            output_dir=output_dir,
            protocol_sha256=protocol_sha256,
            selection_sha256=selection_sha256,
            selection_seal_sha256=selection_seal_sha256,
            selected_cell_id=selected_cell_id,
            selected_active_curvature=selected_active_curvature,
            implementation=implementation,
            created_utc=created_utc,
        )
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
    return {**receipt, "transaction_resumed": False}


def select_test_programs(
    *,
    metadata_rows: Iterable[Mapping[str, Any]],
    assignments: Sequence[Mapping[str, Any]],
    beacon_key: bytes,
    dataset_revision: str,
    program_domain: str,
    user_domain: str,
    queries_per_cluster: int,
    gallery_per_cluster: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Apply the frozen user-distinct HMAC rule to test clusters only."""

    if queries_per_cluster <= 0 or gallery_per_cluster <= 0:
        raise ValueError("test query and gallery counts must be positive")
    split_by_cluster: dict[str, str] = {}
    split_index_by_cluster: dict[str, int] = {}
    for assignment in assignments:
        cluster_id = str(assignment.get("cluster_id", ""))
        split = str(assignment.get("split", ""))
        if not cluster_id or split not in {"train", "validation", "test"}:
            raise ValueError("malformed cluster assignment")
        if cluster_id in split_by_cluster:
            raise ValueError(f"duplicate cluster assignment: {cluster_id}")
        split_by_cluster[cluster_id] = split
        split_index_by_cluster[cluster_id] = int(assignment["split_index"])

    test_clusters = {cluster for cluster, split in split_by_cluster.items() if split == "test"}
    grouped: dict[str, dict[str, list[dict[str, Any]]]] = {
        cluster_id: defaultdict(list) for cluster_id in test_clusters
    }
    observed_test_sources: set[str] = set()
    for row in metadata_rows:
        cluster_id = str(row.get("problem_cluster_id", ""))
        if cluster_id not in split_by_cluster:
            raise ValueError(f"metadata row references an unassigned cluster: {cluster_id}")
        if cluster_id not in test_clusters:
            continue
        user_id = str(row.get("user_id_sha256", ""))
        source_relpath = str(row.get("source_relpath", ""))
        problem_id = str(row.get("problem_id", ""))
        submission_id = str(row.get("submission_id", ""))
        if not all((user_id, source_relpath, problem_id, submission_id)):
            raise ValueError("test metadata rows must contain user, source, problem and submission IDs")
        if str(row.get("status", "")) != "Accepted":
            raise ValueError("only accepted CodeNet programs may be sampled")
        if source_relpath in observed_test_sources:
            raise ValueError(f"duplicate test source_relpath: {source_relpath}")
        observed_test_sources.add(source_relpath)
        grouped[cluster_id][user_id].append(dict(row))

    required = queries_per_cluster + gallery_per_cluster
    selected: list[dict[str, Any]] = []
    minimum_available_users: int | None = None
    ordered_assignments = sorted(assignments, key=lambda row: int(row["order_index"]))
    for assignment in ordered_assignments:
        if str(assignment["split"]) != "test":
            continue
        cluster_id = str(assignment["cluster_id"])
        representatives: list[tuple[bytes, str, str, dict[str, Any]]] = []
        for user_id, candidates in grouped[cluster_id].items():
            representative = min(
                candidates,
                key=lambda row: (
                    representative_program_digest(
                        beacon_key=beacon_key,
                        domain=program_domain,
                        dataset_revision=dataset_revision,
                        cluster_id=cluster_id,
                        user_id_sha256=user_id,
                        source_relpath=str(row["source_relpath"]),
                    ),
                    str(row["source_relpath"]),
                ),
            )
            representatives.append(
                (
                    user_order_digest(
                        beacon_key=beacon_key,
                        domain=user_domain,
                        dataset_revision=dataset_revision,
                        cluster_id=cluster_id,
                        user_id_sha256=user_id,
                    ),
                    user_id,
                    str(representative["source_relpath"]),
                    representative,
                )
            )
        representatives.sort(key=lambda item: (item[0], item[1], item[2]))
        if len(representatives) < required:
            raise ValueError(
                f"test cluster {cluster_id} has {len(representatives)} distinct users; {required} required"
            )
        minimum_available_users = (
            len(representatives)
            if minimum_available_users is None
            else min(minimum_available_users, len(representatives))
        )
        chosen = representatives[:required]
        roles = [
            *(("query", index) for index in range(queries_per_cluster)),
            *(("gallery", index) for index in range(gallery_per_cluster)),
        ]
        for sample_index, ((_, _, _, row), (role, role_index)) in enumerate(
            zip(chosen, roles, strict=True)
        ):
            selected.append(
                {
                    "cluster_id": cluster_id,
                    "cluster_split_index": split_index_by_cluster[cluster_id],
                    "problem_id": str(row["problem_id"]),
                    "role": role,
                    "role_index": role_index,
                    "sample_index": sample_index,
                    "source_relpath": str(row["source_relpath"]),
                    "split": "test",
                    "submission_id": str(row["submission_id"]),
                }
            )
    if any("user" in key.casefold() for row in selected for key in row):
        raise ValueError("test sampling outputs cannot publish user identifiers")
    return selected, {
        "test_clusters": len(test_clusters),
        "test_queries": sum(row["role"] == "query" for row in selected),
        "test_gallery": sum(row["role"] == "gallery" for row in selected),
        "test_programs": len(selected),
        "minimum_available_users_test": minimum_available_users or 0,
    }


def materialize_and_audit_test_programs(
    *,
    project_root: Path,
    protocol_path: Path,
    selection_path: Path,
    selection_seal_path: Path,
    source_root: Path,
    output_dir: Path,
    implementation: Mapping[str, Any],
    d5_metadata_index_path: Path | None = None,
    workers: int = 1,
    progress_every: int = 500,
) -> dict[str, Any]:
    """Perform the one registered unseal and audit every selected test source."""

    if workers <= 0:
        raise ValueError("workers must be positive")
    overrides = (
        {"d5_metadata_index": d5_metadata_index_path}
        if d5_metadata_index_path is not None
        else None
    )
    verified = validate_test_execution_protocol(
        project_root=project_root,
        protocol_path=protocol_path,
        input_path_overrides=overrides,
    )
    protocol = verified["protocol"]
    registered_seeds = tuple(int(seed) for seed in protocol["test_evaluation"]["model_seeds"])
    selection_state = validate_validation_selection_for_test(
        selection_path=selection_path,
        selection_seal_path=selection_seal_path,
        registered_seeds=registered_seeds,
    )
    selection = selection_state["selection"]
    receipt = open_or_resume_test_transaction(
        output_dir=output_dir,
        protocol_sha256=verified["protocol_sha256"],
        selection_sha256=selection_state["selection_sha256"],
        selection_seal_sha256=selection_state["seal_sha256"],
        selected_cell_id=str(selection["selected_cell_id"]),
        selected_active_curvature=float(selection["selected_active_curvature"]),
        implementation=implementation,
    )

    inputs = protocol["inputs"]
    registration = _load_json(project_root / str(inputs["registration"]["path"]))
    sampling_protocol = _load_json(project_root / str(inputs["sampling_protocol"]["path"]))
    assignments = list(_iter_jsonl(project_root / str(inputs["cluster_assignments"]["path"])))
    metadata_path = (
        d5_metadata_index_path
        if d5_metadata_index_path is not None
        else project_root / str(inputs["d5_metadata_index"]["path"])
    )
    if stable_sha256(metadata_path.read_bytes()) != str(inputs["d5_metadata_index"]["sha256"]):
        raise ValueError("D5 metadata index differs from the hash frozen before test opening")
    beacon_key = bytes.fromhex(str(registration["nist_randomness_beacon"]["output_value_hex"]))
    sampling_rule = sampling_protocol["randomness"]
    sample_spec = protocol["test_sampling"]
    selected, sampling_summary = select_test_programs(
        metadata_rows=_iter_jsonl(metadata_path),
        assignments=assignments,
        beacon_key=beacon_key,
        dataset_revision=str(registration["design"]["dataset_revision"]),
        program_domain=str(sampling_rule["program_domain"]),
        user_domain=str(sampling_rule["user_domain"]),
        queries_per_cluster=int(sample_spec["queries_per_cluster"]),
        gallery_per_cluster=int(sample_spec["gallery_per_cluster"]),
    )
    expected = {
        "test_clusters": int(sample_spec["clusters"]),
        "test_queries": int(sample_spec["clusters"]) * int(sample_spec["queries_per_cluster"]),
        "test_gallery": int(sample_spec["clusters"]) * int(sample_spec["gallery_per_cluster"]),
        "test_programs": int(sample_spec["program_count"]),
    }
    if any(sampling_summary[name] != value for name, value in expected.items()):
        raise ValueError(f"test sampling cardinality mismatch: {sampling_summary}")

    ast_protocol = _load_json(project_root / str(inputs["ast_path_protocol"]["path"]))
    max_paths = int(ast_protocol["path_selection"]["maximum_paths_per_program"])
    selection_policy = str(ast_protocol["implementation"]["policy_name"])
    source_root = source_root.resolve()
    audit_arguments = [(source_root, row, max_paths, selection_policy) for row in selected]
    if workers == 1:
        audited_iterator = (_audit_test_worker(item) for item in audit_arguments)
        executor = None
    else:
        executor = ProcessPoolExecutor(max_workers=workers)
        audited_iterator = executor.map(_audit_test_worker, audit_arguments, chunksize=32)
    audited: list[dict[str, Any]] = []
    try:
        for index, row in enumerate(audited_iterator, start=1):
            audited.append(row)
            if progress_every > 0 and index % progress_every == 0:
                print(f"audited_test_sources={index}/{len(selected)}", flush=True)
    finally:
        if executor is not None:
            executor.shutdown()
    ast_summary = build_source_audit_summary(
        audited,
        max_paths=max_paths,
        selection_policy=selection_policy,
    )
    if not ast_summary["valid_for_stage_a_modeling"]:
        failure_path = output_dir / "test_source_ast_failure.json"
        _write_once_or_verify(failure_path, canonical_json_bytes(ast_summary))
        raise ValueError("test source AST audit failed before retrieval metrics")

    payloads = {
        "test_programs.jsonl": jsonl_bytes(selected),
        "test_source_ast_index.jsonl": jsonl_bytes(audited),
        "test_source_ast_summary.json": canonical_json_bytes(ast_summary),
    }
    artifacts = []
    for filename, content in payloads.items():
        path = output_dir / filename
        _write_once_or_verify(path, content)
        artifacts.append({
            "path": filename,
            "bytes": len(content),
            "sha256": stable_sha256(content),
        })
    receipt_path = output_dir / "test_opening_receipt.json"
    manifest = {
        "schema_version": TEST_MATERIALIZATION_SCHEMA,
        "experiment_role": "single_registered_test_opening_with_pre_metric_AST_audit",
        "implementation": dict(implementation),
        "inputs": {
            "test_execution_protocol_sha256": verified["protocol_sha256"],
            "validation_selection_sha256": selection_state["selection_sha256"],
            "validation_selection_seal_sha256": selection_state["seal_sha256"],
            "opening_receipt_sha256": stable_sha256(receipt_path.read_bytes()),
        },
        "selected_active_curvature": float(selection["selected_active_curvature"]),
        "selected_cell_id": str(selection["selected_cell_id"]),
        "sampling_summary": sampling_summary,
        "ast_summary": ast_summary,
        "artifacts": sorted(artifacts, key=lambda item: item["path"]),
        "opening": {
            "ordinal": 1,
            "transaction_identity_sha256": receipt["transaction_identity_sha256"],
        },
        "test_program_ids_materialized": True,
        "test_relevance_labels_opened": True,
        "test_retrieval_metrics_computed": False,
    }
    manifest_path = output_dir / "test_materialization_manifest.json"
    _write_once_or_verify(manifest_path, canonical_json_bytes(manifest))
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def _audit_test_worker(item: tuple[Path, Mapping[str, Any], int, str]) -> dict[str, Any]:
    source_root, row, max_paths, selection_policy = item
    return audit_source_program(
        source_root,
        row,
        max_paths=max_paths,
        selection_policy=selection_policy,
    )


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSONL at {path}:{line_number}") from error


def _write_once_or_verify(path: Path, content: bytes) -> None:
    if path.exists():
        if path.read_bytes() != content:
            raise ValueError(f"refusing to overwrite a different test artifact: {path}")
        return
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(content)
    os.replace(temporary, path)
