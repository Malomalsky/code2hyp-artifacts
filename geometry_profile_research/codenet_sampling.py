from __future__ import annotations

import hashlib
import hmac
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from geometry_profile_research.codenet_eligibility import (
    canonical_json_bytes,
    jsonl_bytes,
    portable_manifest_path,
    stable_sha256,
)
from geometry_profile_research.codenet_split import validate_registration


SAMPLING_SCHEMA_VERSION = "codenet-python800-stage-a-program-sampling-v1"
PROTOCOL_SCHEMA_VERSION = "code2hyp-stage-a-program-sampling-protocol-v1"


def _hmac_fields(*, key: bytes, fields: Sequence[str]) -> bytes:
    if len(key) != 64:
        raise ValueError("the NIST Beacon output must decode to exactly 64 bytes")
    if not fields or any(not field or "\x00" in field for field in fields):
        raise ValueError("HMAC fields must be non-empty and cannot contain NUL")
    return hmac.new(key, "\x00".join(fields).encode("utf-8"), hashlib.sha256).digest()


def representative_program_digest(
    *,
    beacon_key: bytes,
    domain: str,
    dataset_revision: str,
    cluster_id: str,
    user_id_sha256: str,
    source_relpath: str,
) -> bytes:
    """Order one user's programs without using content, time or submission order."""

    return _hmac_fields(
        key=beacon_key,
        fields=(domain, dataset_revision, cluster_id, user_id_sha256, source_relpath),
    )


def user_order_digest(
    *,
    beacon_key: bytes,
    domain: str,
    dataset_revision: str,
    cluster_id: str,
    user_id_sha256: str,
) -> bytes:
    """Order user representatives independently from program selection."""

    return _hmac_fields(
        key=beacon_key,
        fields=(domain, dataset_revision, cluster_id, user_id_sha256),
    )


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSONL at {path}:{line_number}") from error


def validate_sampling_protocol(
    *,
    protocol: Mapping[str, Any],
    design: Mapping[str, Any],
    design_bytes: bytes,
    registration: Mapping[str, Any],
    registration_bytes: bytes,
    split_manifest: Mapping[str, Any],
    split_manifest_bytes: bytes,
    d5_manifest_bytes: bytes,
    d5_index_bytes_sha256: str,
) -> bytes:
    """Fail closed unless every frozen sampling input matches its public hash."""

    if protocol.get("schema_version") != PROTOCOL_SCHEMA_VERSION:
        raise ValueError("unsupported program-sampling protocol schema")
    if protocol.get("status") != "frozen_before_program_sampling_or_validation_metrics":
        raise ValueError("program-sampling protocol was not frozen before execution")
    if protocol.get("role") != "post_registration_execution_clarification":
        raise ValueError("program-sampling protocol role is not an execution clarification")

    pinned = {
        "design": str(protocol["registered_design"]["sha256"]),
        "registration": str(protocol["registration_record"]["sha256"]),
        "split_manifest": str(protocol["cluster_split"]["manifest_sha256"]),
        "d5_manifest": str(protocol["metadata_input"]["manifest_sha256"]),
        "d5_index": str(protocol["metadata_input"]["index_sha256"]),
    }
    actual = {
        "design": stable_sha256(design_bytes),
        "registration": stable_sha256(registration_bytes),
        "split_manifest": stable_sha256(split_manifest_bytes),
        "d5_manifest": stable_sha256(d5_manifest_bytes),
        "d5_index": d5_index_bytes_sha256,
    }
    mismatches = {name: (actual[name], expected) for name, expected in pinned.items() if actual[name] != expected}
    if mismatches:
        raise ValueError(f"sampling input hash mismatch: {mismatches}")

    beacon_key = validate_registration(
        design=design,
        registration=registration,
        design_bytes=design_bytes,
    )
    if str(protocol["registered_design"]["doi"]) != str(registration["registration"]["doi"]):
        raise ValueError("sampling protocol DOI differs from the registration")
    if split_manifest.get("schema_version") != "codenet-python800-beacon-split-v1":
        raise ValueError("unsupported cluster-split manifest schema")
    split_protocol = split_manifest.get("protocol", {})
    if split_protocol.get("test_relevance_labels_opened") is not False:
        raise ValueError("test relevance labels were opened before program sampling")
    if split_protocol.get("retrieval_metrics_computed") is not False:
        raise ValueError("retrieval metrics were computed before program sampling")
    assignment_sha = str(split_manifest["summary"]["assignment_sha256"])
    if assignment_sha != str(protocol["cluster_split"]["assignment_sha256"]):
        raise ValueError("cluster assignment hash differs from the frozen sampling protocol")

    sampling = design["sampling"]
    selection = protocol["selection"]
    expected_sizes = {
        "train": int(sampling["train_programs_per_cluster"]),
        "validation_queries": int(sampling["validation_queries_per_cluster"]),
        "validation_gallery": int(sampling["validation_gallery_per_cluster"]),
        "test_queries": int(sampling["test_queries_per_cluster"]),
        "test_gallery": int(sampling["test_gallery_per_cluster"]),
    }
    protocol_sizes = {
        "train": int(selection["train"]["programs_per_cluster"]),
        "validation_queries": int(selection["validation"]["queries_per_cluster"]),
        "validation_gallery": int(selection["validation"]["gallery_per_cluster"]),
        "test_queries": int(selection["test"]["queries_per_cluster"]),
        "test_gallery": int(selection["test"]["gallery_per_cluster"]),
    }
    if expected_sizes != protocol_sizes:
        raise ValueError("program sample sizes differ from the registered design")
    if selection["test"].get("materialize_before_unseal") is not False:
        raise ValueError("the sampling protocol must keep test programs sealed")
    public_outputs = protocol["public_outputs_before_test_unseal"]
    forbidden_public = (
        "test_program_ids",
        "test_relevance_labels",
        "test_metrics",
        "user_hashes_in_selected_program_artifact",
    )
    if any(public_outputs.get(name) is not False for name in forbidden_public):
        raise ValueError("the sampling protocol exposes a forbidden pre-unseal output")
    state = protocol["state_at_freeze"]
    if state != {
        "cluster_split_generated": True,
        "program_sampling_generated": False,
        "validation_retrieval_metrics_computed": False,
        "test_program_ids_materialized": False,
        "test_relevance_labels_opened": False,
        "test_retrieval_metrics_computed": False,
    }:
        raise ValueError("sampling protocol freeze state is not fail-closed")
    randomness = protocol["randomness"]
    if randomness.get("digest") != "HMAC-SHA256" or randomness.get("field_separator") != "NUL":
        raise ValueError("unexpected sampling randomization convention")
    if randomness["program_domain"] == randomness["user_domain"]:
        raise ValueError("program and user HMAC domains must differ")
    return beacon_key


def select_non_test_programs(
    *,
    metadata_rows: Iterable[Mapping[str, Any]],
    assignments: Sequence[Mapping[str, Any]],
    beacon_key: bytes,
    dataset_revision: str,
    program_domain: str,
    user_domain: str,
    train_programs_per_cluster: int,
    validation_queries_per_cluster: int,
    validation_gallery_per_cluster: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Select train/validation programs while discarding test rows immediately."""

    if min(train_programs_per_cluster, validation_queries_per_cluster, validation_gallery_per_cluster) <= 0:
        raise ValueError("all program sample sizes must be positive")
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
    if not split_by_cluster:
        raise ValueError("cluster assignments cannot be empty")

    grouped: dict[str, dict[str, list[dict[str, Any]]]] = {
        cluster_id: defaultdict(list)
        for cluster_id, split in split_by_cluster.items()
        if split != "test"
    }
    observed_non_test_sources: set[str] = set()
    for row in metadata_rows:
        cluster_id = str(row.get("problem_cluster_id", ""))
        if cluster_id not in split_by_cluster:
            raise ValueError(f"metadata row references an unassigned cluster: {cluster_id}")
        if split_by_cluster[cluster_id] == "test":
            continue
        user_id = str(row.get("user_id_sha256", ""))
        source_relpath = str(row.get("source_relpath", ""))
        problem_id = str(row.get("problem_id", ""))
        submission_id = str(row.get("submission_id", ""))
        if not all((user_id, source_relpath, problem_id, submission_id)):
            raise ValueError("non-test metadata rows must contain user, source, problem and submission IDs")
        if str(row.get("status", "")) != "Accepted":
            raise ValueError("only accepted CodeNet programs may be sampled")
        if source_relpath in observed_non_test_sources:
            raise ValueError(f"duplicate source_relpath in metadata index: {source_relpath}")
        observed_non_test_sources.add(source_relpath)
        grouped[cluster_id][user_id].append(dict(row))

    train_rows: list[dict[str, Any]] = []
    validation_rows: list[dict[str, Any]] = []
    selected_users_by_split: dict[str, set[str]] = {"train": set(), "validation": set()}
    available_users_by_split: dict[str, list[int]] = {"train": [], "validation": []}
    ordered_assignments = sorted(assignments, key=lambda row: int(row["order_index"]))
    for assignment in ordered_assignments:
        split = str(assignment["split"])
        if split == "test":
            continue
        cluster_id = str(assignment["cluster_id"])
        users = grouped.get(cluster_id, {})
        representatives: list[tuple[bytes, str, str, dict[str, Any]]] = []
        for user_id, candidates in users.items():
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
        required = (
            train_programs_per_cluster
            if split == "train"
            else validation_queries_per_cluster + validation_gallery_per_cluster
        )
        if len(representatives) < required:
            raise ValueError(
                f"cluster {cluster_id} has {len(representatives)} distinct users; {required} required"
            )
        available_users_by_split[split].append(len(representatives))
        chosen = representatives[:required]
        if split == "train":
            roles = [("train", index) for index in range(train_programs_per_cluster)]
        else:
            roles = [
                *(('query', index) for index in range(validation_queries_per_cluster)),
                *(('gallery', index) for index in range(validation_gallery_per_cluster)),
            ]
        for sample_index, ((_, user_id, _, row), (role, role_index)) in enumerate(
            zip(chosen, roles, strict=True)
        ):
            selected_users_by_split[split].add(user_id)
            public_row = {
                "cluster_id": cluster_id,
                "cluster_split_index": split_index_by_cluster[cluster_id],
                "problem_id": str(row["problem_id"]),
                "role": role,
                "role_index": role_index,
                "sample_index": sample_index,
                "source_relpath": str(row["source_relpath"]),
                "split": split,
                "submission_id": str(row["submission_id"]),
            }
            if split == "train":
                train_rows.append(public_row)
            else:
                validation_rows.append(public_row)

    overlap = selected_users_by_split["train"] & selected_users_by_split["validation"]
    summary = {
        "train_clusters": sum(str(row["split"]) == "train" for row in assignments),
        "validation_clusters": sum(str(row["split"]) == "validation" for row in assignments),
        "test_clusters_sealed": sum(str(row["split"]) == "test" for row in assignments),
        "train_programs": len(train_rows),
        "validation_queries": sum(row["role"] == "query" for row in validation_rows),
        "validation_gallery": sum(row["role"] == "gallery" for row in validation_rows),
        "test_programs_materialized": 0,
        "minimum_available_users_train": min(available_users_by_split["train"], default=0),
        "median_available_users_train": statistics.median(available_users_by_split["train"]),
        "minimum_available_users_validation": min(available_users_by_split["validation"], default=0),
        "median_available_users_validation": statistics.median(available_users_by_split["validation"]),
        "distinct_selected_users_train": len(selected_users_by_split["train"]),
        "distinct_selected_users_validation": len(selected_users_by_split["validation"]),
        "selected_user_overlap_train_validation": len(overlap),
    }
    return train_rows, validation_rows, summary


def _write_once_or_verify(path: Path, content: bytes) -> None:
    if path.exists():
        if path.read_bytes() != content:
            raise FileExistsError(f"refusing to overwrite a different program-sampling artifact: {path}")
        return
    path.write_bytes(content)


def build_program_sampling_artifacts(
    *,
    project_root: Path,
    protocol_path: Path,
    design_path: Path,
    registration_path: Path,
    split_manifest_path: Path,
    assignments_path: Path,
    d5_manifest_path: Path,
    d5_index_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Materialize registered train/validation program IDs without opening test IDs."""

    protocol_bytes = protocol_path.read_bytes()
    protocol = json.loads(protocol_bytes)
    design_bytes = design_path.read_bytes()
    design = json.loads(design_bytes)
    registration_bytes = registration_path.read_bytes()
    registration = json.loads(registration_bytes)
    split_manifest_bytes = split_manifest_path.read_bytes()
    split_manifest = json.loads(split_manifest_bytes)
    d5_manifest_bytes = d5_manifest_path.read_bytes()
    d5_manifest = json.loads(d5_manifest_bytes)
    d5_index_sha = stable_sha256(d5_index_path.read_bytes())
    beacon_key = validate_sampling_protocol(
        protocol=protocol,
        design=design,
        design_bytes=design_bytes,
        registration=registration,
        registration_bytes=registration_bytes,
        split_manifest=split_manifest,
        split_manifest_bytes=split_manifest_bytes,
        d5_manifest_bytes=d5_manifest_bytes,
        d5_index_bytes_sha256=d5_index_sha,
    )
    manifest_index_hash = next(
        (str(item["sha256"]) for item in d5_manifest["artifacts"] if item["path"] == "d5_metadata_index.jsonl"),
        None,
    )
    if manifest_index_hash != d5_index_sha:
        raise ValueError("D5 manifest does not pin the supplied metadata index")

    assignments_bytes = assignments_path.read_bytes()
    if stable_sha256(assignments_bytes) != str(split_manifest["summary"]["assignment_sha256"]):
        raise ValueError("cluster assignment file differs from its split manifest")
    assignments = _load_jsonl(assignments_path)
    randomness = protocol["randomness"]
    sampling = design["sampling"]
    train_rows, validation_rows, summary = select_non_test_programs(
        metadata_rows=iter_jsonl(d5_index_path),
        assignments=assignments,
        beacon_key=beacon_key,
        dataset_revision=str(design["dataset"]["revision"]),
        program_domain=str(randomness["program_domain"]),
        user_domain=str(randomness["user_domain"]),
        train_programs_per_cluster=int(sampling["train_programs_per_cluster"]),
        validation_queries_per_cluster=int(sampling["validation_queries_per_cluster"]),
        validation_gallery_per_cluster=int(sampling["validation_gallery_per_cluster"]),
    )
    expected_train = summary["train_clusters"] * int(sampling["train_programs_per_cluster"])
    expected_queries = summary["validation_clusters"] * int(sampling["validation_queries_per_cluster"])
    expected_gallery = summary["validation_clusters"] * int(sampling["validation_gallery_per_cluster"])
    if summary["train_programs"] != expected_train:
        raise ValueError("train program count differs from the registered design")
    if summary["validation_queries"] != expected_queries or summary["validation_gallery"] != expected_gallery:
        raise ValueError("validation program counts differ from the registered design")
    if any(row["split"] == "test" for row in train_rows + validation_rows):
        raise ValueError("test program identifiers cannot appear in pre-unseal outputs")
    if any("user" in key.casefold() for row in train_rows + validation_rows for key in row):
        raise ValueError("public program-sampling rows cannot contain user hashes")

    output_dir.mkdir(parents=True, exist_ok=True)
    payloads = {
        "train_programs.jsonl": jsonl_bytes(train_rows),
        "validation_programs.jsonl": jsonl_bytes(validation_rows),
        "program_sampling_summary.json": canonical_json_bytes(summary),
    }
    artifacts: list[dict[str, Any]] = []
    for filename, content in payloads.items():
        _write_once_or_verify(output_dir / filename, content)
        artifacts.append({"path": filename, "bytes": len(content), "sha256": stable_sha256(content)})

    manifest = {
        "schema_version": SAMPLING_SCHEMA_VERSION,
        "experiment_role": "registered_train_validation_program_sampling_with_test_ids_sealed",
        "input": {
            "sampling_protocol": {
                "path": portable_manifest_path(protocol_path, project_root=project_root),
                "sha256": stable_sha256(protocol_bytes),
            },
            "registered_design_sha256": stable_sha256(design_bytes),
            "registration_sha256": stable_sha256(registration_bytes),
            "split_manifest_sha256": stable_sha256(split_manifest_bytes),
            "cluster_assignments_sha256": stable_sha256(assignments_bytes),
            "d5_manifest_sha256": stable_sha256(d5_manifest_bytes),
            "d5_metadata_index_sha256": d5_index_sha,
        },
        "protocol": {
            "representative_per_user": protocol["selection"]["representative_per_user"],
            "user_representative_order": protocol["selection"]["user_representative_order"],
            "train_programs_per_cluster": int(sampling["train_programs_per_cluster"]),
            "validation_queries_per_cluster": int(sampling["validation_queries_per_cluster"]),
            "validation_gallery_per_cluster": int(sampling["validation_gallery_per_cluster"]),
            "program_sampling_generated_for": ["train", "validation"],
            "test_program_sampling_generated": False,
            "test_relevance_labels_opened": False,
            "validation_retrieval_metrics_computed": False,
            "test_retrieval_metrics_computed": False,
            "user_hashes_published": False,
        },
        "summary": summary,
        "artifacts": sorted(artifacts, key=lambda item: item["path"]),
    }
    manifest_bytes = canonical_json_bytes(manifest)
    _write_once_or_verify(output_dir / "program_sampling_manifest.json", manifest_bytes)
    _write_once_or_verify(
        output_dir / "program_sampling_manifest.sha256",
        f"{stable_sha256(manifest_bytes)}  program_sampling_manifest.json\n".encode("ascii"),
    )
    return manifest
