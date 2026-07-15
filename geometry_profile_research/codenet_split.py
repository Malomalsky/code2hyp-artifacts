from __future__ import annotations

import hashlib
import hmac
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from geometry_profile_research.codenet_eligibility import (
    canonical_json_bytes,
    jsonl_bytes,
    portable_manifest_path,
    stable_sha256,
)


SPLIT_SCHEMA_VERSION = "codenet-python800-beacon-split-v1"
SPLIT_NAMES = ("train", "validation", "test")


def hamilton_quotas(total: int, weights: Sequence[int]) -> tuple[int, ...]:
    """Apportion integer quotas with a stable left-to-right tie-break."""

    if total <= 0 or not weights or any(weight <= 0 for weight in weights):
        raise ValueError("total and all weights must be positive")
    denominator = sum(weights)
    exact = [total * weight / denominator for weight in weights]
    quotas = [math.floor(value) for value in exact]
    remainder = total - sum(quotas)
    order = sorted(
        range(len(weights)),
        key=lambda index: (-(exact[index] - quotas[index]), index),
    )
    for index in order[:remainder]:
        quotas[index] += 1
    return tuple(quotas)


def hmac_cluster_digest(*, beacon_key: bytes, dataset_revision: str, cluster_id: str) -> bytes:
    """Return the preregistered HMAC ordering key for one problem cluster."""

    if len(beacon_key) != 64:
        raise ValueError("the NIST Beacon output must decode to exactly 64 bytes")
    if not dataset_revision or "\x00" in dataset_revision:
        raise ValueError("dataset_revision must be non-empty and cannot contain NUL")
    if not cluster_id or "\x00" in cluster_id:
        raise ValueError("cluster_id must be non-empty and cannot contain NUL")
    message = dataset_revision.encode("utf-8") + b"\x00" + cluster_id.encode("utf-8")
    return hmac.new(beacon_key, message, hashlib.sha256).digest()


def validate_registration(*, design: Mapping[str, Any], registration: Mapping[str, Any], design_bytes: bytes) -> bytes:
    """Validate the post-registration provenance and return the Beacon key."""

    if registration.get("schema_version") != "code2hyp-codenet-stage-a-registration-v1":
        raise ValueError("unsupported Stage A registration schema")
    expected_design_sha = str(registration["design"]["sha256"])
    actual_design_sha = stable_sha256(design_bytes)
    if actual_design_sha != expected_design_sha:
        raise ValueError(
            f"registered design SHA-256 mismatch: actual={actual_design_sha}, expected={expected_design_sha}"
        )
    dataset_revision = str(design["dataset"]["revision"])
    if dataset_revision != str(registration["design"]["dataset_revision"]):
        raise ValueError("dataset revision differs from the registered value")
    doi = str(registration["registration"].get("doi", ""))
    if not doi.startswith("10.5281/zenodo."):
        raise ValueError("a published Zenodo registration DOI is required")

    created = datetime.fromisoformat(str(registration["registration"]["created_utc"]))
    pulse_timestamp = datetime.fromisoformat(
        str(registration["nist_randomness_beacon"]["timestamp_utc"]).replace("Z", "+00:00")
    )
    if created.tzinfo is None or pulse_timestamp.tzinfo is None:
        raise ValueError("registration and Beacon timestamps must be timezone-aware")
    if pulse_timestamp <= created:
        raise ValueError("the Beacon pulse must be strictly later than the registration")
    query_milliseconds = int(created.timestamp() * 1000)
    if query_milliseconds != int(registration["nist_randomness_beacon"]["query_timestamp_unix_milliseconds"]):
        raise ValueError("Beacon query timestamp does not match the Zenodo creation timestamp")
    if int(registration["nist_randomness_beacon"]["status_code"]) != 0:
        raise ValueError("the selected NIST Beacon pulse is not valid")
    if int(registration["nist_randomness_beacon"]["period_milliseconds"]) != 60000:
        raise ValueError("unexpected NIST Beacon pulse period")
    try:
        beacon_key = bytes.fromhex(str(registration["nist_randomness_beacon"]["output_value_hex"]))
    except ValueError as error:
        raise ValueError("invalid hexadecimal NIST Beacon output") from error
    if len(beacon_key) != 64:
        raise ValueError("the NIST Beacon output must decode to exactly 64 bytes")

    weights = tuple(int(value) for value in design["split"]["weights_train_validation_test"])
    eligible_clusters = int(registration["design"]["eligible_problem_clusters"])
    derived_quotas = hamilton_quotas(eligible_clusters, weights)
    registered_quotas = tuple(int(value) for value in registration["design"]["quotas_train_validation_test"])
    if derived_quotas != registered_quotas:
        raise ValueError(f"registered split quotas {registered_quotas} differ from Hamilton quotas {derived_quotas}")
    if registration["state_at_registration"] != {
        "split_generated": False,
        "test_labels_opened": False,
        "codenet_retrieval_metrics_computed": False,
    }:
        raise ValueError("registration state must confirm that no split, labels, or metrics were opened")
    return beacon_key


def eligible_cluster_ids(rows: Iterable[Mapping[str, Any]], *, expected_count: int) -> list[str]:
    """Select unique eligible cluster identifiers and fail on malformed input."""

    selected: list[str] = []
    observed: set[str] = set()
    for row in rows:
        cluster_id = str(row.get("cluster_id", ""))
        if not cluster_id:
            raise ValueError("every cluster row must contain a non-empty cluster_id")
        if cluster_id in observed:
            raise ValueError(f"duplicate cluster_id: {cluster_id}")
        observed.add(cluster_id)
        if row.get("eligible_minimum_64") is True:
            selected.append(cluster_id)
    if len(selected) != expected_count:
        raise ValueError(f"eligible cluster count mismatch: actual={len(selected)}, expected={expected_count}")
    return selected


def assign_cluster_ids(
    *,
    cluster_ids: Sequence[str],
    beacon_key: bytes,
    dataset_revision: str,
    quotas: Sequence[int],
) -> list[dict[str, Any]]:
    """Order clusters by HMAC and assign consecutive registered quotas."""

    if len(cluster_ids) != len(set(cluster_ids)):
        raise ValueError("cluster_ids must be unique")
    if len(quotas) != len(SPLIT_NAMES) or any(quota <= 0 for quota in quotas):
        raise ValueError("positive train, validation, and test quotas are required")
    if sum(quotas) != len(cluster_ids):
        raise ValueError("split quotas must sum to the number of clusters")

    ordered = sorted(
        (
            hmac_cluster_digest(
                beacon_key=beacon_key,
                dataset_revision=dataset_revision,
                cluster_id=cluster_id,
            ),
            cluster_id,
        )
        for cluster_id in cluster_ids
    )
    rows: list[dict[str, Any]] = []
    offset = 0
    for split_name, quota in zip(SPLIT_NAMES, quotas, strict=True):
        for split_index, (digest, cluster_id) in enumerate(ordered[offset : offset + quota]):
            rows.append(
                {
                    "order_index": offset + split_index,
                    "split": split_name,
                    "split_index": split_index,
                    "cluster_id": cluster_id,
                    "hmac_sha256": digest.hex(),
                }
            )
        offset += quota
    return rows


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _write_once_or_verify(path: Path, content: bytes) -> None:
    if path.exists():
        if path.read_bytes() != content:
            raise FileExistsError(f"refusing to overwrite a different registered split artifact: {path}")
        return
    path.write_bytes(content)


def build_split_artifacts(
    *,
    project_root: Path,
    design_path: Path,
    registration_path: Path,
    clusters_path: Path,
    statement_d4_manifest_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Build an auditable cluster-level split without sampling programs or evaluating models."""

    design_bytes = design_path.read_bytes()
    design = json.loads(design_bytes)
    registration_bytes = registration_path.read_bytes()
    registration = json.loads(registration_bytes)
    beacon_key = validate_registration(design=design, registration=registration, design_bytes=design_bytes)

    d4_manifest_bytes = statement_d4_manifest_path.read_bytes()
    d4_manifest = json.loads(d4_manifest_bytes)
    if d4_manifest["protocol"]["retrieval_metrics_opened"] is not False:
        raise ValueError("D4 input must predate all retrieval metrics")
    if d4_manifest["protocol"]["split_status"] != "not_generated":
        raise ValueError("D4 input must predate the dataset split")

    expected_count = int(registration["design"]["eligible_problem_clusters"])
    cluster_rows = _load_jsonl(clusters_path)
    cluster_ids = eligible_cluster_ids(cluster_rows, expected_count=expected_count)
    weights = tuple(int(value) for value in design["split"]["weights_train_validation_test"])
    quotas = hamilton_quotas(len(cluster_ids), weights)
    registered_quotas = tuple(int(value) for value in registration["design"]["quotas_train_validation_test"])
    if quotas != registered_quotas:
        raise ValueError("derived quotas differ from the registration")

    assignments = assign_cluster_ids(
        cluster_ids=cluster_ids,
        beacon_key=beacon_key,
        dataset_revision=str(design["dataset"]["revision"]),
        quotas=quotas,
    )
    assignment_content = jsonl_bytes(assignments)
    summary = {
        "cluster_count": len(assignments),
        "train_clusters": quotas[0],
        "validation_clusters": quotas[1],
        "test_clusters": quotas[2],
        "assignment_sha256": stable_sha256(assignment_content),
        "first_ordered_cluster": assignments[0]["cluster_id"],
        "last_ordered_cluster": assignments[-1]["cluster_id"],
    }
    summary_content = canonical_json_bytes(summary)

    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_payloads = {
        "cluster_assignments.jsonl": assignment_content,
        "split_summary.json": summary_content,
    }
    artifacts = []
    for filename, content in artifact_payloads.items():
        _write_once_or_verify(output_dir / filename, content)
        artifacts.append({"path": filename, "bytes": len(content), "sha256": stable_sha256(content)})

    manifest = {
        "schema_version": SPLIT_SCHEMA_VERSION,
        "experiment_role": "registered_problem_cluster_split_without_program_sampling_or_retrieval_metrics",
        "input": {
            "design": {
                "path": portable_manifest_path(design_path, project_root=project_root),
                "sha256": stable_sha256(design_bytes),
            },
            "registration": {
                "path": portable_manifest_path(registration_path, project_root=project_root),
                "sha256": stable_sha256(registration_bytes),
                "doi": registration["registration"]["doi"],
            },
            "eligible_clusters": {
                "path": portable_manifest_path(clusters_path, project_root=project_root),
                "sha256": stable_sha256(clusters_path.read_bytes()),
            },
            "statement_d4_manifest_sha256": stable_sha256(d4_manifest_bytes),
        },
        "protocol": {
            "assignment_unit": design["split"]["assignment_unit"],
            "quota_rule": design["split"]["integer_quota_rule"],
            "weights_train_validation_test": list(weights),
            "quotas_train_validation_test": list(quotas),
            "ordering": design["split"]["ordering"],
            "beacon_uri": registration["nist_randomness_beacon"]["uri"],
            "beacon_timestamp_utc": registration["nist_randomness_beacon"]["timestamp_utc"],
            "program_sampling_generated": False,
            "test_relevance_labels_opened": False,
            "retrieval_metrics_computed": False,
        },
        "summary": summary,
        "artifacts": sorted(artifacts, key=lambda item: item["path"]),
    }
    manifest_content = canonical_json_bytes(manifest)
    _write_once_or_verify(output_dir / "split_manifest.json", manifest_content)
    manifest_sha = stable_sha256(manifest_content)
    _write_once_or_verify(
        output_dir / "split_manifest.sha256",
        f"{manifest_sha}  split_manifest.json\n".encode("ascii"),
    )
    return manifest
