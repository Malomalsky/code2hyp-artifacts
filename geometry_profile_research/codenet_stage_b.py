from __future__ import annotations

import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

from geometry_profile_research.codenet_ast_audit import materialize_and_audit_stage_b_java_rows
from geometry_profile_research.codenet_eligibility import (
    canonical_json_bytes,
    jsonl_bytes,
    portable_manifest_path,
    stable_sha256,
)
from geometry_profile_research.codenet_stage_a_runner import (
    all_role_curvature_cell_id,
    curvature_cell_id,
    select_active_curvature,
)
from geometry_profile_research.codenet_stage_a import select_calibration_pairs
from geometry_profile_research.codenet_stage_a_test import (
    open_or_resume_test_transaction,
    select_test_programs,
)
from geometry_profile_research.codenet_sampling import iter_jsonl, select_non_test_programs
from geometry_profile_research.codenet_split import hmac_cluster_digest


STAGE_B_SELECTION_SCHEMA = "code2hyp-codenet-java-stage-b-validation-selection-v1"
STAGE_B_REGISTRATION_SCHEMA = "code2hyp-codenet-java-stage-b-registration-v1"
STAGE_B_SPLIT_SCHEMA = "codenet-java-stage-b-beacon-split-v1"
STAGE_B_TEST_MATERIALIZATION_SCHEMA = "code2hyp-codenet-java-stage-b-test-materialization-v1"


def validate_stage_b_registration(
    *,
    design: Mapping[str, Any],
    registration: Mapping[str, Any],
    design_bytes: bytes,
) -> bytes:
    """Validate the public design provenance and return the NIST Beacon key."""

    if registration.get("schema_version") != STAGE_B_REGISTRATION_SCHEMA:
        raise ValueError("unsupported Stage B registration schema")
    if stable_sha256(design_bytes) != str(registration["design"]["sha256"]):
        raise ValueError("registered Stage B design SHA-256 mismatch")
    if str(design["dataset"]["revision"]) != str(registration["design"]["dataset_revision"]):
        raise ValueError("Stage B dataset revision differs from the registration")
    doi = str(registration["registration"].get("doi", ""))
    if not doi.startswith("10.5281/zenodo."):
        raise ValueError("a published Zenodo registration DOI is required")

    created = datetime.fromisoformat(str(registration["registration"]["created_utc"]).replace("Z", "+00:00"))
    beacon = registration["nist_randomness_beacon"]
    pulse = datetime.fromisoformat(str(beacon["timestamp_utc"]).replace("Z", "+00:00"))
    if created.tzinfo is None or pulse.tzinfo is None or pulse <= created:
        raise ValueError("the NIST Beacon pulse must be strictly later than the public registration")
    if int(created.timestamp() * 1000) != int(beacon["query_timestamp_unix_milliseconds"]):
        raise ValueError("Beacon query timestamp does not match the registration timestamp")
    if int(beacon["status_code"]) != 0 or int(beacon["period_milliseconds"]) != 60_000:
        raise ValueError("invalid or unsupported NIST Beacon pulse")
    try:
        beacon_key = bytes.fromhex(str(beacon["output_value_hex"]))
    except ValueError as error:
        raise ValueError("invalid hexadecimal NIST Beacon output") from error
    if len(beacon_key) != 64:
        raise ValueError("the NIST Beacon output must decode to exactly 64 bytes")

    counts = design["eligibility"]["primary_counts"]
    quotas = design["eligibility"]["primary_role_upper_bound"]
    expected_design = {
        "eligible_evaluation_clusters": int(counts["eligible_evaluation_clusters"]),
        "eligible_train_clusters": int(counts["eligible_train_clusters"]),
        "quotas_train_validation_test": [
            int(quotas["train_clusters"]),
            int(quotas["validation_clusters"]),
            int(quotas["test_clusters"]),
        ],
    }
    observed_design = {key: registration["design"][key] for key in expected_design}
    if observed_design != expected_design:
        raise ValueError("registered Stage B counts or role-constrained quotas differ from the design")
    expected_state = {
        "split_generated": False,
        "java_validation_metrics_opened": False,
        "java_test_program_ids_materialized": False,
        "java_test_relevance_labels_opened": False,
        "java_test_retrieval_metrics_computed": False,
    }
    if registration.get("state_at_registration") != expected_state:
        raise ValueError("Stage B registration state must keep all downstream data and metrics unopened")
    return beacon_key


def build_stage_b_split_artifacts(
    *,
    project_root: Path,
    design_path: Path,
    registration_path: Path,
    clusters_path: Path,
    d4_manifest_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Build the registered role-constrained split without sampling programs."""

    design_bytes = design_path.read_bytes()
    design = json.loads(design_bytes)
    registration_bytes = registration_path.read_bytes()
    registration = json.loads(registration_bytes)
    beacon_key = validate_stage_b_registration(
        design=design,
        registration=registration,
        design_bytes=design_bytes,
    )
    d4_bytes = d4_manifest_path.read_bytes()
    d4 = json.loads(d4_bytes)
    expected_d4_sha = str(design["eligibility"]["artifacts"]["d4_primary_manifest_sha256"])
    if stable_sha256(d4_bytes) != expected_d4_sha:
        raise ValueError("Stage B D4 manifest differs from the registered design")
    if d4["protocol"].get("split_status") != "not_generated" or d4["protocol"].get("retrieval_metrics_opened") is not False:
        raise ValueError("Stage B D4 input must predate the split and all retrieval metrics")
    expected_cluster_sha = next(
        str(item["sha256"])
        for item in d4["artifacts"]
        if item["path"] == "post_d4_problem_clusters.jsonl"
    )
    cluster_bytes = clusters_path.read_bytes()
    if stable_sha256(cluster_bytes) != expected_cluster_sha:
        raise ValueError("Stage B cluster rows differ from the D4 manifest")
    cluster_rows = [json.loads(line) for line in cluster_bytes.splitlines() if line]

    counts = design["eligibility"]["primary_counts"]
    quotas = design["eligibility"]["primary_role_upper_bound"]
    assignments = assign_stage_b_cluster_ids(
        cluster_rows,
        beacon_key=beacon_key,
        dataset_revision=str(design["dataset"]["revision"]),
        train_clusters=int(quotas["train_clusters"]),
        validation_clusters=int(quotas["validation_clusters"]),
        test_clusters=int(quotas["test_clusters"]),
    )
    split_counts = Counter(str(row["split"]) for row in assignments)
    expected_split_counts = {
        "train": int(quotas["train_clusters"]),
        "validation": int(quotas["validation_clusters"]),
        "test": int(quotas["test_clusters"]),
    }
    reserve = int(quotas["reserve_evaluation_clusters"])
    if reserve:
        expected_split_counts["reserve"] = reserve
    if dict(split_counts) != expected_split_counts:
        raise ValueError(f"Stage B split counts differ from the design: {dict(split_counts)}")
    if len(assignments) != int(counts["eligible_evaluation_clusters"]):
        raise ValueError("Stage B assignment count differs from the registered evaluation frame")

    assignment_bytes = jsonl_bytes(assignments)
    summary = {
        "cluster_count": len(assignments),
        "eligible_train_clusters": int(counts["eligible_train_clusters"]),
        "train_clusters": split_counts["train"],
        "validation_clusters": split_counts["validation"],
        "test_clusters": split_counts["test"],
        "reserve_clusters": split_counts["reserve"],
        "assignment_sha256": stable_sha256(assignment_bytes),
    }
    payloads = {
        "cluster_assignments.jsonl": assignment_bytes,
        "split_summary.json": canonical_json_bytes(summary),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts = []
    for filename, content in payloads.items():
        _write_once_or_verify(output_dir / filename, content)
        artifacts.append({"path": filename, "bytes": len(content), "sha256": stable_sha256(content)})
    manifest = {
        "schema_version": STAGE_B_SPLIT_SCHEMA,
        "experiment_role": "registered_role_constrained_cluster_split_without_program_sampling_or_metrics",
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
                "sha256": stable_sha256(cluster_bytes),
            },
            "d4_manifest_sha256": stable_sha256(d4_bytes),
        },
        "protocol": {
            "assignment_unit": design["split"]["assignment_unit"],
            "ordering": design["split"]["ordering"],
            "role_constraint": design["split"]["role_constraint"],
            "beacon_uri": registration["nist_randomness_beacon"]["uri"],
            "program_sampling_generated": False,
            "java_validation_metrics_opened": False,
            "java_test_program_ids_materialized": False,
            "java_test_relevance_labels_opened": False,
            "java_test_retrieval_metrics_computed": False,
        },
        "summary": summary,
        "artifacts": sorted(artifacts, key=lambda item: item["path"]),
    }
    manifest_bytes = canonical_json_bytes(manifest)
    _write_once_or_verify(output_dir / "split_manifest.json", manifest_bytes)
    _write_once_or_verify(
        output_dir / "split_manifest.sha256",
        f"{stable_sha256(manifest_bytes)}  split_manifest.json\n".encode("ascii"),
    )
    return manifest


def build_stage_b_program_sampling_artifacts(
    *,
    project_root: Path,
    design_path: Path,
    registration_path: Path,
    split_manifest_path: Path,
    assignments_path: Path,
    d5_manifest_path: Path,
    d5_index_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Select registered train/validation programs while keeping test IDs sealed."""

    design_bytes = design_path.read_bytes()
    design = json.loads(design_bytes)
    registration_bytes = registration_path.read_bytes()
    registration = json.loads(registration_bytes)
    beacon_key = validate_stage_b_registration(
        design=design,
        registration=registration,
        design_bytes=design_bytes,
    )
    split_bytes = split_manifest_path.read_bytes()
    split_manifest = json.loads(split_bytes)
    if split_manifest.get("schema_version") != STAGE_B_SPLIT_SCHEMA:
        raise ValueError("unsupported Stage B split manifest")
    if split_manifest["input"]["design"]["sha256"] != stable_sha256(design_bytes):
        raise ValueError("Stage B split and design hashes differ")
    if split_manifest["input"]["registration"]["sha256"] != stable_sha256(registration_bytes):
        raise ValueError("Stage B split and registration hashes differ")
    if any(
        split_manifest["protocol"].get(key) is not False
        for key in (
            "program_sampling_generated",
            "java_validation_metrics_opened",
            "java_test_program_ids_materialized",
            "java_test_relevance_labels_opened",
            "java_test_retrieval_metrics_computed",
        )
    ):
        raise ValueError("Stage B split manifest does not preserve the unopened downstream state")

    d5_bytes = d5_manifest_path.read_bytes()
    d5 = json.loads(d5_bytes)
    d5_index_sha = stable_sha256(d5_index_path.read_bytes())
    expected = design["eligibility"]["artifacts"]
    if stable_sha256(d5_bytes) != str(expected["d5_metadata_manifest_sha256"]):
        raise ValueError("Stage B D5 manifest differs from the registered design")
    if d5_index_sha != str(expected["d5_metadata_index_sha256"]):
        raise ValueError("Stage B D5 index differs from the registered design")
    manifest_index_sha = next(
        str(item["sha256"])
        for item in d5["artifacts"]
        if item["path"] == "d5_metadata_index.jsonl"
    )
    if manifest_index_sha != d5_index_sha:
        raise ValueError("Stage B D5 manifest does not pin the supplied metadata index")

    assignment_bytes = assignments_path.read_bytes()
    if stable_sha256(assignment_bytes) != str(split_manifest["summary"]["assignment_sha256"]):
        raise ValueError("Stage B assignments differ from the split manifest")
    assignments = [json.loads(line) for line in assignment_bytes.splitlines() if line]
    sampling = design["sampling"]
    randomness = sampling["randomness"]
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
    quotas = design["eligibility"]["primary_role_upper_bound"]
    expected_counts = {
        "train_programs": int(quotas["train_clusters"]) * int(sampling["train_programs_per_cluster"]),
        "validation_queries": int(quotas["validation_clusters"]) * int(sampling["validation_queries_per_cluster"]),
        "validation_gallery": int(quotas["validation_clusters"]) * int(sampling["validation_gallery_per_cluster"]),
    }
    if any(int(summary[key]) != value for key, value in expected_counts.items()):
        raise ValueError(f"Stage B sampled program counts differ from the design: {summary}")
    public_rows = train_rows + validation_rows
    test_or_reserve = {
        str(row["cluster_id"])
        for row in assignments
        if str(row["split"]) in {"test", "reserve"}
    }
    if {str(row["cluster_id"]) for row in public_rows} & test_or_reserve:
        raise ValueError("Stage B public sampling contains a test or reserve cluster")
    if any("user" in str(key).casefold() for row in public_rows for key in row):
        raise ValueError("Stage B public sampling cannot expose user hashes")

    summary = {
        **summary,
        "reserve_clusters_unused": sum(str(row["split"]) == "reserve" for row in assignments),
        "java_test_program_ids_materialized": False,
        "java_validation_metrics_opened": False,
        "java_test_retrieval_metrics_computed": False,
    }
    payloads = {
        "train_programs.jsonl": jsonl_bytes(train_rows),
        "validation_programs.jsonl": jsonl_bytes(validation_rows),
        "program_sampling_summary.json": canonical_json_bytes(summary),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts = []
    for filename, content in payloads.items():
        _write_once_or_verify(output_dir / filename, content)
        artifacts.append({"path": filename, "bytes": len(content), "sha256": stable_sha256(content)})
    manifest = {
        "schema_version": "codenet-java-stage-b-program-sampling-v1",
        "experiment_role": "registered_train_validation_sampling_with_test_and_reserve_ids_sealed",
        "input": {
            "design_sha256": stable_sha256(design_bytes),
            "registration_sha256": stable_sha256(registration_bytes),
            "split_manifest_sha256": stable_sha256(split_bytes),
            "cluster_assignments_sha256": stable_sha256(assignment_bytes),
            "d5_manifest_sha256": stable_sha256(d5_bytes),
            "d5_metadata_index_sha256": d5_index_sha,
        },
        "selection": {
            "digest": randomness["digest"],
            "field_separator": randomness["field_separator"],
            "program_domain": randomness["program_domain"],
            "user_domain": randomness["user_domain"],
            "one_program_per_user_within_cluster": True,
        },
        "protocol": {
            "java_test_program_ids_materialized": False,
            "java_test_relevance_labels_opened": False,
            "java_validation_metrics_opened": False,
            "java_test_retrieval_metrics_computed": False,
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


def build_stage_b_calibration_pair_artifacts(
    *,
    project_root: Path,
    design_path: Path,
    registration_path: Path,
    sampling_manifest_path: Path,
    train_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Freeze train-only Stage B calibration pairs before validation metrics."""

    design_bytes = design_path.read_bytes()
    design = json.loads(design_bytes)
    registration_bytes = registration_path.read_bytes()
    registration = json.loads(registration_bytes)
    beacon_key = validate_stage_b_registration(
        design=design,
        registration=registration,
        design_bytes=design_bytes,
    )
    sampling_bytes = sampling_manifest_path.read_bytes()
    sampling_manifest = json.loads(sampling_bytes)
    if sampling_manifest.get("schema_version") != "codenet-java-stage-b-program-sampling-v1":
        raise ValueError("unsupported Stage B program-sampling manifest")
    if sampling_manifest["input"]["design_sha256"] != stable_sha256(design_bytes):
        raise ValueError("Stage B sampling and design hashes differ")
    train_bytes = train_path.read_bytes()
    expected_train_sha = next(
        str(item["sha256"])
        for item in sampling_manifest["artifacts"]
        if item["path"] == "train_programs.jsonl"
    )
    if stable_sha256(train_bytes) != expected_train_sha:
        raise ValueError("Stage B training rows differ from the sampling manifest")
    if any(
        sampling_manifest["protocol"].get(key) is not False
        for key in (
            "java_test_program_ids_materialized",
            "java_test_relevance_labels_opened",
            "java_validation_metrics_opened",
            "java_test_retrieval_metrics_computed",
        )
    ):
        raise ValueError("Stage B calibration must precede validation and test opening")

    calibration = design["train_only_calibration"]
    train_rows = [json.loads(line) for line in train_bytes.splitlines() if line]
    pairs = select_calibration_pairs(
        train_rows,
        beacon_key=beacon_key,
        dataset_revision=str(design["dataset"]["revision"]),
        domain=str(design["sampling"]["randomness"]["calibration_pair_domain"]),
        same_cluster_count=int(calibration["same_cluster_pairs"]),
        cross_cluster_count=int(calibration["cross_cluster_pairs"]),
    )
    summary = {
        "pair_count": len(pairs),
        "same_cluster_pair_count": sum(row["pair_type"] == "same_cluster" for row in pairs),
        "cross_cluster_pair_count": sum(row["pair_type"] == "cross_cluster" for row in pairs),
        "unique_program_count": len(
            {
                str(row[key])
                for row in pairs
                for key in ("left_source_relpath", "right_source_relpath")
            }
        ),
        "validation_programs_used": False,
        "java_test_program_ids_materialized": False,
        "java_validation_metrics_opened": False,
        "java_test_retrieval_metrics_computed": False,
    }
    expected_pair_count = int(calibration["same_cluster_pairs"]) + int(calibration["cross_cluster_pairs"])
    if len(pairs) != expected_pair_count:
        raise ValueError("Stage B calibration pair count differs from the design")
    payloads = {
        "calibration_pairs.jsonl": jsonl_bytes(pairs),
        "calibration_pair_summary.json": canonical_json_bytes(summary),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts = []
    for filename, content in payloads.items():
        _write_once_or_verify(output_dir / filename, content)
        artifacts.append({"path": filename, "bytes": len(content), "sha256": stable_sha256(content)})
    manifest = {
        "schema_version": "codenet-java-stage-b-calibration-pairs-v1",
        "experiment_role": "registered_train_only_calibration_before_validation_metrics",
        "input": {
            "design_sha256": stable_sha256(design_bytes),
            "registration_sha256": stable_sha256(registration_bytes),
            "program_sampling_manifest_sha256": stable_sha256(sampling_bytes),
            "train_programs_sha256": stable_sha256(train_bytes),
        },
        "selection": {
            "algorithm": "domain-separated HMAC-SHA256",
            "domain": design["sampling"]["randomness"]["calibration_pair_domain"],
            "replacement": False,
        },
        "summary": summary,
        "artifacts": sorted(artifacts, key=lambda item: item["path"]),
    }
    manifest_bytes = canonical_json_bytes(manifest)
    _write_once_or_verify(output_dir / "calibration_pair_manifest.json", manifest_bytes)
    _write_once_or_verify(
        output_dir / "calibration_pair_manifest.sha256",
        f"{stable_sha256(manifest_bytes)}  calibration_pair_manifest.json\n".encode("ascii"),
    )
    return manifest


def materialize_stage_b_test_programs(
    *,
    design_path: Path,
    registration_path: Path,
    split_manifest_path: Path,
    assignments_path: Path,
    d5_manifest_path: Path,
    d5_index_path: Path,
    selection_path: Path,
    selection_seal_path: Path,
    candidate_manifest_path: Path,
    candidate_archive_path: Path,
    d0_d2_manifest_path: Path,
    d0_d2_inventory_path: Path,
    source_root: Path,
    output_dir: Path,
    implementation: Mapping[str, Any],
    workers: int = 1,
) -> dict[str, Any]:
    """Perform the single registered Stage B test opening and pre-metric Java audit."""

    design_bytes = design_path.read_bytes()
    design = json.loads(design_bytes)
    design_sha = stable_sha256(design_bytes)
    registration_bytes = registration_path.read_bytes()
    registration = json.loads(registration_bytes)
    beacon_key = validate_stage_b_registration(
        design=design,
        registration=registration,
        design_bytes=design_bytes,
    )
    freeze = design["freeze"]
    expected_implementation = {
        "commit": freeze["implementation_commit"],
        "tag": freeze["test_runner_tag"],
        "container_digest": freeze["container_digest"],
    }
    if any(implementation.get(key) != value for key, value in expected_implementation.items()):
        raise ValueError("Stage B test implementation differs from the frozen design")

    selection_bytes = selection_path.read_bytes()
    selection = json.loads(selection_bytes)
    selection_seal_bytes = selection_seal_path.read_bytes()
    selection_seal = json.loads(selection_seal_bytes)
    _validate_stage_b_selection_for_test(
        design=design,
        design_sha256=design_sha,
        selection=selection,
        selection_bytes=selection_bytes,
        seal=selection_seal,
    )

    split_bytes = split_manifest_path.read_bytes()
    split_manifest = json.loads(split_bytes)
    if split_manifest.get("schema_version") != STAGE_B_SPLIT_SCHEMA:
        raise ValueError("unsupported Stage B split manifest")
    if split_manifest["input"]["design"]["sha256"] != design_sha:
        raise ValueError("Stage B split differs from the frozen design")
    if split_manifest["input"]["registration"]["sha256"] != stable_sha256(registration_bytes):
        raise ValueError("Stage B split differs from the registration")
    assignment_bytes = assignments_path.read_bytes()
    if stable_sha256(assignment_bytes) != str(split_manifest["summary"]["assignment_sha256"]):
        raise ValueError("Stage B assignments differ from the split manifest")
    assignments = [json.loads(line) for line in assignment_bytes.splitlines() if line]

    d5_bytes = d5_manifest_path.read_bytes()
    d5 = json.loads(d5_bytes)
    if stable_sha256(d5_bytes) != str(design["eligibility"]["artifacts"]["d5_metadata_manifest_sha256"]):
        raise ValueError("Stage B D5 manifest differs from the frozen design")
    expected_d5_index_sha = next(
        str(item["sha256"])
        for item in d5["artifacts"]
        if item["path"] == "d5_metadata_index.jsonl"
    )
    if expected_d5_index_sha != str(design["eligibility"]["artifacts"]["d5_metadata_index_sha256"]):
        raise ValueError("Stage B D5 index hash is inconsistent before test opening")

    output_dir.mkdir(parents=True, exist_ok=True)
    receipt = open_or_resume_test_transaction(
        output_dir=output_dir,
        protocol_sha256=design_sha,
        selection_sha256=stable_sha256(selection_bytes),
        selection_seal_sha256=stable_sha256(selection_seal_bytes),
        selected_cell_id=str(selection["selected_prefix_HEE_cell_id"]),
        selected_active_curvature=float(selection["selected_active_curvature"]),
        implementation=implementation,
    )

    d5_index_bytes = d5_index_path.read_bytes()
    if stable_sha256(d5_index_bytes) != expected_d5_index_sha:
        raise ValueError("Stage B D5 index differs from the pre-opening hash")
    sampling = design["sampling"]
    randomness = sampling["randomness"]
    selected, sampling_summary = select_test_programs(
        metadata_rows=(json.loads(line) for line in d5_index_bytes.splitlines() if line),
        assignments=assignments,
        beacon_key=beacon_key,
        dataset_revision=str(design["dataset"]["revision"]),
        program_domain=str(randomness["program_domain"]),
        user_domain=str(randomness["user_domain"]),
        queries_per_cluster=int(sampling["test_queries_per_cluster"]),
        gallery_per_cluster=int(sampling["test_gallery_per_cluster"]),
    )
    test_clusters = int(design["eligibility"]["primary_role_upper_bound"]["test_clusters"])
    expected_summary = {
        "test_clusters": test_clusters,
        "test_queries": test_clusters * int(sampling["test_queries_per_cluster"]),
        "test_gallery": test_clusters * int(sampling["test_gallery_per_cluster"]),
        "test_programs": test_clusters
        * (int(sampling["test_queries_per_cluster"]) + int(sampling["test_gallery_per_cluster"])),
    }
    if any(int(sampling_summary[key]) != value for key, value in expected_summary.items()):
        raise ValueError(f"Stage B test sampling cardinality mismatch: {sampling_summary}")

    audited, ast_summary, source_provenance = materialize_and_audit_stage_b_java_rows(
        design=design,
        sample_rows=selected,
        candidate_manifest_path=candidate_manifest_path,
        candidate_archive_path=candidate_archive_path,
        d0_d2_manifest_path=d0_d2_manifest_path,
        d0_d2_inventory_path=d0_d2_inventory_path,
        source_root=source_root,
        workers=workers,
    )
    if ast_summary["valid_for_stage_b_modeling"] is not True:
        raise ValueError("Stage B test AST audit failed before retrieval metrics")

    payloads = {
        "test_programs.jsonl": jsonl_bytes(selected),
        "test_source_ast_index.jsonl": jsonl_bytes(audited),
        "test_source_ast_summary.json": canonical_json_bytes(ast_summary),
    }
    artifacts = []
    for filename, content in payloads.items():
        _write_once_or_verify(output_dir / filename, content)
        artifacts.append({"path": filename, "bytes": len(content), "sha256": stable_sha256(content)})
    receipt_path = output_dir / "test_opening_receipt.json"
    manifest = {
        "schema_version": STAGE_B_TEST_MATERIALIZATION_SCHEMA,
        "experiment_role": "single_registered_Java_test_opening_with_pre_metric_AST_identity_audit",
        "implementation": dict(implementation),
        "inputs": {
            "design_sha256": design_sha,
            "registration_sha256": stable_sha256(registration_bytes),
            "split_manifest_sha256": stable_sha256(split_bytes),
            "cluster_assignments_sha256": stable_sha256(assignment_bytes),
            "d5_manifest_sha256": stable_sha256(d5_bytes),
            "d5_metadata_index_sha256": stable_sha256(d5_index_bytes),
            "validation_selection_sha256": stable_sha256(selection_bytes),
            "validation_selection_seal_sha256": stable_sha256(selection_seal_bytes),
            "opening_receipt_sha256": stable_sha256(receipt_path.read_bytes()),
            **source_provenance,
        },
        "selected_active_curvature": float(selection["selected_active_curvature"]),
        "selected_prefix_HEE_cell_id": str(selection["selected_prefix_HEE_cell_id"]),
        "selected_prefix_HHH_cell_id": str(selection["selected_prefix_HHH_cell_id"]),
        "test_cell_plan": selection["test_cell_plan"],
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
    manifest_bytes = canonical_json_bytes(manifest)
    _write_once_or_verify(output_dir / "test_materialization_manifest.json", manifest_bytes)
    return manifest


def _validate_stage_b_selection_for_test(
    *,
    design: Mapping[str, Any],
    design_sha256: str,
    selection: Mapping[str, Any],
    selection_bytes: bytes,
    seal: Mapping[str, Any],
) -> None:
    if selection.get("schema_version") != STAGE_B_SELECTION_SCHEMA:
        raise ValueError("unexpected Stage B validation-selection schema")
    if selection.get("status") != "validation_selection_complete_test_unopened":
        raise ValueError("Stage B validation selection is incomplete or test-opened")
    if seal.get("schema_version") != "code2hyp-codenet-java-stage-b-validation-selection-seal-v1":
        raise ValueError("unexpected Stage B validation-selection seal schema")
    if seal.get("status") != "validation_selection_sealed_test_unopened":
        raise ValueError("Stage B validation-selection seal is not test-naive")
    if seal.get("inputs", {}).get("selection", {}).get("sha256") != stable_sha256(selection_bytes):
        raise ValueError("Stage B validation selection differs from its seal")
    required_checks = {
        "registered_seed_set_complete_for_both_models",
        "all_seed_results_recomputed_and_sealed",
        "selection_recomputed_from_frozen_rule",
        "validation_only",
    }
    if any(seal.get("checks", {}).get(check) is not True for check in required_checks):
        raise ValueError("Stage B validation-selection seal is incomplete")
    if selection.get("input", {}).get("design_sha256") != design_sha256:
        raise ValueError("Stage B validation selection used a different design")
    if tuple(int(seed) for seed in selection.get("registered_seeds", ())) != tuple(
        int(seed) for seed in design["encoder_training"]["model_seeds"]
    ):
        raise ValueError("Stage B validation selection has a different seed sequence")
    selected_curvature = float(selection["selected_active_curvature"])
    active_cell = curvature_cell_id(selected_curvature)
    hhh_cell = all_role_curvature_cell_id(selected_curvature)
    if selected_curvature not in tuple(float(value) for value in design["geometry"]["active_curvature_candidates"]):
        raise ValueError("Stage B selected curvature is outside the frozen candidate set")
    if selection.get("selected_prefix_HEE_cell_id") != active_cell or selection.get("selected_prefix_HHH_cell_id") != hhh_cell:
        raise ValueError("Stage B selected cell identifiers do not match the selected curvature")
    expected_plan = _stage_b_test_cell_plan(active_cell=active_cell, hhh_cell=hhh_cell)
    if tuple(selection.get("test_cell_plan", ())) != expected_plan or tuple(
        row["cell_id"] for row in expected_plan
    ) != tuple(design["geometry"]["test_cells"]):
        raise ValueError("Stage B validation selection changed the frozen seven-cell mapping")
    if float(selection["selected_active_curvature"]) != float(seal["selected_active_curvature"]):
        raise ValueError("Stage B selected curvature differs from its seal")
    if (
        seal.get("selected_prefix_HEE_cell_id") != active_cell
        or seal.get("selected_prefix_HHH_cell_id") != hhh_cell
        or tuple(seal.get("test_cell_plan", ())) != expected_plan
    ):
        raise ValueError("Stage B seven-cell mapping differs from its seal")
    forbidden = (
        "java_test_program_ids_materialized",
        "java_test_relevance_labels_opened",
        "java_test_retrieval_metrics_computed",
    )
    if any(bool(selection.get(flag)) or bool(seal.get(flag)) for flag in forbidden):
        raise ValueError("Stage B validation artifacts indicate prior test access")


def assign_stage_b_cluster_ids(
    cluster_rows: Sequence[Mapping[str, Any]],
    *,
    beacon_key: bytes,
    dataset_revision: str,
    train_clusters: int,
    validation_clusters: int,
    test_clusters: int,
) -> list[dict[str, Any]]:
    """Assign role-constrained Java clusters after the public Beacon pulse."""

    if min(train_clusters, validation_clusters, test_clusters) <= 0:
        raise ValueError("Stage B train, validation, and test quotas must be positive")
    rows_by_id: dict[str, Mapping[str, Any]] = {}
    for row in cluster_rows:
        cluster_id = str(row.get("cluster_id", ""))
        if not cluster_id or cluster_id in rows_by_id:
            raise ValueError("Stage B cluster IDs must be non-empty and unique")
        rows_by_id[cluster_id] = row
    evaluation_ids = {
        cluster_id
        for cluster_id, row in rows_by_id.items()
        if row.get("eligible_evaluation_minimum_16") is True
    }
    train_ids = {
        cluster_id
        for cluster_id, row in rows_by_id.items()
        if row.get("eligible_train") is True
    }
    if not train_ids <= evaluation_ids:
        raise ValueError("every train-eligible Stage B cluster must also be evaluation-eligible")
    if len(train_ids) < train_clusters:
        raise ValueError("insufficient train-eligible Stage B clusters")

    ordered = lambda ids: sorted(
        (
            hmac_cluster_digest(
                beacon_key=beacon_key,
                dataset_revision=dataset_revision,
                cluster_id=cluster_id,
            ),
            cluster_id,
        )
        for cluster_id in ids
    )
    selected_train = ordered(train_ids)[:train_clusters]
    selected_train_ids = {cluster_id for _, cluster_id in selected_train}
    evaluation_pool = evaluation_ids - selected_train_ids
    required_evaluation = validation_clusters + test_clusters
    if len(evaluation_pool) < required_evaluation:
        raise ValueError("insufficient evaluation clusters after the Stage B training assignment")
    ordered_evaluation = ordered(evaluation_pool)
    selected_validation = ordered_evaluation[:validation_clusters]
    selected_test = ordered_evaluation[validation_clusters:required_evaluation]
    reserve = ordered_evaluation[required_evaluation:]

    result: list[dict[str, Any]] = []
    order_index = 0
    for split, values in (
        ("train", selected_train),
        ("validation", selected_validation),
        ("test", selected_test),
        ("reserve", reserve),
    ):
        for split_index, (digest, cluster_id) in enumerate(values):
            result.append(
                {
                    "order_index": order_index,
                    "split": split,
                    "split_index": split_index,
                    "cluster_id": cluster_id,
                    "hmac_sha256": digest.hex(),
                }
            )
            order_index += 1
    return result


def build_stage_b_validation_selection(
    prefix_payloads: Sequence[Mapping[str, Any]],
    label_only_payloads: Sequence[Mapping[str, Any]],
    *,
    expected_seeds: Sequence[int],
    active_curvatures: Sequence[float],
) -> dict[str, Any]:
    """Freeze the seven test cells after prefix-only curvature selection."""

    seeds = tuple(int(seed) for seed in expected_seeds)
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("registered Stage B seeds must be non-empty and unique")
    curvatures = tuple(float(value) for value in active_curvatures)
    if not curvatures or len(curvatures) != len(set(curvatures)) or any(c <= 0 for c in curvatures):
        raise ValueError("active Stage B curvatures must be positive and unique")

    prefix_config = _validate_model_payloads(
        prefix_payloads,
        expected_seeds=seeds,
        expected_mode="label_depth_prefix",
        active_curvatures=curvatures,
        include_all_role_hyperbolic=True,
    )
    label_config = _validate_model_payloads(
        label_only_payloads,
        expected_seeds=seeds,
        expected_mode="label_only",
        active_curvatures=curvatures,
        include_all_role_hyperbolic=False,
    )
    ignored = {"node_input_mode", "include_all_role_hyperbolic"}
    if {key: value for key, value in prefix_config.items() if key not in ignored} != {
        key: value for key, value in label_config.items() if key not in ignored
    }:
        raise ValueError("prefix and label-only Stage B models do not use the same training budget")

    prefix_selection = select_active_curvature(
        prefix_payloads,
        active_curvatures=curvatures,
        expected_seeds=seeds,
    )
    selected = float(prefix_selection["selected_active_curvature"])
    active_cell = curvature_cell_id(selected)
    hhh_cell = all_role_curvature_cell_id(selected)
    test_plan = _stage_b_test_cell_plan(active_cell=active_cell, hhh_cell=hhh_cell)
    payloads_by_model = {"prefix": prefix_payloads, "label_only": label_only_payloads}
    validation_means = {
        row["cell_id"]: _mean_seed_averaged_task_score(
            payloads_by_model[row["model"]], row["validation_cell_id"]
        )
        for row in test_plan
    }
    return {
        "schema_version": STAGE_B_SELECTION_SCHEMA,
        "status": "validation_selection_complete_test_unopened",
        "registered_seeds": list(seeds),
        "selection_rule": prefix_selection["selection_rule"],
        "selection_model": "label_depth_prefix",
        "selected_active_curvature": selected,
        "selected_prefix_HEE_cell_id": active_cell,
        "selected_prefix_HHH_cell_id": hhh_cell,
        "candidate_mean_validation_problem_macro_MAP_at_8": prefix_selection[
            "candidate_mean_validation_problem_macro_MAP_at_8"
        ],
        "test_cell_plan": list(test_plan),
        "descriptive_validation_mean_problem_macro_MAP_at_8": validation_means,
        "java_test_program_ids_materialized": False,
        "java_test_relevance_labels_opened": False,
        "java_test_retrieval_metrics_computed": False,
    }


def _stage_b_test_cell_plan(*, active_cell: str, hhh_cell: str) -> tuple[dict[str, str], ...]:
    return (
        {"cell_id": "prefix_EEE_true_LCA", "model": "prefix", "validation_cell_id": "EEE_true_LCA"},
        {"cell_id": "prefix_EEE_zero_anchor", "model": "prefix", "validation_cell_id": "EEE_zero_anchor"},
        {
            "cell_id": "prefix_HEE_near_zero_true_LCA",
            "model": "prefix",
            "validation_cell_id": "HEE_near_zero_true_LCA",
        },
        {"cell_id": "prefix_HEE_active_true_LCA", "model": "prefix", "validation_cell_id": active_cell},
        {"cell_id": "label_only_EEE_true_LCA", "model": "label_only", "validation_cell_id": "EEE_true_LCA"},
        {
            "cell_id": "label_only_HEE_active_true_LCA",
            "model": "label_only",
            "validation_cell_id": active_cell,
        },
        {"cell_id": "prefix_HHH_active_true_LCA", "model": "prefix", "validation_cell_id": hhh_cell},
    )


def _validate_model_payloads(
    payloads: Sequence[Mapping[str, Any]],
    *,
    expected_seeds: Sequence[int],
    expected_mode: str,
    active_curvatures: Sequence[float],
    include_all_role_hyperbolic: bool,
) -> dict[str, Any]:
    if tuple(int(payload.get("seed", -1)) for payload in payloads) != tuple(expected_seeds):
        raise ValueError(f"{expected_mode} validation payloads do not follow the registered seed order")
    if any(payload.get("status") != "complete" for payload in payloads):
        raise ValueError(f"{expected_mode} validation payloads are incomplete")
    configs = [dict(payload["execution_config"]) for payload in payloads]
    if any(config != configs[0] for config in configs[1:]):
        raise ValueError(f"{expected_mode} execution config differs across seeds")
    config = configs[0]
    if str(config.get("node_input_mode", "label_only")) != expected_mode:
        raise ValueError(f"unexpected node input mode for {expected_mode} validation")
    if config.get("fit_all_roles_to_active_ball") is not True:
        raise ValueError("Stage B requires train-only ball fitting for all three roles")
    if bool(config.get("include_all_role_hyperbolic", False)) != include_all_role_hyperbolic:
        raise ValueError(f"unexpected HHH validation setting for {expected_mode}")
    if any(
        payload.get("calibration_manifest_sha256") != payloads[0].get("calibration_manifest_sha256")
        for payload in payloads[1:]
    ):
        raise ValueError(f"{expected_mode} calibration manifest differs across seeds")

    expected_cells = {
        "EEE_true_LCA",
        "EEE_zero_anchor",
        "HEE_near_zero_true_LCA",
        *(curvature_cell_id(value) for value in active_curvatures),
    }
    if include_all_role_hyperbolic:
        expected_cells.update(all_role_curvature_cell_id(value) for value in active_curvatures)
    for payload in payloads:
        if set(payload["cells"]) != expected_cells:
            raise ValueError(f"{expected_mode} validation does not contain the frozen geometry cells")
    return config


def _mean_seed_averaged_task_score(
    payloads: Sequence[Mapping[str, Any]],
    cell_id: str,
) -> float:
    per_task: dict[str, list[float]] = {}
    reference: set[str] | None = None
    for payload in payloads:
        scores = payload["cells"][cell_id]["metrics"]["task_scores"]
        tasks = set(scores)
        if reference is None:
            reference = tasks
        elif tasks != reference:
            raise ValueError(f"validation task set differs across seeds for {cell_id}")
        for task, value in scores.items():
            per_task.setdefault(str(task), []).append(float(value))
    if not per_task:
        raise ValueError(f"validation cell {cell_id} contains no task scores")
    return sum(sum(values) / len(values) for values in per_task.values()) / len(per_task)


def _write_once_or_verify(path: Path, content: bytes) -> None:
    if path.exists():
        if path.read_bytes() != content:
            raise FileExistsError(f"refusing to overwrite a different Stage B artifact: {path}")
        return
    path.write_bytes(content)
