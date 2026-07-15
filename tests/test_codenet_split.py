from __future__ import annotations

import hashlib
import hmac
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from geometry_profile_research.codenet_eligibility import canonical_json_bytes
from geometry_profile_research.codenet_split import (
    assign_cluster_ids,
    build_split_artifacts,
    eligible_cluster_ids,
    hamilton_quotas,
    hmac_cluster_digest,
    validate_registration,
)
from scripts.check_codenet_stage_a_split import audit_split


def _design(cluster_count: int = 8) -> dict[str, object]:
    return {
        "dataset": {"revision": "1.0.0"},
        "split": {
            "assignment_unit": "duplicate_closed_problem_cluster",
            "weights_train_validation_test": [3, 1, 4],
            "integer_quota_rule": "hamilton_train_validation_test_tie_order",
            "ordering": "HMAC-SHA256(beacon, dataset_revision_NUL_cluster_id)",
        },
        "test_policy": {"test_labels_opened": False},
        "eligibility": {"minimum_clusters_for_practical_claim": cluster_count},
    }


def _registration(design_bytes: bytes, cluster_count: int = 8) -> dict[str, object]:
    created = datetime(2026, 7, 15, 7, 53, 27, 528133, tzinfo=timezone.utc)
    pulse = created + timedelta(seconds=32.471867)
    return {
        "schema_version": "code2hyp-codenet-stage-a-registration-v1",
        "registration": {
            "doi": "10.5281/zenodo.21371188",
            "created_utc": created.isoformat(),
        },
        "design": {
            "sha256": hashlib.sha256(design_bytes).hexdigest(),
            "dataset_revision": "1.0.0",
            "eligible_problem_clusters": cluster_count,
            "quotas_train_validation_test": list(hamilton_quotas(cluster_count, (3, 1, 4))),
        },
        "nist_randomness_beacon": {
            "timestamp_utc": pulse.isoformat().replace("+00:00", "Z"),
            "query_timestamp_unix_milliseconds": int(created.timestamp() * 1000),
            "status_code": 0,
            "period_milliseconds": 60000,
            "output_value_hex": bytes(range(64)).hex(),
            "uri": "https://beacon.nist.gov/beacon/2.0/chain/2/pulse/1",
        },
        "state_at_registration": {
            "split_generated": False,
            "test_labels_opened": False,
            "codenet_retrieval_metrics_computed": False,
        },
    }


def test_hamilton_quotas_match_registered_773_cluster_split() -> None:
    assert hamilton_quotas(773, (3, 1, 4)) == (290, 97, 386)
    assert hamilton_quotas(8, (3, 1, 4)) == (3, 1, 4)


def test_hmac_encoding_matches_independent_reference() -> None:
    key = bytes(range(64))
    expected = hmac.new(key, b"1.0.0\x00problem-001", hashlib.sha256).digest()
    assert hmac_cluster_digest(
        beacon_key=key,
        dataset_revision="1.0.0",
        cluster_id="problem-001",
    ) == expected


def test_assignment_is_deterministic_and_respects_quotas() -> None:
    cluster_ids = [f"problem-{index:03d}" for index in range(8)]
    kwargs = {
        "cluster_ids": cluster_ids,
        "beacon_key": bytes(range(64)),
        "dataset_revision": "1.0.0",
        "quotas": (3, 1, 4),
    }
    first = assign_cluster_ids(**kwargs)
    second = assign_cluster_ids(**kwargs)
    assert first == second
    assert [row["split"] for row in first].count("train") == 3
    assert [row["split"] for row in first].count("validation") == 1
    assert [row["split"] for row in first].count("test") == 4
    assert [row["order_index"] for row in first] == list(range(8))
    assert [row["hmac_sha256"] for row in first] == sorted(row["hmac_sha256"] for row in first)


def test_registration_rejects_non_future_beacon() -> None:
    design_bytes = canonical_json_bytes(_design())
    registration = _registration(design_bytes)
    registration["nist_randomness_beacon"]["timestamp_utc"] = registration["registration"]["created_utc"]
    with pytest.raises(ValueError, match="strictly later"):
        validate_registration(
            design=json.loads(design_bytes),
            registration=registration,
            design_bytes=design_bytes,
        )


def test_eligible_cluster_ids_fail_closed_on_duplicates_and_count() -> None:
    with pytest.raises(ValueError, match="duplicate cluster_id"):
        eligible_cluster_ids(
            [
                {"cluster_id": "same", "eligible_minimum_64": True},
                {"cluster_id": "same", "eligible_minimum_64": True},
            ],
            expected_count=2,
        )
    with pytest.raises(ValueError, match="count mismatch"):
        eligible_cluster_ids(
            [{"cluster_id": "only", "eligible_minimum_64": True}],
            expected_count=2,
        )


def test_build_split_is_idempotent_and_contains_no_program_labels_or_metrics(tmp_path: Path) -> None:
    design_path = tmp_path / "design.json"
    design_bytes = canonical_json_bytes(_design())
    design_path.write_bytes(design_bytes)
    registration_path = tmp_path / "registration.json"
    registration_path.write_bytes(canonical_json_bytes(_registration(design_bytes)))
    clusters_path = tmp_path / "clusters.jsonl"
    clusters_path.write_text(
        "".join(
            json.dumps(
                {
                    "cluster_id": f"problem-{index:03d}",
                    "problem_ids": [f"p{index:05d}"],
                    "eligible_minimum_64": True,
                }
            )
            + "\n"
            for index in range(8)
        ),
        encoding="utf-8",
    )
    d4_manifest_path = tmp_path / "d4.json"
    d4_manifest_path.write_bytes(
        canonical_json_bytes(
            {
                "protocol": {
                    "retrieval_metrics_opened": False,
                    "split_status": "not_generated",
                }
            }
        )
    )
    output_dir = tmp_path / "split"
    kwargs = {
        "project_root": tmp_path,
        "design_path": design_path,
        "registration_path": registration_path,
        "clusters_path": clusters_path,
        "statement_d4_manifest_path": d4_manifest_path,
        "output_dir": output_dir,
    }
    first = build_split_artifacts(**kwargs)
    first_bytes = {path.name: path.read_bytes() for path in output_dir.iterdir()}
    second = build_split_artifacts(**kwargs)
    second_bytes = {path.name: path.read_bytes() for path in output_dir.iterdir()}
    assert first == second
    assert first_bytes == second_bytes
    assert first["summary"]["cluster_count"] == 8
    assert first["protocol"]["program_sampling_generated"] is False
    assert first["protocol"]["test_relevance_labels_opened"] is False
    assert first["protocol"]["retrieval_metrics_computed"] is False
    assignments = (output_dir / "cluster_assignments.jsonl").read_text(encoding="utf-8")
    assert "problem_ids" not in assignments
    assert "MAP@R" not in assignments
    assert "relevance" not in assignments


def test_build_split_refuses_to_overwrite_different_artifact(tmp_path: Path) -> None:
    design_path = tmp_path / "design.json"
    design_bytes = canonical_json_bytes(_design())
    design_path.write_bytes(design_bytes)
    registration_path = tmp_path / "registration.json"
    registration_path.write_bytes(canonical_json_bytes(_registration(design_bytes)))
    clusters_path = tmp_path / "clusters.jsonl"
    clusters_path.write_text(
        "".join(
            json.dumps({"cluster_id": f"problem-{index:03d}", "eligible_minimum_64": True}) + "\n"
            for index in range(8)
        ),
        encoding="utf-8",
    )
    d4_manifest_path = tmp_path / "d4.json"
    d4_manifest_path.write_bytes(
        canonical_json_bytes(
            {"protocol": {"retrieval_metrics_opened": False, "split_status": "not_generated"}}
        )
    )
    output_dir = tmp_path / "split"
    output_dir.mkdir()
    (output_dir / "cluster_assignments.jsonl").write_text("different\n", encoding="utf-8")
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        build_split_artifacts(
            project_root=tmp_path,
            design_path=design_path,
            registration_path=registration_path,
            clusters_path=clusters_path,
            statement_d4_manifest_path=d4_manifest_path,
            output_dir=output_dir,
        )


def test_read_only_audit_rederives_split_and_detects_tampering(tmp_path: Path) -> None:
    design_path = tmp_path / "design.json"
    design_bytes = canonical_json_bytes(_design())
    design_path.write_bytes(design_bytes)
    registration_path = tmp_path / "registration.json"
    registration_path.write_bytes(canonical_json_bytes(_registration(design_bytes)))
    clusters_path = tmp_path / "clusters.jsonl"
    clusters_path.write_text(
        "".join(
            json.dumps({"cluster_id": f"problem-{index:03d}", "eligible_minimum_64": True}) + "\n"
            for index in range(8)
        ),
        encoding="utf-8",
    )
    d4_manifest_path = tmp_path / "d4.json"
    d4_manifest_path.write_bytes(
        canonical_json_bytes(
            {"protocol": {"retrieval_metrics_opened": False, "split_status": "not_generated"}}
        )
    )
    split_dir = tmp_path / "split"
    build_split_artifacts(
        project_root=tmp_path,
        design_path=design_path,
        registration_path=registration_path,
        clusters_path=clusters_path,
        statement_d4_manifest_path=d4_manifest_path,
        output_dir=split_dir,
    )
    kwargs = {
        "design_path": design_path,
        "registration_path": registration_path,
        "clusters_path": clusters_path,
        "statement_d4_manifest_path": d4_manifest_path,
        "split_dir": split_dir,
    }
    valid = audit_split(**kwargs)
    assert valid["valid_for_program_sampling"] is True
    assert valid["blocking_failures"] == []

    assignments = (split_dir / "cluster_assignments.jsonl").read_text(encoding="utf-8")
    (split_dir / "cluster_assignments.jsonl").write_text(
        assignments.replace('"split":"train"', '"split":"test"', 1),
        encoding="utf-8",
    )
    tampered = audit_split(**kwargs)
    assert tampered["valid_for_program_sampling"] is False
    assert "assignment_rederived" in tampered["blocking_failures"]
