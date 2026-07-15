from __future__ import annotations

import json
from pathlib import Path

from geometry_profile_research.codenet_eligibility import canonical_json_bytes, jsonl_bytes, stable_sha256
from geometry_profile_research.codenet_stage_a import (
    build_calibration_pair_artifacts,
    select_calibration_pairs,
)


def _train_rows(cluster_count: int = 4, programs_per_cluster: int = 5) -> list[dict[str, str]]:
    return [
        {
            "cluster_id": f"cluster-{cluster:02d}",
            "role": "train",
            "source_relpath": f"p{cluster:02d}/s{program:03d}.py",
            "split": "train",
        }
        for cluster in range(cluster_count)
        for program in range(programs_per_cluster)
    ]


def test_calibration_pairs_are_deterministic_unique_and_train_only() -> None:
    rows = _train_rows()
    key = bytes(range(64))

    first = select_calibration_pairs(
        rows,
        beacon_key=key,
        dataset_revision="1.0.0",
        domain="code2hyp/test/calibration/v1",
        same_cluster_count=12,
        cross_cluster_count=6,
    )
    second = select_calibration_pairs(
        tuple(reversed(rows)),
        beacon_key=key,
        dataset_revision="1.0.0",
        domain="code2hyp/test/calibration/v1",
        same_cluster_count=12,
        cross_cluster_count=6,
    )

    assert first == second
    assert len(first) == 18
    pairs = {(row["left_source_relpath"], row["right_source_relpath"]) for row in first}
    assert len(pairs) == 18
    assert [row["pair_index"] for row in first] == list(range(18))
    assert sum(row["pair_type"] == "same_cluster" for row in first) == 12
    assert sum(row["pair_type"] == "cross_cluster" for row in first) == 6
    assert all(len(row["selection_digest"]) == 64 for row in first)


def test_calibration_pairs_reject_validation_rows() -> None:
    rows = _train_rows()
    rows[0]["split"] = "validation"

    try:
        select_calibration_pairs(
            rows,
            beacon_key=bytes(range(64)),
            dataset_revision="1.0.0",
            domain="code2hyp/test/calibration/v1",
            same_cluster_count=1,
            cross_cluster_count=1,
        )
    except ValueError as error:
        assert "only frozen training" in str(error)
    else:
        raise AssertionError("validation rows must be rejected")


def test_build_calibration_artifacts_is_idempotent_and_train_only(tmp_path: Path) -> None:
    rows = _train_rows()
    train_path = tmp_path / "data" / "train.jsonl"
    registration_path = tmp_path / "registrations" / "registration.json"
    protocol_path = tmp_path / "configs" / "protocol.json"
    output_dir = tmp_path / "data" / "calibration"
    train_path.parent.mkdir(parents=True)
    registration_path.parent.mkdir(parents=True)
    protocol_path.parent.mkdir(parents=True)
    train_bytes = jsonl_bytes(rows)
    train_path.write_bytes(train_bytes)
    registration = {
        "design": {"dataset_revision": "1.0.0"},
        "nist_randomness_beacon": {"output_value_hex": bytes(range(64)).hex()},
        "state_at_registration": {"codenet_retrieval_metrics_computed": False},
    }
    registration_bytes = canonical_json_bytes(registration)
    registration_path.write_bytes(registration_bytes)
    protocol = {
        "schema_version": "code2hyp-stage-a-model-analysis-protocol-v1",
        "status": "frozen_before_calibration_pair_materialization_or_validation_metrics",
        "registration_record": {"sha256": stable_sha256(registration_bytes)},
        "frozen_inputs": {"train_programs": {"sha256": stable_sha256(train_bytes)}},
        "train_only_calibration": {
            "domain": "code2hyp/test/calibration-artifacts/v1",
            "same_cluster_pairs": 8,
            "cross_cluster_pairs": 4,
            "pair_count": 12,
        },
        "state_at_freeze": {"validation_retrieval_metrics_computed": False},
    }
    protocol_path.write_bytes(canonical_json_bytes(protocol))

    first = build_calibration_pair_artifacts(
        project_root=tmp_path,
        protocol_path=protocol_path,
        registration_path=registration_path,
        train_path=train_path,
        output_dir=output_dir,
    )
    second = build_calibration_pair_artifacts(
        project_root=tmp_path,
        protocol_path=protocol_path,
        registration_path=registration_path,
        train_path=train_path,
        output_dir=output_dir,
    )

    assert first == second
    assert first["summary"]["pair_count"] == 12
    assert first["summary"]["validation_programs_used"] is False
    assert first["summary"]["test_program_ids_materialized"] is False
    pairs = [
        json.loads(line)
        for line in (output_dir / "calibration_pairs.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert all(row["left_source_relpath"] in {item["source_relpath"] for item in rows} for row in pairs)
