from __future__ import annotations

import copy
import json
from collections import Counter
from pathlib import Path

import pytest

from geometry_profile_research.codenet_sampling import (
    representative_program_digest,
    select_non_test_programs,
    user_order_digest,
    validate_sampling_protocol,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _assignments() -> list[dict[str, object]]:
    return [
        {"order_index": 0, "split": "train", "split_index": 0, "cluster_id": "train-cluster"},
        {"order_index": 1, "split": "validation", "split_index": 0, "cluster_id": "val-cluster"},
        {"order_index": 2, "split": "test", "split_index": 0, "cluster_id": "test-cluster"},
    ]


def _metadata() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for cluster_id, users in (("train-cluster", 5), ("val-cluster", 6), ("test-cluster", 6)):
        for user_index in range(users):
            for program_index in range(2):
                submission = f"{cluster_id}-{user_index}-{program_index}"
                rows.append(
                    {
                        "problem_cluster_id": cluster_id,
                        "problem_id": f"problem-{cluster_id}",
                        "source_relpath": f"problem-{cluster_id}/{submission}.py",
                        "status": "Accepted",
                        "submission_id": submission,
                        "user_id_sha256": f"user-{cluster_id}-{user_index}",
                    }
                )
    return rows


def _select(rows: list[dict[str, object]]) -> tuple[list[dict], list[dict], dict]:
    return select_non_test_programs(
        metadata_rows=rows,
        assignments=_assignments(),
        beacon_key=bytes(range(64)),
        dataset_revision="1.0.0",
        program_domain="program-domain",
        user_domain="user-domain",
        train_programs_per_cluster=3,
        validation_queries_per_cluster=2,
        validation_gallery_per_cluster=2,
    )


def test_program_and_user_hmac_domains_are_independent() -> None:
    kwargs = {
        "beacon_key": bytes(range(64)),
        "dataset_revision": "1.0.0",
        "cluster_id": "cluster",
        "user_id_sha256": "user",
    }
    program = representative_program_digest(
        **kwargs,
        domain="program-domain",
        source_relpath="problem/submission.py",
    )
    user = user_order_digest(**kwargs, domain="user-domain")
    assert len(program) == 32
    assert len(user) == 32
    assert program != user


def test_sampling_is_order_invariant_user_distinct_and_keeps_test_sealed() -> None:
    rows = _metadata()
    first = _select(rows)
    second = _select(list(reversed(rows)))
    assert first == second
    train, validation, summary = first
    assert len(train) == 3
    assert Counter(row["role"] for row in validation) == {"query": 2, "gallery": 2}
    assert not any(row["split"] == "test" for row in train + validation)
    assert not any("user" in key for row in train + validation for key in row)
    assert summary["test_programs_materialized"] == 0
    assert summary["test_clusters_sealed"] == 1
    assert len({row["source_relpath"] for row in validation}) == 4


def test_sampling_rejects_insufficient_distinct_users() -> None:
    rows = [row for row in _metadata() if row["problem_cluster_id"] != "val-cluster"]
    try:
        _select(rows)
    except ValueError as error:
        assert "distinct users" in str(error)
    else:
        raise AssertionError("sampling must fail closed when a cluster has too few users")


def test_frozen_sampling_protocol_matches_registered_inputs_and_rejects_open_test() -> None:
    protocol_path = PROJECT_ROOT / "configs/codenet_python800_stage_a_sampling_protocol_v1.json"
    design_path = PROJECT_ROOT / "configs/codenet_python800_stage_a_draft.json"
    registration_path = PROJECT_ROOT / "registrations/codenet_python800_stage_a_registration_v1.json"
    split_path = PROJECT_ROOT / "data/codenet_python800_stage_a_split/split_manifest.json"
    d5_path = PROJECT_ROOT / "data/codenet_python800_d5_metadata/d5_metadata_manifest.json"
    protocol = json.loads(protocol_path.read_bytes())
    design_bytes = design_path.read_bytes()
    registration_bytes = registration_path.read_bytes()
    split_bytes = split_path.read_bytes()
    d5_bytes = d5_path.read_bytes()
    kwargs = {
        "design": json.loads(design_bytes),
        "design_bytes": design_bytes,
        "registration": json.loads(registration_bytes),
        "registration_bytes": registration_bytes,
        "split_manifest": json.loads(split_bytes),
        "split_manifest_bytes": split_bytes,
        "d5_manifest_bytes": d5_bytes,
        "d5_index_bytes_sha256": protocol["metadata_input"]["index_sha256"],
    }
    assert len(validate_sampling_protocol(protocol=protocol, **kwargs)) == 64

    opened = copy.deepcopy(protocol)
    opened["selection"]["test"]["materialize_before_unseal"] = True
    with pytest.raises(ValueError, match="test programs sealed"):
        validate_sampling_protocol(protocol=opened, **kwargs)
