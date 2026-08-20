from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from geometry_profile_research.codenet_eligibility import canonical_json_bytes, jsonl_bytes, stable_sha256
from geometry_profile_research.codenet_stage_a_runner import (
    all_role_curvature_cell_id,
    curvature_cell_id,
)
from geometry_profile_research.codenet_stage_b import (
    assign_stage_b_cluster_ids,
    build_stage_b_split_artifacts,
    build_stage_b_validation_selection,
    validate_stage_b_registration,
)
from geometry_profile_research.codenet_sampling import select_non_test_programs


CURVATURES = (0.1, 0.3, 1.0, 3.0)
SEEDS = (11, 12)


def _payload(seed: int, *, mode: str, hhh: bool) -> dict:
    config = {
        "epochs": 5,
        "batch_size": 8,
        "fit_all_roles_to_active_ball": True,
    }
    if mode != "label_only":
        config["node_input_mode"] = mode
    if hhh:
        config["include_all_role_hyperbolic"] = True
    cells = {
        "EEE_true_LCA": _cell(0.30),
        "EEE_zero_anchor": _cell(0.20),
        "HEE_near_zero_true_LCA": _cell(0.31),
    }
    for curvature, score in zip(CURVATURES, (0.32, 0.34, 0.40, 0.39), strict=True):
        cells[curvature_cell_id(curvature)] = _cell(score + seed / 10_000)
        if hhh:
            cells[all_role_curvature_cell_id(curvature)] = _cell(score - 0.01)
    return {
        "seed": seed,
        "status": "complete",
        "execution_config": config,
        "calibration_manifest_sha256": "calibration",
        "cells": cells,
    }


def _cell(score: float) -> dict:
    return {"metrics": {"task_scores": {"task-a": score, "task-b": score + 0.02}}}


def test_stage_b_selection_uses_prefix_validation_and_freezes_seven_test_cells() -> None:
    prefix = tuple(_payload(seed, mode="label_depth_prefix", hhh=True) for seed in SEEDS)
    label_only = tuple(_payload(seed, mode="label_only", hhh=False) for seed in SEEDS)

    result = build_stage_b_validation_selection(
        prefix,
        label_only,
        expected_seeds=SEEDS,
        active_curvatures=CURVATURES,
    )

    assert result["selected_active_curvature"] == 1.0
    assert len(result["test_cell_plan"]) == 7
    assert result["test_cell_plan"][-1] == {
        "cell_id": "prefix_HHH_active_true_LCA",
        "model": "prefix",
        "validation_cell_id": "HHH_c1_true_LCA",
    }
    assert result["java_test_program_ids_materialized"] is False


def test_stage_b_selection_rejects_unmatched_training_budget() -> None:
    prefix = tuple(_payload(seed, mode="label_depth_prefix", hhh=True) for seed in SEEDS)
    label_only = tuple(_payload(seed, mode="label_only", hhh=False) for seed in SEEDS)
    label_only[0]["execution_config"]["epochs"] = 4

    with pytest.raises(ValueError, match="differs across seeds"):
        build_stage_b_validation_selection(
            prefix,
            label_only,
            expected_seeds=SEEDS,
            active_curvatures=CURVATURES,
        )


def test_stage_b_split_respects_role_eligibility_and_retains_an_auditable_reserve() -> None:
    rows = [
        {
            "cluster_id": f"cluster-{index}",
            "eligible_evaluation_minimum_16": True,
            "eligible_train": index < 3,
        }
        for index in range(8)
    ]

    assignments = assign_stage_b_cluster_ids(
        rows,
        beacon_key=bytes(range(64)),
        dataset_revision="1.0.0",
        train_clusters=2,
        validation_clusters=1,
        test_clusters=2,
    )

    by_split = {
        split: [row for row in assignments if row["split"] == split]
        for split in ("train", "validation", "test", "reserve")
    }
    assert {split: len(values) for split, values in by_split.items()} == {
        "train": 2,
        "validation": 1,
        "test": 2,
        "reserve": 3,
    }
    assert all(int(row["cluster_id"].split("-")[-1]) < 3 for row in by_split["train"])
    assert len({row["cluster_id"] for row in assignments}) == len(assignments)


def test_stage_b_registration_and_split_are_bound_to_the_public_design(tmp_path: Path) -> None:
    cluster_rows = [
        {
            "cluster_id": f"cluster-{index}",
            "eligible_evaluation_minimum_16": True,
            "eligible_train": index < 3,
        }
        for index in range(8)
    ]
    clusters_path = tmp_path / "post_d4_problem_clusters.jsonl"
    clusters_path.write_bytes(jsonl_bytes(cluster_rows))
    d4 = {
        "protocol": {"split_status": "not_generated", "retrieval_metrics_opened": False},
        "artifacts": [
            {
                "path": clusters_path.name,
                "sha256": stable_sha256(clusters_path.read_bytes()),
            }
        ],
    }
    d4_path = tmp_path / "d4_manifest.json"
    d4_path.write_bytes(canonical_json_bytes(d4))
    design = {
        "dataset": {"revision": "1.0.0"},
        "eligibility": {
            "primary_counts": {
                "eligible_evaluation_clusters": 8,
                "eligible_train_clusters": 3,
            },
            "primary_role_upper_bound": {
                "train_clusters": 2,
                "validation_clusters": 1,
                "test_clusters": 2,
                "reserve_evaluation_clusters": 3,
            },
            "artifacts": {"d4_primary_manifest_sha256": stable_sha256(d4_path.read_bytes())},
        },
        "split": {
            "assignment_unit": "duplicate_closed_problem_cluster",
            "ordering": "role-constrained HMAC-SHA256",
            "role_constraint": "fixture",
        },
    }
    design_path = tmp_path / "design.json"
    design_path.write_bytes(canonical_json_bytes(design))
    created = datetime(2026, 8, 20, 10, 0, tzinfo=timezone.utc)
    pulse = created + timedelta(minutes=1)
    registration = {
        "schema_version": "code2hyp-codenet-java-stage-b-registration-v1",
        "registration": {"doi": "10.5281/zenodo.123456", "created_utc": created.isoformat()},
        "design": {
            "sha256": stable_sha256(design_path.read_bytes()),
            "dataset_revision": "1.0.0",
            "eligible_evaluation_clusters": 8,
            "eligible_train_clusters": 3,
            "quotas_train_validation_test": [2, 1, 2],
        },
        "nist_randomness_beacon": {
            "timestamp_utc": pulse.isoformat(),
            "query_timestamp_unix_milliseconds": int(created.timestamp() * 1000),
            "status_code": 0,
            "period_milliseconds": 60_000,
            "output_value_hex": bytes(range(64)).hex(),
            "uri": "https://beacon.example/pulse/1",
        },
        "state_at_registration": {
            "split_generated": False,
            "java_validation_metrics_opened": False,
            "java_test_program_ids_materialized": False,
            "java_test_relevance_labels_opened": False,
            "java_test_retrieval_metrics_computed": False,
        },
    }
    registration_path = tmp_path / "registration.json"
    registration_path.write_bytes(canonical_json_bytes(registration))

    assert validate_stage_b_registration(
        design=design,
        registration=registration,
        design_bytes=design_path.read_bytes(),
    ) == bytes(range(64))
    manifest = build_stage_b_split_artifacts(
        project_root=tmp_path,
        design_path=design_path,
        registration_path=registration_path,
        clusters_path=clusters_path,
        d4_manifest_path=d4_path,
        output_dir=tmp_path / "split",
    )

    assert manifest["summary"]["train_clusters"] == 2
    assert manifest["summary"]["reserve_clusters"] == 3
    assert manifest["protocol"]["java_test_program_ids_materialized"] is False


def test_stage_b_program_selection_discards_test_and_reserve_before_public_output() -> None:
    assignments = [
        {"order_index": 0, "split": "train", "split_index": 0, "cluster_id": "train"},
        {"order_index": 1, "split": "validation", "split_index": 0, "cluster_id": "validation"},
        {"order_index": 2, "split": "test", "split_index": 0, "cluster_id": "test"},
        {"order_index": 3, "split": "reserve", "split_index": 0, "cluster_id": "reserve"},
    ]
    metadata = [
        {
            "problem_cluster_id": cluster,
            "problem_id": cluster,
            "source_relpath": f"{cluster}/{index}.java",
            "status": "Accepted",
            "submission_id": f"{cluster}-{index}",
            "user_id_sha256": f"{cluster}-user-{index}",
        }
        for cluster in ("train", "validation", "test", "reserve")
        for index in range(4)
    ]

    train, validation, _ = select_non_test_programs(
        metadata_rows=metadata,
        assignments=assignments,
        beacon_key=bytes(range(64)),
        dataset_revision="1.0.0",
        program_domain="program",
        user_domain="user",
        train_programs_per_cluster=2,
        validation_queries_per_cluster=1,
        validation_gallery_per_cluster=1,
    )

    assert {row["cluster_id"] for row in train + validation} == {"train", "validation"}
