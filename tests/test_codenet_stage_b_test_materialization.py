from __future__ import annotations

import io
import json
import tarfile
from datetime import datetime
from pathlib import Path

import pytest

from geometry_profile_research.codenet_eligibility import (
    canonical_json_bytes,
    jsonl_bytes,
    normalize_java_source,
    stable_sha256,
)
from geometry_profile_research.codenet_stage_b import materialize_stage_b_test_programs
from geometry_profile_research.java_raw_ast import parse_java_ast_tree


def test_stage_b_test_opening_materializes_only_registered_java_rows(tmp_path: Path) -> None:
    sources = {
        "p1/s1.java": b"class A { int f(int x) { return x + 1; } }\n",
        "p1/s2.java": b"class B { int g(int y) { return y * 2; } }\n",
    }
    archive_path = tmp_path / "sources.tar"
    with tarfile.open(archive_path, "w") as archive:
        for name, content in sources.items():
            member = tarfile.TarInfo(name)
            member.size = len(content)
            archive.addfile(member, io.BytesIO(content))
    candidate_path = tmp_path / "candidate.json"
    candidate_path.write_bytes(
        canonical_json_bytes({"candidate_archive": {"sha256": stable_sha256(archive_path.read_bytes())}})
    )

    inventory = []
    for name, content in sources.items():
        canonical = normalize_java_source(content)
        tree = parse_java_ast_tree(canonical.text)
        inventory.append(
            {
                "source_relpath": name,
                "d0_sha256": stable_sha256(canonical.text),
                "ast_node_count": len(tree.labels),
                "retained_after_d0_d2": True,
                "canonical_source_relpath": name,
            }
        )
    inventory_path = tmp_path / "inventory.jsonl"
    inventory_path.write_bytes(jsonl_bytes(inventory))
    d0_path = tmp_path / "d0.json"
    d0_path.write_bytes(
        canonical_json_bytes(
            {"artifacts": [{"path": "file_inventory.jsonl", "sha256": stable_sha256(inventory_path.read_bytes())}]}
        )
    )

    d5_rows = [
        {
            "problem_cluster_id": "cluster-A",
            "user_id_sha256": f"user-{index}",
            "source_relpath": source,
            "problem_id": "p1",
            "submission_id": f"s{index}",
            "status": "Accepted",
        }
        for index, source in enumerate(sources, start=1)
    ]
    d5_index_path = tmp_path / "d5.jsonl"
    d5_index_path.write_bytes(jsonl_bytes(d5_rows))
    d5_manifest_path = tmp_path / "d5_manifest.json"
    d5_manifest_path.write_bytes(
        canonical_json_bytes(
            {"artifacts": [{"path": "d5_metadata_index.jsonl", "sha256": stable_sha256(d5_index_path.read_bytes())}]}
        )
    )

    design = json.loads((Path(__file__).resolve().parents[1] / "configs/codenet_java_stage_b_draft_v1.json").read_text())
    design["eligibility"]["primary_role_upper_bound"] = {
        "train_clusters": 1,
        "validation_clusters": 1,
        "test_clusters": 1,
        "reserve_evaluation_clusters": 0,
    }
    design["eligibility"]["primary_counts"]["eligible_evaluation_clusters"] = 3
    design["eligibility"]["primary_counts"]["eligible_train_clusters"] = 1
    design["eligibility"]["artifacts"].update(
        {
            "candidate_materialization_manifest_sha256": stable_sha256(candidate_path.read_bytes()),
            "d0_d2_manifest_sha256": stable_sha256(d0_path.read_bytes()),
            "d5_metadata_manifest_sha256": stable_sha256(d5_manifest_path.read_bytes()),
            "d5_metadata_index_sha256": stable_sha256(d5_index_path.read_bytes()),
        }
    )
    design["sampling"]["test_queries_per_cluster"] = 1
    design["sampling"]["test_gallery_per_cluster"] = 1
    design["encoder_training"]["model_seeds"] = [7]
    design["freeze"].update(
        {
            "implementation_commit": "a" * 40,
            "container_digest": "sha256:" + "b" * 64,
        }
    )
    design_path = tmp_path / "design.json"
    design_path.write_bytes(canonical_json_bytes(design))
    design_sha = stable_sha256(design_path.read_bytes())

    created = "2026-01-01T00:00:00+00:00"
    registration = {
        "schema_version": "code2hyp-codenet-java-stage-b-registration-v1",
        "registration": {"doi": "10.5281/zenodo.1", "created_utc": created},
        "design": {
            "sha256": design_sha,
            "dataset_revision": design["dataset"]["revision"],
            "eligible_evaluation_clusters": 3,
            "eligible_train_clusters": 1,
            "quotas_train_validation_test": [1, 1, 1],
        },
        "nist_randomness_beacon": {
            "timestamp_utc": "2026-01-01T00:01:00+00:00",
            "query_timestamp_unix_milliseconds": int(datetime.fromisoformat(created).timestamp() * 1000),
            "status_code": 0,
            "period_milliseconds": 60_000,
            "output_value_hex": bytes(range(64)).hex(),
            "uri": "https://beacon.nist.gov/test",
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

    assignments = [
        {"order_index": 0, "split": "train", "split_index": 0, "cluster_id": "train-A"},
        {"order_index": 1, "split": "validation", "split_index": 0, "cluster_id": "validation-A"},
        {"order_index": 2, "split": "test", "split_index": 0, "cluster_id": "cluster-A"},
    ]
    assignments_path = tmp_path / "assignments.jsonl"
    assignments_path.write_bytes(jsonl_bytes(assignments))
    split_path = tmp_path / "split.json"
    split_path.write_bytes(
        canonical_json_bytes(
            {
                "schema_version": "codenet-java-stage-b-beacon-split-v1",
                "input": {
                    "design": {"sha256": design_sha},
                    "registration": {"sha256": stable_sha256(registration_path.read_bytes())},
                },
                "summary": {"assignment_sha256": stable_sha256(assignments_path.read_bytes())},
            }
        )
    )

    active_cell = "HEE_c0p1_true_LCA"
    hhh_cell = "HHH_c0p1_true_LCA"
    plan = [
        {"cell_id": "prefix_EEE_true_LCA", "model": "prefix", "validation_cell_id": "EEE_true_LCA"},
        {"cell_id": "prefix_EEE_zero_anchor", "model": "prefix", "validation_cell_id": "EEE_zero_anchor"},
        {
            "cell_id": "prefix_HEE_near_zero_true_LCA",
            "model": "prefix",
            "validation_cell_id": "HEE_near_zero_true_LCA",
        },
        {"cell_id": "prefix_HEE_active_true_LCA", "model": "prefix", "validation_cell_id": active_cell},
        {"cell_id": "label_only_EEE_true_LCA", "model": "label_only", "validation_cell_id": "EEE_true_LCA"},
        {"cell_id": "label_only_HEE_active_true_LCA", "model": "label_only", "validation_cell_id": active_cell},
        {"cell_id": "prefix_HHH_active_true_LCA", "model": "prefix", "validation_cell_id": hhh_cell},
    ]
    selection = {
        "schema_version": "code2hyp-codenet-java-stage-b-validation-selection-v1",
        "status": "validation_selection_complete_test_unopened",
        "input": {"design_sha256": design_sha},
        "registered_seeds": [7],
        "selected_active_curvature": 0.1,
        "selected_prefix_HEE_cell_id": active_cell,
        "selected_prefix_HHH_cell_id": hhh_cell,
        "test_cell_plan": [dict(row) for row in plan],
        "java_test_program_ids_materialized": False,
        "java_test_relevance_labels_opened": False,
        "java_test_retrieval_metrics_computed": False,
    }
    selection_path = tmp_path / "selection.json"
    selection_path.write_bytes(canonical_json_bytes(selection))
    seal = {
        "schema_version": "code2hyp-codenet-java-stage-b-validation-selection-seal-v1",
        "status": "validation_selection_sealed_test_unopened",
        "inputs": {"selection": {"sha256": stable_sha256(selection_path.read_bytes())}},
        "selected_active_curvature": 0.1,
        "selected_prefix_HEE_cell_id": active_cell,
        "selected_prefix_HHH_cell_id": hhh_cell,
        "test_cell_plan": [dict(row) for row in plan],
        "checks": {
            "registered_seed_set_complete_for_both_models": True,
            "all_seed_results_recomputed_and_sealed": True,
            "selection_recomputed_from_frozen_rule": True,
            "validation_only": True,
        },
        "java_test_program_ids_materialized": False,
        "java_test_relevance_labels_opened": False,
        "java_test_retrieval_metrics_computed": False,
    }
    seal_path = tmp_path / "selection_seal.json"
    seal_path.write_bytes(canonical_json_bytes(seal))
    implementation = {
        "commit": design["freeze"]["implementation_commit"],
        "tag": design["freeze"]["test_runner_tag"],
        "container_digest": design["freeze"]["container_digest"],
    }

    output_dir = tmp_path / "test_output"
    manifest = materialize_stage_b_test_programs(
        design_path=design_path,
        registration_path=registration_path,
        split_manifest_path=split_path,
        assignments_path=assignments_path,
        d5_manifest_path=d5_manifest_path,
        d5_index_path=d5_index_path,
        selection_path=selection_path,
        selection_seal_path=seal_path,
        candidate_manifest_path=candidate_path,
        candidate_archive_path=archive_path,
        d0_d2_manifest_path=d0_path,
        d0_d2_inventory_path=inventory_path,
        source_root=tmp_path / "test_sources",
        output_dir=output_dir,
        implementation=implementation,
    )

    assert manifest["sampling_summary"]["test_programs"] == 2
    assert manifest["ast_summary"]["valid_for_stage_b_modeling"] is True
    assert manifest["test_retrieval_metrics_computed"] is False

    selection["test_cell_plan"][0]["validation_cell_id"] = "EEE_zero_anchor"
    selection_path.write_bytes(canonical_json_bytes(selection))
    seal["inputs"]["selection"]["sha256"] = stable_sha256(selection_path.read_bytes())
    seal["test_cell_plan"] = selection["test_cell_plan"]
    seal_path.write_bytes(canonical_json_bytes(seal))
    with pytest.raises(ValueError, match="seven-cell mapping"):
        materialize_stage_b_test_programs(
            design_path=design_path,
            registration_path=registration_path,
            split_manifest_path=split_path,
            assignments_path=assignments_path,
            d5_manifest_path=d5_manifest_path,
            d5_index_path=d5_index_path,
            selection_path=selection_path,
            selection_seal_path=seal_path,
            candidate_manifest_path=candidate_path,
            candidate_archive_path=archive_path,
            d0_d2_manifest_path=d0_path,
            d0_d2_inventory_path=inventory_path,
            source_root=tmp_path / "test_sources",
            output_dir=output_dir,
            implementation=implementation,
        )

    selection["test_cell_plan"] = [dict(row) for row in plan]
    selection_path.write_bytes(canonical_json_bytes(selection))
    seal["inputs"]["selection"]["sha256"] = stable_sha256(selection_path.read_bytes())
    seal["test_cell_plan"] = [dict(row) for row in plan]
    seal_path.write_bytes(canonical_json_bytes(seal))

    d5_index_path.write_bytes(d5_index_path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="D5 index differs"):
        materialize_stage_b_test_programs(
            design_path=design_path,
            registration_path=registration_path,
            split_manifest_path=split_path,
            assignments_path=assignments_path,
            d5_manifest_path=d5_manifest_path,
            d5_index_path=d5_index_path,
            selection_path=selection_path,
            selection_seal_path=seal_path,
            candidate_manifest_path=candidate_path,
            candidate_archive_path=archive_path,
            d0_d2_manifest_path=d0_path,
            d0_d2_inventory_path=inventory_path,
            source_root=tmp_path / "test_sources",
            output_dir=output_dir,
            implementation=implementation,
        )
