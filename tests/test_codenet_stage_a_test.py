from __future__ import annotations

import json
from pathlib import Path

import pytest

from geometry_profile_research.codenet_eligibility import canonical_json_bytes, stable_sha256
from geometry_profile_research.codenet_stage_a import StageAProgram, load_stage_a_test_split
from geometry_profile_research.codenet_stage_a_test import (
    open_or_resume_test_transaction,
    select_test_programs,
    validate_test_execution_protocol,
    validate_validation_selection_for_test,
)


def _metadata_row(cluster: str, user: str, source: str) -> dict[str, str]:
    return {
        "problem_cluster_id": cluster,
        "user_id_sha256": user,
        "source_relpath": source,
        "problem_id": f"problem-{cluster}",
        "submission_id": source.removesuffix(".py"),
        "status": "Accepted",
    }


def test_test_sampling_uses_distinct_users_and_publishes_no_user_hash() -> None:
    assignments = [
        {"cluster_id": "train-cluster", "split": "train", "split_index": 0, "order_index": 0},
        {"cluster_id": "test-cluster", "split": "test", "split_index": 0, "order_index": 1},
    ]
    metadata = [
        _metadata_row("train-cluster", "train-user", "train.py"),
        _metadata_row("test-cluster", "user-a", "a1.py"),
        _metadata_row("test-cluster", "user-a", "a2.py"),
        _metadata_row("test-cluster", "user-b", "b.py"),
        _metadata_row("test-cluster", "user-c", "c.py"),
    ]

    selected, summary = select_test_programs(
        metadata_rows=metadata,
        assignments=assignments,
        beacon_key=bytes(range(64)),
        dataset_revision="1.0.0",
        program_domain="program-domain",
        user_domain="user-domain",
        queries_per_cluster=1,
        gallery_per_cluster=1,
    )

    assert [row["role"] for row in selected] == ["query", "gallery"]
    assert len({row["source_relpath"][0] for row in selected}) == 2
    assert all(row["split"] == "test" for row in selected)
    assert all(not any("user" in key.casefold() for key in row) for row in selected)
    assert summary == {
        "test_clusters": 1,
        "test_queries": 1,
        "test_gallery": 1,
        "test_programs": 2,
        "minimum_available_users_test": 3,
    }


def test_opening_receipt_is_single_and_resumable_only_for_same_identity(tmp_path: Path) -> None:
    common = {
        "output_dir": tmp_path,
        "protocol_sha256": "a" * 64,
        "selection_sha256": "b" * 64,
        "selection_seal_sha256": "c" * 64,
        "selected_cell_id": "HEE_c1_true_LCA",
        "selected_active_curvature": 1.0,
        "implementation": {"commit": "deadbeef", "tag": "test-runner-v1"},
        "created_utc": "2026-07-15T12:00:00+00:00",
    }
    first = open_or_resume_test_transaction(**common)
    second = open_or_resume_test_transaction(**common)

    assert first["transaction_resumed"] is False
    assert second["transaction_resumed"] is True
    assert first["transaction_identity_sha256"] == second["transaction_identity_sha256"]
    with pytest.raises(ValueError, match="second or incompatible"):
        open_or_resume_test_transaction(
            **{**common, "selected_active_curvature": 3.0}
        )


def test_metadata_index_hash_read_is_deferred_until_after_opening_receipt(tmp_path: Path) -> None:
    inputs = {}
    file_names = (
        "registration",
        "sampling_protocol",
        "ast_path_protocol",
        "model_analysis_protocol",
        "test_inference_protocol",
        "split_manifest",
        "cluster_assignments",
        "d5_metadata_manifest",
    )
    for name in file_names:
        path = tmp_path / f"{name}.json"
        path.write_bytes(name.encode("ascii"))
        inputs[name] = {"path": path.name, "sha256": stable_sha256(path.read_bytes())}
    missing_d5 = tmp_path / "sealed-d5-index.jsonl"
    inputs["d5_metadata_index"] = {
        "path": missing_d5.name,
        "sha256": "d" * 64,
    }
    inputs["validation_selection_requirement"] = "a valid seal"
    protocol = {
        "schema_version": "code2hyp-stage-a-test-execution-protocol-v1",
        "status": "frozen_during_validation_before_validation_selection_or_test_unseal",
        "inputs": inputs,
        "state_at_freeze": {
            "validation_started": True,
            "validation_selection_complete": False,
            "test_program_ids_materialized": False,
            "test_relevance_labels_opened": False,
            "test_retrieval_metrics_computed": False,
        },
    }
    protocol_path = tmp_path / "test_protocol.json"
    protocol_path.write_bytes(canonical_json_bytes(protocol))

    verified = validate_test_execution_protocol(
        project_root=tmp_path,
        protocol_path=protocol_path,
    )

    assert not missing_d5.exists()
    assert verified["inputs"]["d5_metadata_index"]["verification"] == (
        "deferred_until_after_opening_receipt"
    )


def test_selection_must_match_its_seal_and_registered_seed_sequence(tmp_path: Path) -> None:
    selection = {
        "schema_version": "code2hyp-stage-a-validation-selection-v1",
        "registered_seeds": [11, 12],
        "selected_active_curvature": 0.3,
        "selected_cell_id": "HEE_c0p3_true_LCA",
        "test_program_ids_materialized": False,
        "test_relevance_labels_opened": False,
        "test_retrieval_metrics_computed": False,
    }
    selection_path = tmp_path / "selection.json"
    selection_path.write_bytes(canonical_json_bytes(selection))
    seal = {
        "schema_version": "code2hyp-stage-a-validation-selection-seal-v1",
        "inputs": {
            "selection": {"sha256": stable_sha256(selection_path.read_bytes())},
            "seeds": [{"seed": 11}, {"seed": 12}],
        },
        "selected_active_curvature": 0.3,
        "selected_cell_id": "HEE_c0p3_true_LCA",
        "checks": {
            "registered_seed_set_complete": True,
            "all_seed_results_match_their_seals": True,
            "selection_recomputed_from_frozen_rule": True,
            "validation_only": True,
        },
        "test_program_ids_materialized": False,
        "test_relevance_labels_opened": False,
        "test_retrieval_metrics_computed": False,
    }
    seal_path = tmp_path / "selection_seal.json"
    seal_path.write_bytes(canonical_json_bytes(seal))

    verified = validate_validation_selection_for_test(
        selection_path=selection_path,
        selection_seal_path=seal_path,
        registered_seeds=(11, 12),
    )
    assert verified["selection"]["selected_active_curvature"] == 0.3

    seal["selected_active_curvature"] = 1.0
    seal_path.write_text(json.dumps(seal), encoding="utf-8")
    with pytest.raises(ValueError, match="curvature differs"):
        validate_validation_selection_for_test(
            selection_path=selection_path,
            selection_seal_path=seal_path,
            registered_seeds=(11, 12),
        )


def test_test_split_loader_enforces_registered_cluster_cardinalities(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = []
    ast_rows = []
    for cluster_index in range(386):
        cluster = f"cluster-{cluster_index:03d}"
        for role in ("query", "gallery"):
            for role_index in range(8):
                source = f"{cluster}/{role}-{role_index}.py"
                rows.append(
                    {
                        "cluster_id": cluster,
                        "problem_id": f"problem-{cluster}",
                        "role": role,
                        "source_relpath": source,
                        "split": "test",
                        "submission_id": f"{cluster}-{role}-{role_index}",
                    }
                )
                ast_rows.append({"source_relpath": source})
    test_path = tmp_path / "test.jsonl"
    ast_path = tmp_path / "ast.jsonl"
    test_path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    ast_path.write_text("".join(json.dumps(row) + "\n" for row in ast_rows), encoding="utf-8")

    def fake_load_program(*, source_root: Path, sample_row: dict, ast_row: dict) -> StageAProgram:
        assert ast_row["source_relpath"] == sample_row["source_relpath"]
        return StageAProgram(
            item_id=sample_row["source_relpath"],
            cluster_id=sample_row["cluster_id"],
            problem_id=sample_row["problem_id"],
            split=sample_row["split"],
            role=sample_row["role"],
            tree=None,  # type: ignore[arg-type]
        )

    monkeypatch.setattr("geometry_profile_research.codenet_stage_a._load_program", fake_load_program)
    split = load_stage_a_test_split(
        source_root=tmp_path,
        test_path=test_path,
        ast_index_path=ast_path,
    )

    assert len(split.query) == 3_088
    assert len(split.gallery) == 3_088
