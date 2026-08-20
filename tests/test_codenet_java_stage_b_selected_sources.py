from __future__ import annotations

import io
import json
import tarfile
from pathlib import Path

from geometry_profile_research.codenet_ast_audit import audit_stage_b_selected_sources
from geometry_profile_research.codenet_eligibility import (
    canonical_json_bytes,
    jsonl_bytes,
    normalize_java_source,
    stable_sha256,
)
from geometry_profile_research.java_raw_ast import parse_java_ast_tree


def test_stage_b_selected_source_audit_rechecks_pre_split_content_and_ast(tmp_path: Path) -> None:
    sources = {
        "p1/s1.java": b"class A { int f(int x) { return x + 1; } }\n",
        "p2/s2.java": b"class B { int g(int y) { return y * 2; } }\n",
    }
    archive_path = tmp_path / "sources.tar"
    with tarfile.open(archive_path, "w") as archive:
        for name, content in sources.items():
            member = tarfile.TarInfo(name)
            member.size = len(content)
            archive.addfile(member, io.BytesIO(content))
    candidate_manifest = {
        "candidate_archive": {
            "path": archive_path.name,
            "sha256": stable_sha256(archive_path.read_bytes()),
        }
    }
    candidate_manifest_path = tmp_path / "candidate_manifest.json"
    candidate_manifest_path.write_bytes(canonical_json_bytes(candidate_manifest))

    inventory = []
    for name, content in sources.items():
        canonical = normalize_java_source(content)
        tree = parse_java_ast_tree(canonical.text)
        inventory.append(
            {
                "source_relpath": name,
                "source_bytes": len(content),
                "d0_sha256": stable_sha256(canonical.text),
                "ast_node_count": len(tree.labels),
                "retained_after_d0_d2": True,
                "canonical_source_relpath": name,
            }
        )
    inventory_path = tmp_path / "file_inventory.jsonl"
    inventory_path.write_bytes(jsonl_bytes(inventory))
    d0_d2_manifest = {
        "artifacts": [
            {"path": inventory_path.name, "sha256": stable_sha256(inventory_path.read_bytes())}
        ]
    }
    d0_d2_manifest_path = tmp_path / "d0_d2_manifest.json"
    d0_d2_manifest_path.write_bytes(canonical_json_bytes(d0_d2_manifest))

    design = {
        "eligibility": {
            "artifacts": {
                "candidate_materialization_manifest_sha256": stable_sha256(candidate_manifest_path.read_bytes()),
                "d0_d2_manifest_sha256": stable_sha256(d0_d2_manifest_path.read_bytes()),
            }
        },
        "sampling": {
            "paths_per_program": 64,
            "path_selection_policy": "lca_depth_affine_sampled",
        },
    }
    design_path = tmp_path / "design.json"
    design_path.write_bytes(canonical_json_bytes(design))
    rows = [
        {
            "cluster_id": "cluster-1",
            "problem_id": "p1",
            "role": "train",
            "source_relpath": "p1/s1.java",
            "split": "train",
            "submission_id": "s1",
        },
        {
            "cluster_id": "cluster-2",
            "problem_id": "p2",
            "role": "query",
            "source_relpath": "p2/s2.java",
            "split": "validation",
            "submission_id": "s2",
        },
    ]
    train_path = tmp_path / "train.jsonl"
    validation_path = tmp_path / "validation.jsonl"
    train_path.write_bytes(jsonl_bytes(rows[:1]))
    validation_path.write_bytes(jsonl_bytes(rows[1:]))
    sampling_manifest = {
        "schema_version": "codenet-java-stage-b-program-sampling-v1",
        "input": {"design_sha256": stable_sha256(design_path.read_bytes())},
        "protocol": {
            "java_test_program_ids_materialized": False,
            "java_test_relevance_labels_opened": False,
            "java_validation_metrics_opened": False,
            "java_test_retrieval_metrics_computed": False,
        },
        "artifacts": [
            {"path": "train_programs.jsonl", "sha256": stable_sha256(train_path.read_bytes())},
            {"path": "validation_programs.jsonl", "sha256": stable_sha256(validation_path.read_bytes())},
        ],
    }
    sampling_manifest_path = tmp_path / "sampling_manifest.json"
    sampling_manifest_path.write_bytes(canonical_json_bytes(sampling_manifest))

    manifest = audit_stage_b_selected_sources(
        design_path=design_path,
        sampling_manifest_path=sampling_manifest_path,
        train_path=train_path,
        validation_path=validation_path,
        candidate_manifest_path=candidate_manifest_path,
        candidate_archive_path=archive_path,
        d0_d2_manifest_path=d0_d2_manifest_path,
        d0_d2_inventory_path=inventory_path,
        source_root=tmp_path / "selected_sources",
        output_dir=tmp_path / "audit",
    )

    assert manifest["valid_for_stage_b_modeling"] is True
    assert manifest["summary"]["program_count"] == 2
    audited = [json.loads(line) for line in (tmp_path / "audit/selected_source_ast_index.jsonl").read_text().splitlines()]
    assert all(row["pre_split_D0_match"] and row["pre_split_AST_node_count_match"] for row in audited)
