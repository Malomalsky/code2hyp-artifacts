from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import pytest
import torch

from geometry_profile_research.codenet_eligibility import canonical_json_bytes, jsonl_bytes, stable_sha256
from geometry_profile_research.codenet_stage_a_evaluation import summarize_problem_macro_retrieval
from geometry_profile_research.codenet_stage_b import STAGE_B_TEST_MATERIALIZATION_SCHEMA
from scripts.seal_codenet_java_stage_b_test import RESULT_SCHEMA, seal_stage_b_test_seed


def test_stage_b_test_seed_seal_recomputes_metrics(tmp_path: Path) -> None:
    rows = [
        {
            "role": role,
            "source_relpath": f"{role}/{index}.java",
            "cluster_id": "cluster-A",
            "split": "test",
        }
        for role in ("query", "gallery")
        for index in range(8)
    ]
    programs_path = tmp_path / "test_programs.jsonl"
    programs_path.write_bytes(jsonl_bytes(rows))
    implementation = {"commit": "a" * 40, "tag": "test", "container_digest": "sha256:" + "b" * 64}
    materialization = {
        "schema_version": STAGE_B_TEST_MATERIALIZATION_SCHEMA,
        "implementation": implementation,
        "sampling_summary": {"test_clusters": 1, "test_queries": 8, "test_gallery": 8},
        "opening": {"ordinal": 1},
        "artifacts": [{"path": "test_programs.jsonl", "sha256": stable_sha256(programs_path.read_bytes())}],
        "test_program_ids_materialized": True,
        "test_relevance_labels_opened": True,
        "test_retrieval_metrics_computed": False,
    }
    materialization_path = tmp_path / "materialization.json"
    materialization_path.write_bytes(canonical_json_bytes(materialization))

    distances = torch.zeros((8, 8), dtype=torch.float64)
    distance_path = tmp_path / "distances.pt"
    torch.save(distances, distance_path)
    metrics = asdict(
        summarize_problem_macro_retrieval(
            distances,
            query_ids=tuple(f"query/{index}.java" for index in range(8)),
            query_cluster_ids=("cluster-A",) * 8,
            gallery_ids=tuple(f"gallery/{index}.java" for index in range(8)),
            gallery_cluster_ids=("cluster-A",) * 8,
            r=8,
        )
    )
    cells = ("EEE_true_LCA", "HEE_c0p1_true_LCA")
    distance_contract = {
        "path": distance_path.name,
        "shape": [8, 8],
        "dtype": "float64",
        "sha256": stable_sha256(distance_path.read_bytes()),
        "minimum": 0.0,
        "maximum": 0.0,
        "negative_count": 0,
    }
    design_sha = "d" * 64
    result = {
        "schema_version": RESULT_SCHEMA,
        "status": "complete",
        "seed": 7,
        "identity": {
            "validation_result_sha256": "e" * 64,
            "validation_seed_seal_sha256": "f" * 64,
            "test_materialization_manifest_sha256": stable_sha256(materialization_path.read_bytes()),
            "test_execution_protocol_sha256": design_sha,
            "test_runtime_addendum_sha256": design_sha,
            "test_resumability_addendum_sha256": design_sha,
            "relevance_identity_addendum_sha256": design_sha,
            "test_cell_ids": list(cells),
            "implementation": implementation,
        },
        "cells": {
            cell: {"metrics": metrics, "distance_matrix": distance_contract}
            for cell in cells
        },
        "test_program_ids_materialized": True,
        "test_relevance_labels_opened": True,
        "test_retrieval_metrics_computed": True,
    }
    result_path = tmp_path / "seed_7_test.json"
    result_path.write_bytes(canonical_json_bytes(result))

    seal = seal_stage_b_test_seed(
        result_path=result_path,
        test_programs_path=programs_path,
        materialization_manifest_path=materialization_path,
        model="label_only",
        expected_seed=7,
        expected_cell_ids=cells,
        expected_validation_result_sha256="e" * 64,
        expected_validation_seed_seal_sha256="f" * 64,
        design_sha256=design_sha,
        output_path=tmp_path / "seal.json",
    )
    assert seal["checks"]["all_metrics_recomputed_from_distances"] is True

    result["cells"]["EEE_true_LCA"]["metrics"]["mrr"] = 0.0
    result_path.write_bytes(canonical_json_bytes(result))
    with pytest.raises(ValueError, match="metrics do not match"):
        seal_stage_b_test_seed(
            result_path=result_path,
            test_programs_path=programs_path,
            materialization_manifest_path=materialization_path,
            model="label_only",
            expected_seed=7,
            expected_cell_ids=cells,
            expected_validation_result_sha256="e" * 64,
            expected_validation_seed_seal_sha256="f" * 64,
            design_sha256=design_sha,
            output_path=tmp_path / "tampered-seal.json",
        )
