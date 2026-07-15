from __future__ import annotations

import json
from pathlib import Path

import pytest

from geometry_profile_research.codenet_eligibility import canonical_json_bytes, stable_sha256
from geometry_profile_research.codenet_stage_a_runner import curvature_cell_id
from scripts.seal_codenet_stage_a_validation_seed import PROJECT_ROOT, seal_seed_result


def test_seed_seal_verifies_hashes_cardinalities_and_test_boundary(tmp_path: Path) -> None:
    protocol_path = PROJECT_ROOT / "configs/codenet_python800_stage_a_model_analysis_protocol_v1.json"
    calibration_path = (
        PROJECT_ROOT
        / "data/codenet_python800_stage_a_calibration_pairs/calibration_pair_manifest.json"
    )
    gate0_path = PROJECT_ROOT / "reports/codenet_python800_stage_a_gate0_numerical_v1.json"
    protocol_bytes = protocol_path.read_bytes()
    calibration_bytes = calibration_path.read_bytes()
    protocol = json.loads(protocol_bytes)
    checkpoint_path = tmp_path / "encoder.pt"
    distance_path = tmp_path / "distance.pt"
    checkpoint_path.write_bytes(b"checkpoint")
    distance_path.write_bytes(b"distance")
    cell_ids = {
        "EEE_true_LCA",
        "EEE_zero_anchor",
        "HEE_near_zero_true_LCA",
        *(
            curvature_cell_id(float(cell["factor_curvatures"][0]))
            for cell in protocol["geometry_cells"]["gate_C_active_candidates"]
        ),
    }
    metrics = {
        "query_count": 776,
        "problem_count": 97,
        "query_scores": {f"q{index}": 0.0 for index in range(776)},
        "task_scores": {f"p{index}": 0.0 for index in range(97)},
    }
    cells = {
        cell_id: {
            "metrics": metrics,
            "distance_matrix": {
                "path": distance_path.name,
                "shape": [776, 776],
                "dtype": "float64",
                "sha256": stable_sha256(distance_path.read_bytes()),
            },
        }
        for cell_id in cell_ids
    }
    result = {
        "status": "complete",
        "seed": 20260711,
        "protocol_sha256": stable_sha256(protocol_bytes),
        "calibration_manifest_sha256": stable_sha256(calibration_bytes),
        "execution_config": {
            "dim": 8,
            "epochs": 5,
            "batch_size": 8,
            "learning_rate": 0.003,
            "gradient_clip_norm": 1.0,
            "lambda_edge": 1.0,
            "lambda_gromov": 0.1,
            "lambda_branch": 1.0,
            "max_paths": 64,
            "max_ball_fraction": 0.35,
            "active_curvatures": [0.1, 0.3, 1.0, 3.0],
            "near_zero_curvature": 0.0001,
            "sinkhorn_kappa": 0.05,
            "sinkhorn_iterations": 128,
            "projection_iterations": 2048,
            "marginal_tolerance": 1e-7,
            "query_batch_size": 4,
            "gallery_batch_size": 32,
            "torch_num_threads": 1,
        },
        "checkpoint": {
            "path": checkpoint_path.name,
            "sha256": stable_sha256(checkpoint_path.read_bytes()),
        },
        "cells": cells,
        "test_program_ids_materialized": False,
        "test_relevance_labels_opened": False,
        "test_retrieval_metrics_computed": False,
    }
    result_path = tmp_path / "seed.json"
    result_path.write_bytes(canonical_json_bytes(result))
    output_path = tmp_path / "seal.json"

    seal = seal_seed_result(
        result_path=result_path,
        protocol_path=protocol_path,
        calibration_manifest_path=calibration_path,
        gate0_path=gate0_path,
        output_path=output_path,
    )

    assert seal["checks"]["all_hashes_match"] is True
    assert seal["checks"]["validation_only"] is True
    assert seal["test_relevance_labels_opened"] is False

    distance_path.write_bytes(b"tampered-distance")
    with pytest.raises(ValueError, match="distance matrix hash mismatch"):
        seal_seed_result(
            result_path=result_path,
            protocol_path=protocol_path,
            calibration_manifest_path=calibration_path,
            gate0_path=gate0_path,
            output_path=tmp_path / "tampered-seal.json",
        )

    distance_path.write_bytes(b"distance")
    result["test_relevance_labels_opened"] = True
    result_path.write_bytes(canonical_json_bytes(result))
    with pytest.raises(ValueError, match="forbidden test access"):
        seal_seed_result(
            result_path=result_path,
            protocol_path=protocol_path,
            calibration_manifest_path=calibration_path,
            gate0_path=gate0_path,
            output_path=tmp_path / "forbidden-test-seal.json",
        )
