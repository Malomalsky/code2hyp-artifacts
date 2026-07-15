from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from geometry_profile_research.codenet_eligibility import canonical_json_bytes, stable_sha256
from geometry_profile_research.codenet_stage_a_runner import curvature_cell_id
from scripts.seal_codenet_stage_a_validation_seed import (
    PROJECT_ROOT,
    RUNNER_COMMIT,
    RUNNER_TAG,
    seal_seed_result,
)


def test_seed_seal_verifies_hashes_cardinalities_and_test_boundary(tmp_path: Path) -> None:
    protocol_path = PROJECT_ROOT / "configs/codenet_python800_stage_a_model_analysis_protocol_v1.json"
    calibration_path = (
        PROJECT_ROOT
        / "data/codenet_python800_stage_a_calibration_pairs/calibration_pair_manifest.json"
    )
    gate0_path = PROJECT_ROOT / "reports/codenet_python800_stage_a_gate0_numerical_rounding_v2.json"
    rounding_addendum_path = (
        PROJECT_ROOT / "configs/codenet_python800_stage_a_transport_rounding_addendum_v1.json"
    )
    protocol_bytes = protocol_path.read_bytes()
    calibration_bytes = calibration_path.read_bytes()
    protocol = json.loads(protocol_bytes)
    checkpoint_path = tmp_path / "seed_20260711_encoder.pt"
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
        "problem_macro_map_at_r": 0.0,
        "query_macro_map_at_r": 0.0,
        "mrr": 0.0,
        "recall_at_1": 0.0,
        "recall_at_5": 0.0,
        "recall_at_10": 0.0,
        "mean_first_relevant_rank": 1.0,
        "query_count": 776,
        "problem_count": 97,
        "query_scores": {f"q{index}": 0.0 for index in range(776)},
        "task_scores": {f"p{index}": 0.0 for index in range(97)},
    }
    cell_curvatures = {
        "EEE_true_LCA": [0.0, 0.0, 0.0],
        "EEE_zero_anchor": [0.0, 0.0, 0.0],
        "HEE_near_zero_true_LCA": [0.0001, 0.0, 0.0],
        **{
            curvature_cell_id(float(cell["factor_curvatures"][0])): list(cell["factor_curvatures"])
            for cell in protocol["geometry_cells"]["gate_C_active_candidates"]
        },
    }
    cells = {}
    distance_paths = []
    for cell_id in cell_ids:
        distance_path = tmp_path / f"seed_20260711_{cell_id}_distances.pt"
        torch.save(torch.zeros((776, 776), dtype=torch.float64), distance_path)
        distance_paths.append(distance_path)
        cells[cell_id] = {
            "factor_curvatures": cell_curvatures[cell_id],
            "factor_weights": [1.0, 1.0, 1.0],
            "metrics": metrics,
            "distance_matrix": {
                "path": distance_path.name,
                "shape": [776, 776],
                "dtype": "float64",
                "sha256": stable_sha256(distance_path.read_bytes()),
                "minimum": 0.0,
                "maximum": 0.0,
                "negative_count": 0,
            },
        }
    execution_config = {
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
    }
    history = [
        {
            "epoch": float(epoch),
            "loss": 1.0,
            "edge": 1.0,
            "gromov_lca": 1.0,
            "gromov_lca_mean_abs_residual": 1.0,
            "branch_length": 1.0,
            "reversal": 0.0,
            "program_count": 18_560.0,
            "update_count": 2_320.0,
            "max_preclip_gradient_norm": 1.0,
        }
        for epoch in range(1, 6)
    ]
    implementation = {
        "repository": "https://github.com/Malomalsky/code2hyp-artifacts",
        "commit": RUNNER_COMMIT,
        "tag": RUNNER_TAG,
        "tracked_worktree_clean": True,
    }
    checkpoint_payload = {
        "seed": 20260711,
        "protocol_sha256": stable_sha256(protocol_bytes),
        "calibration_manifest_sha256": stable_sha256(calibration_bytes),
        "execution_config": execution_config,
        "implementation": implementation,
        "training_history": history,
        "model_state_dict": {"weight": torch.tensor([1.0])},
    }
    torch.save(checkpoint_payload, checkpoint_path)
    result = {
        "status": "complete",
        "seed": 20260711,
        "protocol_sha256": stable_sha256(protocol_bytes),
        "calibration_manifest_sha256": stable_sha256(calibration_bytes),
        "execution_config": execution_config,
        "implementation": implementation,
        "checkpoint": {
            "path": checkpoint_path.name,
            "sha256": stable_sha256(checkpoint_path.read_bytes()),
        },
        "training_history": history,
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
        rounding_addendum_path=rounding_addendum_path,
        output_path=output_path,
    )

    assert seal["checks"]["all_hashes_match"] is True
    assert seal["checks"]["validation_only"] is True
    assert seal["test_relevance_labels_opened"] is False

    distance_path = distance_paths[0]
    original_distance_bytes = distance_path.read_bytes()
    distance_path.write_bytes(b"tampered-distance")
    with pytest.raises(ValueError, match="distance matrix hash mismatch"):
        seal_seed_result(
            result_path=result_path,
            protocol_path=protocol_path,
            calibration_manifest_path=calibration_path,
            gate0_path=gate0_path,
            rounding_addendum_path=rounding_addendum_path,
            output_path=tmp_path / "tampered-seal.json",
        )

    distance_path.write_bytes(original_distance_bytes)
    result["test_relevance_labels_opened"] = True
    result_path.write_bytes(canonical_json_bytes(result))
    with pytest.raises(ValueError, match="forbidden test access"):
        seal_seed_result(
            result_path=result_path,
            protocol_path=protocol_path,
            calibration_manifest_path=calibration_path,
            gate0_path=gate0_path,
            rounding_addendum_path=rounding_addendum_path,
            output_path=tmp_path / "forbidden-test-seal.json",
        )
