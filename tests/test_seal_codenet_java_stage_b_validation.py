from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import pytest
import torch

from geometry_profile_research.codenet_eligibility import canonical_json_bytes, stable_sha256
from geometry_profile_research.codenet_stage_a_evaluation import summarize_problem_macro_retrieval
from geometry_profile_research.codenet_stage_a_runner import all_role_curvature_cell_id, curvature_cell_id
from scripts.run_codenet_java_stage_b_validation import RUNNER_TAG
from scripts.seal_codenet_java_stage_b_validation import seal_stage_b_seed_result


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_stage_b_seed_seal_recomputes_metrics_and_rejects_tampering(tmp_path: Path) -> None:
    design = json.loads((PROJECT_ROOT / "configs/codenet_java_stage_b_draft_v1.json").read_text())
    design["eligibility"]["primary_role_upper_bound"]["validation_clusters"] = 1
    design["freeze"]["implementation_commit"] = "a" * 40
    design["freeze"]["container_digest"] = "sha256:" + "b" * 64
    design_sha = stable_sha256(canonical_json_bytes(design))
    calibration_sha = "c" * 64

    rows = [
        {"role": role, "source_relpath": f"{role}/{index}.java", "cluster_id": "cluster-A"}
        for role in ("query", "gallery")
        for index in range(8)
    ]
    validation_path = tmp_path / "validation_programs.jsonl"
    validation_path.write_bytes(b"".join(canonical_json_bytes(row) for row in rows))

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
    active = tuple(float(value) for value in design["geometry"]["active_curvature_candidates"])
    cell_ids = {
        "EEE_true_LCA",
        "EEE_zero_anchor",
        "HEE_near_zero_true_LCA",
        *(curvature_cell_id(value) for value in active),
        *(all_role_curvature_cell_id(value) for value in active),
    }
    distance_contract = {
        "path": distance_path.name,
        "shape": [8, 8],
        "dtype": "float64",
        "sha256": stable_sha256(distance_path.read_bytes()),
        "minimum": 0.0,
        "maximum": 0.0,
        "negative_count": 0,
    }
    execution_config = _execution_config(design)
    checkpoint_path = tmp_path / "seed_7_encoder.pt"
    torch.save(
        {
            "seed": 7,
            "execution_config": execution_config,
            "training_metadata": {"compute_device": design["encoder_training"]["compute_device"]},
            "model_state_dict": {"weight": torch.tensor([1.0])},
        },
        checkpoint_path,
    )
    result = {
        "status": "complete",
        "seed": 7,
        "protocol_sha256": design_sha,
        "calibration_manifest_sha256": calibration_sha,
        "execution_config": execution_config,
        "implementation": {
            "commit": design["freeze"]["implementation_commit"],
            "tag": RUNNER_TAG,
            "container_digest": design["freeze"]["container_digest"],
        },
        "checkpoint": {"path": checkpoint_path.name, "sha256": stable_sha256(checkpoint_path.read_bytes())},
        "cells": {
            cell_id: {"metrics": metrics, "distance_matrix": distance_contract}
            for cell_id in cell_ids
        },
    }
    result_path = tmp_path / "seed_7_validation.json"
    result_path.write_bytes(canonical_json_bytes(result))

    seal = seal_stage_b_seed_result(
        result_path=result_path,
        design=design,
        design_sha256=design_sha,
        calibration_manifest_sha256=calibration_sha,
        validation_programs_path=validation_path,
        model="prefix",
        output_path=tmp_path / "seed_7_validation_seal.json",
    )
    assert seal["checks"]["all_metrics_recomputed_from_distance_matrices"] is True

    result["cells"]["EEE_true_LCA"]["metrics"]["problem_macro_map_at_r"] = 0.0
    result_path.write_bytes(canonical_json_bytes(result))
    with pytest.raises(ValueError, match="metrics do not match"):
        seal_stage_b_seed_result(
            result_path=result_path,
            design=design,
            design_sha256=design_sha,
            calibration_manifest_sha256=calibration_sha,
            validation_programs_path=validation_path,
            model="prefix",
            output_path=tmp_path / "tampered_seal.json",
        )


def _execution_config(design: dict) -> dict:
    encoder = design["encoder_training"]
    calibration = design["train_only_calibration"]
    transport = design["transport"]
    sampling = design["sampling"]
    return {
        "dim": encoder["dimension_per_role"],
        "epochs": encoder["epochs"],
        "batch_size": encoder["batch_size_programs"],
        "learning_rate": encoder["learning_rate"],
        "gradient_clip_norm": encoder["gradient_clip_global_norm"],
        "lambda_edge": encoder["loss"]["edge_length_weight"],
        "lambda_gromov": encoder["loss"]["soft_gromov_LCA_distortion_weight"],
        "lambda_branch": encoder["loss"]["branch_length_weight"],
        "max_paths": sampling["paths_per_program"],
        "max_ball_fraction": calibration["maximum_ball_radius_fraction"],
        "active_curvatures": design["geometry"]["active_curvature_candidates"],
        "near_zero_curvature": design["geometry"]["near_zero_curvature"],
        "sinkhorn_kappa": calibration["sinkhorn_kappa"],
        "sinkhorn_iterations": transport["sinkhorn_iterations"],
        "projection_iterations": transport["projection_iterations_max"],
        "marginal_tolerance": transport["maximum_marginal_residual"],
        "query_batch_size": transport["query_batch_size"],
        "gallery_batch_size": transport["gallery_batch_size"],
        "torch_num_threads": encoder["torch_num_threads_per_process"],
        "compute_device": encoder["compute_device"],
        "fit_all_roles_to_active_ball": True,
        "node_input_mode": "label_depth_prefix",
        "include_all_role_hyperbolic": True,
    }
