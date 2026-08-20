from __future__ import annotations

import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from geometry_profile_research.codenet_eligibility import canonical_json_bytes, stable_sha256
from geometry_profile_research.codenet_stage_a_evaluation import summarize_problem_macro_retrieval
from geometry_profile_research.codenet_stage_a_inference import analyze_stage_b_confirmatory_test
from geometry_profile_research.codenet_stage_a_test_runner import aggregate_all_test_cells
from geometry_profile_research.codenet_stage_b import (
    STAGE_B_TEST_MATERIALIZATION_SCHEMA,
    _validate_stage_b_selection_for_test,
    validate_stage_b_registration,
)


RESULT_SCHEMA = "code2hyp-codenet-java-stage-b-test-seed-v1"


def seal_stage_b_test_seed(
    *,
    result_path: Path,
    test_programs_path: Path,
    materialization_manifest_path: Path,
    model: str,
    expected_seed: int,
    expected_cell_ids: Sequence[str],
    expected_validation_result_sha256: str,
    expected_validation_seed_seal_sha256: str,
    design_sha256: str,
    output_path: Path,
) -> dict[str, Any]:
    """Recompute one Stage B test seed from its full distance matrices."""

    if model not in {"prefix", "label_only"}:
        raise ValueError("Stage B test model must be prefix or label_only")
    result_bytes = result_path.read_bytes()
    result = json.loads(result_bytes)
    if result.get("schema_version") != RESULT_SCHEMA or result.get("status") != "complete":
        raise ValueError("Stage B test seed is incomplete or has an unexpected schema")
    if int(result.get("seed", -1)) != expected_seed:
        raise ValueError("Stage B test seed differs from the frozen sequence")
    identity = result["identity"]
    if identity.get("validation_result_sha256") != expected_validation_result_sha256:
        raise ValueError("Stage B test seed references an unexpected validation result")
    if identity.get("validation_seed_seal_sha256") != expected_validation_seed_seal_sha256:
        raise ValueError("Stage B test seed references an unexpected validation seal")
    if any(identity.get(key) != design_sha256 for key in (
        "test_execution_protocol_sha256",
        "test_runtime_addendum_sha256",
        "test_resumability_addendum_sha256",
        "relevance_identity_addendum_sha256",
    )):
        raise ValueError("Stage B test seed differs from the frozen design")
    if tuple(identity.get("test_cell_ids", ())) != tuple(expected_cell_ids):
        raise ValueError("Stage B test seed changed the frozen model-specific cells")
    if set(result["cells"]) != set(expected_cell_ids):
        raise ValueError("Stage B test result does not contain its exact planned cells")
    if any(result.get(flag) is not True for flag in (
        "test_program_ids_materialized",
        "test_relevance_labels_opened",
        "test_retrieval_metrics_computed",
    )):
        raise ValueError("Stage B test result does not record the completed opening")

    materialization_bytes = materialization_manifest_path.read_bytes()
    materialization = json.loads(materialization_bytes)
    if materialization.get("schema_version") != STAGE_B_TEST_MATERIALIZATION_SCHEMA:
        raise ValueError("unexpected Stage B test materialization schema")
    if identity.get("test_materialization_manifest_sha256") != stable_sha256(materialization_bytes):
        raise ValueError("Stage B test seed used a different materialization")
    if identity.get("implementation") != materialization.get("implementation"):
        raise ValueError("Stage B test seed and materialization implementations differ")
    if (
        materialization.get("opening", {}).get("ordinal") != 1
        or materialization.get("test_program_ids_materialized") is not True
        or materialization.get("test_relevance_labels_opened") is not True
        or materialization.get("test_retrieval_metrics_computed") is not False
    ):
        raise ValueError("Stage B test materialization does not represent the single pre-metric opening")
    test_programs_sha = stable_sha256(test_programs_path.read_bytes())
    materialized_test_programs_sha = next(
        (
            str(artifact["sha256"])
            for artifact in materialization.get("artifacts", ())
            if artifact.get("path") == "test_programs.jsonl"
        ),
        None,
    )
    if test_programs_sha != materialized_test_programs_sha:
        raise ValueError("Stage B test programs differ from the materialization manifest")

    rows = [json.loads(line) for line in test_programs_path.read_bytes().splitlines() if line]
    query = [row for row in rows if row["role"] == "query"]
    gallery = [row for row in rows if row["role"] == "gallery"]
    query_ids = tuple(str(row["source_relpath"]) for row in query)
    query_clusters = tuple(str(row["cluster_id"]) for row in query)
    gallery_ids = tuple(str(row["source_relpath"]) for row in gallery)
    gallery_clusters = tuple(str(row["cluster_id"]) for row in gallery)
    relevant_count = int(materialization["sampling_summary"]["test_gallery"]) // int(
        materialization["sampling_summary"]["test_clusters"]
    )
    expected_shape = [len(query), len(gallery)]
    artifacts = []
    for cell_id in expected_cell_ids:
        cell = result["cells"][cell_id]
        metadata = cell["distance_matrix"]
        if metadata["shape"] != expected_shape or metadata["dtype"] != "float64":
            raise ValueError(f"Stage B test distance contract mismatch: {cell_id}")
        distance_path = result_path.parent / str(metadata["path"])
        distance_bytes = distance_path.read_bytes()
        if stable_sha256(distance_bytes) != str(metadata["sha256"]):
            raise ValueError(f"Stage B test distance hash mismatch: {cell_id}")
        distances = torch.load(distance_path, map_location="cpu", weights_only=True)
        if not isinstance(distances, torch.Tensor) or distances.dtype != torch.float64:
            raise ValueError(f"Stage B test distance tensor must be float64: {cell_id}")
        if list(distances.shape) != expected_shape or not bool(torch.isfinite(distances).all()):
            raise ValueError(f"Stage B test distance tensor is malformed: {cell_id}")
        observed = {
            "minimum": float(distances.min()),
            "maximum": float(distances.max()),
            "negative_count": int((distances < 0.0).sum()),
        }
        if any(not math.isclose(observed[key], float(metadata[key]), rel_tol=0.0, abs_tol=0.0) for key in ("minimum", "maximum")):
            raise ValueError(f"Stage B test distance extrema mismatch: {cell_id}")
        if observed["negative_count"] != int(metadata["negative_count"]):
            raise ValueError(f"Stage B test negative-distance count mismatch: {cell_id}")
        recomputed = summarize_problem_macro_retrieval(
            distances,
            query_ids=query_ids,
            query_cluster_ids=query_clusters,
            gallery_ids=gallery_ids,
            gallery_cluster_ids=gallery_clusters,
            r=relevant_count,
        )
        if canonical_json_bytes(asdict(recomputed)) != canonical_json_bytes(cell["metrics"]):
            raise ValueError(f"Stage B test metrics do not match stored distances: {cell_id}")
        artifacts.append({"cell_id": cell_id, "path": distance_path.name, "sha256": stable_sha256(distance_bytes)})

    seal = {
        "schema_version": "code2hyp-codenet-java-stage-b-test-seed-seal-v1",
        "model": model,
        "seed": expected_seed,
        "inputs": {
            "result": {"path": result_path.name, "sha256": stable_sha256(result_bytes)},
            "test_programs_sha256": test_programs_sha,
            "test_materialization_manifest_sha256": stable_sha256(materialization_bytes),
        },
        "cells": artifacts,
        "checks": {
            "exact_model_specific_cell_set": True,
            "all_distance_hashes_match": True,
            "all_distances_are_finite_float64": True,
            "all_metrics_recomputed_from_distances": True,
        },
        "test_program_ids_materialized": True,
        "test_relevance_labels_opened": True,
        "test_retrieval_metrics_computed": True,
    }
    _write_once_or_verify(output_path, canonical_json_bytes(seal))
    return seal


def seal_stage_b_confirmatory_report(
    *,
    design_path: Path,
    registration_path: Path,
    selection_path: Path,
    selection_seal_path: Path,
    materialization_manifest_path: Path,
    test_programs_path: Path,
    validation_output_dir: Path,
    test_output_dir: Path,
    report_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Recompute and seal the complete Stage B confirmatory report."""

    design_bytes = design_path.read_bytes()
    design = json.loads(design_bytes)
    design_sha = stable_sha256(design_bytes)
    registration_bytes = registration_path.read_bytes()
    registration = json.loads(registration_bytes)
    validate_stage_b_registration(
        design=design,
        registration=registration,
        design_bytes=design_bytes,
    )
    selection_bytes = selection_path.read_bytes()
    selection = json.loads(selection_bytes)
    selection_seal_bytes = selection_seal_path.read_bytes()
    selection_seal = json.loads(selection_seal_bytes)
    _validate_stage_b_selection_for_test(
        design=design,
        design_sha256=design_sha,
        selection=selection,
        selection_bytes=selection_bytes,
        seal=selection_seal,
    )
    materialization_bytes = materialization_manifest_path.read_bytes()
    materialization = json.loads(materialization_bytes)
    if (
        materialization.get("selected_active_curvature") != selection.get("selected_active_curvature")
        or materialization.get("selected_prefix_HEE_cell_id") != selection.get("selected_prefix_HEE_cell_id")
        or materialization.get("selected_prefix_HHH_cell_id") != selection.get("selected_prefix_HHH_cell_id")
        or materialization.get("test_cell_plan") != selection.get("test_cell_plan")
    ):
        raise ValueError("Stage B test materialization changed the sealed seven-cell mapping")
    seeds = tuple(int(seed) for seed in design["encoder_training"]["model_seeds"])
    seed_inputs = {
        (str(row["model"]), int(row["seed"])): row
        for row in selection_seal["inputs"]["seeds"]
    }
    active_cell = str(selection["selected_prefix_HEE_cell_id"])
    hhh_cell = str(selection["selected_prefix_HHH_cell_id"])
    cells_by_model = {
        "prefix": ("EEE_true_LCA", "EEE_zero_anchor", "HEE_near_zero_true_LCA", active_cell, hhh_cell),
        "label_only": ("EEE_true_LCA", active_cell),
    }
    payloads: dict[str, list[dict[str, Any]]] = {"prefix": [], "label_only": []}
    sealed_inputs = []
    for model in ("prefix", "label_only"):
        for seed in seeds:
            seed_input = seed_inputs[(model, seed)]
            validation_result_path = validation_output_dir / str(seed_input["result_path"])
            validation_seed_seal_path = validation_output_dir / str(seed_input["seal_path"])
            if stable_sha256(validation_result_path.read_bytes()) != str(seed_input["result_sha256"]):
                raise ValueError("Stage B validation result differs from the selection seal")
            if stable_sha256(validation_seed_seal_path.read_bytes()) != str(seed_input["seal_sha256"]):
                raise ValueError("Stage B validation seed seal differs from the selection seal")
            result_path = test_output_dir / model / f"seed_{seed}_test.json"
            seal_path = test_output_dir / model / f"seed_{seed}_test_seal.json"
            seed_seal = seal_stage_b_test_seed(
                result_path=result_path,
                test_programs_path=test_programs_path,
                materialization_manifest_path=materialization_manifest_path,
                model=model,
                expected_seed=seed,
                expected_cell_ids=cells_by_model[model],
                expected_validation_result_sha256=str(seed_input["result_sha256"]),
                expected_validation_seed_seal_sha256=str(seed_input["seal_sha256"]),
                design_sha256=design_sha,
                output_path=seal_path,
            )
            payloads[model].append(json.loads(result_path.read_text(encoding="utf-8")))
            sealed_inputs.append(
                {
                    "model": model,
                    "seed": seed,
                    "result_sha256": seed_seal["inputs"]["result"]["sha256"],
                    "seal_sha256": stable_sha256(seal_path.read_bytes()),
                }
            )

    inference_config = design["inference"]
    inference = analyze_stage_b_confirmatory_test(
        payloads["prefix"],
        payloads["label_only"],
        selected_active_cell_id=active_cell,
        selected_hhh_cell_id=hhh_cell,
        expected_seeds=seeds,
        beacon_output_hex=str(registration["nist_randomness_beacon"]["output_value_hex"]),
        bootstrap_domain=str(inference_config["bootstrap_domain"]),
        bootstrap_resamples=int(inference_config["bootstrap_resamples"]),
        practical_delta=float(inference_config["minimum_practically_significant_delta_MAP_at_8"]),
    )
    aggregates = {
        model: aggregate_all_test_cells(values, expected_seeds=seeds)
        for model, values in payloads.items()
    }
    public_cells = {
        str(row["cell_id"]): aggregates[str(row["model"])][str(row["validation_cell_id"])]
        for row in selection["test_cell_plan"]
    }
    report_bytes = report_path.read_bytes()
    report = json.loads(report_bytes)
    if report.get("schema_version") != "code2hyp-codenet-java-stage-b-confirmatory-test-v1":
        raise ValueError("unexpected Stage B confirmatory report schema")
    if report.get("status") != "complete" or any(report.get(flag) is not True for flag in (
        "test_program_ids_materialized",
        "test_relevance_labels_opened",
        "test_retrieval_metrics_computed",
    )):
        raise ValueError("Stage B confirmatory report is incomplete")
    if canonical_json_bytes(report.get("all_seven_planned_cells")) != canonical_json_bytes(public_cells):
        raise ValueError("Stage B report cell aggregation differs from recomputation")
    if canonical_json_bytes(report.get("confirmatory_inference")) != canonical_json_bytes(inference):
        raise ValueError("Stage B report inference differs from frozen-rule recomputation")
    expected_inputs = {
        "design_sha256": design_sha,
        "registration_sha256": stable_sha256(registration_bytes),
        "validation_selection_sha256": stable_sha256(selection_bytes),
        "validation_selection_seal_sha256": stable_sha256(selection_seal_bytes),
        "test_materialization_manifest_sha256": stable_sha256(materialization_bytes),
    }
    if any(report.get("inputs", {}).get(key) != value for key, value in expected_inputs.items()):
        raise ValueError("Stage B report input hashes differ from frozen artifacts")
    expected_result_inputs = [
        {
            "model": row["model"],
            "seed": row["seed"],
            "path": f"{row['model']}/seed_{row['seed']}_test.json",
            "sha256": row["result_sha256"],
        }
        for row in sealed_inputs
    ]
    if report.get("inputs", {}).get("test_seed_results") != expected_result_inputs:
        raise ValueError("Stage B report does not bind every test seed result")
    if report.get("implementation") != materialization.get("implementation") or int(report.get("opening_count", 0)) != 1:
        raise ValueError("Stage B report implementation or opening count is invalid")

    seal = {
        "schema_version": "code2hyp-codenet-java-stage-b-confirmatory-test-seal-v1",
        "inputs": {
            "report": {"path": report_path.name, "sha256": stable_sha256(report_bytes)},
            "seed_results_and_seals": sealed_inputs,
        },
        "confirmatory_inference": inference,
        "checks": {
            "all_registered_model_seed_results_recomputed": len(sealed_inputs) == 2 * len(seeds),
            "all_seven_cells_reaggregated": True,
            "cluster_bootstrap_recomputed_from_registered_beacon": True,
            "single_test_opening": True,
        },
    }
    _write_once_or_verify(output_path, canonical_json_bytes(seal))
    return seal


def _write_once_or_verify(path: Path, content: bytes) -> None:
    if path.exists() and path.read_bytes() != content:
        raise FileExistsError(f"refusing to overwrite a different Stage B test seal: {path}")
    path.write_bytes(content)
