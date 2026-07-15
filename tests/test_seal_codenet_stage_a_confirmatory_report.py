from __future__ import annotations

import json
from pathlib import Path

import pytest

from geometry_profile_research.codenet_eligibility import canonical_json_bytes, stable_sha256
from geometry_profile_research.codenet_stage_a_inference import analyze_confirmatory_test
from geometry_profile_research.codenet_stage_a_test_runner import aggregate_all_test_cells
from scripts.seal_codenet_stage_a_confirmatory_report import seal_confirmatory_report
from scripts.seal_codenet_stage_a_test_seed import TEST_RUNNER_COMMIT, TEST_RUNNER_TAG


CELL_IDS = (
    "EEE_true_LCA",
    "EEE_zero_anchor",
    "HEE_near_zero_true_LCA",
    "HEE_c0p1_true_LCA",
    "HEE_c0p3_true_LCA",
    "HEE_c1_true_LCA",
    "HEE_c3_true_LCA",
)


def _cell_metrics(seed: int, cell_index: int) -> dict:
    offset = 0.01 * cell_index + 0.001 * seed
    return {
        "task_scores": {"A": 0.2 + offset, "B": 0.4 + offset},
        "mrr": 0.3 + offset,
        "recall_at_1": 0.1 + offset,
        "recall_at_5": 0.5 + offset,
        "recall_at_10": 0.8 + offset,
        "mean_first_relevant_rank": 4.0 - offset,
    }


def test_confirmatory_report_seal_recomputes_aggregation_and_bootstrap(tmp_path: Path) -> None:
    execution_path = tmp_path / "execution.json"
    execution_path.write_text("execution", encoding="utf-8")
    model_path = tmp_path / "model.json"
    model_path.write_bytes(canonical_json_bytes({"encoder_training": {"model_seeds": [1, 2]}}))
    inference_protocol = {
        "bootstrap": {
            "resamples": 100,
            "rng_seed_derivation": {"domain": "test-domain"},
            "two_sided_interval": {"lower_quantile": 0.025, "upper_quantile": 0.975},
        },
        "decision_rules": {"minimum_practically_significant_delta_MAP_at_8": 0.01},
    }
    inference_path = tmp_path / "inference.json"
    inference_path.write_bytes(canonical_json_bytes(inference_protocol))
    registration = {"nist_randomness_beacon": {"output_value_hex": "00" * 64}}
    registration_path = tmp_path / "registration.json"
    registration_path.write_bytes(canonical_json_bytes(registration))
    selection = {
        "selected_active_curvature": 0.1,
        "selected_cell_id": "HEE_c0p1_true_LCA",
    }
    selection_path = tmp_path / "selection.json"
    selection_path.write_bytes(canonical_json_bytes(selection))
    selection_seal_path = tmp_path / "selection_seal.json"
    selection_seal_path.write_text("selection seal", encoding="utf-8")
    implementation = {
        "repository": "https://github.com/Malomalsky/code2hyp-artifacts",
        "commit": TEST_RUNNER_COMMIT,
        "tag": TEST_RUNNER_TAG,
        "tracked_worktree_clean": True,
    }
    materialization_path = tmp_path / "test_materialization_manifest.json"
    materialization_path.write_bytes(canonical_json_bytes({"implementation": implementation}))

    seed_payloads = []
    test_seed_inputs = []
    for seed in (1, 2):
        payload = {
            "status": "complete",
            "seed": seed,
            "cells": {
                cell_id: {"metrics": _cell_metrics(seed, cell_index)}
                for cell_index, cell_id in enumerate(CELL_IDS)
            },
        }
        seed_payloads.append(payload)
        result_path = tmp_path / f"seed_{seed}_test.json"
        result_path.write_bytes(canonical_json_bytes(payload))
        seal = {
            "schema_version": "code2hyp-stage-a-test-seed-seal-v1",
            "inputs": {"result": {"sha256": stable_sha256(result_path.read_bytes())}},
            "checks": {
                "all_seven_cells_present": True,
                "all_distance_hashes_match": True,
                "all_distance_matrices_are_finite_float64": True,
                "all_metrics_recomputed_from_distances": True,
                "registered_test_cardinalities_match": True,
            },
        }
        seal_path = tmp_path / f"seed_{seed}_test_seal.json"
        seal_path.write_bytes(canonical_json_bytes(seal))
        test_seed_inputs.append(
            {
                "seed": seed,
                "path": result_path.name,
                "sha256": stable_sha256(result_path.read_bytes()),
            }
        )

    confirmatory = analyze_confirmatory_test(
        seed_payloads,
        selected_active_cell_id="HEE_c0p1_true_LCA",
        expected_seeds=(1, 2),
        beacon_output_hex="00" * 64,
        bootstrap_domain="test-domain",
        bootstrap_resamples=100,
        practical_delta=0.01,
    )
    report = {
        "schema_version": "code2hyp-stage-a-confirmatory-test-v1",
        "status": "complete",
        "implementation": implementation,
        "inputs": {
            "test_execution_protocol_sha256": stable_sha256(execution_path.read_bytes()),
            "model_analysis_protocol_sha256": stable_sha256(model_path.read_bytes()),
            "test_inference_protocol_sha256": stable_sha256(inference_path.read_bytes()),
            "registration_sha256": stable_sha256(registration_path.read_bytes()),
            "validation_selection_sha256": stable_sha256(selection_path.read_bytes()),
            "validation_selection_seal_sha256": stable_sha256(selection_seal_path.read_bytes()),
            "test_materialization_manifest_sha256": stable_sha256(materialization_path.read_bytes()),
            "test_seed_results": test_seed_inputs,
        },
        "opening_count": 1,
        "selected_active_curvature": 0.1,
        "selected_cell_id": "HEE_c0p1_true_LCA",
        "all_planned_cells": aggregate_all_test_cells(seed_payloads, expected_seeds=(1, 2)),
        "confirmatory_inference": confirmatory,
        "test_program_ids_materialized": True,
        "test_relevance_labels_opened": True,
        "test_retrieval_metrics_computed": True,
    }
    report_path = tmp_path / "confirmatory_test_report.json"
    report_path.write_bytes(canonical_json_bytes(report))
    output_path = tmp_path / "confirmatory_test_report_seal.json"

    seal = seal_confirmatory_report(
        report_path=report_path,
        test_execution_protocol_path=execution_path,
        model_protocol_path=model_path,
        inference_protocol_path=inference_path,
        registration_path=registration_path,
        validation_selection_path=selection_path,
        validation_selection_seal_path=selection_seal_path,
        test_materialization_manifest_path=materialization_path,
        output_path=output_path,
    )
    assert seal["checks"]["cluster_bootstrap_recomputed"] is True

    report["confirmatory_inference"]["decisions"]["H1_confirmatory_success"] = True
    report_path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError, match="frozen-rule recomputation"):
        seal_confirmatory_report(
            report_path=report_path,
            test_execution_protocol_path=execution_path,
            model_protocol_path=model_path,
            inference_protocol_path=inference_path,
            registration_path=registration_path,
            validation_selection_path=selection_path,
            validation_selection_seal_path=selection_seal_path,
            test_materialization_manifest_path=materialization_path,
            output_path=tmp_path / "tampered_seal.json",
        )
