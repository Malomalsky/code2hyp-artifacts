from __future__ import annotations

import json
from pathlib import Path

import pytest

from geometry_profile_research.codenet_eligibility import canonical_json_bytes, stable_sha256
from geometry_profile_research.codenet_stage_a_runner import build_validation_selection_record
from scripts.seal_codenet_stage_a_validation_seed import RUNNER_COMMIT
from scripts.seal_codenet_stage_a_validation_selection import seal_validation_selection


def test_selection_seal_recomputes_rule_and_verifies_all_seed_seals(tmp_path: Path) -> None:
    source_protocol = Path(
        "configs/codenet_python800_stage_a_model_analysis_protocol_v1.json"
    )
    protocol = json.loads(source_protocol.read_text(encoding="utf-8"))
    protocol["encoder_training"]["model_seeds"] = [1, 2]
    protocol_path = tmp_path / "protocol.json"
    protocol_path.write_bytes(canonical_json_bytes(protocol))
    calibration_path = tmp_path / "calibration.json"
    calibration_path.write_bytes(canonical_json_bytes({"kind": "frozen-calibration"}))

    cell_ids = (
        "EEE_true_LCA",
        "EEE_zero_anchor",
        "HEE_near_zero_true_LCA",
        "HEE_c0p1_true_LCA",
        "HEE_c0p3_true_LCA",
        "HEE_c1_true_LCA",
        "HEE_c3_true_LCA",
    )
    payloads = []
    for seed in (1, 2):
        cells = {}
        for cell_index, cell_id in enumerate(cell_ids):
            cells[cell_id] = {
                "metrics": {
                    "task_scores": {
                        "problem_A": 0.10 * seed + 0.01 * cell_index,
                        "problem_B": 0.20 * seed + 0.01 * cell_index,
                    }
                }
            }
        payload = {
            "status": "complete",
            "seed": seed,
            "cells": cells,
            "test_program_ids_materialized": False,
            "test_relevance_labels_opened": False,
            "test_retrieval_metrics_computed": False,
        }
        payloads.append(payload)
        result_path = tmp_path / f"seed_{seed}_validation.json"
        result_path.write_bytes(canonical_json_bytes(payload))
        seed_seal = {
            "seed": seed,
            "implementation": {"commit": RUNNER_COMMIT},
            "inputs": {"result": {"sha256": stable_sha256(result_path.read_bytes())}},
            "checks": {"validation_only": True},
            "test_program_ids_materialized": False,
            "test_relevance_labels_opened": False,
            "test_retrieval_metrics_computed": False,
        }
        (tmp_path / f"seed_{seed}_validation_seal.json").write_bytes(
            canonical_json_bytes(seed_seal)
        )

    selection = build_validation_selection_record(
        payloads,
        protocol_bytes=protocol_path.read_bytes(),
        calibration_manifest_bytes=calibration_path.read_bytes(),
    )
    selection_path = tmp_path / "validation_selection_record.json"
    selection_path.write_bytes(canonical_json_bytes(selection))
    output_path = tmp_path / "validation_selection_record_seal.json"

    seal = seal_validation_selection(
        selection_path=selection_path,
        protocol_path=protocol_path,
        calibration_manifest_path=calibration_path,
        output_path=output_path,
    )

    assert seal["checks"]["registered_seed_set_complete"] is True
    assert seal["checks"]["selection_recomputed_from_frozen_rule"] is True
    assert len(seal["inputs"]["seeds"]) == 2

    selection["selected_active_curvature"] = 0.1
    selection_path.write_bytes(canonical_json_bytes(selection))
    with pytest.raises(ValueError, match="frozen-rule recomputation"):
        seal_validation_selection(
            selection_path=selection_path,
            protocol_path=protocol_path,
            calibration_manifest_path=calibration_path,
            output_path=tmp_path / "tampered-selection-seal.json",
        )
