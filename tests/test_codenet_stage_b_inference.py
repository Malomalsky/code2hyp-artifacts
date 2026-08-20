from __future__ import annotations

import pytest

from geometry_profile_research.codenet_stage_a_inference import analyze_stage_b_confirmatory_test


def _payload(seed: int, scores: dict[str, float]) -> dict:
    return {
        "status": "complete",
        "seed": seed,
        "cells": {
            cell: {"metrics": {"task_scores": {"A": score, "B": score, "C": score}}}
            for cell, score in scores.items()
        },
    }


def test_stage_b_inference_applies_the_frozen_sequence_and_difference_in_differences() -> None:
    prefix_scores = {
        "EEE_true_LCA": 0.20,
        "EEE_zero_anchor": 0.10,
        "HEE_near_zero_true_LCA": 0.25,
        "HEE_c1_true_LCA": 0.40,
        "HHH_c1_true_LCA": 0.35,
    }
    label_scores = {
        "EEE_true_LCA": 0.20,
        "HEE_c1_true_LCA": 0.25,
    }
    prefix = tuple(_payload(seed, prefix_scores) for seed in (11, 12))
    label = tuple(_payload(seed, label_scores) for seed in (11, 12))

    result = analyze_stage_b_confirmatory_test(
        prefix,
        label,
        selected_active_cell_id="HEE_c1_true_LCA",
        selected_hhh_cell_id="HHH_c1_true_LCA",
        expected_seeds=(11, 12),
        beacon_output_hex=bytes(range(64)).hex(),
        bootstrap_domain="stage-b-test",
        bootstrap_resamples=100,
        practical_delta=0.01,
    )

    assert result["fixed_sequence"]["H_B3_success"] is True
    assert result["contrasts"]["H_B3_difference_in_differences"]["point_estimate_delta_problem_macro_MAP_at_8"] == pytest.approx(0.15)
    assert result["H_B4_two_sided_descriptive"]["interval_excludes_zero"] is True
