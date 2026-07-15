from __future__ import annotations

import pytest

from geometry_profile_research.codenet_stage_a_inference import (
    analyze_confirmatory_test,
    derive_cluster_bootstrap_seed,
    seed_averaged_problem_scores,
)


def _seed_payload(seed: int, offset: float = 0.0) -> dict:
    def scores(a: float, b: float) -> dict:
        return {"problem_A": a + offset, "problem_B": b + offset}

    return {
        "status": "complete",
        "seed": seed,
        "cells": {
            "EEE_zero_anchor": {"metrics": {"task_scores": scores(0.20, 0.30)}},
            "EEE_true_LCA": {"metrics": {"task_scores": scores(0.23, 0.33)}},
            "HEE_near_zero_true_LCA": {"metrics": {"task_scores": scores(0.24, 0.34)}},
            "HEE_c1_true_LCA": {"metrics": {"task_scores": scores(0.27, 0.37)}},
        },
    }


def test_bootstrap_seed_is_domain_separated_and_deterministic() -> None:
    beacon = "ab" * 64

    first = derive_cluster_bootstrap_seed(beacon, "domain-A")

    assert first == derive_cluster_bootstrap_seed(beacon, "domain-A")
    assert first != derive_cluster_bootstrap_seed(beacon, "domain-B")
    assert 0 <= first < 2**63 - 1


def test_seed_aggregation_averages_within_problem_not_over_seed_rows() -> None:
    payloads = (_seed_payload(1, 0.0), _seed_payload(2, 0.02))

    scores = seed_averaged_problem_scores(
        payloads,
        cell_id="EEE_true_LCA",
        expected_seeds=(1, 2),
    )

    assert scores == pytest.approx({"problem_A": 0.24, "problem_B": 0.34})


def test_confirmatory_analysis_uses_shared_problem_bootstrap_and_iut_rule() -> None:
    payloads = (_seed_payload(1, 0.0), _seed_payload(2, 0.02))

    result = analyze_confirmatory_test(
        payloads,
        selected_active_cell_id="HEE_c1_true_LCA",
        expected_seeds=(1, 2),
        beacon_output_hex="ab" * 64,
        bootstrap_domain="test-domain",
        bootstrap_resamples=200,
    )

    assert result["problem_count"] == 2
    assert result["contrasts"]["H1_EEE_true_LCA_minus_EEE_zero_anchor"][
        "point_estimate_delta_problem_macro_MAP_at_8"
    ] == pytest.approx(0.03)
    assert result["decisions"]["H1_confirmatory_success"] is True
    assert result["decisions"]["H3_confirmatory_success"] is True


def test_confirmatory_analysis_requires_exact_registered_seed_set() -> None:
    with pytest.raises(ValueError, match="registered model seed set"):
        analyze_confirmatory_test(
            (_seed_payload(1),),
            selected_active_cell_id="HEE_c1_true_LCA",
            expected_seeds=(1, 2),
            beacon_output_hex="ab" * 64,
            bootstrap_domain="test-domain",
            bootstrap_resamples=10,
        )
