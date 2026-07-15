from __future__ import annotations

import pytest

from geometry_profile_research.codenet_stage_a_identifiability import (
    program_identifiability_diagnostics,
    summarize_identifiability_diagnostics,
)
from geometry_profile_research.raw_ast import RawAstTree


def _tree() -> RawAstTree:
    return RawAstTree.from_edges(
        root_id=0,
        edges=((0, 1), (0, 2), (1, 3), (1, 4), (2, 5), (2, 6)),
        labels={
            0: "Method",
            1: "IfStatement",
            2: "ReturnStatement",
            3: "Identifier",
            4: "Literal",
            5: "Identifier",
            6: "Literal",
        },
    )


def test_label_only_audit_counts_node_lca_and_path_signature_collisions() -> None:
    result = program_identifiability_diagnostics(_tree())

    assert result["node_count"] == 7
    assert result["distinct_node_signature_count"] == 5
    assert result["colliding_node_count"] == 2
    assert result["node_collision_rate"] == pytest.approx(2 / 7)
    assert result["distinct_true_lca_node_count"] == 3
    assert result["distinct_lca_signature_count"] == 3
    assert result["selected_path_object_count"] == 6
    assert result["distinct_path_signature_count"] == 5
    assert result["path_object_collision_rate"] == pytest.approx(1 / 6)


def test_prefix_context_removes_toy_encoder_input_collisions() -> None:
    result = program_identifiability_diagnostics(
        _tree(),
        node_input_mode="label_depth_prefix",
    )

    assert result["distinct_node_signature_count"] == 7
    assert result["colliding_node_count"] == 0
    assert result["distinct_path_signature_count"] == 6
    assert result["colliding_path_object_count"] == 0


def test_identifiability_summary_uses_micro_and_equal_program_macro_aggregation() -> None:
    first = program_identifiability_diagnostics(_tree())
    second = program_identifiability_diagnostics(_tree(), node_input_mode="label_depth_prefix")

    summary = summarize_identifiability_diagnostics((first, second))

    assert summary["program_count"] == 2
    assert summary["micro_counts"]["node_count"] == 14
    assert summary["micro_rates"]["node_collision_rate"] == pytest.approx(1 / 7)
    assert summary["program_macro_rates"]["node_collision_rate"] == pytest.approx(1 / 7)
    assert summary["program_rate_quantiles"]["path_object_collision_rate"]["q050"] == pytest.approx(1 / 12)
