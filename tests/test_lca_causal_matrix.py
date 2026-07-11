from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from geometry_profile_research.constant_curvature import ProductMeasure
from scripts.run_dta_level_b_c_retrieval import TaskSource
from scripts.run_lca_causal_matrix import (
    _average_precision_at_r,
    _matched_role_weights,
    _permute_anchors_between_measures,
    run_lca_causal_matrix,
)


def test_lca_causal_matrix_freezes_encoder_and_normalization_across_cells(tmp_path: Path) -> None:
    task_a = _write_task(tmp_path, "task-a", "a")
    task_b = _write_task(tmp_path, "task-b", "b")
    output = tmp_path / "lca-causal.json"

    payload = run_lca_causal_matrix(
        tasks=(TaskSource("task-a", task_a), TaskSource("task-b", task_b)),
        output_path=output,
        train_per_task=2,
        query_per_task=1,
        gallery_per_task=2,
        max_methods_per_task=5,
        max_paths=8,
        epochs=1,
        sinkhorn_iterations=3,
        sinkhorn_projection_iterations=512,
        hard_negatives_per_query=2,
    )

    loaded = json.loads(output.read_text(encoding="utf-8"))
    assert payload["status"] == "complete"
    assert loaded["completed_runs"] == loaded["expected_runs"] == 11
    assert loaded["config"]["neutral_encoder"] is True
    assert loaded["config"]["retrieval_loss_weight"] == 0.0
    assert loaded["config"]["training_history"][0]["retrieval_weight"] == 0.0
    assert loaded["config"]["study_stage"] == "pilot"
    assert loaded["config"]["normalization"]["source"] == "true_lca_train_split_only"
    assert loaded["config"]["normalization"]["endpoint_weight_constraint"] == "w_start = w_end"
    shuffle_diagnostics = loaded["config"]["treatment_diagnostics"]["depth_matched_shuffled"]
    assert shuffle_diagnostics["train"]["path_count"] > 0
    assert 0.0 <= shuffle_diagnostics["train"]["fallback_fraction"] <= 1.0
    assert len(loaded["config"]["role_point_scales"]) == 3
    assert all(0.0 < value <= 1.0 for value in loaded["config"]["role_point_scales"])

    by_id = {row["cell_id"]: row for row in loaded["runs"]}
    assert set(by_id) == {
        "EEE__true_lca__measure",
        "EEE__zero_anchor__measure",
        "EEE__root_anchor__measure",
        "EEE__depth_matched_shuffled__measure",
        "EEE__program_shuffled_lca__measure",
        "EEE__full_path_no_explicit_lca__measure",
        "EEE__endpoint_only__measure",
        "EEE_concat__true_lca__measure",
        "HEE_near_zero__true_lca__measure",
        "HEE__true_lca__measure",
        "HHH__true_lca__measure",
    }
    euclidean_weights = by_id["EEE__true_lca__measure"]["factor_weights"]
    canonical_weights = loaded["config"]["canonical_factor_weights"]
    assert euclidean_weights[1] == euclidean_weights[2]
    assert euclidean_weights == pytest.approx([4.0 * weight for weight in canonical_weights])
    endpoint_only = by_id["EEE__endpoint_only__measure"]["factor_weights"]
    assert endpoint_only[0] == 0.0
    assert endpoint_only[1:] == euclidean_weights[1:]
    assert by_id["HEE__true_lca__measure"]["factor_weights"] == pytest.approx(
        [canonical_weights[0], euclidean_weights[1], euclidean_weights[2]]
    )
    assert by_id["HHH__true_lca__measure"]["factor_weights"] == pytest.approx(canonical_weights)
    assert by_id["EEE__zero_anchor__measure"]["map_at_r"] == pytest.approx(
        by_id["EEE__endpoint_only__measure"]["map_at_r"]
    )
    assert by_id["EEE_concat__true_lca__measure"]["map_at_r"] == pytest.approx(
        by_id["EEE__true_lca__measure"]["map_at_r"]
    )
    assert by_id["HEE__true_lca__measure"]["effective_lca_sectional_curvature"] < 0.0
    assert {contrast["name"] for contrast in loaded["contrasts"]} == {
        "H1_true_lca_vs_zero_anchor",
        "true_lca_vs_endpoint_only",
        "true_lca_vs_program_shuffled_lca",
        "true_lca_vs_full_path_pool",
        "product_vs_equal_capacity_concat_identity",
        "HEE_vs_EEE",
        "HEE_vs_near_zero_HEE",
        "HEE_vs_HHH",
    }


def test_average_precision_at_r_uses_fixed_relevant_denominator() -> None:
    assert _average_precision_at_r((True, False, True, True), total_positives=3, r=3) == pytest.approx(
        (1.0 + 2.0 / 3.0) / 3.0
    )
    assert _average_precision_at_r((False, True), total_positives=1, r=8) == pytest.approx(0.5)


def test_role_weights_match_the_standard_poincare_zero_curvature_limit() -> None:
    canonical = (1.0, 2.0, 2.0)

    assert _matched_role_weights(canonical, factor_curvatures=(0.0, 0.0, 0.0)) == (4.0, 8.0, 8.0)
    assert _matched_role_weights(canonical, factor_curvatures=(1e-4, 0.0, 0.0)) == (1.0, 8.0, 8.0)
    assert _matched_role_weights(canonical, factor_curvatures=(1.0, 1.0, 1.0)) == canonical


def test_confirmatory_stage_rejects_pilot_budget(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="confirmatory stage requires"):
        run_lca_causal_matrix(
            tasks=(),
            output_path=tmp_path / "invalid.json",
            study_stage="confirmatory",
            max_paths=16,
            ap_at_r=2,
            gallery_per_task=2,
        )


def test_program_shuffle_preserves_anchor_marginal_but_moves_anchors_across_measures() -> None:
    measures = [
        ProductMeasure(
            points=torch.tensor([[[float(10 * group + path)], [100.0 + path], [200.0 + path]]]),
            mass=torch.ones(1),
        )
        for group, path in ((0, 0), (1, 0), (2, 0))
    ]

    shuffled = _permute_anchors_between_measures(measures, seed=7)

    original_anchors = sorted(float(measure.points[0, 0, 0]) for measure in measures)
    shuffled_anchors = sorted(float(measure.points[0, 0, 0]) for measure in shuffled)
    assert shuffled_anchors == original_anchors
    for original, changed in zip(measures, shuffled):
        assert float(original.points[0, 0, 0]) != float(changed.points[0, 0, 0])
        torch.testing.assert_close(original.points[:, 1:], changed.points[:, 1:])


def _write_task(root: Path, task_name: str, prefix: str) -> Path:
    task_dir = root / task_name
    task_dir.mkdir()
    bodies = (
        "total = 0\n    for x in xs:\n        total += x\n    return total\n",
        "acc = 1\n    for x in xs:\n        acc *= (x + 1)\n    return acc\n",
        "out = []\n    for x in xs:\n        out.append(x + 1)\n    return out\n",
        "count = 0\n    for x in xs:\n        if x > 0:\n            count += 1\n    return count\n",
        "best = None\n    for x in xs:\n        if best is None or x > best:\n            best = x\n    return best\n",
    )
    for index, body in enumerate(bodies):
        (task_dir / f"{prefix}_{index}.py").write_text(
            f"def f_{prefix}_{index}(xs):\n    {body}",
            encoding="utf-8",
        )
    return task_dir
