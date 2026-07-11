from __future__ import annotations

import json
from pathlib import Path

import torch

from geometry_profile_research.constant_curvature import ProductMeasure
from scripts.run_dta_factor_matrix import TaskSource, _normalized_factor_weights, _train_factor_cost_scales, run_dta_factor_matrix


def test_factor_matrix_runs_all_requested_cells_with_disjoint_split(tmp_path: Path) -> None:
    task_a = _write_task(tmp_path, "task-a", "a")
    task_b = _write_task(tmp_path, "task-b", "b")
    output = tmp_path / "factor.json"

    payload = run_dta_factor_matrix(
        tasks=(TaskSource("task-a", task_a), TaskSource("task-b", task_b)),
        output_path=output,
        geometries=("E", "H_1"),
        path_object_modes=("single_point", "lca_product"),
        method_aggregations=("centroid", "measure"),
        train_per_task=2,
        query_per_task=1,
        gallery_per_task=2,
        max_methods_per_task=5,
        max_paths=8,
        sinkhorn_iterations=4,
        sinkhorn_projection_iterations=512,
    )

    loaded = json.loads(output.read_text(encoding="utf-8"))
    assert payload["status"] == "complete"
    assert loaded["expected_runs"] == 8
    assert len(loaded["runs"]) == 8
    assert {run["path_object_mode"] for run in loaded["runs"]} == {"single_point", "lca_product"}
    assert {run["method_aggregation"] for run in loaded["runs"]} == {"centroid", "measure"}
    assert {run["geometry"] for run in loaded["runs"]} == {"E", "H_1"}
    first_run = loaded["runs"][0]
    assert "embedding_norm_diagnostics" in first_run
    assert "cost_component_diagnostics" in first_run
    assert "scaled_norm_max" in first_run["embedding_norm_diagnostics"]
    assert "point_cost_share" in first_run["cost_component_diagnostics"]
    train = set(loaded["split"]["train_ids"])
    query = set(loaded["split"]["query_ids"])
    gallery = set(loaded["split"]["gallery_ids"])
    assert not (train & query)
    assert not (train & gallery)
    assert not (query & gallery)


def test_factor_matrix_can_train_geometry_aware_encoder_per_cell(tmp_path: Path) -> None:
    task_a = _write_task(tmp_path, "task-a", "a")
    task_b = _write_task(tmp_path, "task-b", "b")
    output = tmp_path / "geometry-aware-factor.json"

    payload = run_dta_factor_matrix(
        tasks=(TaskSource("task-a", task_a), TaskSource("task-b", task_b)),
        output_path=output,
        geometries=("H_1",),
        path_object_modes=("lca_product",),
        method_aggregations=("measure",),
        train_per_task=2,
        query_per_task=1,
        gallery_per_task=1,
        max_methods_per_task=4,
        max_paths=6,
        epochs=1,
        sinkhorn_iterations=3,
        sinkhorn_projection_iterations=512,
        encoder_policy="geometry_aware",
    )

    run = payload["runs"][0]
    assert payload["config"]["encoder_policy"] == "geometry_aware"
    assert run["encoder_policy"] == "geometry_aware"
    assert run["trained_manifold"] == "poincare"
    assert run["trained_curvature"] == 1.0
    assert run["point_scale"] == 1.0


def test_factor_matrix_can_sweep_side_weights_without_retraining_per_weight(tmp_path: Path) -> None:
    task_a = _write_task(tmp_path, "task-a", "a")
    task_b = _write_task(tmp_path, "task-b", "b")
    output = tmp_path / "side-weight-sweep.json"

    payload = run_dta_factor_matrix(
        tasks=(TaskSource("task-a", task_a), TaskSource("task-b", task_b)),
        output_path=output,
        geometries=("H_1",),
        path_object_modes=("lca_product",),
        method_aggregations=("measure",),
        train_per_task=2,
        query_per_task=1,
        gallery_per_task=1,
        max_methods_per_task=4,
        max_paths=6,
        epochs=1,
        sinkhorn_iterations=3,
        sinkhorn_projection_iterations=512,
        encoder_policy="shared_euclidean",
        side_weights=(0.0, 1.0),
    )

    loaded = json.loads(output.read_text(encoding="utf-8"))
    assert payload["status"] == "complete"
    assert loaded["expected_runs"] == 2
    assert len(loaded["runs"]) == 2
    assert loaded["config"]["side_weights"] == [0.0, 1.0]
    assert [run["side_weight"] for run in loaded["runs"]] == [0.0, 1.0]
    histories = [run["training_history"] for run in loaded["runs"]]
    assert histories[0] == histories[1]


def test_factor_matrix_can_audit_prespecified_cost_modes(tmp_path: Path) -> None:
    task_a = _write_task(tmp_path, "task-a", "a")
    task_b = _write_task(tmp_path, "task-b", "b")
    output = tmp_path / "cost-mode-audit.json"

    payload = run_dta_factor_matrix(
        tasks=(TaskSource("task-a", task_a), TaskSource("task-b", task_b)),
        output_path=output,
        geometries=("H_1",),
        path_object_modes=("lca_product",),
        method_aggregations=("measure",),
        train_per_task=2,
        query_per_task=1,
        gallery_per_task=1,
        max_methods_per_task=4,
        max_paths=6,
        epochs=1,
        sinkhorn_iterations=3,
        sinkhorn_projection_iterations=512,
        encoder_policy="shared_euclidean",
        cost_modes=("point_only", "side_only", "unnormalized_combined", "train_normalized_combined"),
    )

    loaded = json.loads(output.read_text(encoding="utf-8"))
    assert payload["status"] == "complete"
    assert loaded["expected_runs"] == 4
    assert loaded["config"]["reproducibility"]["python_random_seed"] == 20260625
    assert loaded["config"]["reproducibility"]["torch_manual_seed"] == 20260625
    assert loaded["config"]["cost_modes"] == ["point_only", "side_only", "unnormalized_combined", "train_normalized_combined"]
    by_mode = {run["cost_mode"]: run for run in loaded["runs"]}
    assert set(by_mode) == {"point_only", "side_only", "unnormalized_combined", "train_normalized_combined"}
    assert by_mode["point_only"]["side_weight"] == 0.0
    normalized = by_mode["train_normalized_combined"]
    assert all(weight >= 0.0 for weight in normalized["factor_weights"])
    assert normalized["cost_normalization"]["active_factor_count"] == sum(
        weight > 0.0 for weight in normalized["factor_weights"]
    )
    assert by_mode["train_normalized_combined"]["side_weight"] > 0.0
    assert "cost_normalization" in by_mode["train_normalized_combined"]
    assert by_mode["train_normalized_combined"]["cost_normalization"]["source"] == "train_split"
    diagnostics = by_mode["train_normalized_combined"]["retrieval_diagnostics"]
    assert "total_distance_side_expected_cost_spearman" in diagnostics
    assert "total_distance_point_expected_cost_spearman" in diagnostics
    assert "transport_entropy_mean" in diagnostics
    assert diagnostics["transport_entropy_pair_count"] > 0


def test_factor_normalization_matches_block_budget_and_suppresses_degenerate_factors() -> None:
    weights, diagnostics = _normalized_factor_weights((2.0, 1e-16, 4.0))

    assert weights == (0.25, 0.0, 0.125)
    assert diagnostics["active_factor_indices"] == [0, 2]
    assert diagnostics["degenerate_factor_indices"] == [1]
    assert diagnostics["active_factor_count"] == 2
    assert sum(weight * scale for weight, scale in zip(weights, (2.0, 1e-16, 4.0))) == 1.0


def test_factor_scale_reports_zero_for_a_constant_factor() -> None:
    left = ProductMeasure(points=torch.tensor([[[0.0], [0.5], [0.0]]]), mass=torch.ones(1))
    right = ProductMeasure(points=torch.tensor([[[1.0], [0.5], [2.0]]]), mass=torch.ones(1))

    scales = _train_factor_cost_scales((left, right), curvature=0.0)

    assert scales[0] > 0.0
    assert scales[1] == 0.0
    assert scales[2] > 0.0


def test_factor_matrix_can_select_product_cost_weight_on_train_split(tmp_path: Path) -> None:
    task_a = _write_task(tmp_path, "task-a", "a")
    task_b = _write_task(tmp_path, "task-b", "b")
    output = tmp_path / "validation-selected-cost.json"

    payload = run_dta_factor_matrix(
        tasks=(TaskSource("task-a", task_a), TaskSource("task-b", task_b)),
        output_path=output,
        geometries=("H_1",),
        path_object_modes=("lca_product",),
        method_aggregations=("measure",),
        train_per_task=3,
        query_per_task=1,
        gallery_per_task=1,
        max_methods_per_task=5,
        max_paths=6,
        epochs=1,
        sinkhorn_iterations=3,
        sinkhorn_projection_iterations=512,
        encoder_policy="shared_euclidean",
        cost_modes=("validation_selected_combined",),
    )

    loaded = json.loads(output.read_text(encoding="utf-8"))
    assert payload["status"] == "complete"
    assert loaded["config"]["cost_modes"] == ["validation_selected_combined"]
    run = loaded["runs"][0]
    assert run["cost_mode"] == "validation_selected_combined"
    normalization = run["cost_normalization"]
    assert normalization["source"] == "train_split_internal_validation"
    assert 0.0 <= normalization["selected_point_weight"] <= 1.0
    assert normalization["selected_side_weight"] == 1.0 - normalization["selected_point_weight"]
    assert normalization["selection_query_count"] > 0
    assert normalization["selection_strategy"] == "one_train_gallery_item_per_task"
    assert normalization["selection_grid"]


def test_factor_matrix_can_sweep_fixed_train_normalized_point_weights(tmp_path: Path) -> None:
    task_a = _write_task(tmp_path, "task-a", "a")
    task_b = _write_task(tmp_path, "task-b", "b")
    output = tmp_path / "fixed-point-weight-grid.json"

    payload = run_dta_factor_matrix(
        tasks=(TaskSource("task-a", task_a), TaskSource("task-b", task_b)),
        output_path=output,
        geometries=("H_1",),
        path_object_modes=("lca_product",),
        method_aggregations=("measure",),
        train_per_task=2,
        query_per_task=1,
        gallery_per_task=1,
        max_methods_per_task=4,
        max_paths=6,
        epochs=1,
        sinkhorn_iterations=3,
        sinkhorn_projection_iterations=512,
        encoder_policy="shared_euclidean",
        cost_modes=("train_weighted_combined",),
        point_weights=(0.0, 0.5, 1.0),
    )

    loaded = json.loads(output.read_text(encoding="utf-8"))
    assert payload["status"] == "complete"
    assert loaded["expected_runs"] == 3
    assert loaded["config"]["point_weights"] == [0.0, 0.5, 1.0]
    by_mode = {run["cost_mode"]: run for run in loaded["runs"]}
    assert set(by_mode) == {
        "train_weighted_combined_p0p00",
        "train_weighted_combined_p0p50",
        "train_weighted_combined_p1p00",
    }
    assert by_mode["train_weighted_combined_p0p00"]["cost_normalization"]["point_weight"] == 0.0
    assert by_mode["train_weighted_combined_p0p50"]["cost_normalization"]["side_weight"] == 0.5
    assert by_mode["train_weighted_combined_p1p00"]["cost_normalization"]["point_weight"] == 1.0


def _write_task(root: Path, task_name: str, prefix: str) -> Path:
    task_dir = root / task_name
    task_dir.mkdir()
    bodies = (
        "total = 0\n    for x in xs:\n        total += x\n    return total\n",
        "acc = 1\n    for x in xs:\n        acc *= (x + 1)\n    return acc\n",
        "out = []\n    for x in xs:\n        out.append(x + 1)\n    return out\n",
        "count = 0\n    for x in xs:\n        if x > 0:\n            count += 1\n    return count\n",
        "best = None\n    for x in xs:\n        if best is None or x > best:\n            best = x\n    return best\n",
        "return [x for x in xs if x % 2 == 0]\n",
    )
    for index, body in enumerate(bodies):
        (task_dir / f"{prefix}_{index}.py").write_text(f"def f_{prefix}_{index}(xs):\n    {body}", encoding="utf-8")
    return task_dir
