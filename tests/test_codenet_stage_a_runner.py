from __future__ import annotations

from pathlib import Path

import pytest
import torch

from geometry_profile_research.codenet_stage_a import StageAProgram, StageASplit
from geometry_profile_research.codenet_stage_a_runner import (
    all_role_curvature_cell_id,
    configure_torch_runtime,
    curvature_cell_id,
    matched_role_weights,
    run_stage_a_validation_seed,
    select_active_curvature,
    train_stage_a_encoder,
)
from geometry_profile_research.python_raw_ast import parse_python_ast_tree


def _program(item_id: str, problem_id: str, role: str, source: str) -> StageAProgram:
    return StageAProgram(
        item_id=item_id,
        cluster_id=f"cluster-{problem_id}",
        problem_id=problem_id,
        split="train" if role == "train" else "validation",
        role=role,
        tree=parse_python_ast_tree(source),
    )


def _small_split() -> StageASplit:
    train = (
        _program("train/a.py", "T1", "train", "x = 1\nprint(x)\n"),
        _program("train/b.py", "T1", "train", "if x:\n    print(x)\nelse:\n    print(0)\n"),
        _program("train/c.py", "T2", "train", "for x in range(3):\n    print(x)\n"),
        _program("train/d.py", "T2", "train", "def f(x):\n    return x + 1\n"),
    )
    queries = (
        _program("validation/query-a.py", "A", "query", "x = 1\nprint(x)\n"),
        _program("validation/query-b.py", "B", "query", "for x in range(2):\n    print(x)\n"),
    )
    gallery = tuple(
        _program(
            f"validation/gallery-{problem}-{index}.py",
            problem,
            "gallery",
            (
                f"x = {index}\nprint(x)\n"
                if problem == "A"
                else f"for x in range({index + 1}):\n    print(x)\n"
            ),
        )
        for problem in ("A", "B")
        for index in range(8)
    )
    return StageASplit(train=train, query=queries, gallery=gallery)


def test_stage_a_seed_runner_writes_complete_resumable_result(tmp_path: Path) -> None:
    split = _small_split()
    pairs = (
        {
            "pair_type": "same_cluster",
            "left_source_relpath": "train/a.py",
            "right_source_relpath": "train/b.py",
        },
        {
            "pair_type": "cross_cluster",
            "left_source_relpath": "train/a.py",
            "right_source_relpath": "train/c.py",
        },
        {
            "pair_type": "cross_cluster",
            "left_source_relpath": "train/b.py",
            "right_source_relpath": "train/d.py",
        },
    )

    result = run_stage_a_validation_seed(
        split=split,
        calibration_pairs=pairs,
        seed=20260711,
        output_dir=tmp_path,
        protocol_sha256="protocol",
        calibration_manifest_sha256="calibration",
        dim=2,
        epochs=1,
        batch_size=2,
        learning_rate=0.001,
        max_paths=4,
        active_curvatures=(0.1,),
        near_zero_curvature=1e-4,
        sinkhorn_iterations=12,
        projection_iterations=128,
        marginal_tolerance=1e-6,
        query_batch_size=2,
        gallery_batch_size=8,
        torch_num_threads=1,
    )
    resumed = run_stage_a_validation_seed(
        split=split,
        calibration_pairs=pairs,
        seed=20260711,
        output_dir=tmp_path,
        protocol_sha256="protocol",
        calibration_manifest_sha256="calibration",
        dim=2,
        epochs=1,
        batch_size=2,
        learning_rate=0.001,
        max_paths=4,
        active_curvatures=(0.1,),
        near_zero_curvature=1e-4,
        sinkhorn_iterations=12,
        projection_iterations=128,
        marginal_tolerance=1e-6,
        query_batch_size=2,
        gallery_batch_size=8,
        torch_num_threads=1,
    )

    assert result == resumed
    assert result["status"] == "complete"
    assert result["implementation"] == {"mode": "library_call_without_repository_provenance"}
    assert set(result["cells"]) == {
        "EEE_true_LCA",
        "EEE_zero_anchor",
        "HEE_near_zero_true_LCA",
        "HEE_c0p1_true_LCA",
    }
    assert all(cell["metrics"]["problem_count"] == 2 for cell in result["cells"].values())
    assert (tmp_path / "seed_20260711_encoder.pt").exists()
    assert not (tmp_path / "seed_20260711_validation.partial.json").exists()


def test_validation_curvature_selection_uses_mean_then_smallest_tie() -> None:
    payloads = [
        {
            "status": "complete",
            "seed": 1,
            "cells": {
                "EEE_true_LCA": {"metrics": {"task_scores": {"A": 0.4, "B": 0.6}}},
                "EEE_zero_anchor": {"metrics": {"task_scores": {"A": 0.3, "B": 0.5}}},
                "HEE_near_zero_true_LCA": {"metrics": {"task_scores": {"A": 0.4, "B": 0.6}}},
                "HEE_c0p1_true_LCA": {"metrics": {"task_scores": {"A": 0.3, "B": 0.5}}},
                "HEE_c0p3_true_LCA": {"metrics": {"task_scores": {"A": 0.4, "B": 0.6}}},
            },
        },
        {
            "status": "complete",
            "seed": 2,
            "cells": {
                "EEE_true_LCA": {"metrics": {"task_scores": {"A": 0.5, "B": 0.7}}},
                "EEE_zero_anchor": {"metrics": {"task_scores": {"A": 0.4, "B": 0.6}}},
                "HEE_near_zero_true_LCA": {"metrics": {"task_scores": {"A": 0.5, "B": 0.7}}},
                "HEE_c0p1_true_LCA": {"metrics": {"task_scores": {"A": 0.5, "B": 0.7}}},
                "HEE_c0p3_true_LCA": {"metrics": {"task_scores": {"A": 0.4, "B": 0.6}}},
            },
        },
    ]

    result = select_active_curvature(
        payloads,
        active_curvatures=(0.1, 0.3),
        expected_seeds=(1, 2),
    )

    assert result["selected_active_curvature"] == 0.1
    assert result["problem_count"] == 2
    assert (
        result["descriptive_validation_contrasts"]["H1_EEE_true_LCA_minus_EEE_zero_anchor"][
            "mean_delta_problem_macro_MAP_at_8"
        ]
        == pytest.approx(0.1)
    )
    assert result["test_relevance_labels_opened"] is False

    with pytest.raises(ValueError, match="registered model seed set"):
        select_active_curvature(
            payloads,
            active_curvatures=(0.1, 0.3),
            expected_seeds=(1, 3),
        )


def test_matched_weights_preserve_standard_poincare_limit_convention() -> None:
    canonical = (0.2, 0.3, 0.3)

    eee = matched_role_weights(canonical, factor_curvatures=(0.0, 0.0, 0.0))
    hee = matched_role_weights(canonical, factor_curvatures=(1.0, 0.0, 0.0))

    assert eee == (0.8, 1.2, 1.2)
    assert hee == (0.2, 1.2, 1.2)
    assert curvature_cell_id(0.3) == "HEE_c0p3_true_LCA"
    assert torch.isfinite(torch.tensor(hee)).all()


def test_encoder_accepts_prefix_identifiable_node_input() -> None:
    model, _, metadata = train_stage_a_encoder(
        _small_split().train,
        seed=7,
        dim=2,
        epochs=1,
        batch_size=2,
        max_paths=4,
        node_input_mode="label_depth_prefix",
        torch_num_threads=1,
    )

    assert model.node_input_mode == "label_depth_prefix"
    assert any(token.startswith("edge:") for token in model.token_to_id)
    assert metadata["trainable_parameter_count"] > 0
    assert metadata["compute_device"] == "cpu"


def test_cuda_runtime_fails_closed_when_cuda_is_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    with pytest.raises(RuntimeError, match="requires an available CUDA device"):
        configure_torch_runtime(compute_device="cuda", torch_num_threads=1, seed=7)


def test_cuda_runtime_rejects_conflicting_cublas_workspace(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":16:8")

    with pytest.raises(RuntimeError, match="CUBLAS_WORKSPACE_CONFIG=:4096:8"):
        configure_torch_runtime(compute_device="cuda", torch_num_threads=1, seed=7)


def test_runner_requires_rolewise_ball_fit_and_emits_hhh_control(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="ball fitting"):
        run_stage_a_validation_seed(
            split=_small_split(),
            calibration_pairs=(),
            seed=7,
            output_dir=tmp_path,
            protocol_sha256="p",
            calibration_manifest_sha256="c",
            include_all_role_hyperbolic=True,
        )

    pairs = (
        {"pair_type": "same_cluster", "left_source_relpath": "train/a.py", "right_source_relpath": "train/b.py"},
        {"pair_type": "cross_cluster", "left_source_relpath": "train/a.py", "right_source_relpath": "train/c.py"},
    )
    result = run_stage_a_validation_seed(
        split=_small_split(),
        calibration_pairs=pairs,
        seed=7,
        output_dir=tmp_path / "hhh",
        protocol_sha256="p",
        calibration_manifest_sha256="c",
        dim=2,
        epochs=1,
        batch_size=2,
        max_paths=4,
        node_input_mode="label_depth_prefix",
        fit_all_roles_to_active_ball=True,
        include_all_role_hyperbolic=True,
        active_curvatures=(0.1,),
        sinkhorn_iterations=12,
        projection_iterations=128,
        marginal_tolerance=1e-6,
        query_batch_size=2,
        gallery_batch_size=8,
        torch_num_threads=1,
    )

    hhh = result["cells"][all_role_curvature_cell_id(0.1)]
    assert hhh["factor_curvatures"] == [0.1, 0.1, 0.1]
    assert len(hhh["effective_factor_sectional_curvatures"]) == 3
    maxima = result["coordinate_scaling"]["maximum_unscaled_training_role_norms"]
    scales = result["coordinate_scaling"]["role_scales"]
    assert all(maximum * scale <= 0.35 / (0.1**0.5) + 1e-7 for maximum, scale in zip(maxima, scales))
