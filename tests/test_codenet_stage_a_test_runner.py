from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from geometry_profile_research.codenet_eligibility import canonical_json_bytes, stable_sha256
from geometry_profile_research.codenet_stage_a import StageAProgram, StageASplit, StageATestSplit
from geometry_profile_research.codenet_stage_a_evaluation import full_gallery_sinkhorn_divergence
from geometry_profile_research.codenet_stage_a_runner import run_stage_a_validation_seed
from geometry_profile_research.codenet_stage_a_test_runner import (
    VALIDATION_RUNNER_COMMIT,
    VALIDATION_RUNNER_TAG,
    aggregate_all_test_cells,
    resumable_full_gallery_sinkhorn_divergence,
    run_stage_a_test_seed,
)
from geometry_profile_research.constant_curvature import ProductMeasure, RoleProductGeometry
from geometry_profile_research.python_raw_ast import parse_python_ast_tree
from scripts.seal_codenet_stage_a_test_seed import (
    TEST_RUNNER_COMMIT,
    TEST_RUNNER_TAG,
    seal_test_seed_result,
)


def _program(item_id: str, problem_id: str, split: str, role: str, source: str) -> StageAProgram:
    return StageAProgram(
        item_id=item_id,
        cluster_id=f"cluster-{problem_id}",
        problem_id=problem_id,
        split=split,
        role=role,
        tree=parse_python_ast_tree(source),
    )


def _validation_split() -> StageASplit:
    train = (
        _program("train/a.py", "T1", "train", "train", "x = 1\nprint(x)\n"),
        _program("train/b.py", "T1", "train", "train", "if x:\n print(x)\n"),
        _program("train/c.py", "T2", "train", "train", "for x in range(3):\n print(x)\n"),
        _program("train/d.py", "T2", "train", "train", "def f(x):\n return x + 1\n"),
    )
    query = (
        _program("validation/query-a.py", "A", "validation", "query", "x = 1\nprint(x)\n"),
        _program("validation/query-b.py", "B", "validation", "query", "for x in range(2):\n print(x)\n"),
    )
    gallery = tuple(
        _program(
            f"validation/{problem}-{index}.py",
            problem,
            "validation",
            "gallery",
            f"x = {index}\nprint(x)\n" if problem == "A" else f"for x in range({index + 1}):\n print(x)\n",
        )
        for problem in ("A", "B")
        for index in range(8)
    )
    return StageASplit(train=train, query=query, gallery=gallery)


def _test_split() -> StageATestSplit:
    query = (
        _program("test/query-a.py", "A", "test", "query", "x = 2\nprint(x)\n"),
        _program("test/query-b.py", "B", "test", "query", "for x in range(4):\n print(x)\n"),
    )
    gallery = tuple(
        _program(
            f"test/{problem}-{index}.py",
            problem,
            "test",
            "gallery",
            f"x = {index + 1}\nprint(x)\n"
            if problem == "A"
            else f"for x in range({index + 2}):\n print(x)\n",
        )
        for problem in ("A", "B")
        for index in range(8)
    )
    return StageATestSplit(query=query, gallery=gallery)


def test_test_seed_reuses_sealed_validation_calibration_and_runs_all_cells(tmp_path: Path) -> None:
    validation_dir = tmp_path / "validation"
    validation_implementation = {
        "repository": "https://github.com/Malomalsky/code2hyp-artifacts",
        "commit": VALIDATION_RUNNER_COMMIT,
        "tag": VALIDATION_RUNNER_TAG,
        "tracked_worktree_clean": True,
    }
    validation = run_stage_a_validation_seed(
        split=_validation_split(),
        calibration_pairs=(
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
        ),
        seed=20260711,
        output_dir=validation_dir,
        protocol_sha256="protocol",
        calibration_manifest_sha256="calibration",
        dim=2,
        epochs=1,
        batch_size=2,
        max_paths=4,
        active_curvatures=(0.1, 0.3, 1.0, 3.0),
        sinkhorn_iterations=8,
        projection_iterations=64,
        marginal_tolerance=1e-6,
        query_batch_size=2,
        gallery_batch_size=8,
        implementation=validation_implementation,
    )
    validation_path = validation_dir / "seed_20260711_validation.json"
    validation_sha = stable_sha256(validation_path.read_bytes())
    seed_seal_path = validation_dir / "seed_20260711_validation_seal.json"
    seed_seal_path.write_bytes(
        canonical_json_bytes(
            {
                "inputs": {"result": {"sha256": validation_sha}},
                "checks": {"validation_only": True},
            }
        )
    )
    test_implementation = {
        "repository": "https://github.com/Malomalsky/code2hyp-artifacts",
        "commit": TEST_RUNNER_COMMIT,
        "tag": TEST_RUNNER_TAG,
        "tracked_worktree_clean": True,
    }
    test_protocol_path = tmp_path / "test_execution_protocol.json"
    test_protocol_path.write_text("test execution protocol", encoding="utf-8")
    test_protocol_sha = stable_sha256(test_protocol_path.read_bytes())
    runtime_addendum_path = tmp_path / "runtime_addendum.json"
    runtime_addendum_path.write_text("runtime addendum", encoding="utf-8")
    runtime_addendum_sha = stable_sha256(runtime_addendum_path.read_bytes())
    resumability_addendum_path = tmp_path / "resumability_addendum.json"
    resumability_addendum_path.write_text("resumability addendum", encoding="utf-8")
    resumability_addendum_sha = stable_sha256(resumability_addendum_path.read_bytes())
    relevance_addendum_path = tmp_path / "relevance_addendum.json"
    relevance_addendum_path.write_bytes(
        canonical_json_bytes(
            {
                "schema_version": "code2hyp-stage-a-relevance-identity-addendum-v1",
                "correction": {"relevance_key_after": "cluster_id"},
            }
        )
    )
    relevance_addendum_sha = stable_sha256(relevance_addendum_path.read_bytes())
    materialization_path = tmp_path / "test_materialization_manifest.json"
    materialization_path.write_bytes(
        canonical_json_bytes(
            {
                "schema_version": "code2hyp-stage-a-test-materialization-v1",
                "implementation": test_implementation,
                "test_program_ids_materialized": True,
                "test_relevance_labels_opened": True,
                "test_retrieval_metrics_computed": False,
            }
        )
    )

    result = run_stage_a_test_seed(
        test_split=_test_split(),
        seed=20260711,
        validation_result_path=validation_path,
        validation_result_expected_sha256=validation_sha,
        validation_seed_seal_path=seed_seal_path,
        test_materialization_manifest_path=materialization_path,
        output_dir=tmp_path / "test",
        test_execution_protocol_sha256=test_protocol_sha,
        test_runtime_addendum_sha256=runtime_addendum_sha,
        test_resumability_addendum_sha256=resumability_addendum_sha,
        relevance_identity_addendum_sha256=relevance_addendum_sha,
        implementation=test_implementation,
    )
    resumed = run_stage_a_test_seed(
        test_split=_test_split(),
        seed=20260711,
        validation_result_path=validation_path,
        validation_result_expected_sha256=validation_sha,
        validation_seed_seal_path=seed_seal_path,
        test_materialization_manifest_path=materialization_path,
        output_dir=tmp_path / "test",
        test_execution_protocol_sha256=test_protocol_sha,
        test_runtime_addendum_sha256=runtime_addendum_sha,
        test_resumability_addendum_sha256=resumability_addendum_sha,
        relevance_identity_addendum_sha256=relevance_addendum_sha,
        implementation=test_implementation,
    )

    assert result == resumed
    assert result["status"] == "complete"
    assert set(result["cells"]) == set(validation["cells"])
    assert all(cell["metrics"]["problem_count"] == 2 for cell in result["cells"].values())
    assert all(cell["distance_matrix"]["shape"] == [2, 16] for cell in result["cells"].values())

    selection_seal_path = tmp_path / "validation_selection_seal.json"
    selection_seal_path.write_bytes(
        canonical_json_bytes(
            {
                "schema_version": "code2hyp-stage-a-validation-selection-seal-v1",
                "inputs": {
                    "seeds": [
                        {
                            "seed": 20260711,
                            "result_sha256": validation_sha,
                            "seal_sha256": stable_sha256(seed_seal_path.read_bytes()),
                        }
                    ]
                },
            }
        )
    )
    test_programs_path = tmp_path / "test" / "test_programs.jsonl"
    test_rows = [
        {
            "source_relpath": program.item_id,
            "cluster_id": program.cluster_id,
            "problem_id": program.problem_id,
            "split": "test",
            "role": program.role,
        }
        for program in (*_test_split().query, *_test_split().gallery)
    ]
    test_programs_path.write_text(
        "".join(json.dumps(row) + "\n" for row in test_rows),
        encoding="utf-8",
    )
    test_result_path = tmp_path / "test" / "seed_20260711_test.json"
    seal = seal_test_seed_result(
        result_path=test_result_path,
        test_execution_protocol_path=test_protocol_path,
        test_runtime_addendum_path=runtime_addendum_path,
        test_resumability_addendum_path=resumability_addendum_path,
        relevance_addendum_path=relevance_addendum_path,
        test_materialization_manifest_path=materialization_path,
        test_programs_path=test_programs_path,
        validation_selection_seal_path=selection_seal_path,
        output_path=tmp_path / "test" / "seed_20260711_test_seal.json",
        expected_query_count=2,
        expected_gallery_count=16,
        expected_problem_count=2,
        relevant_count=8,
    )
    assert seal["checks"]["all_metrics_recomputed_from_distances"] is True


def test_query_sharding_is_bitwise_equal_to_monolithic_full_gallery(tmp_path: Path) -> None:
    generator = torch.Generator().manual_seed(17)

    def measure(size: int) -> ProductMeasure:
        points = 0.05 * torch.randn((size, 3, 2), generator=generator, dtype=torch.float64)
        return ProductMeasure(
            points=points,
            mass=torch.full((size,), 1.0 / size, dtype=torch.float64),
        )

    queries = tuple(measure(size) for size in (2, 3, 2, 4, 3))
    gallery = tuple(measure(size) for size in (2, 4, 3, 2, 3, 4))
    geometry = RoleProductGeometry(
        factor_curvatures=(0.0, 0.0, 0.0),
        factor_weights=(1.0, 1.0, 1.0),
        side_weight=0.0,
        unoriented=True,
    )
    common = {
        "epsilon": 0.05,
        "query_batch_size": 1,
        "gallery_batch_size": 2,
        "sinkhorn_iterations": 16,
        "projection_iterations": 128,
        "marginal_tolerance": 1e-7,
    }
    monolithic = full_gallery_sinkhorn_divergence(
        queries,
        gallery,
        geometry,
        **common,
    )
    sharded, shard_paths = resumable_full_gallery_sinkhorn_divergence(
        queries,
        gallery,
        geometry,
        output_dir=tmp_path,
        shard_prefix="equivalence",
        query_shard_size=2,
        **common,
    )
    resumed, resumed_paths = resumable_full_gallery_sinkhorn_divergence(
        queries,
        gallery,
        geometry,
        output_dir=tmp_path,
        shard_prefix="equivalence",
        query_shard_size=2,
        **common,
    )

    assert torch.equal(sharded, monolithic)
    assert torch.equal(resumed, monolithic)
    assert resumed_paths == shard_paths
    assert all(path.with_suffix(path.suffix + ".sha256").exists() for path in shard_paths)


def test_all_cell_aggregation_averages_seeds_within_problem_first() -> None:
    payloads = []
    for seed, values in ((1, {"A": 0.2, "B": 0.6}), (2, {"A": 0.4, "B": 0.8})):
        payloads.append(
            {
                "seed": seed,
                "cells": {
                    "EEE_true_LCA": {
                        "metrics": {
                            "task_scores": values,
                            "mrr": 0.5,
                            "recall_at_1": 0.25,
                            "recall_at_5": 0.75,
                            "recall_at_10": 1.0,
                            "mean_first_relevant_rank": 3.0,
                        }
                    }
                },
            }
        )

    summary = aggregate_all_test_cells(payloads, expected_seeds=(1, 2))

    assert summary["EEE_true_LCA"]["problem_macro_MAP_at_8"] == pytest.approx(0.5)
    assert summary["EEE_true_LCA"]["problem_scores_after_seed_averaging"] == {
        "A": pytest.approx(0.3),
        "B": pytest.approx(0.7),
    }
