from __future__ import annotations

import json
import math
import os
import platform
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from geometry_profile_research.codenet_eligibility import canonical_json_bytes, stable_sha256
from geometry_profile_research.codenet_stage_a import StageATestSplit
from geometry_profile_research.codenet_stage_a_evaluation import (
    full_gallery_sinkhorn_divergence,
    precompute_self_objectives,
    summarize_problem_macro_retrieval,
)
from geometry_profile_research.codenet_stage_a_runner import (
    encode_stage_a_programs,
    scale_product_measure,
)
from geometry_profile_research.constant_curvature import RoleProductGeometry
from geometry_profile_research.raw_ast_code2hyp import RawASTCode2Hyp


TEST_SEED_SCHEMA = "code2hyp-stage-a-test-seed-v1"
VALIDATION_RUNNER_COMMIT = "469cbabc6692d1bc6cfde8cbb33c7ad79f8c9093"
VALIDATION_RUNNER_TAG = "codenet-stage-a-validation-runner-v3"


def run_stage_a_test_seed(
    *,
    test_split: StageATestSplit,
    seed: int,
    validation_result_path: Path,
    validation_result_expected_sha256: str,
    validation_seed_seal_path: Path,
    test_materialization_manifest_path: Path,
    output_dir: Path,
    test_execution_protocol_sha256: str,
    test_runtime_addendum_sha256: str,
    test_resumability_addendum_sha256: str,
    implementation: Mapping[str, Any],
    progress_callback: Any | None = None,
) -> dict[str, Any]:
    """Evaluate all frozen cells using one sealed validation checkpoint."""

    validation_bytes = validation_result_path.read_bytes()
    validation_sha256 = stable_sha256(validation_bytes)
    if validation_sha256 != validation_result_expected_sha256:
        raise ValueError("validation result differs from the validation-selection seal")
    validation = json.loads(validation_bytes)
    seed_seal_bytes = validation_seed_seal_path.read_bytes()
    seed_seal = json.loads(seed_seal_bytes)
    if validation.get("status") != "complete" or int(validation.get("seed", -1)) != seed:
        raise ValueError("validation seed result is missing or incomplete")
    if seed_seal.get("inputs", {}).get("result", {}).get("sha256") != validation_sha256:
        raise ValueError("validation seed result differs from its seed seal")
    if seed_seal.get("checks", {}).get("validation_only") is not True:
        raise ValueError("validation seed is not sealed as validation-only")
    if validation.get("implementation", {}).get("commit") != VALIDATION_RUNNER_COMMIT:
        raise ValueError("validation checkpoint was produced by an unexpected runner commit")
    if validation.get("implementation", {}).get("tag") != VALIDATION_RUNNER_TAG:
        raise ValueError("validation checkpoint was produced by an unexpected runner tag")

    materialization_bytes = test_materialization_manifest_path.read_bytes()
    materialization = json.loads(materialization_bytes)
    if materialization.get("schema_version") != "code2hyp-stage-a-test-materialization-v1":
        raise ValueError("unexpected test-materialization schema")
    if materialization.get("test_program_ids_materialized") is not True:
        raise ValueError("test programs were not materialized by the registered opening")
    if materialization.get("test_relevance_labels_opened") is not True:
        raise ValueError("test relevance labels were not opened by the registered transaction")
    if materialization.get("test_retrieval_metrics_computed") is not False:
        raise ValueError("test materialization indicates prior retrieval metrics")
    if materialization.get("implementation") != dict(implementation):
        raise ValueError("test materialization and test runner implementation differ")

    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / f"seed_{seed}_test.json"
    partial_path = result_path.with_suffix(".partial.json")
    result_identity = {
        "seed": seed,
        "validation_result_sha256": validation_sha256,
        "validation_seed_seal_sha256": stable_sha256(seed_seal_bytes),
        "test_materialization_manifest_sha256": stable_sha256(materialization_bytes),
        "test_execution_protocol_sha256": str(test_execution_protocol_sha256),
        "implementation": dict(implementation),
    }
    execution = dict(validation["execution_config"])
    torch_num_threads = int(execution["torch_num_threads"])
    if torch_num_threads != 1:
        raise ValueError("the frozen Stage A test runtime requires torch_num_threads=1")
    torch.set_num_threads(torch_num_threads)
    try:
        torch.use_deterministic_algorithms(True, warn_only=False)
    except Exception as error:
        raise RuntimeError("unable to enable deterministic algorithms for Stage A test") from error
    test_runtime = {
        "torch_version": torch.__version__,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "torch_num_threads": torch.get_num_threads(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
    }
    if test_runtime["torch_num_threads"] != 1 or test_runtime["deterministic_algorithms"] is not True:
        raise RuntimeError("Stage A test runtime did not apply the frozen deterministic configuration")
    result_identity["test_runtime_addendum_sha256"] = str(test_runtime_addendum_sha256)
    result_identity["test_resumability_addendum_sha256"] = str(
        test_resumability_addendum_sha256
    )
    result_identity["test_runtime"] = test_runtime
    if result_path.exists():
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        if payload.get("status") == "complete" and payload.get("identity") == result_identity:
            return payload
        raise ValueError(f"refusing to overwrite incompatible test result: {result_path}")

    checkpoint_path = validation_result_path.parent / str(validation["checkpoint"]["path"])
    if stable_sha256(checkpoint_path.read_bytes()) != str(validation["checkpoint"]["sha256"]):
        raise ValueError("validation checkpoint differs from its sealed result")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("implementation") != validation.get("implementation"):
        raise ValueError("checkpoint implementation provenance differs from validation result")
    if int(checkpoint.get("seed", -1)) != seed:
        raise ValueError("checkpoint seed differs from requested test seed")
    model = _model_from_checkpoint(checkpoint)

    cells: dict[str, dict[str, Any]] = {}
    if partial_path.exists():
        partial = json.loads(partial_path.read_text(encoding="utf-8"))
        if partial.get("status") != "partial" or partial.get("identity") != result_identity:
            raise ValueError(f"refusing to reuse incompatible test partial: {partial_path}")
        cells.update(partial.get("cells", {}))

    role_scales = tuple(float(value) for value in validation["coordinate_scaling"]["role_scales"])
    epsilon = float(validation["sinkhorn_calibration"]["epsilon"])
    if not math.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError("sealed validation epsilon must be finite and positive")
    query_true = encode_stage_a_programs(
        model,
        test_split.query,
        anchor_mode="true_lca",
        progress_callback=progress_callback,
        phase="encoding_test_query_true_lca",
    )
    gallery_true = encode_stage_a_programs(
        model,
        test_split.gallery,
        anchor_mode="true_lca",
        progress_callback=progress_callback,
        phase="encoding_test_gallery_true_lca",
    )
    query_zero = encode_stage_a_programs(
        model,
        test_split.query,
        anchor_mode="zero_anchor",
        progress_callback=progress_callback,
        phase="encoding_test_query_zero_anchor",
    )
    gallery_zero = encode_stage_a_programs(
        model,
        test_split.gallery,
        anchor_mode="zero_anchor",
        progress_callback=progress_callback,
        phase="encoding_test_gallery_zero_anchor",
    )
    query_true_values = tuple(
        scale_product_measure(query_true[item.item_id], role_scales=role_scales)
        for item in test_split.query
    )
    gallery_true_values = tuple(
        scale_product_measure(gallery_true[item.item_id], role_scales=role_scales)
        for item in test_split.gallery
    )
    query_zero_values = tuple(
        scale_product_measure(query_zero[item.item_id], role_scales=role_scales)
        for item in test_split.query
    )
    gallery_zero_values = tuple(
        scale_product_measure(gallery_zero[item.item_id], role_scales=role_scales)
        for item in test_split.gallery
    )
    del query_true, gallery_true, query_zero, gallery_zero

    expected_cell_ids = (
        "EEE_true_LCA",
        "EEE_zero_anchor",
        "HEE_near_zero_true_LCA",
        "HEE_c0p1_true_LCA",
        "HEE_c0p3_true_LCA",
        "HEE_c1_true_LCA",
        "HEE_c3_true_LCA",
    )
    if set(validation["cells"]) != set(expected_cell_ids):
        raise ValueError("sealed validation result does not contain the exact seven-cell design")
    cell_ids = expected_cell_ids
    for cell_index, cell_id in enumerate(cell_ids, start=1):
        if cell_id in cells:
            _cleanup_distance_shards(output_dir=output_dir, seed=seed, cell_id=cell_id)
            continue
        validation_cell = validation["cells"][cell_id]
        curvatures = tuple(float(value) for value in validation_cell["factor_curvatures"])
        weights = tuple(float(value) for value in validation_cell["factor_weights"])
        geometry = RoleProductGeometry(
            factor_curvatures=curvatures,
            factor_weights=weights,
            side_weight=0.0,
            unoriented=True,
        )
        queries = query_zero_values if cell_id == "EEE_zero_anchor" else query_true_values
        gallery = gallery_zero_values if cell_id == "EEE_zero_anchor" else gallery_true_values
        if progress_callback is not None:
            progress_callback(
                {
                    "phase": "test_cell",
                    "seed": seed,
                    "cell_id": cell_id,
                    "cell_index": cell_index,
                    "cell_count": len(cell_ids),
                }
            )
        distances, shard_paths = resumable_full_gallery_sinkhorn_divergence(
            queries,
            gallery,
            geometry,
            epsilon=epsilon,
            output_dir=output_dir,
            shard_prefix=f"seed_{seed}_{cell_id}",
            query_shard_size=32,
            query_batch_size=int(execution["query_batch_size"]),
            gallery_batch_size=int(execution["gallery_batch_size"]),
            sinkhorn_iterations=int(execution["sinkhorn_iterations"]),
            projection_iterations=int(execution["projection_iterations"]),
            marginal_tolerance=float(execution["marginal_tolerance"]),
            progress_callback=progress_callback,
        )
        summary = summarize_problem_macro_retrieval(
            distances,
            query_ids=tuple(item.item_id for item in test_split.query),
            query_cluster_ids=tuple(item.cluster_id for item in test_split.query),
            gallery_ids=tuple(item.item_id for item in test_split.gallery),
            gallery_cluster_ids=tuple(item.cluster_id for item in test_split.gallery),
            r=8,
        )
        distance_path = output_dir / f"seed_{seed}_{cell_id}_test_distances.pt"
        _atomic_torch_save(distance_path, distances.to(dtype=torch.float64, device="cpu"))
        cells[cell_id] = {
            "factor_curvatures": list(curvatures),
            "factor_weights": list(weights),
            "effective_lca_sectional_curvature": geometry.factor_sectional_curvatures[0],
            "metrics": asdict(summary),
            "distance_matrix": {
                "path": distance_path.name,
                "shape": list(distances.shape),
                "dtype": "float64",
                "sha256": stable_sha256(distance_path.read_bytes()),
                "minimum": float(distances.min()),
                "maximum": float(distances.max()),
                "negative_count": int((distances < 0.0).sum()),
            },
        }
        _atomic_json_write(
            partial_path,
            _test_seed_payload(identity=result_identity, cells=cells, status="partial"),
        )
        for shard_path in shard_paths:
            shard_path.unlink(missing_ok=True)
            shard_path.with_suffix(shard_path.suffix + ".sha256").unlink(missing_ok=True)

    payload = _test_seed_payload(identity=result_identity, cells=cells, status="complete")
    _atomic_json_write(result_path, payload)
    partial_path.unlink(missing_ok=True)
    return json.loads(result_path.read_text(encoding="utf-8"))


def resumable_full_gallery_sinkhorn_divergence(
    queries: Sequence[Any],
    gallery: Sequence[Any],
    geometry: RoleProductGeometry,
    *,
    epsilon: float,
    output_dir: Path,
    shard_prefix: str,
    query_shard_size: int = 32,
    query_batch_size: int = 4,
    gallery_batch_size: int = 32,
    sinkhorn_iterations: int = 128,
    projection_iterations: int = 2048,
    marginal_tolerance: float = 1e-7,
    progress_callback: Any | None = None,
) -> tuple[torch.Tensor, tuple[Path, ...]]:
    """Compute the exact full gallery matrix in hash-verified query shards."""

    if not queries or not gallery:
        raise ValueError("queries and gallery must not be empty")
    if query_shard_size <= 0 or query_shard_size % query_batch_size != 0:
        raise ValueError("query shard size must be a positive multiple of query batch size")
    output_dir.mkdir(parents=True, exist_ok=True)
    query_self = precompute_self_objectives(
        queries,
        geometry,
        epsilon=epsilon,
        batch_size=query_batch_size,
        sinkhorn_iterations=sinkhorn_iterations,
        projection_iterations=projection_iterations,
        marginal_tolerance=marginal_tolerance,
    )
    gallery_self = precompute_self_objectives(
        gallery,
        geometry,
        epsilon=epsilon,
        batch_size=gallery_batch_size,
        sinkhorn_iterations=sinkhorn_iterations,
        projection_iterations=projection_iterations,
        marginal_tolerance=marginal_tolerance,
    )
    shard_paths: list[Path] = []
    shards: list[torch.Tensor] = []
    shard_count = math.ceil(len(queries) / query_shard_size)
    for shard_index, start in enumerate(range(0, len(queries), query_shard_size), start=1):
        stop = min(start + query_shard_size, len(queries))
        shard_path = output_dir / f"{shard_prefix}_q{start:04d}_{stop:04d}.pt"
        sidecar_path = shard_path.with_suffix(shard_path.suffix + ".sha256")
        expected_shape = (stop - start, len(gallery))
        if shard_path.exists() or sidecar_path.exists():
            if not shard_path.exists() or not sidecar_path.exists():
                raise ValueError(f"incomplete distance shard artifact: {shard_path}")
            expected_sha = sidecar_path.read_text(encoding="ascii").strip()
            actual_sha = stable_sha256(shard_path.read_bytes())
            if expected_sha != actual_sha:
                raise ValueError(f"distance shard hash mismatch: {shard_path}")
            shard = torch.load(shard_path, map_location="cpu", weights_only=True)
        else:
            shard = full_gallery_sinkhorn_divergence(
                queries[start:stop],
                gallery,
                geometry,
                epsilon=epsilon,
                query_batch_size=query_batch_size,
                gallery_batch_size=gallery_batch_size,
                sinkhorn_iterations=sinkhorn_iterations,
                projection_iterations=projection_iterations,
                marginal_tolerance=marginal_tolerance,
                query_self_objectives=query_self[start:stop],
                gallery_self_objectives=gallery_self,
            ).to(dtype=torch.float64, device="cpu")
            _atomic_torch_save(shard_path, shard)
            _atomic_bytes_write(
                sidecar_path,
                (stable_sha256(shard_path.read_bytes()) + "\n").encode("ascii"),
            )
        if not isinstance(shard, torch.Tensor) or shard.dtype != torch.float64:
            raise ValueError(f"distance shard must be a float64 tensor: {shard_path}")
        if tuple(shard.shape) != expected_shape:
            raise ValueError(f"distance shard shape mismatch: {shard_path}")
        if not torch.isfinite(shard).all():
            raise ValueError(f"distance shard contains non-finite values: {shard_path}")
        shard_paths.append(shard_path)
        shards.append(shard)
        if progress_callback is not None:
            progress_callback(
                {
                    "phase": "test_distance_shard",
                    "shard_prefix": shard_prefix,
                    "shard_index": shard_index,
                    "shard_count": shard_count,
                    "query_start": start,
                    "query_stop": stop,
                }
            )
    distances = torch.cat(shards, dim=0)
    if tuple(distances.shape) != (len(queries), len(gallery)):
        raise ValueError("assembled distance matrix shape mismatch")
    return distances, tuple(shard_paths)


def aggregate_all_test_cells(
    seed_payloads: Sequence[Mapping[str, Any]],
    *,
    expected_seeds: Sequence[int],
) -> dict[str, Any]:
    """Report every planned cell with seed-then-problem primary aggregation."""

    observed_seeds = tuple(int(payload["seed"]) for payload in seed_payloads)
    if observed_seeds != tuple(int(seed) for seed in expected_seeds):
        raise ValueError("test payload order differs from the registered seed order")
    reference_cells = tuple(seed_payloads[0]["cells"])
    if any(tuple(payload["cells"]) != reference_cells for payload in seed_payloads):
        raise ValueError("planned test cell set differs across seeds")
    result: dict[str, Any] = {}
    for cell_id in reference_cells:
        per_problem: dict[str, list[float]] = {}
        secondary: dict[str, list[float]] = {
            "mrr": [],
            "recall_at_1": [],
            "recall_at_5": [],
            "recall_at_10": [],
            "mean_first_relevant_rank": [],
        }
        for payload in seed_payloads:
            metrics = payload["cells"][cell_id]["metrics"]
            for problem, score in metrics["task_scores"].items():
                per_problem.setdefault(str(problem), []).append(float(score))
            for name in secondary:
                secondary[name].append(float(metrics[name]))
        averaged_problems = {
            problem: sum(values) / len(values)
            for problem, values in sorted(per_problem.items())
        }
        result[cell_id] = {
            "problem_macro_MAP_at_8": sum(averaged_problems.values()) / len(averaged_problems),
            "problem_count": len(averaged_problems),
            "problem_scores_after_seed_averaging": averaged_problems,
            "secondary_metrics_seed_mean_descriptive_only": {
                name: sum(values) / len(values) for name, values in secondary.items()
            },
        }
    return result


def _test_seed_payload(
    *,
    identity: Mapping[str, Any],
    cells: Mapping[str, Any],
    status: str,
) -> dict[str, Any]:
    return {
        "schema_version": TEST_SEED_SCHEMA,
        "status": status,
        "seed": int(identity["seed"]),
        "identity": dict(identity),
        "cells": dict(cells),
        "validation_metrics_computed": True,
        "test_program_ids_materialized": True,
        "test_relevance_labels_opened": True,
        "test_retrieval_metrics_computed": bool(cells),
    }


def _model_from_checkpoint(checkpoint: Mapping[str, Any]) -> RawASTCode2Hyp:
    model = RawASTCode2Hyp(checkpoint["token_to_id"], **dict(checkpoint["model_config"]))
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    return model


def _atomic_json_write(path: Path, payload: Mapping[str, Any]) -> None:
    content = canonical_json_bytes(payload)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(content)
    os.replace(temporary, path)


def _atomic_torch_save(path: Path, payload: Any) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _atomic_bytes_write(path: Path, content: bytes) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(content)
    os.replace(temporary, path)


def _cleanup_distance_shards(*, output_dir: Path, seed: int, cell_id: str) -> None:
    for shard_path in output_dir.glob(f"seed_{seed}_{cell_id}_q????_????.pt"):
        shard_path.unlink(missing_ok=True)
        shard_path.with_suffix(shard_path.suffix + ".sha256").unlink(missing_ok=True)
