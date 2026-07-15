from __future__ import annotations

import json
import math
import os
import platform
import random
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch

from geometry_profile_research.codenet_eligibility import canonical_json_bytes, stable_sha256
from geometry_profile_research.codenet_stage_a import StageAProgram, StageASplit
from geometry_profile_research.codenet_stage_a_evaluation import (
    calibrate_euclidean_role_weights,
    calibration_cost_scale,
    full_gallery_sinkhorn_divergence,
    summarize_problem_macro_retrieval,
)
from geometry_profile_research.constant_curvature import (
    ProductMeasure,
    RoleProductGeometry,
    scaled_sinkhorn_epsilon,
)
from geometry_profile_research.raw_ast_code2hyp import RawASTCode2Hyp, build_raw_ast_token_vocab


ACTIVE_CURVATURES: tuple[float, ...] = (0.1, 0.3, 1.0, 3.0)
NEAR_ZERO_CURVATURE = 1e-4


def train_stage_a_encoder(
    train_programs: Sequence[StageAProgram],
    *,
    seed: int,
    dim: int = 8,
    epochs: int = 5,
    batch_size: int = 8,
    learning_rate: float = 0.003,
    gradient_clip_norm: float = 1.0,
    lambda_edge: float = 1.0,
    lambda_gromov: float = 0.1,
    lambda_branch: float = 1.0,
    max_paths: int = 64,
    path_selection_policy: str = "lca_depth_affine_sampled",
    torch_num_threads: int = 1,
    progress_callback: Any | None = None,
) -> tuple[RawASTCode2Hyp, list[dict[str, float]], dict[str, Any]]:
    """Train the frozen structural-only encoder without validation information."""

    if not train_programs:
        raise ValueError("train_programs must not be empty")
    if epochs <= 0 or batch_size <= 0:
        raise ValueError("epochs and batch_size must be positive")
    if gradient_clip_norm <= 0.0:
        raise ValueError("gradient_clip_norm must be positive")
    if torch_num_threads <= 0:
        raise ValueError("torch_num_threads must be positive")
    random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(torch_num_threads)
    try:
        torch.use_deterministic_algorithms(True, warn_only=False)
        deterministic = True
    except Exception:
        deterministic = False
        raise

    trees = tuple(program.tree for program in train_programs)
    vocab = build_raw_ast_token_vocab(
        trees,
        terminal_policy="class",
        node_input_mode="label_only",
    )
    model = RawASTCode2Hyp(
        vocab,
        dim=dim,
        token_dim=dim,
        manifold="euclidean",
        curvature=1.0,
        max_paths=max_paths,
        terminal_policy="class",
        node_input_mode="label_only",
        path_object_mode="lca_product",
        method_aggregation="measure",
        path_cost_orientation="unoriented",
        path_selection_policy=path_selection_policy,
        anchor_mode="true_lca",
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.0)
    history: list[dict[str, float]] = []
    for epoch in range(epochs):
        model.train()
        order = list(range(len(trees)))
        random.Random(seed + epoch).shuffle(order)
        totals = {
            "loss": 0.0,
            "edge": 0.0,
            "gromov_lca": 0.0,
            "gromov_lca_mean_abs_residual": 0.0,
            "branch_length": 0.0,
            "reversal": 0.0,
        }
        observed = 0
        update_count = 0
        max_observed_gradient_norm = 0.0
        for start in range(0, len(order), batch_size):
            indices = order[start : start + batch_size]
            batch_trees = tuple(trees[index] for index in indices)
            optimizer.zero_grad(set_to_none=True)
            loss = model.structural_training_loss(
                batch_trees,
                lambda_edge=lambda_edge,
                lambda_gromov=lambda_gromov,
                lambda_branch=lambda_branch,
                lambda_reversal=0.0,
            )
            if not torch.isfinite(loss["loss"]):
                raise FloatingPointError(f"non-finite Stage A loss at epoch={epoch + 1}, batch={update_count + 1}")
            loss["loss"].backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip_norm)
            if not torch.isfinite(gradient_norm):
                raise FloatingPointError(f"non-finite Stage A gradient at epoch={epoch + 1}, batch={update_count + 1}")
            optimizer.step()
            weight = len(indices)
            for name in totals:
                totals[name] += weight * float(loss[name].detach())
            observed += weight
            update_count += 1
            max_observed_gradient_norm = max(max_observed_gradient_norm, float(gradient_norm.detach()))
            if progress_callback is not None and (update_count % 250 == 0 or observed == len(order)):
                progress_callback(
                    {
                        "phase": "training",
                        "seed": seed,
                        "epoch": epoch + 1,
                        "epochs": epochs,
                        "programs_observed": observed,
                        "program_count": len(order),
                        "update_count": update_count,
                    }
                )
        history.append(
            {
                "epoch": float(epoch + 1),
                "loss": totals["loss"] / observed,
                "edge": totals["edge"] / observed,
                "gromov_lca": totals["gromov_lca"] / observed,
                "gromov_lca_mean_abs_residual": totals["gromov_lca_mean_abs_residual"] / observed,
                "branch_length": totals["branch_length"] / observed,
                "reversal": totals["reversal"] / observed,
                "program_count": float(observed),
                "update_count": float(update_count),
                "max_preclip_gradient_norm": max_observed_gradient_norm,
            }
        )
    metadata = {
        "seed": seed,
        "vocabulary_size": len(vocab),
        "trainable_parameter_count": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
        "torch_version": torch.__version__,
        "torch_num_threads": torch.get_num_threads(),
        "deterministic_algorithms": deterministic,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
    }
    return model, history, metadata


def encode_stage_a_programs(
    model: RawASTCode2Hyp,
    programs: Sequence[StageAProgram],
    *,
    anchor_mode: str,
    progress_callback: Any | None = None,
    phase: str = "encoding",
) -> dict[str, ProductMeasure]:
    """Encode frozen programs as point-only product measures."""

    if anchor_mode not in {"true_lca", "zero_anchor"}:
        raise ValueError("Stage A encoding supports only true_lca and zero_anchor")
    previous_anchor = model.anchor_mode
    model.anchor_mode = anchor_mode
    model.eval()
    result: dict[str, ProductMeasure] = {}
    try:
        with torch.no_grad():
            for index, program in enumerate(programs, start=1):
                if program.item_id in result:
                    raise ValueError(f"duplicate Stage A program ID: {program.item_id}")
                measure = model.encode_product_measure(program.tree)
                if not torch.isfinite(measure.points).all():
                    raise FloatingPointError(f"non-finite encoded measure: {program.item_id}")
                result[program.item_id] = measure
                if progress_callback is not None and (index % 500 == 0 or index == len(programs)):
                    progress_callback(
                        {
                            "phase": phase,
                            "anchor_mode": anchor_mode,
                            "programs_observed": index,
                            "program_count": len(programs),
                        }
                    )
    finally:
        model.anchor_mode = previous_anchor
    return result


def run_stage_a_validation_seed(
    *,
    split: StageASplit,
    calibration_pairs: Sequence[Mapping[str, Any]],
    seed: int,
    output_dir: Path,
    protocol_sha256: str,
    calibration_manifest_sha256: str,
    dim: int = 8,
    epochs: int = 5,
    batch_size: int = 8,
    learning_rate: float = 0.003,
    gradient_clip_norm: float = 1.0,
    lambda_edge: float = 1.0,
    lambda_gromov: float = 0.1,
    lambda_branch: float = 1.0,
    max_paths: int = 64,
    max_ball_fraction: float = 0.35,
    active_curvatures: Sequence[float] = ACTIVE_CURVATURES,
    near_zero_curvature: float = NEAR_ZERO_CURVATURE,
    sinkhorn_kappa: float = 0.05,
    sinkhorn_iterations: int = 128,
    projection_iterations: int = 2048,
    marginal_tolerance: float = 1e-7,
    query_batch_size: int = 4,
    gallery_batch_size: int = 32,
    torch_num_threads: int = 1,
    implementation: Mapping[str, Any] | None = None,
    progress_callback: Any | None = None,
) -> dict[str, Any]:
    """Train one seed and evaluate every frozen Stage A validation cell."""

    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / f"seed_{seed}_validation.json"
    partial_path = result_path.with_suffix(".partial.json")
    execution_config = {
        "dim": dim,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "gradient_clip_norm": gradient_clip_norm,
        "lambda_edge": lambda_edge,
        "lambda_gromov": lambda_gromov,
        "lambda_branch": lambda_branch,
        "max_paths": max_paths,
        "max_ball_fraction": max_ball_fraction,
        "active_curvatures": [float(value) for value in active_curvatures],
        "near_zero_curvature": near_zero_curvature,
        "sinkhorn_kappa": sinkhorn_kappa,
        "sinkhorn_iterations": sinkhorn_iterations,
        "projection_iterations": projection_iterations,
        "marginal_tolerance": marginal_tolerance,
        "query_batch_size": query_batch_size,
        "gallery_batch_size": gallery_batch_size,
        "torch_num_threads": torch_num_threads,
    }
    implementation_record = dict(implementation or {"mode": "library_call_without_repository_provenance"})
    if result_path.exists():
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        if (
            payload.get("status") == "complete"
            and payload.get("seed") == seed
            and payload.get("protocol_sha256") == protocol_sha256
            and payload.get("calibration_manifest_sha256") == calibration_manifest_sha256
            and payload.get("execution_config") == execution_config
            and payload.get("implementation") == implementation_record
        ):
            return payload
        raise ValueError(f"refusing to overwrite incompatible seed result: {result_path}")

    checkpoint_path = output_dir / f"seed_{seed}_encoder.pt"
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if (
            checkpoint.get("seed") != seed
            or checkpoint.get("protocol_sha256") != protocol_sha256
            or checkpoint.get("calibration_manifest_sha256") != calibration_manifest_sha256
            or checkpoint.get("execution_config") != execution_config
            or checkpoint.get("implementation") != implementation_record
        ):
            raise ValueError(f"refusing to reuse incompatible checkpoint: {checkpoint_path}")
        model = _model_from_checkpoint(checkpoint)
        training_history = list(checkpoint["training_history"])
        training_metadata = dict(checkpoint["training_metadata"])
    else:
        model, training_history, training_metadata = train_stage_a_encoder(
            split.train,
            seed=seed,
            dim=dim,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            gradient_clip_norm=gradient_clip_norm,
            lambda_edge=lambda_edge,
            lambda_gromov=lambda_gromov,
            lambda_branch=lambda_branch,
            max_paths=max_paths,
            torch_num_threads=torch_num_threads,
            progress_callback=progress_callback,
        )
        checkpoint = {
            "seed": seed,
            "protocol_sha256": protocol_sha256,
            "calibration_manifest_sha256": calibration_manifest_sha256,
            "execution_config": execution_config,
            "implementation": implementation_record,
            "model_config": _model_config(model),
            "model_state_dict": model.state_dict(),
            "token_to_id": model.token_to_id,
            "training_history": training_history,
            "training_metadata": training_metadata,
        }
        _atomic_torch_save(checkpoint_path, checkpoint)
    checkpoint_sha256 = stable_sha256(checkpoint_path.read_bytes())
    cells: dict[str, dict[str, Any]] = {}
    if partial_path.exists():
        partial = json.loads(partial_path.read_text(encoding="utf-8"))
        if (
            partial.get("status") != "partial"
            or partial.get("seed") != seed
            or partial.get("protocol_sha256") != protocol_sha256
            or partial.get("calibration_manifest_sha256") != calibration_manifest_sha256
            or partial.get("execution_config") != execution_config
            or partial.get("implementation") != implementation_record
            or partial.get("checkpoint", {}).get("sha256") != checkpoint_sha256
        ):
            raise ValueError(f"refusing to reuse incompatible partial result: {partial_path}")
        cells.update(partial.get("cells", {}))

    train_measures = encode_stage_a_programs(
        model,
        split.train,
        anchor_mode="true_lca",
        progress_callback=progress_callback,
        phase="encoding_train_true_lca",
    )
    max_active_curvature = max(float(value) for value in active_curvatures)
    maximum_lca_norm = max(
        float(torch.linalg.vector_norm(measure.points[:, 0], dim=-1).max())
        for measure in train_measures.values()
    )
    allowed_norm = max_ball_fraction / math.sqrt(max_active_curvature)
    lca_coordinate_scale = 1.0 if maximum_lca_norm <= 0.0 else min(1.0, allowed_norm / maximum_lca_norm)
    role_scales = (lca_coordinate_scale, 1.0, 1.0)
    train_measures = {
        item_id: scale_product_measure(measure, role_scales=role_scales)
        for item_id, measure in train_measures.items()
    }
    role_calibration = calibrate_euclidean_role_weights(
        train_measures,
        calibration_pairs,
    )
    eee_geometry = RoleProductGeometry(
        factor_curvatures=(0.0, 0.0, 0.0),
        factor_weights=role_calibration.euclidean_weights,
        side_weight=0.0,
        unoriented=True,
    )
    cost_scale = calibration_cost_scale(
        train_measures,
        calibration_pairs,
        eee_geometry,
    )
    epsilon = scaled_sinkhorn_epsilon(cost_scale.cost_scale, kappa=sinkhorn_kappa)

    query_true = encode_stage_a_programs(
        model,
        split.query,
        anchor_mode="true_lca",
        progress_callback=progress_callback,
        phase="encoding_validation_query_true_lca",
    )
    gallery_true = encode_stage_a_programs(
        model,
        split.gallery,
        anchor_mode="true_lca",
        progress_callback=progress_callback,
        phase="encoding_validation_gallery_true_lca",
    )
    query_zero = encode_stage_a_programs(
        model,
        split.query,
        anchor_mode="zero_anchor",
        progress_callback=progress_callback,
        phase="encoding_validation_query_zero_anchor",
    )
    gallery_zero = encode_stage_a_programs(
        model,
        split.gallery,
        anchor_mode="zero_anchor",
        progress_callback=progress_callback,
        phase="encoding_validation_gallery_zero_anchor",
    )
    query_true_values = tuple(scale_product_measure(query_true[item.item_id], role_scales=role_scales) for item in split.query)
    gallery_true_values = tuple(scale_product_measure(gallery_true[item.item_id], role_scales=role_scales) for item in split.gallery)
    query_zero_values = tuple(scale_product_measure(query_zero[item.item_id], role_scales=role_scales) for item in split.query)
    gallery_zero_values = tuple(scale_product_measure(gallery_zero[item.item_id], role_scales=role_scales) for item in split.gallery)
    del train_measures, query_true, gallery_true, query_zero, gallery_zero

    cell_specs = [
        ("EEE_true_LCA", (0.0, 0.0, 0.0), query_true_values, gallery_true_values),
        ("EEE_zero_anchor", (0.0, 0.0, 0.0), query_zero_values, gallery_zero_values),
        (
            "HEE_near_zero_true_LCA",
            (near_zero_curvature, 0.0, 0.0),
            query_true_values,
            gallery_true_values,
        ),
    ]
    cell_specs.extend(
        (
            curvature_cell_id(curvature),
            (float(curvature), 0.0, 0.0),
            query_true_values,
            gallery_true_values,
        )
        for curvature in active_curvatures
    )
    for cell_index, (cell_id, curvatures, queries, gallery) in enumerate(cell_specs, start=1):
        if cell_id in cells:
            continue
        geometry = RoleProductGeometry(
            factor_curvatures=curvatures,
            factor_weights=matched_role_weights(
                role_calibration.canonical_weights,
                factor_curvatures=curvatures,
            ),
            side_weight=0.0,
            unoriented=True,
        )
        if progress_callback is not None:
            progress_callback(
                {
                    "phase": "validation_cell",
                    "seed": seed,
                    "cell_id": cell_id,
                    "cell_index": cell_index,
                    "cell_count": len(cell_specs),
                }
            )
        distances = full_gallery_sinkhorn_divergence(
            queries,
            gallery,
            geometry,
            epsilon=epsilon,
            query_batch_size=query_batch_size,
            gallery_batch_size=gallery_batch_size,
            sinkhorn_iterations=sinkhorn_iterations,
            projection_iterations=projection_iterations,
            marginal_tolerance=marginal_tolerance,
        )
        summary = summarize_problem_macro_retrieval(
            distances,
            query_ids=tuple(item.item_id for item in split.query),
            query_problem_ids=tuple(item.problem_id for item in split.query),
            gallery_ids=tuple(item.item_id for item in split.gallery),
            gallery_problem_ids=tuple(item.problem_id for item in split.gallery),
            r=8,
        )
        distance_path = output_dir / f"seed_{seed}_{cell_id}_distances.pt"
        _atomic_torch_save(distance_path, distances.to(dtype=torch.float64, device="cpu"))
        cells[cell_id] = {
            "factor_curvatures": list(curvatures),
            "factor_weights": list(geometry.factor_weights),
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
        partial = _seed_payload(
            seed=seed,
            protocol_sha256=protocol_sha256,
            calibration_manifest_sha256=calibration_manifest_sha256,
            execution_config=execution_config,
            implementation=implementation_record,
            checkpoint_path=checkpoint_path,
            checkpoint_sha256=checkpoint_sha256,
            training_history=training_history,
            training_metadata=training_metadata,
            role_scales=role_scales,
            maximum_lca_norm=maximum_lca_norm,
            role_calibration=asdict(role_calibration),
            cost_scale=asdict(cost_scale),
            epsilon=epsilon,
            cells=cells,
            status="partial",
        )
        _atomic_json_write(partial_path, partial)

    payload = _seed_payload(
        seed=seed,
        protocol_sha256=protocol_sha256,
        calibration_manifest_sha256=calibration_manifest_sha256,
        execution_config=execution_config,
        implementation=implementation_record,
        checkpoint_path=checkpoint_path,
        checkpoint_sha256=checkpoint_sha256,
        training_history=training_history,
        training_metadata=training_metadata,
        role_scales=role_scales,
        maximum_lca_norm=maximum_lca_norm,
        role_calibration=asdict(role_calibration),
        cost_scale=asdict(cost_scale),
        epsilon=epsilon,
        cells=cells,
        status="complete",
    )
    _atomic_json_write(result_path, payload)
    partial_path.unlink(missing_ok=True)
    return json.loads(result_path.read_text(encoding="utf-8"))


def select_active_curvature(
    seed_payloads: Sequence[Mapping[str, Any]],
    *,
    active_curvatures: Sequence[float] = ACTIVE_CURVATURES,
    expected_seeds: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Apply the frozen validation rule and write no test-facing conclusion."""

    if not seed_payloads or any(payload.get("status") != "complete" for payload in seed_payloads):
        raise ValueError("all registered seed payloads must be complete")
    seed_ids = [int(payload["seed"]) for payload in seed_payloads]
    if len(seed_ids) != len(set(seed_ids)):
        raise ValueError("validation selection received duplicate model seeds")
    if expected_seeds is not None and set(seed_ids) != {int(seed) for seed in expected_seeds}:
        raise ValueError("validation selection does not contain the registered model seed set")
    candidate_scores: dict[float, float] = {}
    candidate_task_scores: dict[float, dict[str, float]] = {}
    for curvature in active_curvatures:
        cell_id = curvature_cell_id(curvature)
        task_scores = _seed_averaged_task_scores(seed_payloads, cell_id=cell_id)
        candidate_task_scores[float(curvature)] = task_scores
        candidate_scores[float(curvature)] = sum(task_scores.values()) / len(task_scores)
    selected = min(
        candidate_scores,
        key=lambda curvature: (-candidate_scores[curvature], curvature),
    )
    selected_cell = curvature_cell_id(selected)
    cell_ids = (
        "EEE_true_LCA",
        "EEE_zero_anchor",
        "HEE_near_zero_true_LCA",
        *(curvature_cell_id(curvature) for curvature in active_curvatures),
    )
    cell_scores = {
        cell_id: _seed_averaged_task_scores(seed_payloads, cell_id=cell_id)
        for cell_id in cell_ids
    }
    contrasts = {
        "H1_EEE_true_LCA_minus_EEE_zero_anchor": _task_score_contrast(
            cell_scores["EEE_true_LCA"],
            cell_scores["EEE_zero_anchor"],
        ),
        "H3_selected_active_minus_EEE_true_LCA": _task_score_contrast(
            cell_scores[selected_cell],
            cell_scores["EEE_true_LCA"],
        ),
        "H3_selected_active_minus_HEE_near_zero_true_LCA": _task_score_contrast(
            cell_scores[selected_cell],
            cell_scores["HEE_near_zero_true_LCA"],
        ),
    }
    return {
        "selection_rule": "maximum_mean_validation_problem_macro_MAP_at_8_across_model_seeds_then_smallest_curvature",
        "selected_active_curvature": selected,
        "selected_cell_id": selected_cell,
        "candidate_mean_validation_problem_macro_MAP_at_8": {
            str(curvature): candidate_scores[curvature]
            for curvature in sorted(candidate_scores)
        },
        "cell_mean_validation_problem_macro_MAP_at_8": {
            cell_id: sum(task_scores.values()) / len(task_scores)
            for cell_id, task_scores in cell_scores.items()
        },
        "descriptive_validation_contrasts": contrasts,
        "seed_count": len(seed_payloads),
        "problem_count": len(next(iter(candidate_task_scores.values()))),
        "test_program_ids_materialized": False,
        "test_relevance_labels_opened": False,
        "test_retrieval_metrics_computed": False,
    }


def build_validation_selection_record(
    seed_payloads: Sequence[Mapping[str, Any]],
    *,
    protocol_bytes: bytes,
    calibration_manifest_bytes: bytes,
) -> dict[str, Any]:
    """Recompute the complete validation-only curvature selection record."""

    protocol = json.loads(protocol_bytes)
    registered_seeds = tuple(int(seed) for seed in protocol["encoder_training"]["model_seeds"])
    active_curvatures = tuple(
        float(cell["factor_curvatures"][0])
        for cell in protocol["geometry_cells"]["gate_C_active_candidates"]
    )
    selection = select_active_curvature(
        seed_payloads,
        active_curvatures=active_curvatures,
        expected_seeds=registered_seeds,
    )
    selection.update(
        {
            "schema_version": "code2hyp-stage-a-validation-selection-v1",
            "protocol_sha256": stable_sha256(protocol_bytes),
            "calibration_manifest_sha256": stable_sha256(calibration_manifest_bytes),
            "registered_seeds": list(registered_seeds),
        }
    )
    return selection


def scale_product_measure(
    measure: ProductMeasure,
    *,
    role_scales: Sequence[float],
) -> ProductMeasure:
    if len(role_scales) != measure.points.shape[1]:
        raise ValueError("role scale count must match product factor count")
    scale = measure.points.new_tensor(tuple(float(value) for value in role_scales)).view(1, -1, 1)
    return ProductMeasure(points=measure.points * scale, mass=measure.mass)


def matched_role_weights(
    canonical_weights: Sequence[float],
    *,
    factor_curvatures: Sequence[float],
) -> tuple[float, ...]:
    if len(canonical_weights) != len(factor_curvatures):
        raise ValueError("weight and curvature counts must match")
    return tuple(
        float(weight) if float(curvature) > 0.0 else 4.0 * float(weight)
        for weight, curvature in zip(canonical_weights, factor_curvatures)
    )


def curvature_cell_id(curvature: float) -> str:
    token = f"{float(curvature):g}".replace(".", "p")
    return f"HEE_c{token}_true_LCA"


def _seed_averaged_task_scores(
    seed_payloads: Sequence[Mapping[str, Any]],
    *,
    cell_id: str,
) -> dict[str, float]:
    per_task: dict[str, list[float]] = {}
    reference_tasks: set[str] | None = None
    for payload in seed_payloads:
        try:
            task_scores = payload["cells"][cell_id]["metrics"]["task_scores"]
        except KeyError as error:
            raise ValueError(f"validation seed is missing cell {cell_id!r}") from error
        tasks = set(task_scores)
        if reference_tasks is None:
            reference_tasks = tasks
        elif tasks != reference_tasks:
            raise ValueError(f"task set differs across seeds for cell {cell_id!r}")
        for task, value in task_scores.items():
            per_task.setdefault(str(task), []).append(float(value))
    return {
        task: sum(values) / len(values)
        for task, values in sorted(per_task.items())
    }


def _task_score_contrast(
    treatment: Mapping[str, float],
    control: Mapping[str, float],
) -> dict[str, Any]:
    if set(treatment) != set(control):
        raise ValueError("contrast cells do not contain the same validation tasks")
    differences = {
        task: float(treatment[task]) - float(control[task])
        for task in sorted(treatment)
    }
    values = tuple(differences.values())
    return {
        "mean_delta_problem_macro_MAP_at_8": sum(values) / len(values),
        "positive_problem_count": sum(value > 0.0 for value in values),
        "zero_problem_count": sum(value == 0.0 for value in values),
        "negative_problem_count": sum(value < 0.0 for value in values),
        "task_deltas": differences,
        "inference_status": "validation_descriptive_only",
    }


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSONL at {path}:{line_number}") from error


def _seed_payload(
    *,
    seed: int,
    protocol_sha256: str,
    calibration_manifest_sha256: str,
    execution_config: Mapping[str, Any],
    implementation: Mapping[str, Any],
    checkpoint_path: Path,
    checkpoint_sha256: str,
    training_history: Sequence[Mapping[str, float]],
    training_metadata: Mapping[str, Any],
    role_scales: Sequence[float],
    maximum_lca_norm: float,
    role_calibration: Mapping[str, Any],
    cost_scale: Mapping[str, Any],
    epsilon: float,
    cells: Mapping[str, Any],
    status: str,
) -> dict[str, Any]:
    return {
        "schema_version": "code2hyp-stage-a-validation-seed-v1",
        "status": status,
        "seed": seed,
        "protocol_sha256": protocol_sha256,
        "calibration_manifest_sha256": calibration_manifest_sha256,
        "execution_config": dict(execution_config),
        "implementation": dict(implementation),
        "checkpoint": {
            "path": checkpoint_path.name,
            "sha256": checkpoint_sha256,
        },
        "training_history": list(training_history),
        "training_metadata": dict(training_metadata),
        "coordinate_scaling": {
            "role_scales": list(role_scales),
            "maximum_unscaled_training_LCA_norm": maximum_lca_norm,
        },
        "role_calibration": dict(role_calibration),
        "sinkhorn_calibration": {
            "cost_scale": dict(cost_scale),
            "epsilon": epsilon,
        },
        "cells": dict(cells),
        "validation_metrics_computed": bool(cells),
        "test_program_ids_materialized": False,
        "test_relevance_labels_opened": False,
        "test_retrieval_metrics_computed": False,
    }


def _atomic_json_write(path: Path, payload: Mapping[str, Any]) -> None:
    content = canonical_json_bytes(payload)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(content)
    os.replace(temporary, path)


def _atomic_torch_save(path: Path, payload: Any) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _model_config(model: RawASTCode2Hyp) -> dict[str, Any]:
    return {
        "dim": model.dim,
        "token_dim": model.token_dim,
        "manifold": model.manifold,
        "curvature": model.curvature,
        "max_paths": model.max_paths,
        "terminal_policy": model.terminal_policy,
        "node_input_mode": model.node_input_mode,
        "path_object_mode": model.path_object_mode,
        "method_aggregation": model.method_aggregation,
        "path_cost_orientation": model.path_cost_orientation,
        "path_selection_policy": model.path_selection_policy,
        "anchor_mode": "true_lca",
    }


def _model_from_checkpoint(checkpoint: Mapping[str, Any]) -> RawASTCode2Hyp:
    config = dict(checkpoint["model_config"])
    model = RawASTCode2Hyp(checkpoint["token_to_id"], **config)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    return model
