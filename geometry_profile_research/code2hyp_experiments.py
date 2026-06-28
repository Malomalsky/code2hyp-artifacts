from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from .code2hyp_data import (
    apply_lexical_ablation,
    encode_records_to_multilabel_batch,
    filter_records_by_known_label_subtokens,
    label_subtoken_coverage,
    load_code2vec_records,
    sample_code2vec_records,
)
from .code2hyp_synthetic import SyntheticCode2HypConfig, make_synthetic_code2hyp_dataset
from .code2hyp_torch import (
    Code2HypTorchConfig,
    Code2HypTorchModel,
    batch_method_transport_diagnostics,
    batch_poincare_frechet_diagnostics,
    batch_poincare_radius_utilization,
    batch_structural_distance_level_summary,
    batch_structural_distance_regularizer,
    batch_structural_neighbor_exact_overlap_at_k,
    batch_structural_neighbor_overlap_at_k,
    batch_structural_normalized_stress,
    batch_structural_rank_regularizer,
    batch_structural_relation_conditioned_diagnostics,
    batch_structural_spearman_correlation,
)
from .code2hyp_training import (
    compute_multilabel_pos_weight,
    evaluate_accuracy,
    evaluate_multilabel_metrics,
    fit_multilabel_supervised,
    fit_supervised,
    slice_batch,
)


METHOD_TRANSPORT_DIAGNOSTIC_TARGETS = {
    "prefix": "prefix_tree",
    "edit": "edit",
    "jaccard": "jaccard_bigrams",
}


@dataclass(frozen=True)
class SyntheticComparisonConfig:
    examples: int = 128
    contexts_per_method: int = 6
    max_path_length: int = 5
    branches: int = 4
    token_dim: int = 8
    structural_dim: int = 8
    epochs: int = 30
    batch_size: int = 32
    learning_rate: float = 0.04
    structural_loss_weight: float = 0.1
    dataset_seed: int = 13
    model_seeds: tuple[int, ...] = (101, 202, 303)


@dataclass(frozen=True)
class RealCode2HypPilotConfig:
    train_limit: int = 2048
    val_limit: int = 512
    max_contexts: int = 100
    max_path_length: int = 8
    token_dim: int = 32
    structural_dim: int = 32
    curvature: float = 1.0
    path_encoder: str = "mean"
    representation_transform: str = "identity"
    epochs: int = 3
    batch_size: int = 32
    learning_rate: float = 0.003
    structural_loss_weight: float = 0.05
    structural_regularizer: str = "distance"
    lexical_ablation: str = "original"
    use_positive_weighting: bool = True
    max_positive_weight: float = 20.0
    model_seeds: tuple[int, ...] = (101, 202, 303)
    variant_filter: tuple[str, ...] | None = None
    sample_seed: int | None = None
    context_sample_seed: int | None = None
    structural_eval_limit: int | None = 512
    structural_eval_seed: int = 314159


def _model_config(
    base: Code2HypTorchConfig,
    trainable_curvature: bool,
    curvature_override: float | None = None,
) -> Code2HypTorchConfig:
    return Code2HypTorchConfig(
        token_vocab_size=base.token_vocab_size,
        ast_node_vocab_size=base.ast_node_vocab_size,
        label_vocab_size=base.label_vocab_size,
        token_dim=base.token_dim,
        structural_dim=base.structural_dim,
        curvature=base.curvature if curvature_override is None else curvature_override,
        trainable_curvature=trainable_curvature,
        path_encoder=base.path_encoder,
        representation_transform=base.representation_transform,
        frechet_steps=base.frechet_steps,
        frechet_step_size=base.frechet_step_size,
        factorized_mixer_rank=base.factorized_mixer_rank,
        path_message_passing_steps=base.path_message_passing_steps,
    )


def _run_multilabel_variant(
    variant_name: str,
    torch_variant: str,
    trainable_curvature: bool,
    structural_loss_weight: float,
    base_config: Code2HypTorchConfig,
    train_dataset,
    validation_dataset,
    model_seed: int,
    config: RealCode2HypPilotConfig,
    pos_weight: torch.Tensor | None,
    structural_loss_schedule: str = "constant",
    structural_regularizer: str | None = None,
    curvature_override: float | None = None,
) -> dict[str, Any]:
    torch.manual_seed(model_seed)
    model = Code2HypTorchModel(
        _model_config(
            base_config,
            trainable_curvature=trainable_curvature,
            curvature_override=curvature_override,
        ),
        variant=torch_variant,  # type: ignore[arg-type]
    )
    initial_metrics = evaluate_multilabel_metrics(
        model,
        validation_dataset.batch,
        validation_dataset.labels,
        validation_dataset.target_sizes,
        batch_size=config.batch_size,
    )
    effective_structural_regularizer = structural_regularizer or config.structural_regularizer
    history = fit_multilabel_supervised(
        model,
        train_dataset.batch,
        train_dataset.labels,
        train_dataset.target_sizes,
        epochs=config.epochs,
        batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        structural_loss_weight=structural_loss_weight,
        structural_loss_schedule=structural_loss_schedule,
        structural_regularizer=effective_structural_regularizer,
        pos_weight=pos_weight,
        seed=model_seed,
    )
    validation_metrics = evaluate_multilabel_metrics(
        model,
        validation_dataset.batch,
        validation_dataset.labels,
        validation_dataset.target_sizes,
        batch_size=config.batch_size,
    )
    validation_fixed_top3_metrics = evaluate_multilabel_metrics(
        model,
        validation_dataset.batch,
        validation_dataset.labels,
        validation_dataset.target_sizes,
        batch_size=config.batch_size,
        selection="fixed_topk",
        fixed_k=3,
    )
    validation_threshold05_metrics = evaluate_multilabel_metrics(
        model,
        validation_dataset.batch,
        validation_dataset.labels,
        validation_dataset.target_sizes,
        batch_size=config.batch_size,
        selection="threshold",
        threshold=0.0,
    )
    diagnostic_batch = validation_dataset.batch
    structural_diagnostic_records = int(validation_dataset.labels.shape[0])
    if config.structural_eval_limit is not None:
        if config.structural_eval_limit <= 0:
            raise ValueError("structural_eval_limit must be positive or None")
        if structural_diagnostic_records > config.structural_eval_limit:
            generator = torch.Generator(device=validation_dataset.labels.device).manual_seed(config.structural_eval_seed)
            indices = torch.randperm(
                structural_diagnostic_records,
                generator=generator,
                device=validation_dataset.labels.device,
            )[: config.structural_eval_limit]
            diagnostic_batch = slice_batch(validation_dataset.batch, indices)
            structural_diagnostic_records = int(indices.numel())
    with torch.no_grad():
        output = model(diagnostic_batch)
        validation_structural_loss = batch_structural_distance_regularizer(output, diagnostic_batch)
        validation_structural_normalized_stress = batch_structural_normalized_stress(
            output,
            diagnostic_batch,
        )
        validation_structural_rank_loss = batch_structural_rank_regularizer(output, diagnostic_batch)
        validation_structural_spearman = batch_structural_spearman_correlation(output, diagnostic_batch)
        validation_structural_edit_spearman = batch_structural_spearman_correlation(
            output,
            diagnostic_batch,
            target_distance="edit",
        )
        validation_structural_edit_normalized_stress = batch_structural_normalized_stress(
            output,
            diagnostic_batch,
            target_distance="edit",
        )
        validation_structural_jaccard_spearman = batch_structural_spearman_correlation(
            output,
            diagnostic_batch,
            target_distance="jaccard_bigrams",
        )
        validation_structural_jaccard_normalized_stress = batch_structural_normalized_stress(
            output,
            diagnostic_batch,
            target_distance="jaccard_bigrams",
        )
        validation_structural_neighbor_overlap_at_1 = batch_structural_neighbor_overlap_at_k(
            output,
            diagnostic_batch,
            k=1,
        )
        validation_structural_neighbor_overlap_at_3 = batch_structural_neighbor_overlap_at_k(
            output,
            diagnostic_batch,
            k=3,
        )
        validation_structural_neighbor_exact_overlap_at_1 = batch_structural_neighbor_exact_overlap_at_k(
            output,
            diagnostic_batch,
            k=1,
        )
        validation_structural_neighbor_exact_overlap_at_3 = batch_structural_neighbor_exact_overlap_at_k(
            output,
            diagnostic_batch,
            k=3,
        )
        validation_structural_prefix_distance_level_summary = batch_structural_distance_level_summary(
            output,
            diagnostic_batch,
            target_distance="prefix_tree",
        )
        validation_poincare_frechet_diagnostics = batch_poincare_frechet_diagnostics(output)
        validation_poincare_radius_utilization = batch_poincare_radius_utilization(output, diagnostic_batch)
        validation_relation_conditioned_diagnostics = batch_structural_relation_conditioned_diagnostics(
            output,
            diagnostic_batch,
        )
        validation_method_transport_diagnostics = batch_method_transport_diagnostics(output, diagnostic_batch)
        validation_method_transport_target_diagnostics = {
            label: batch_method_transport_diagnostics(
                output,
                diagnostic_batch,
                target_distance=target_distance,  # type: ignore[arg-type]
            )
            for label, target_distance in METHOD_TRANSPORT_DIAGNOSTIC_TARGETS.items()
        }
    frechet_diagnostics = validation_poincare_frechet_diagnostics or {}
    radius_utilization = validation_poincare_radius_utilization or {}
    relation_conditioned_diagnostics = validation_relation_conditioned_diagnostics or {}
    method_transport_diagnostics = validation_method_transport_diagnostics or {}
    method_transport_target_diagnostics = validation_method_transport_target_diagnostics or {}
    method_transport_target_fields: dict[str, float | int | None] = {}
    for label, diagnostics in method_transport_target_diagnostics.items():
        method_transport_target_fields[f"validation_method_transport_{label}_pair_count"] = (
            int(diagnostics["method_pair_count"].detach()) if "method_pair_count" in diagnostics else None
        )
        method_transport_target_fields[f"validation_method_transport_{label}_spearman"] = (
            float(diagnostics["transport_spearman"].detach()) if "transport_spearman" in diagnostics else None
        )
        method_transport_target_fields[f"validation_method_transport_{label}_normalized_stress"] = (
            float(diagnostics["transport_normalized_stress"].detach())
            if "transport_normalized_stress" in diagnostics
            else None
        )
        method_transport_target_fields[f"validation_method_aggregate_{label}_spearman"] = (
            float(diagnostics["aggregate_spearman"].detach()) if "aggregate_spearman" in diagnostics else None
        )
        method_transport_target_fields[f"validation_method_aggregate_{label}_normalized_stress"] = (
            float(diagnostics["aggregate_normalized_stress"].detach())
            if "aggregate_normalized_stress" in diagnostics
            else None
        )
    return {
        "variant": variant_name,
        "model_seed": model_seed,
        "parameter_count": model.parameter_count(),
        "initial_validation_f1": initial_metrics["f1"],
        "validation_precision": validation_metrics["precision"],
        "validation_recall": validation_metrics["recall"],
        "validation_f1": validation_metrics["f1"],
        "validation_oracle_topk_precision": validation_metrics["precision"],
        "validation_oracle_topk_recall": validation_metrics["recall"],
        "validation_oracle_topk_f1": validation_metrics["f1"],
        "validation_oracle_topk_predicted_positive_count_mean": validation_metrics[
            "predicted_positive_count_mean"
        ],
        "validation_fixed_top3_precision": validation_fixed_top3_metrics["precision"],
        "validation_fixed_top3_recall": validation_fixed_top3_metrics["recall"],
        "validation_fixed_top3_f1": validation_fixed_top3_metrics["f1"],
        "validation_fixed_top3_predicted_positive_count_mean": validation_fixed_top3_metrics[
            "predicted_positive_count_mean"
        ],
        "validation_threshold05_precision": validation_threshold05_metrics["precision"],
        "validation_threshold05_recall": validation_threshold05_metrics["recall"],
        "validation_threshold05_f1": validation_threshold05_metrics["f1"],
        "validation_threshold05_predicted_positive_count_mean": validation_threshold05_metrics[
            "predicted_positive_count_mean"
        ],
        "final_train_loss": history[-1]["loss"],
        "final_train_f1": history[-1]["f1"],
        "history": history,
        "validation_structural_loss": float(validation_structural_loss.detach()),
        "validation_structural_normalized_stress": float(validation_structural_normalized_stress.detach()),
        "validation_structural_rank_loss": float(validation_structural_rank_loss.detach()),
        "validation_structural_spearman": float(validation_structural_spearman.detach()),
        "validation_structural_edit_spearman": float(validation_structural_edit_spearman.detach()),
        "validation_structural_edit_normalized_stress": float(validation_structural_edit_normalized_stress.detach()),
        "validation_structural_jaccard_spearman": float(validation_structural_jaccard_spearman.detach()),
        "validation_structural_jaccard_normalized_stress": float(
            validation_structural_jaccard_normalized_stress.detach()
        ),
        "validation_structural_diagnostic_records": structural_diagnostic_records,
        "validation_structural_neighbor_overlap_at_1": float(validation_structural_neighbor_overlap_at_1.detach()),
        "validation_structural_neighbor_overlap_at_3": float(validation_structural_neighbor_overlap_at_3.detach()),
        "validation_structural_neighbor_exact_overlap_at_1": float(
            validation_structural_neighbor_exact_overlap_at_1.detach()
        ),
        "validation_structural_neighbor_exact_overlap_at_3": float(
            validation_structural_neighbor_exact_overlap_at_3.detach()
        ),
        "validation_structural_neighbor_recall_at_1": float(validation_structural_neighbor_overlap_at_1.detach()),
        "validation_structural_neighbor_recall_at_3": float(validation_structural_neighbor_overlap_at_3.detach()),
        "validation_structural_prefix_distance_level_summary": validation_structural_prefix_distance_level_summary,
        "validation_relation_conditioned_prefix_spearman": (
            float(relation_conditioned_diagnostics["prefix_spearman"].detach())
            if "prefix_spearman" in relation_conditioned_diagnostics
            else None
        ),
        "validation_relation_conditioned_prefix_normalized_stress": (
            float(relation_conditioned_diagnostics["prefix_normalized_stress"].detach())
            if "prefix_normalized_stress" in relation_conditioned_diagnostics
            else None
        ),
        "validation_relation_conditioned_edit_spearman": (
            float(relation_conditioned_diagnostics["edit_spearman"].detach())
            if "edit_spearman" in relation_conditioned_diagnostics
            else None
        ),
        "validation_relation_conditioned_edit_normalized_stress": (
            float(relation_conditioned_diagnostics["edit_normalized_stress"].detach())
            if "edit_normalized_stress" in relation_conditioned_diagnostics
            else None
        ),
        "validation_relation_conditioned_jaccard_spearman": (
            float(relation_conditioned_diagnostics["jaccard_spearman"].detach())
            if "jaccard_spearman" in relation_conditioned_diagnostics
            else None
        ),
        "validation_relation_conditioned_jaccard_normalized_stress": (
            float(relation_conditioned_diagnostics["jaccard_normalized_stress"].detach())
            if "jaccard_normalized_stress" in relation_conditioned_diagnostics
            else None
        ),
        "validation_method_transport_pair_count": (
            int(method_transport_diagnostics["method_pair_count"].detach())
            if "method_pair_count" in method_transport_diagnostics
            else None
        ),
        "validation_method_transport_spearman": (
            float(method_transport_diagnostics["transport_spearman"].detach())
            if "transport_spearman" in method_transport_diagnostics
            else None
        ),
        "validation_method_transport_normalized_stress": (
            float(method_transport_diagnostics["transport_normalized_stress"].detach())
            if "transport_normalized_stress" in method_transport_diagnostics
            else None
        ),
        "validation_method_aggregate_spearman": (
            float(method_transport_diagnostics["aggregate_spearman"].detach())
            if "aggregate_spearman" in method_transport_diagnostics
            else None
        ),
        "validation_method_aggregate_normalized_stress": (
            float(method_transport_diagnostics["aggregate_normalized_stress"].detach())
            if "aggregate_normalized_stress" in method_transport_diagnostics
            else None
        ),
        **method_transport_target_fields,
        "validation_poincare_frechet_residual_mean": (
            float(frechet_diagnostics["residual_mean"].detach()) if "residual_mean" in frechet_diagnostics else None
        ),
        "validation_poincare_frechet_residual_max": (
            float(frechet_diagnostics["residual_max"].detach()) if "residual_max" in frechet_diagnostics else None
        ),
        "validation_poincare_frechet_objective_mean": (
            float(frechet_diagnostics["objective_mean"].detach()) if "objective_mean" in frechet_diagnostics else None
        ),
        "validation_poincare_context_radius_ratio_mean": (
            float(radius_utilization["context_radius_ratio_mean"].detach())
            if "context_radius_ratio_mean" in radius_utilization
            else None
        ),
        "validation_poincare_context_radius_ratio_max": (
            float(radius_utilization["context_radius_ratio_max"].detach())
            if "context_radius_ratio_max" in radius_utilization
            else None
        ),
        "validation_poincare_context_near_boundary_rate": (
            float(radius_utilization["context_near_boundary_rate"].detach())
            if "context_near_boundary_rate" in radius_utilization
            else None
        ),
        "validation_poincare_aggregate_radius_ratio_mean": (
            float(radius_utilization["aggregate_radius_ratio_mean"].detach())
            if "aggregate_radius_ratio_mean" in radius_utilization
            else None
        ),
        "validation_poincare_aggregate_radius_ratio_max": (
            float(radius_utilization["aggregate_radius_ratio_max"].detach())
            if "aggregate_radius_ratio_max" in radius_utilization
            else None
        ),
        "structural_loss_weight": structural_loss_weight,
        "structural_loss_schedule": structural_loss_schedule,
        "structural_regularizer": effective_structural_regularizer,
        "configured_curvature": base_config.curvature if curvature_override is None else curvature_override,
        "curvature_override": curvature_override,
        "curvature": float(output.curvature.detach()),
        "factorized_metric_weights": [
            float(weight.detach()) for weight in model.factorized_metric_weights()
        ],
        "product_attention_bias_weight": float(model.product_attention_bias_weight().detach()),
        "factorized_channel_mixer_rank": model.factorized_channel_mixer_rank(),
    }


def _real_variant_specs(config: RealCode2HypPilotConfig) -> dict[str, dict[str, Any]]:
    return {
        "B1_euclidean": {
            "torch_variant": "euclidean",
            "trainable_curvature": False,
            "structural_loss_weight": 0.0,
        },
        "B2_product_fixed_curvature": {
            "torch_variant": "product",
            "trainable_curvature": False,
            "structural_loss_weight": 0.0,
        },
        "B3_product": {
            "torch_variant": "product",
            "trainable_curvature": True,
            "structural_loss_weight": 0.0,
        },
        "B4_hyperbolic_code2vec": {
            "torch_variant": "hyperbolic",
            "trainable_curvature": False,
            "structural_loss_weight": 0.0,
        },
        "B4T_hyperbolic_code2vec_trainable_curvature": {
            "torch_variant": "hyperbolic",
            "trainable_curvature": True,
            "structural_loss_weight": 0.0,
        },
        "B8_hyperbolic_frechet_code2vec": {
            "torch_variant": "hyperbolic_frechet",
            "trainable_curvature": False,
            "structural_loss_weight": 0.0,
        },
        "B17_hyperbolic_path_mp_code2vec": {
            "torch_variant": "hyperbolic_path_message_passing",
            "trainable_curvature": False,
            "structural_loss_weight": 0.0,
        },
        "B18_hyperbolic_path_mp_struct_rank": {
            "torch_variant": "hyperbolic_path_message_passing",
            "trainable_curvature": False,
            "structural_loss_weight": config.structural_loss_weight,
        },
        "B19_hyperbolic_path_mp_rank_annealed": {
            "torch_variant": "hyperbolic_path_message_passing",
            "trainable_curvature": False,
            "structural_loss_weight": config.structural_loss_weight,
            "structural_loss_schedule": "linear",
        },
        "B20_hyperbolic_path_mp_rank_delayed": {
            "torch_variant": "hyperbolic_path_message_passing",
            "trainable_curvature": False,
            "structural_loss_weight": config.structural_loss_weight,
            "structural_loss_schedule": "delayed_linear",
        },
        "B21_hyperbolic_path_mp_rank_cosine": {
            "torch_variant": "hyperbolic_path_message_passing",
            "trainable_curvature": False,
            "structural_loss_weight": config.structural_loss_weight,
            "structural_loss_schedule": "cosine",
        },
        "B22_hyperbolic_path_mp_rank_warmup_decay": {
            "torch_variant": "hyperbolic_path_message_passing",
            "trainable_curvature": False,
            "structural_loss_weight": config.structural_loss_weight,
            "structural_loss_schedule": "warmup_decay",
        },
        "B23_hyperbolic_path_attention_mp_code2vec": {
            "torch_variant": "hyperbolic_path_attention_message_passing",
            "trainable_curvature": False,
            "structural_loss_weight": 0.0,
        },
        "B24_hyperbolic_path_attention_mp_rank_annealed": {
            "torch_variant": "hyperbolic_path_attention_message_passing",
            "trainable_curvature": False,
            "structural_loss_weight": config.structural_loss_weight,
            "structural_loss_schedule": "linear",
        },
        "B25_hyperbolic_path_depth_attention_mp_code2vec": {
            "torch_variant": "hyperbolic_path_depth_attention_message_passing",
            "trainable_curvature": False,
            "structural_loss_weight": 0.0,
        },
        "B26_hyperbolic_path_depth_attention_mp_rank_annealed": {
            "torch_variant": "hyperbolic_path_depth_attention_message_passing",
            "trainable_curvature": False,
            "structural_loss_weight": config.structural_loss_weight,
            "structural_loss_schedule": "linear",
        },
        "B27_hyperbolic_path_attention_mp_monotone": {
            "torch_variant": "hyperbolic_path_attention_message_passing",
            "trainable_curvature": False,
            "structural_loss_weight": config.structural_loss_weight,
            "structural_loss_schedule": "linear",
            "structural_regularizer": "path_attention_monotone",
        },
        "B28_hyperbolic_path_attention_mp_tree_distance": {
            "torch_variant": "hyperbolic_path_attention_message_passing",
            "trainable_curvature": False,
            "structural_loss_weight": config.structural_loss_weight,
            "structural_loss_schedule": "linear",
            "structural_regularizer": "path_attention_tree_distance",
        },
        "B29_hyperbolic_path_dual_attention_mp_separated": {
            "torch_variant": "hyperbolic_path_dual_attention_message_passing",
            "trainable_curvature": False,
            "structural_loss_weight": config.structural_loss_weight,
            "structural_loss_schedule": "linear",
            "structural_regularizer": "path_dual_attention_separation",
        },
        "B30_hyperbolic_path_dual_attention_mp_rank_separated": {
            "torch_variant": "hyperbolic_path_dual_attention_message_passing",
            "trainable_curvature": False,
            "structural_loss_weight": config.structural_loss_weight,
            "structural_loss_schedule": "linear",
            "structural_regularizer": "path_dual_attention_separation_rank",
        },
        "B31_hyperbolic_path_dual_attention_mp_soft_rank": {
            "torch_variant": "hyperbolic_path_dual_attention_message_passing",
            "trainable_curvature": False,
            "structural_loss_weight": config.structural_loss_weight,
            "structural_loss_schedule": "linear",
            "structural_regularizer": "path_dual_attention_separation_soft_rank",
        },
        "B32_lorentz_path_dual_attention_mp_soft_rank": {
            "torch_variant": "lorentz_path_dual_attention_message_passing",
            "trainable_curvature": False,
            "structural_loss_weight": config.structural_loss_weight,
            "structural_loss_schedule": "linear",
            "structural_regularizer": "path_dual_attention_separation_soft_rank",
        },
        "B34_hyperbolic_path_dual_attention_mp_adaptive_rank": {
            "torch_variant": "hyperbolic_path_dual_attention_message_passing",
            "trainable_curvature": False,
            "structural_loss_weight": config.structural_loss_weight,
            "structural_loss_schedule": "linear",
            "structural_regularizer": "path_dual_attention_separation_adaptive_rank",
        },
        "B9_lorentz_code2vec": {
            "torch_variant": "lorentz",
            "trainable_curvature": False,
            "structural_loss_weight": 0.0,
        },
        "B15_lorentz_product_code2vec": {
            "torch_variant": "lorentz_product",
            "trainable_curvature": False,
            "structural_loss_weight": 0.0,
        },
        "B10_factorized_product_code2vec": {
            "torch_variant": "factorized_product",
            "trainable_curvature": False,
            "structural_loss_weight": 0.0,
        },
        "B11_factorized_product_struct_rank": {
            "torch_variant": "factorized_product",
            "trainable_curvature": False,
            "structural_loss_weight": config.structural_loss_weight,
        },
        "B12_factorized_product_learned_metric_rank": {
            "torch_variant": "factorized_product_learned_metric",
            "trainable_curvature": False,
            "structural_loss_weight": config.structural_loss_weight,
        },
        "B16_factorized_product_three_metric_rank": {
            "torch_variant": "factorized_product_three_metric",
            "trainable_curvature": False,
            "structural_loss_weight": config.structural_loss_weight,
        },
        "B35_code2hyp_product_frechet_adaptive": {
            "torch_variant": "code2hyp_product_frechet",
            "trainable_curvature": True,
            "structural_loss_weight": config.structural_loss_weight,
            "structural_loss_schedule": "delayed_linear",
        },
        "B36_code2hyp_product_frechet_neighbor": {
            "torch_variant": "code2hyp_product_frechet",
            "trainable_curvature": True,
            "structural_loss_weight": config.structural_loss_weight,
            "structural_loss_schedule": "delayed_linear",
            "structural_regularizer": "neighbor_distribution",
        },
        "B37_code2hyp_code2vec_attention_frechet": {
            "torch_variant": "code2hyp_code2vec_attention_frechet",
            "trainable_curvature": True,
            "structural_loss_weight": config.structural_loss_weight,
            "structural_loss_schedule": "delayed_linear",
        },
        "B38_code2hyp_code2vec_attention_neighbor": {
            "torch_variant": "code2hyp_code2vec_attention_frechet",
            "trainable_curvature": True,
            "structural_loss_weight": config.structural_loss_weight,
            "structural_loss_schedule": "delayed_linear",
            "structural_regularizer": "neighbor_distribution",
        },
        "B39_code2vec_context_transform_baseline": {
            "torch_variant": "code2vec_context_transform",
            "trainable_curvature": False,
            "structural_loss_weight": 0.0,
        },
        "B46_code2vec_context_transform_neighbor_control": {
            "torch_variant": "code2vec_context_transform",
            "trainable_curvature": False,
            "structural_loss_weight": config.structural_loss_weight,
            "structural_loss_schedule": "delayed_linear",
            "structural_regularizer": "neighbor_distribution",
        },
        "B47_code2vec_context_transform_distance_control": {
            "torch_variant": "code2vec_context_transform",
            "trainable_curvature": False,
            "structural_loss_weight": config.structural_loss_weight,
            "structural_loss_schedule": "delayed_linear",
            "structural_regularizer": "distance",
        },
        "B64_code2vec_context_transform_multi_metric_control": {
            "torch_variant": "code2vec_context_transform",
            "trainable_curvature": False,
            "structural_loss_weight": config.structural_loss_weight,
            "structural_loss_schedule": "delayed_linear",
            "structural_regularizer": "multi_metric_distance",
        },
        "B50_code2vec_context_transform_l1_baseline": {
            "torch_variant": "code2vec_context_transform_l1",
            "trainable_curvature": False,
            "structural_loss_weight": 0.0,
        },
        "B51_code2vec_context_transform_l1_distance_control": {
            "torch_variant": "code2vec_context_transform_l1",
            "trainable_curvature": False,
            "structural_loss_weight": config.structural_loss_weight,
            "structural_loss_schedule": "delayed_linear",
            "structural_regularizer": "distance",
        },
        "B65_code2vec_context_transform_l1_multi_metric_control": {
            "torch_variant": "code2vec_context_transform_l1",
            "trainable_curvature": False,
            "structural_loss_weight": config.structural_loss_weight,
            "structural_loss_schedule": "delayed_linear",
            "structural_regularizer": "multi_metric_distance",
        },
        "B40_code2hyp_context_transform_frechet": {
            "torch_variant": "code2hyp_context_transform_frechet",
            "trainable_curvature": True,
            "structural_loss_weight": config.structural_loss_weight,
            "structural_loss_schedule": "delayed_linear",
        },
        "B41_code2hyp_context_transform_neighbor": {
            "torch_variant": "code2hyp_context_transform_frechet",
            "trainable_curvature": True,
            "structural_loss_weight": config.structural_loss_weight,
            "structural_loss_schedule": "delayed_linear",
            "structural_regularizer": "neighbor_distribution",
        },
        "B42_code2hyp_product_context_transform_frechet": {
            "torch_variant": "code2hyp_product_context_transform_frechet",
            "trainable_curvature": True,
            "structural_loss_weight": config.structural_loss_weight,
            "structural_loss_schedule": "delayed_linear",
        },
        "B43_code2hyp_product_context_transform_neighbor": {
            "torch_variant": "code2hyp_product_context_transform_frechet",
            "trainable_curvature": True,
            "structural_loss_weight": config.structural_loss_weight,
            "structural_loss_schedule": "delayed_linear",
            "structural_regularizer": "neighbor_distribution",
        },
        "B44_code2hyp_context_transform_product_bias_frechet": {
            "torch_variant": "code2hyp_context_transform_product_bias_frechet",
            "trainable_curvature": True,
            "structural_loss_weight": config.structural_loss_weight,
            "structural_loss_schedule": "delayed_linear",
            "structural_regularizer": "distance",
        },
        "B63_code2hyp_context_transform_product_bias_multi_metric_frechet": {
            "torch_variant": "code2hyp_context_transform_product_bias_frechet",
            "trainable_curvature": True,
            "structural_loss_weight": config.structural_loss_weight,
            "structural_loss_schedule": "delayed_linear",
            "structural_regularizer": "multi_metric_distance",
        },
        "B52_code2hyp_branch_product_context_transform_frechet": {
            "torch_variant": "code2hyp_branch_product_context_transform_frechet",
            "trainable_curvature": True,
            "structural_loss_weight": config.structural_loss_weight,
            "structural_loss_schedule": "delayed_linear",
            "structural_regularizer": "distance",
        },
        "B53_code2hyp_branch_product_context_transform_no_struct": {
            "torch_variant": "code2hyp_branch_product_context_transform_frechet",
            "trainable_curvature": True,
            "structural_loss_weight": 0.0,
        },
        "B54_code2hyp_context_transform_branch_product_bias_frechet": {
            "torch_variant": "code2hyp_context_transform_branch_product_bias_frechet",
            "trainable_curvature": True,
            "structural_loss_weight": config.structural_loss_weight,
            "structural_loss_schedule": "delayed_linear",
            "structural_regularizer": "distance",
        },
        "B55_code2hyp_context_transform_branch_product_bias_no_struct": {
            "torch_variant": "code2hyp_context_transform_branch_product_bias_frechet",
            "trainable_curvature": True,
            "structural_loss_weight": 0.0,
        },
        "B56_code2hyp_context_transform_latent_lca_branch_product_bias_frechet": {
            "torch_variant": "code2hyp_context_transform_latent_lca_branch_product_bias_frechet",
            "trainable_curvature": True,
            "structural_loss_weight": config.structural_loss_weight,
            "structural_loss_schedule": "delayed_linear",
            "structural_regularizer": "distance",
        },
        "B57_code2hyp_context_transform_latent_lca_branch_product_bias_no_struct": {
            "torch_variant": "code2hyp_context_transform_latent_lca_branch_product_bias_frechet",
            "trainable_curvature": True,
            "structural_loss_weight": 0.0,
        },
        "B58_code2hyp_context_transform_latent_lca_prior_branch_product_bias_frechet": {
            "torch_variant": "code2hyp_context_transform_latent_lca_prior_branch_product_bias_frechet",
            "trainable_curvature": True,
            "structural_loss_weight": config.structural_loss_weight,
            "structural_loss_schedule": "delayed_linear",
            "structural_regularizer": "distance",
        },
        "B59_code2hyp_context_transform_latent_lca_prior_branch_product_bias_no_struct": {
            "torch_variant": "code2hyp_context_transform_latent_lca_prior_branch_product_bias_frechet",
            "trainable_curvature": True,
            "structural_loss_weight": 0.0,
        },
        "B60_code2hyp_context_transform_branch_sequence_product_bias_frechet": {
            "torch_variant": "code2hyp_context_transform_branch_sequence_product_bias_frechet",
            "trainable_curvature": True,
            "structural_loss_weight": config.structural_loss_weight,
            "structural_loss_schedule": "delayed_linear",
            "structural_regularizer": "distance",
        },
        "B62_code2hyp_context_transform_branch_sequence_product_bias_multi_metric_frechet": {
            "torch_variant": "code2hyp_context_transform_branch_sequence_product_bias_frechet",
            "trainable_curvature": True,
            "structural_loss_weight": config.structural_loss_weight,
            "structural_loss_schedule": "delayed_linear",
            "structural_regularizer": "multi_metric_distance",
        },
        "B66_branch_sequence_euclidean_product_l2_multi_metric_control": {
            "torch_variant": "code2hyp_context_transform_branch_sequence_euclidean_product_l2",
            "trainable_curvature": False,
            "structural_loss_weight": config.structural_loss_weight,
            "structural_loss_schedule": "delayed_linear",
            "structural_regularizer": "multi_metric_distance",
        },
        "B67_branch_sequence_euclidean_product_l1_multi_metric_control": {
            "torch_variant": "code2hyp_context_transform_branch_sequence_euclidean_product_l1",
            "trainable_curvature": False,
            "structural_loss_weight": config.structural_loss_weight,
            "structural_loss_schedule": "delayed_linear",
            "structural_regularizer": "multi_metric_distance",
        },
        "B68_branch_sequence_product_bias_near_euclidean_multi_metric": {
            "torch_variant": "code2hyp_context_transform_branch_sequence_product_bias_frechet",
            "trainable_curvature": False,
            "curvature": 1e-4,
            "structural_loss_weight": config.structural_loss_weight,
            "structural_loss_schedule": "delayed_linear",
            "structural_regularizer": "multi_metric_distance",
        },
        "B69_branch_sequence_product_bias_fixed_curvature_multi_metric": {
            "torch_variant": "code2hyp_context_transform_branch_sequence_product_bias_frechet",
            "trainable_curvature": False,
            "curvature": 1.0,
            "structural_loss_weight": config.structural_loss_weight,
            "structural_loss_schedule": "delayed_linear",
            "structural_regularizer": "multi_metric_distance",
        },
        "B70_branch_sequence_single_hyperbolic_multi_metric_control": {
            "torch_variant": "code2hyp_context_transform_branch_sequence_single_bias_frechet",
            "trainable_curvature": True,
            "structural_loss_weight": config.structural_loss_weight,
            "structural_loss_schedule": "delayed_linear",
            "structural_regularizer": "multi_metric_distance",
        },
        "B80_geocodepath_endpoint_geodesic_product_proxy": {
            "torch_variant": "code2hyp_context_transform_endpoint_geodesic_product_bias_frechet",
            "trainable_curvature": True,
            "structural_loss_weight": config.structural_loss_weight,
            "structural_loss_schedule": "delayed_linear",
            "structural_regularizer": "multi_metric_distance",
        },
        "B81_geocodepath_endpoint_lca_product_proxy": {
            "torch_variant": "code2hyp_context_transform_endpoint_lca_product_bias_frechet",
            "trainable_curvature": True,
            "structural_loss_weight": config.structural_loss_weight,
            "structural_loss_schedule": "delayed_linear",
            "structural_regularizer": "multi_metric_distance",
        },
        "B82_geocodepath_endpoint_lca_prior_product_proxy": {
            "torch_variant": "code2hyp_context_transform_endpoint_lca_prior_product_bias_frechet",
            "trainable_curvature": True,
            "structural_loss_weight": config.structural_loss_weight,
            "structural_loss_schedule": "delayed_linear",
            "structural_regularizer": "multi_metric_distance",
        },
        "B83_geocodepath_endpoint_lca_axiom_product_proxy": {
            "torch_variant": "code2hyp_context_transform_endpoint_lca_axiom_product_bias_frechet",
            "trainable_curvature": True,
            "structural_loss_weight": config.structural_loss_weight,
            "structural_loss_schedule": "delayed_linear",
            "structural_regularizer": "multi_metric_lca_axiom",
        },
        "B84_geocodepath_relation_conditioned_product_proxy": {
            "torch_variant": "code2hyp_context_transform_relation_conditioned_product_bias_frechet",
            "trainable_curvature": True,
            "structural_loss_weight": config.structural_loss_weight,
            "structural_loss_schedule": "delayed_linear",
            "structural_regularizer": "relation_conditioned_lca_axiom",
        },
        "B85_geocodepath_relation_conditioned_aux_product_proxy": {
            "torch_variant": "code2hyp_context_transform_relation_conditioned_aux_product_bias_frechet",
            "trainable_curvature": True,
            "structural_loss_weight": config.structural_loss_weight,
            "structural_loss_schedule": "delayed_linear",
            "structural_regularizer": "relation_conditioned_lca_axiom",
        },
        "B86_geocodepath_method_transport_aux_product_proxy": {
            "torch_variant": "code2hyp_context_transform_relation_conditioned_aux_product_bias_frechet",
            "trainable_curvature": True,
            "structural_loss_weight": config.structural_loss_weight,
            "structural_loss_schedule": "delayed_linear",
            "structural_regularizer": "method_transport",
        },
        "B87_geocodepath_multi_metric_method_transport_aux_product_proxy": {
            "torch_variant": "code2hyp_context_transform_relation_conditioned_aux_product_bias_frechet",
            "trainable_curvature": True,
            "structural_loss_weight": config.structural_loss_weight,
            "structural_loss_schedule": "delayed_linear",
            "structural_regularizer": "method_transport_multi_metric",
        },
        "B71_branch_sequence_product_bias_prefix_only": {
            "torch_variant": "code2hyp_context_transform_branch_sequence_product_bias_frechet",
            "trainable_curvature": True,
            "structural_loss_weight": config.structural_loss_weight,
            "structural_loss_schedule": "delayed_linear",
            "structural_regularizer": "distance_prefix",
        },
        "B72_branch_sequence_product_bias_edit_only": {
            "torch_variant": "code2hyp_context_transform_branch_sequence_product_bias_frechet",
            "trainable_curvature": True,
            "structural_loss_weight": config.structural_loss_weight,
            "structural_loss_schedule": "delayed_linear",
            "structural_regularizer": "distance_edit",
        },
        "B73_branch_sequence_product_bias_jaccard_only": {
            "torch_variant": "code2hyp_context_transform_branch_sequence_product_bias_frechet",
            "trainable_curvature": True,
            "structural_loss_weight": config.structural_loss_weight,
            "structural_loss_schedule": "delayed_linear",
            "structural_regularizer": "distance_jaccard",
        },
        "B74_branch_sequence_product_bias_prefix_edit": {
            "torch_variant": "code2hyp_context_transform_branch_sequence_product_bias_frechet",
            "trainable_curvature": True,
            "structural_loss_weight": config.structural_loss_weight,
            "structural_loss_schedule": "delayed_linear",
            "structural_regularizer": "multi_metric_prefix_edit",
        },
        "B75_branch_sequence_product_bias_prefix_jaccard": {
            "torch_variant": "code2hyp_context_transform_branch_sequence_product_bias_frechet",
            "trainable_curvature": True,
            "structural_loss_weight": config.structural_loss_weight,
            "structural_loss_schedule": "delayed_linear",
            "structural_regularizer": "multi_metric_prefix_jaccard",
        },
        "B76_branch_sequence_product_bias_edit_jaccard": {
            "torch_variant": "code2hyp_context_transform_branch_sequence_product_bias_frechet",
            "trainable_curvature": True,
            "structural_loss_weight": config.structural_loss_weight,
            "structural_loss_schedule": "delayed_linear",
            "structural_regularizer": "multi_metric_edit_jaccard",
        },
        "B61_code2hyp_context_transform_branch_sequence_product_bias_no_struct": {
            "torch_variant": "code2hyp_context_transform_branch_sequence_product_bias_frechet",
            "trainable_curvature": True,
            "structural_loss_weight": 0.0,
        },
        "B49_code2hyp_context_transform_product_bias_near_euclidean": {
            "torch_variant": "code2hyp_context_transform_product_bias_frechet",
            "trainable_curvature": False,
            "curvature": 1e-4,
            "structural_loss_weight": config.structural_loss_weight,
            "structural_loss_schedule": "delayed_linear",
            "structural_regularizer": "distance",
        },
        "B48_code2hyp_context_transform_product_bias_no_struct": {
            "torch_variant": "code2hyp_context_transform_product_bias_frechet",
            "trainable_curvature": True,
            "structural_loss_weight": 0.0,
        },
        "B45_code2hyp_context_transform_product_bias_neighbor": {
            "torch_variant": "code2hyp_context_transform_product_bias_frechet",
            "trainable_curvature": True,
            "structural_loss_weight": config.structural_loss_weight,
            "structural_loss_schedule": "delayed_linear",
            "structural_regularizer": "neighbor_distribution",
        },
        "B13_factorized_product_channel_mixer_rank": {
            "torch_variant": "factorized_product_channel_mixer",
            "trainable_curvature": False,
            "structural_loss_weight": config.structural_loss_weight,
        },
        "B7_hyperbolic_attention_only": {
            "torch_variant": "hyperbolic_attention",
            "trainable_curvature": False,
            "structural_loss_weight": 0.0,
        },
        "B5_euclidean_struct_loss": {
            "torch_variant": "euclidean",
            "trainable_curvature": False,
            "structural_loss_weight": config.structural_loss_weight,
        },
        "B6_euclidean_metric_code2vec": {
            "torch_variant": "euclidean_metric",
            "trainable_curvature": False,
            "structural_loss_weight": 0.0,
        },
        "B14_bounded_euclidean_metric_code2vec": {
            "torch_variant": "bounded_euclidean_metric",
            "trainable_curvature": False,
            "structural_loss_weight": 0.0,
        },
        "B_tree_euclidean_lca_bias": {
            "torch_variant": "euclidean_tree",
            "trainable_curvature": False,
            "structural_loss_weight": 0.0,
        },
    }


def real_variant_specs(config: RealCode2HypPilotConfig) -> dict[str, dict[str, Any]]:
    return _real_variant_specs(config)


def _build_real_pilot_result(
    train_path: str | Path,
    validation_path: str | Path,
    train_records: list[Any],
    raw_validation_records: list[Any],
    validation_records: list[Any],
    train_dataset: Any,
    config: RealCode2HypPilotConfig,
    pos_weight: torch.Tensor | None,
    runs: list[dict[str, Any]],
) -> dict[str, Any]:
    target_coverage = label_subtoken_coverage(raw_validation_records, train_dataset.target_vocab)
    return {
        "experiment": "real_code2hyp_multilabel_pilot",
        "interpretation_status": "real_data_pilot_not_final_claim",
        "dataset": {
            "train_path": str(train_path),
            "validation_path": str(validation_path),
            "train_records": len(train_records),
            "validation_records_loaded": len(raw_validation_records),
            "validation_records_after_known_target_filter": len(validation_records),
            "validation_known_target_record_coverage": target_coverage["record_coverage"],
            "validation_target_subtokens_loaded": target_coverage["subtokens"],
            "validation_known_target_subtokens": target_coverage["known_subtokens"],
            "validation_known_target_subtoken_coverage": target_coverage["subtoken_coverage"],
            "target_subtoken_vocab_size": len(train_dataset.target_vocab),
            "token_vocab_size": len(train_dataset.token_vocab),
            "ast_node_vocab_size": len(train_dataset.ast_node_vocab),
            "path_encoder": config.path_encoder,
            "representation_transform": config.representation_transform,
            "lexical_ablation": config.lexical_ablation,
        },
        "training": {
            "epochs": config.epochs,
            "batch_size": config.batch_size,
            "learning_rate": config.learning_rate,
            "structural_loss_weight_for_B5": config.structural_loss_weight,
            "structural_regularizer_for_B5": config.structural_regularizer,
            "use_positive_weighting": config.use_positive_weighting,
            "max_positive_weight": config.max_positive_weight,
            "positive_weight_mean": float(pos_weight.mean()) if pos_weight is not None else None,
            "positive_weight_max": float(pos_weight.max()) if pos_weight is not None else None,
            "curvature": config.curvature,
            "model_seeds": list(config.model_seeds),
            "variant_filter": list(config.variant_filter) if config.variant_filter is not None else None,
            "metric": "target-subtoken micro precision/recall/F1 with top-k = true target subtoken count",
            "additional_prediction_metrics": (
                "fixed_top3 and threshold@0.5 are reported as non-oracle diagnostics; "
                "oracle_topk remains for continuity with earlier controlled results."
            ),
            "sample_seed": config.sample_seed,
            "context_sample_seed": config.context_sample_seed,
            "structural_eval_limit": config.structural_eval_limit,
            "structural_eval_seed": config.structural_eval_seed,
        },
        "runs": runs,
        "claim_boundary": (
            "This is a real-data pilot on code2vec/code2seq preprocessed files. "
            "It is suitable for debugging the scientific pipeline, but final "
            "article claims require the preregistered full split and statistical "
            "analysis."
        ),
    }


def _run_variant(
    variant_name: str,
    torch_variant: str,
    trainable_curvature: bool,
    structural_loss_weight: float,
    base_config: Code2HypTorchConfig,
    dataset_seed: int,
    model_seed: int,
    comparison_config: SyntheticComparisonConfig,
    structural_loss_schedule: str = "constant",
    structural_regularizer: str = "distance",
) -> dict[str, Any]:
    dataset = make_synthetic_code2hyp_dataset(
        SyntheticCode2HypConfig(
            examples=comparison_config.examples,
            contexts_per_method=comparison_config.contexts_per_method,
            max_path_length=comparison_config.max_path_length,
            branches=comparison_config.branches,
            token_dim=comparison_config.token_dim,
            structural_dim=comparison_config.structural_dim,
            seed=dataset_seed,
        )
    )
    torch.manual_seed(model_seed)
    model = Code2HypTorchModel(
        _model_config(base_config, trainable_curvature=trainable_curvature),
        variant=torch_variant,  # type: ignore[arg-type]
    )
    initial_accuracy = evaluate_accuracy(model, dataset.batch, dataset.labels, batch_size=comparison_config.batch_size)
    history = fit_supervised(
        model,
        dataset.batch,
        dataset.labels,
        epochs=comparison_config.epochs,
        batch_size=comparison_config.batch_size,
        learning_rate=comparison_config.learning_rate,
        structural_loss_weight=structural_loss_weight,
        structural_loss_schedule=structural_loss_schedule,
        structural_regularizer=structural_regularizer,
        seed=model_seed,
    )
    final_accuracy = evaluate_accuracy(model, dataset.batch, dataset.labels, batch_size=comparison_config.batch_size)
    with torch.no_grad():
        output = model(dataset.batch)
    return {
        "variant": variant_name,
        "dataset_seed": dataset_seed,
        "model_seed": model_seed,
        "parameter_count": model.parameter_count(),
        "initial_accuracy": initial_accuracy,
        "final_accuracy": final_accuracy,
        "final_loss": history[-1]["loss"],
        "final_epoch_accuracy": history[-1]["accuracy"],
        "final_structural_loss": history[-1]["structural_loss"],
        "history": history,
        "structural_loss_weight": structural_loss_weight,
        "structural_loss_schedule": structural_loss_schedule,
        "structural_regularizer": structural_regularizer,
        "curvature": float(output.curvature.detach()),
        "factorized_metric_weights": [
            float(weight.detach()) for weight in model.factorized_metric_weights()
        ],
        "product_attention_bias_weight": float(model.product_attention_bias_weight().detach()),
        "factorized_channel_mixer_rank": model.factorized_channel_mixer_rank(),
    }


def run_synthetic_comparison(config: SyntheticComparisonConfig) -> dict[str, Any]:
    reference_dataset = make_synthetic_code2hyp_dataset(
        SyntheticCode2HypConfig(
            examples=config.examples,
            contexts_per_method=config.contexts_per_method,
            max_path_length=config.max_path_length,
            branches=config.branches,
            token_dim=config.token_dim,
            structural_dim=config.structural_dim,
            seed=config.dataset_seed,
        )
    )

    runs: list[dict[str, Any]] = []
    for model_seed in config.model_seeds:
        runs.append(
            _run_variant(
                variant_name="B1_euclidean",
                torch_variant="euclidean",
                trainable_curvature=False,
                structural_loss_weight=0.0,
                base_config=reference_dataset.model_config,
                dataset_seed=config.dataset_seed,
                model_seed=model_seed,
                comparison_config=config,
            )
        )
        runs.append(
            _run_variant(
                variant_name="B2_product_fixed_curvature",
                torch_variant="product",
                trainable_curvature=False,
                structural_loss_weight=0.0,
                base_config=reference_dataset.model_config,
                dataset_seed=config.dataset_seed,
                model_seed=model_seed,
                comparison_config=config,
            )
        )
        runs.append(
            _run_variant(
                variant_name="B3_product",
                torch_variant="product",
                trainable_curvature=True,
                structural_loss_weight=0.0,
                base_config=reference_dataset.model_config,
                dataset_seed=config.dataset_seed,
                model_seed=model_seed,
                comparison_config=config,
            )
        )
        runs.append(
            _run_variant(
                variant_name="B4_hyperbolic_code2vec",
                torch_variant="hyperbolic",
                trainable_curvature=False,
                structural_loss_weight=0.0,
                base_config=reference_dataset.model_config,
                dataset_seed=config.dataset_seed,
                model_seed=model_seed,
                comparison_config=config,
            )
        )
        runs.append(
            _run_variant(
                variant_name="B4T_hyperbolic_code2vec_trainable_curvature",
                torch_variant="hyperbolic",
                trainable_curvature=True,
                structural_loss_weight=0.0,
                base_config=reference_dataset.model_config,
                dataset_seed=config.dataset_seed,
                model_seed=model_seed,
                comparison_config=config,
            )
        )
        runs.append(
            _run_variant(
                variant_name="B8_hyperbolic_frechet_code2vec",
                torch_variant="hyperbolic_frechet",
                trainable_curvature=False,
                structural_loss_weight=0.0,
                base_config=reference_dataset.model_config,
                dataset_seed=config.dataset_seed,
                model_seed=model_seed,
                comparison_config=config,
            )
        )
        runs.append(
            _run_variant(
                variant_name="B17_hyperbolic_path_mp_code2vec",
                torch_variant="hyperbolic_path_message_passing",
                trainable_curvature=False,
                structural_loss_weight=0.0,
                base_config=reference_dataset.model_config,
                dataset_seed=config.dataset_seed,
                model_seed=model_seed,
                comparison_config=config,
            )
        )
        runs.append(
            _run_variant(
                variant_name="B18_hyperbolic_path_mp_struct_rank",
                torch_variant="hyperbolic_path_message_passing",
                trainable_curvature=False,
                structural_loss_weight=config.structural_loss_weight,
                base_config=reference_dataset.model_config,
                dataset_seed=config.dataset_seed,
                model_seed=model_seed,
                comparison_config=config,
            )
        )
        runs.append(
            _run_variant(
                variant_name="B19_hyperbolic_path_mp_rank_annealed",
                torch_variant="hyperbolic_path_message_passing",
                trainable_curvature=False,
                structural_loss_weight=config.structural_loss_weight,
                base_config=reference_dataset.model_config,
                dataset_seed=config.dataset_seed,
                model_seed=model_seed,
                comparison_config=config,
                structural_loss_schedule="linear",
            )
        )
        runs.append(
            _run_variant(
                variant_name="B20_hyperbolic_path_mp_rank_delayed",
                torch_variant="hyperbolic_path_message_passing",
                trainable_curvature=False,
                structural_loss_weight=config.structural_loss_weight,
                base_config=reference_dataset.model_config,
                dataset_seed=config.dataset_seed,
                model_seed=model_seed,
                comparison_config=config,
                structural_loss_schedule="delayed_linear",
            )
        )
        runs.append(
            _run_variant(
                variant_name="B21_hyperbolic_path_mp_rank_cosine",
                torch_variant="hyperbolic_path_message_passing",
                trainable_curvature=False,
                structural_loss_weight=config.structural_loss_weight,
                base_config=reference_dataset.model_config,
                dataset_seed=config.dataset_seed,
                model_seed=model_seed,
                comparison_config=config,
                structural_loss_schedule="cosine",
            )
        )
        runs.append(
            _run_variant(
                variant_name="B22_hyperbolic_path_mp_rank_warmup_decay",
                torch_variant="hyperbolic_path_message_passing",
                trainable_curvature=False,
                structural_loss_weight=config.structural_loss_weight,
                base_config=reference_dataset.model_config,
                dataset_seed=config.dataset_seed,
                model_seed=model_seed,
                comparison_config=config,
                structural_loss_schedule="warmup_decay",
            )
        )
        runs.append(
            _run_variant(
                variant_name="B23_hyperbolic_path_attention_mp_code2vec",
                torch_variant="hyperbolic_path_attention_message_passing",
                trainable_curvature=False,
                structural_loss_weight=0.0,
                base_config=reference_dataset.model_config,
                dataset_seed=config.dataset_seed,
                model_seed=model_seed,
                comparison_config=config,
            )
        )
        runs.append(
            _run_variant(
                variant_name="B24_hyperbolic_path_attention_mp_rank_annealed",
                torch_variant="hyperbolic_path_attention_message_passing",
                trainable_curvature=False,
                structural_loss_weight=config.structural_loss_weight,
                base_config=reference_dataset.model_config,
                dataset_seed=config.dataset_seed,
                model_seed=model_seed,
                comparison_config=config,
                structural_loss_schedule="linear",
            )
        )
        runs.append(
            _run_variant(
                variant_name="B25_hyperbolic_path_depth_attention_mp_code2vec",
                torch_variant="hyperbolic_path_depth_attention_message_passing",
                trainable_curvature=False,
                structural_loss_weight=0.0,
                base_config=reference_dataset.model_config,
                dataset_seed=config.dataset_seed,
                model_seed=model_seed,
                comparison_config=config,
            )
        )
        runs.append(
            _run_variant(
                variant_name="B26_hyperbolic_path_depth_attention_mp_rank_annealed",
                torch_variant="hyperbolic_path_depth_attention_message_passing",
                trainable_curvature=False,
                structural_loss_weight=config.structural_loss_weight,
                base_config=reference_dataset.model_config,
                dataset_seed=config.dataset_seed,
                model_seed=model_seed,
                comparison_config=config,
                structural_loss_schedule="linear",
            )
        )
        runs.append(
            _run_variant(
                variant_name="B27_hyperbolic_path_attention_mp_monotone",
                torch_variant="hyperbolic_path_attention_message_passing",
                trainable_curvature=False,
                structural_loss_weight=config.structural_loss_weight,
                base_config=reference_dataset.model_config,
                dataset_seed=config.dataset_seed,
                model_seed=model_seed,
                comparison_config=config,
                structural_loss_schedule="linear",
                structural_regularizer="path_attention_monotone",
            )
        )
        runs.append(
            _run_variant(
                variant_name="B28_hyperbolic_path_attention_mp_tree_distance",
                torch_variant="hyperbolic_path_attention_message_passing",
                trainable_curvature=False,
                structural_loss_weight=config.structural_loss_weight,
                base_config=reference_dataset.model_config,
                dataset_seed=config.dataset_seed,
                model_seed=model_seed,
                comparison_config=config,
                structural_loss_schedule="linear",
                structural_regularizer="path_attention_tree_distance",
            )
        )
        runs.append(
            _run_variant(
                variant_name="B29_hyperbolic_path_dual_attention_mp_separated",
                torch_variant="hyperbolic_path_dual_attention_message_passing",
                trainable_curvature=False,
                structural_loss_weight=config.structural_loss_weight,
                base_config=reference_dataset.model_config,
                dataset_seed=config.dataset_seed,
                model_seed=model_seed,
                comparison_config=config,
                structural_loss_schedule="linear",
                structural_regularizer="path_dual_attention_separation",
            )
        )
        runs.append(
            _run_variant(
                variant_name="B30_hyperbolic_path_dual_attention_mp_rank_separated",
                torch_variant="hyperbolic_path_dual_attention_message_passing",
                trainable_curvature=False,
                structural_loss_weight=config.structural_loss_weight,
                base_config=reference_dataset.model_config,
                dataset_seed=config.dataset_seed,
                model_seed=model_seed,
                comparison_config=config,
                structural_loss_schedule="linear",
                structural_regularizer="path_dual_attention_separation_rank",
            )
        )
        runs.append(
            _run_variant(
                variant_name="B31_hyperbolic_path_dual_attention_mp_soft_rank",
                torch_variant="hyperbolic_path_dual_attention_message_passing",
                trainable_curvature=False,
                structural_loss_weight=config.structural_loss_weight,
                base_config=reference_dataset.model_config,
                dataset_seed=config.dataset_seed,
                model_seed=model_seed,
                comparison_config=config,
                structural_loss_schedule="linear",
                structural_regularizer="path_dual_attention_separation_soft_rank",
            )
        )
        runs.append(
            _run_variant(
                variant_name="B32_lorentz_path_dual_attention_mp_soft_rank",
                torch_variant="lorentz_path_dual_attention_message_passing",
                trainable_curvature=False,
                structural_loss_weight=config.structural_loss_weight,
                base_config=reference_dataset.model_config,
                dataset_seed=config.dataset_seed,
                model_seed=model_seed,
                comparison_config=config,
                structural_loss_schedule="linear",
                structural_regularizer="path_dual_attention_separation_soft_rank",
            )
        )
        runs.append(
            _run_variant(
                variant_name="B34_hyperbolic_path_dual_attention_mp_adaptive_rank",
                torch_variant="hyperbolic_path_dual_attention_message_passing",
                trainable_curvature=False,
                structural_loss_weight=config.structural_loss_weight,
                base_config=reference_dataset.model_config,
                dataset_seed=config.dataset_seed,
                model_seed=model_seed,
                comparison_config=config,
                structural_loss_schedule="linear",
                structural_regularizer="path_dual_attention_separation_adaptive_rank",
            )
        )
        runs.append(
            _run_variant(
                variant_name="B9_lorentz_code2vec",
                torch_variant="lorentz",
                trainable_curvature=False,
                structural_loss_weight=0.0,
                base_config=reference_dataset.model_config,
                dataset_seed=config.dataset_seed,
                model_seed=model_seed,
                comparison_config=config,
            )
        )
        runs.append(
            _run_variant(
                variant_name="B15_lorentz_product_code2vec",
                torch_variant="lorentz_product",
                trainable_curvature=False,
                structural_loss_weight=0.0,
                base_config=reference_dataset.model_config,
                dataset_seed=config.dataset_seed,
                model_seed=model_seed,
                comparison_config=config,
            )
        )
        runs.append(
            _run_variant(
                variant_name="B10_factorized_product_code2vec",
                torch_variant="factorized_product",
                trainable_curvature=False,
                structural_loss_weight=0.0,
                base_config=reference_dataset.model_config,
                dataset_seed=config.dataset_seed,
                model_seed=model_seed,
                comparison_config=config,
            )
        )
        runs.append(
            _run_variant(
                variant_name="B11_factorized_product_struct_rank",
                torch_variant="factorized_product",
                trainable_curvature=False,
                structural_loss_weight=config.structural_loss_weight,
                base_config=reference_dataset.model_config,
                dataset_seed=config.dataset_seed,
                model_seed=model_seed,
                comparison_config=config,
            )
        )
        runs.append(
            _run_variant(
                variant_name="B12_factorized_product_learned_metric_rank",
                torch_variant="factorized_product_learned_metric",
                trainable_curvature=False,
                structural_loss_weight=config.structural_loss_weight,
                base_config=reference_dataset.model_config,
                dataset_seed=config.dataset_seed,
                model_seed=model_seed,
                comparison_config=config,
            )
        )
        runs.append(
            _run_variant(
                variant_name="B16_factorized_product_three_metric_rank",
                torch_variant="factorized_product_three_metric",
                trainable_curvature=False,
                structural_loss_weight=config.structural_loss_weight,
                base_config=reference_dataset.model_config,
                dataset_seed=config.dataset_seed,
                model_seed=model_seed,
                comparison_config=config,
            )
        )
        runs.append(
            _run_variant(
                variant_name="B35_code2hyp_product_frechet_adaptive",
                torch_variant="code2hyp_product_frechet",
                trainable_curvature=True,
                structural_loss_weight=config.structural_loss_weight,
                base_config=reference_dataset.model_config,
                dataset_seed=config.dataset_seed,
                model_seed=model_seed,
                comparison_config=config,
                structural_loss_schedule="delayed_linear",
            )
        )
        runs.append(
            _run_variant(
                variant_name="B13_factorized_product_channel_mixer_rank",
                torch_variant="factorized_product_channel_mixer",
                trainable_curvature=False,
                structural_loss_weight=config.structural_loss_weight,
                base_config=reference_dataset.model_config,
                dataset_seed=config.dataset_seed,
                model_seed=model_seed,
                comparison_config=config,
            )
        )
        runs.append(
            _run_variant(
                variant_name="B7_hyperbolic_attention_only",
                torch_variant="hyperbolic_attention",
                trainable_curvature=False,
                structural_loss_weight=0.0,
                base_config=reference_dataset.model_config,
                dataset_seed=config.dataset_seed,
                model_seed=model_seed,
                comparison_config=config,
            )
        )
        runs.append(
            _run_variant(
                variant_name="B5_euclidean_struct_loss",
                torch_variant="euclidean",
                trainable_curvature=False,
                structural_loss_weight=config.structural_loss_weight,
                base_config=reference_dataset.model_config,
                dataset_seed=config.dataset_seed,
                model_seed=model_seed,
                comparison_config=config,
            )
        )
        runs.append(
            _run_variant(
                variant_name="B6_euclidean_metric_code2vec",
                torch_variant="euclidean_metric",
                trainable_curvature=False,
                structural_loss_weight=0.0,
                base_config=reference_dataset.model_config,
                dataset_seed=config.dataset_seed,
                model_seed=model_seed,
                comparison_config=config,
            )
        )
        runs.append(
            _run_variant(
                variant_name="B14_bounded_euclidean_metric_code2vec",
                torch_variant="bounded_euclidean_metric",
                trainable_curvature=False,
                structural_loss_weight=0.0,
                base_config=reference_dataset.model_config,
                dataset_seed=config.dataset_seed,
                model_seed=model_seed,
                comparison_config=config,
            )
        )
        runs.append(
            _run_variant(
                variant_name="B_tree_euclidean_lca_bias",
                torch_variant="euclidean_tree",
                trainable_curvature=False,
                structural_loss_weight=0.0,
                base_config=reference_dataset.model_config,
                dataset_seed=config.dataset_seed,
                model_seed=model_seed,
                comparison_config=config,
            )
        )

    b1 = [run["final_accuracy"] for run in runs if run["variant"] == "B1_euclidean"]
    b2 = [run["final_accuracy"] for run in runs if run["variant"] == "B2_product_fixed_curvature"]
    b3 = [run["final_accuracy"] for run in runs if run["variant"] == "B3_product"]
    b4 = [run["final_accuracy"] for run in runs if run["variant"] == "B4_hyperbolic_code2vec"]
    b4t = [run["final_accuracy"] for run in runs if run["variant"] == "B4T_hyperbolic_code2vec_trainable_curvature"]
    b8 = [run["final_accuracy"] for run in runs if run["variant"] == "B8_hyperbolic_frechet_code2vec"]
    b17 = [run["final_accuracy"] for run in runs if run["variant"] == "B17_hyperbolic_path_mp_code2vec"]
    b18 = [run["final_accuracy"] for run in runs if run["variant"] == "B18_hyperbolic_path_mp_struct_rank"]
    b19 = [run["final_accuracy"] for run in runs if run["variant"] == "B19_hyperbolic_path_mp_rank_annealed"]
    b20 = [run["final_accuracy"] for run in runs if run["variant"] == "B20_hyperbolic_path_mp_rank_delayed"]
    b21 = [run["final_accuracy"] for run in runs if run["variant"] == "B21_hyperbolic_path_mp_rank_cosine"]
    b22 = [run["final_accuracy"] for run in runs if run["variant"] == "B22_hyperbolic_path_mp_rank_warmup_decay"]
    b9 = [run["final_accuracy"] for run in runs if run["variant"] == "B9_lorentz_code2vec"]
    b15 = [run["final_accuracy"] for run in runs if run["variant"] == "B15_lorentz_product_code2vec"]
    b10 = [run["final_accuracy"] for run in runs if run["variant"] == "B10_factorized_product_code2vec"]
    b11 = [run["final_accuracy"] for run in runs if run["variant"] == "B11_factorized_product_struct_rank"]
    b12 = [run["final_accuracy"] for run in runs if run["variant"] == "B12_factorized_product_learned_metric_rank"]
    b16 = [run["final_accuracy"] for run in runs if run["variant"] == "B16_factorized_product_three_metric_rank"]
    b35 = [run["final_accuracy"] for run in runs if run["variant"] == "B35_code2hyp_product_frechet_adaptive"]
    b13 = [run["final_accuracy"] for run in runs if run["variant"] == "B13_factorized_product_channel_mixer_rank"]
    b7 = [run["final_accuracy"] for run in runs if run["variant"] == "B7_hyperbolic_attention_only"]
    b5 = [run["final_accuracy"] for run in runs if run["variant"] == "B5_euclidean_struct_loss"]
    b6 = [run["final_accuracy"] for run in runs if run["variant"] == "B6_euclidean_metric_code2vec"]
    b14 = [run["final_accuracy"] for run in runs if run["variant"] == "B14_bounded_euclidean_metric_code2vec"]
    btree = [run["final_accuracy"] for run in runs if run["variant"] == "B_tree_euclidean_lca_bias"]
    b23 = [run["final_accuracy"] for run in runs if run["variant"] == "B23_hyperbolic_path_attention_mp_code2vec"]
    b24 = [
        run["final_accuracy"]
        for run in runs
        if run["variant"] == "B24_hyperbolic_path_attention_mp_rank_annealed"
    ]
    b25 = [run["final_accuracy"] for run in runs if run["variant"] == "B25_hyperbolic_path_depth_attention_mp_code2vec"]
    b26 = [
        run["final_accuracy"]
        for run in runs
        if run["variant"] == "B26_hyperbolic_path_depth_attention_mp_rank_annealed"
    ]
    b27 = [run["final_accuracy"] for run in runs if run["variant"] == "B27_hyperbolic_path_attention_mp_monotone"]
    b28 = [
        run["final_accuracy"]
        for run in runs
        if run["variant"] == "B28_hyperbolic_path_attention_mp_tree_distance"
    ]
    b29 = [
        run["final_accuracy"]
        for run in runs
        if run["variant"] == "B29_hyperbolic_path_dual_attention_mp_separated"
    ]
    b30 = [
        run["final_accuracy"]
        for run in runs
        if run["variant"] == "B30_hyperbolic_path_dual_attention_mp_rank_separated"
    ]
    b31 = [
        run["final_accuracy"]
        for run in runs
        if run["variant"] == "B31_hyperbolic_path_dual_attention_mp_soft_rank"
    ]
    b1_mean = sum(b1) / len(b1)
    b2_mean = sum(b2) / len(b2)
    b3_mean = sum(b3) / len(b3)
    b4_mean = sum(b4) / len(b4)
    b4t_mean = sum(b4t) / len(b4t)
    b8_mean = sum(b8) / len(b8)
    b17_mean = sum(b17) / len(b17)
    b18_mean = sum(b18) / len(b18)
    b19_mean = sum(b19) / len(b19)
    b20_mean = sum(b20) / len(b20)
    b21_mean = sum(b21) / len(b21)
    b22_mean = sum(b22) / len(b22)
    b9_mean = sum(b9) / len(b9)
    b15_mean = sum(b15) / len(b15)
    b10_mean = sum(b10) / len(b10)
    b11_mean = sum(b11) / len(b11)
    b12_mean = sum(b12) / len(b12)
    b16_mean = sum(b16) / len(b16)
    b35_mean = sum(b35) / len(b35)
    b13_mean = sum(b13) / len(b13)
    b7_mean = sum(b7) / len(b7)
    b5_mean = sum(b5) / len(b5)
    b6_mean = sum(b6) / len(b6)
    b14_mean = sum(b14) / len(b14)
    btree_mean = sum(btree) / len(btree)
    b23_mean = sum(b23) / len(b23)
    b24_mean = sum(b24) / len(b24)
    b25_mean = sum(b25) / len(b25)
    b26_mean = sum(b26) / len(b26)
    b27_mean = sum(b27) / len(b27)
    b28_mean = sum(b28) / len(b28)
    b29_mean = sum(b29) / len(b29)
    b30_mean = sum(b30) / len(b30)
    b31_mean = sum(b31) / len(b31)
    return {
        "experiment": "synthetic_code2hyp_b1_b3_comparison",
        "interpretation_status": "sanity_check_not_scientific_claim",
        "dataset": {
            "examples": config.examples,
            "contexts_per_method": config.contexts_per_method,
            "max_path_length": config.max_path_length,
            "branches": config.branches,
            "dataset_seed": config.dataset_seed,
            "description": reference_dataset.description,
        },
        "training": {
            "epochs": config.epochs,
            "batch_size": config.batch_size,
            "learning_rate": config.learning_rate,
            "structural_loss_weight_for_B5": config.structural_loss_weight,
            "model_seeds": list(config.model_seeds),
        },
        "summary": {
            "B1_final_accuracy_mean": b1_mean,
            "B2_final_accuracy_mean": b2_mean,
            "B3_final_accuracy_mean": b3_mean,
            "B4_final_accuracy_mean": b4_mean,
            "B4T_final_accuracy_mean": b4t_mean,
            "B8_final_accuracy_mean": b8_mean,
            "B17_final_accuracy_mean": b17_mean,
            "B18_final_accuracy_mean": b18_mean,
            "B19_final_accuracy_mean": b19_mean,
            "B20_final_accuracy_mean": b20_mean,
            "B21_final_accuracy_mean": b21_mean,
            "B22_final_accuracy_mean": b22_mean,
            "B9_final_accuracy_mean": b9_mean,
            "B15_final_accuracy_mean": b15_mean,
            "B10_final_accuracy_mean": b10_mean,
            "B11_final_accuracy_mean": b11_mean,
            "B12_final_accuracy_mean": b12_mean,
            "B16_final_accuracy_mean": b16_mean,
            "B35_final_accuracy_mean": b35_mean,
            "B13_final_accuracy_mean": b13_mean,
            "B7_final_accuracy_mean": b7_mean,
            "B5_final_accuracy_mean": b5_mean,
            "B6_final_accuracy_mean": b6_mean,
            "B14_final_accuracy_mean": b14_mean,
            "B_tree_final_accuracy_mean": btree_mean,
            "B23_final_accuracy_mean": b23_mean,
            "B24_final_accuracy_mean": b24_mean,
            "B25_final_accuracy_mean": b25_mean,
            "B26_final_accuracy_mean": b26_mean,
            "B27_final_accuracy_mean": b27_mean,
            "B28_final_accuracy_mean": b28_mean,
            "B29_final_accuracy_mean": b29_mean,
            "B30_final_accuracy_mean": b30_mean,
            "B31_final_accuracy_mean": b31_mean,
            "B2_minus_B1_final_accuracy_mean": b2_mean - b1_mean,
            "B3_minus_B1_final_accuracy_mean": b3_mean - b1_mean,
            "B3_minus_B2_final_accuracy_mean": b3_mean - b2_mean,
            "B4_minus_B1_final_accuracy_mean": b4_mean - b1_mean,
            "B4_minus_B3_final_accuracy_mean": b4_mean - b3_mean,
            "B4T_minus_B1_final_accuracy_mean": b4t_mean - b1_mean,
            "B4T_minus_B4_final_accuracy_mean": b4t_mean - b4_mean,
            "B8_minus_B1_final_accuracy_mean": b8_mean - b1_mean,
            "B8_minus_B4_final_accuracy_mean": b8_mean - b4_mean,
            "B4_minus_B8_final_accuracy_mean": b4_mean - b8_mean,
            "B9_minus_B1_final_accuracy_mean": b9_mean - b1_mean,
            "B9_minus_B4_final_accuracy_mean": b9_mean - b4_mean,
            "B4_minus_B9_final_accuracy_mean": b4_mean - b9_mean,
            "B15_minus_B1_final_accuracy_mean": b15_mean - b1_mean,
            "B15_minus_B4_final_accuracy_mean": b15_mean - b4_mean,
            "B4_minus_B15_final_accuracy_mean": b4_mean - b15_mean,
            "B10_minus_B1_final_accuracy_mean": b10_mean - b1_mean,
            "B10_minus_B4_final_accuracy_mean": b10_mean - b4_mean,
            "B4_minus_B10_final_accuracy_mean": b4_mean - b10_mean,
            "B11_minus_B1_final_accuracy_mean": b11_mean - b1_mean,
            "B11_minus_B4_final_accuracy_mean": b11_mean - b4_mean,
            "B4_minus_B11_final_accuracy_mean": b4_mean - b11_mean,
            "B12_minus_B1_final_accuracy_mean": b12_mean - b1_mean,
            "B12_minus_B4_final_accuracy_mean": b12_mean - b4_mean,
            "B4_minus_B12_final_accuracy_mean": b4_mean - b12_mean,
            "B16_minus_B1_final_accuracy_mean": b16_mean - b1_mean,
            "B16_minus_B10_final_accuracy_mean": b16_mean - b10_mean,
            "B35_minus_B1_final_accuracy_mean": b35_mean - b1_mean,
            "B35_minus_B10_final_accuracy_mean": b35_mean - b10_mean,
            "B35_minus_B16_final_accuracy_mean": b35_mean - b16_mean,
            "B13_minus_B1_final_accuracy_mean": b13_mean - b1_mean,
            "B13_minus_B4_final_accuracy_mean": b13_mean - b4_mean,
            "B4_minus_B13_final_accuracy_mean": b4_mean - b13_mean,
            "B7_minus_B1_final_accuracy_mean": b7_mean - b1_mean,
            "B7_minus_B6_final_accuracy_mean": b7_mean - b6_mean,
            "B4_minus_B7_final_accuracy_mean": b4_mean - b7_mean,
            "B3_minus_B5_final_accuracy_mean": b3_mean - b5_mean,
            "B4_minus_B6_final_accuracy_mean": b4_mean - b6_mean,
            "B14_minus_B6_final_accuracy_mean": b14_mean - b6_mean,
            "B4_minus_B14_final_accuracy_mean": b4_mean - b14_mean,
            "B4_minus_B_tree_final_accuracy_mean": b4_mean - btree_mean,
            "B_tree_minus_B6_final_accuracy_mean": btree_mean - b6_mean,
            "B23_minus_B17_final_accuracy_mean": b23_mean - b17_mean,
            "B24_minus_B23_final_accuracy_mean": b24_mean - b23_mean,
            "B24_minus_B17_final_accuracy_mean": b24_mean - b17_mean,
            "B25_minus_B23_final_accuracy_mean": b25_mean - b23_mean,
            "B25_minus_B17_final_accuracy_mean": b25_mean - b17_mean,
            "B26_minus_B25_final_accuracy_mean": b26_mean - b25_mean,
            "B27_minus_B23_final_accuracy_mean": b27_mean - b23_mean,
            "B27_minus_B24_final_accuracy_mean": b27_mean - b24_mean,
            "B28_minus_B23_final_accuracy_mean": b28_mean - b23_mean,
            "B28_minus_B24_final_accuracy_mean": b28_mean - b24_mean,
            "B28_minus_B27_final_accuracy_mean": b28_mean - b27_mean,
            "B29_minus_B23_final_accuracy_mean": b29_mean - b23_mean,
            "B29_minus_B28_final_accuracy_mean": b29_mean - b28_mean,
            "B30_minus_B23_final_accuracy_mean": b30_mean - b23_mean,
            "B30_minus_B29_final_accuracy_mean": b30_mean - b29_mean,
            "B31_minus_B23_final_accuracy_mean": b31_mean - b23_mean,
            "B31_minus_B29_final_accuracy_mean": b31_mean - b29_mean,
            "B31_minus_B30_final_accuracy_mean": b31_mean - b30_mean,
            "B26_minus_B17_final_accuracy_mean": b26_mean - b17_mean,
        },
        "runs": runs,
        "claim_boundary": (
            "This synthetic experiment validates the training and comparison "
            "pipeline only. It must not be used as evidence that hyperbolic "
            "geometry improves real code2vec/code2seq tasks."
        ),
    }


def run_real_code2hyp_pilot(
    train_path: str | Path,
    validation_path: str | Path,
    config: RealCode2HypPilotConfig,
) -> dict[str, Any]:
    train_loader = (
        sample_code2vec_records(train_path, limit=config.train_limit, seed=config.sample_seed)
        if config.sample_seed is not None
        else load_code2vec_records(train_path, limit=config.train_limit)
    )
    validation_loader = (
        sample_code2vec_records(validation_path, limit=config.val_limit, seed=config.sample_seed + 1)
        if config.sample_seed is not None
        else load_code2vec_records(validation_path, limit=config.val_limit)
    )
    train_records = apply_lexical_ablation(
        train_loader,
        config.lexical_ablation,  # type: ignore[arg-type]
    )
    raw_validation_records = apply_lexical_ablation(
        validation_loader,
        config.lexical_ablation,  # type: ignore[arg-type]
    )
    train_dataset = encode_records_to_multilabel_batch(
        train_records,
        max_contexts=config.max_contexts,
        max_path_length=config.max_path_length,
        token_dim=config.token_dim,
        structural_dim=config.structural_dim,
        curvature=config.curvature,
        path_encoder=config.path_encoder,
        representation_transform=config.representation_transform,
        context_sample_seed=config.context_sample_seed,
    )
    validation_records = filter_records_by_known_label_subtokens(
        raw_validation_records,
        train_dataset.target_vocab,
    )
    if not validation_records:
        raise ValueError("validation split has no records with target subtokens seen in train split")
    validation_dataset = encode_records_to_multilabel_batch(
        validation_records,
        max_contexts=config.max_contexts,
        max_path_length=config.max_path_length,
        token_dim=config.token_dim,
        structural_dim=config.structural_dim,
        curvature=config.curvature,
        path_encoder=config.path_encoder,
        representation_transform=config.representation_transform,
        token_vocab=train_dataset.token_vocab,
        ast_node_vocab=train_dataset.ast_node_vocab,
        target_vocab=train_dataset.target_vocab,
    )
    pos_weight = (
        compute_multilabel_pos_weight(train_dataset.labels, max_weight=config.max_positive_weight)
        if config.use_positive_weighting
        else None
    )

    runs: list[dict[str, Any]] = []
    if config.variant_filter is not None:
        variant_specs = _real_variant_specs(config)
        unknown_variants = [variant for variant in config.variant_filter if variant not in variant_specs]
        if unknown_variants:
            raise ValueError(f"unknown variant_filter entries: {', '.join(unknown_variants)}")
        for model_seed in config.model_seeds:
            for variant_name in config.variant_filter:
                spec = variant_specs[variant_name]
                runs.append(
                    _run_multilabel_variant(
                        variant_name=variant_name,
                        torch_variant=spec["torch_variant"],
                        trainable_curvature=spec["trainable_curvature"],
                        structural_loss_weight=spec["structural_loss_weight"],
                        base_config=train_dataset.model_config,
                        train_dataset=train_dataset,
                        validation_dataset=validation_dataset,
                        model_seed=model_seed,
                        config=config,
                        pos_weight=pos_weight,
                        structural_loss_schedule=spec.get("structural_loss_schedule", "constant"),
                        structural_regularizer=spec.get("structural_regularizer"),
                        curvature_override=spec.get("curvature"),
                    )
                )
        return _build_real_pilot_result(
            train_path,
            validation_path,
            train_records,
            raw_validation_records,
            validation_records,
            train_dataset,
            config,
            pos_weight,
            runs,
        )

    for model_seed in config.model_seeds:
        runs.append(
            _run_multilabel_variant(
                variant_name="B1_euclidean",
                torch_variant="euclidean",
                trainable_curvature=False,
                structural_loss_weight=0.0,
                base_config=train_dataset.model_config,
                train_dataset=train_dataset,
                validation_dataset=validation_dataset,
                model_seed=model_seed,
                config=config,
                pos_weight=pos_weight,
            )
        )
        runs.append(
            _run_multilabel_variant(
                variant_name="B2_product_fixed_curvature",
                torch_variant="product",
                trainable_curvature=False,
                structural_loss_weight=0.0,
                base_config=train_dataset.model_config,
                train_dataset=train_dataset,
                validation_dataset=validation_dataset,
                model_seed=model_seed,
                config=config,
                pos_weight=pos_weight,
            )
        )
        runs.append(
            _run_multilabel_variant(
                variant_name="B3_product",
                torch_variant="product",
                trainable_curvature=True,
                structural_loss_weight=0.0,
                base_config=train_dataset.model_config,
                train_dataset=train_dataset,
                validation_dataset=validation_dataset,
                model_seed=model_seed,
                config=config,
                pos_weight=pos_weight,
            )
        )
        runs.append(
            _run_multilabel_variant(
                variant_name="B4_hyperbolic_code2vec",
                torch_variant="hyperbolic",
                trainable_curvature=False,
                structural_loss_weight=0.0,
                base_config=train_dataset.model_config,
                train_dataset=train_dataset,
                validation_dataset=validation_dataset,
                model_seed=model_seed,
                config=config,
                pos_weight=pos_weight,
            )
        )
        runs.append(
            _run_multilabel_variant(
                variant_name="B4T_hyperbolic_code2vec_trainable_curvature",
                torch_variant="hyperbolic",
                trainable_curvature=True,
                structural_loss_weight=0.0,
                base_config=train_dataset.model_config,
                train_dataset=train_dataset,
                validation_dataset=validation_dataset,
                model_seed=model_seed,
                config=config,
                pos_weight=pos_weight,
            )
        )
        runs.append(
            _run_multilabel_variant(
                variant_name="B8_hyperbolic_frechet_code2vec",
                torch_variant="hyperbolic_frechet",
                trainable_curvature=False,
                structural_loss_weight=0.0,
                base_config=train_dataset.model_config,
                train_dataset=train_dataset,
                validation_dataset=validation_dataset,
                model_seed=model_seed,
                config=config,
                pos_weight=pos_weight,
            )
        )
        runs.append(
            _run_multilabel_variant(
                variant_name="B17_hyperbolic_path_mp_code2vec",
                torch_variant="hyperbolic_path_message_passing",
                trainable_curvature=False,
                structural_loss_weight=0.0,
                base_config=train_dataset.model_config,
                train_dataset=train_dataset,
                validation_dataset=validation_dataset,
                model_seed=model_seed,
                config=config,
                pos_weight=pos_weight,
            )
        )
        runs.append(
            _run_multilabel_variant(
                variant_name="B18_hyperbolic_path_mp_struct_rank",
                torch_variant="hyperbolic_path_message_passing",
                trainable_curvature=False,
                structural_loss_weight=config.structural_loss_weight,
                base_config=train_dataset.model_config,
                train_dataset=train_dataset,
                validation_dataset=validation_dataset,
                model_seed=model_seed,
                config=config,
                pos_weight=pos_weight,
            )
        )
        runs.append(
            _run_multilabel_variant(
                variant_name="B19_hyperbolic_path_mp_rank_annealed",
                torch_variant="hyperbolic_path_message_passing",
                trainable_curvature=False,
                structural_loss_weight=config.structural_loss_weight,
                base_config=train_dataset.model_config,
                train_dataset=train_dataset,
                validation_dataset=validation_dataset,
                model_seed=model_seed,
                config=config,
                pos_weight=pos_weight,
                structural_loss_schedule="linear",
            )
        )
        runs.append(
            _run_multilabel_variant(
                variant_name="B20_hyperbolic_path_mp_rank_delayed",
                torch_variant="hyperbolic_path_message_passing",
                trainable_curvature=False,
                structural_loss_weight=config.structural_loss_weight,
                base_config=train_dataset.model_config,
                train_dataset=train_dataset,
                validation_dataset=validation_dataset,
                model_seed=model_seed,
                config=config,
                pos_weight=pos_weight,
                structural_loss_schedule="delayed_linear",
            )
        )
        runs.append(
            _run_multilabel_variant(
                variant_name="B21_hyperbolic_path_mp_rank_cosine",
                torch_variant="hyperbolic_path_message_passing",
                trainable_curvature=False,
                structural_loss_weight=config.structural_loss_weight,
                base_config=train_dataset.model_config,
                train_dataset=train_dataset,
                validation_dataset=validation_dataset,
                model_seed=model_seed,
                config=config,
                pos_weight=pos_weight,
                structural_loss_schedule="cosine",
            )
        )
        runs.append(
            _run_multilabel_variant(
                variant_name="B22_hyperbolic_path_mp_rank_warmup_decay",
                torch_variant="hyperbolic_path_message_passing",
                trainable_curvature=False,
                structural_loss_weight=config.structural_loss_weight,
                base_config=train_dataset.model_config,
                train_dataset=train_dataset,
                validation_dataset=validation_dataset,
                model_seed=model_seed,
                config=config,
                pos_weight=pos_weight,
                structural_loss_schedule="warmup_decay",
            )
        )
        runs.append(
            _run_multilabel_variant(
                variant_name="B23_hyperbolic_path_attention_mp_code2vec",
                torch_variant="hyperbolic_path_attention_message_passing",
                trainable_curvature=False,
                structural_loss_weight=0.0,
                base_config=train_dataset.model_config,
                train_dataset=train_dataset,
                validation_dataset=validation_dataset,
                model_seed=model_seed,
                config=config,
                pos_weight=pos_weight,
            )
        )
        runs.append(
            _run_multilabel_variant(
                variant_name="B24_hyperbolic_path_attention_mp_rank_annealed",
                torch_variant="hyperbolic_path_attention_message_passing",
                trainable_curvature=False,
                structural_loss_weight=config.structural_loss_weight,
                base_config=train_dataset.model_config,
                train_dataset=train_dataset,
                validation_dataset=validation_dataset,
                model_seed=model_seed,
                config=config,
                pos_weight=pos_weight,
                structural_loss_schedule="linear",
            )
        )
        runs.append(
            _run_multilabel_variant(
                variant_name="B25_hyperbolic_path_depth_attention_mp_code2vec",
                torch_variant="hyperbolic_path_depth_attention_message_passing",
                trainable_curvature=False,
                structural_loss_weight=0.0,
                base_config=train_dataset.model_config,
                train_dataset=train_dataset,
                validation_dataset=validation_dataset,
                model_seed=model_seed,
                config=config,
                pos_weight=pos_weight,
            )
        )
        runs.append(
            _run_multilabel_variant(
                variant_name="B26_hyperbolic_path_depth_attention_mp_rank_annealed",
                torch_variant="hyperbolic_path_depth_attention_message_passing",
                trainable_curvature=False,
                structural_loss_weight=config.structural_loss_weight,
                base_config=train_dataset.model_config,
                train_dataset=train_dataset,
                validation_dataset=validation_dataset,
                model_seed=model_seed,
                config=config,
                pos_weight=pos_weight,
                structural_loss_schedule="linear",
            )
        )
        runs.append(
            _run_multilabel_variant(
                variant_name="B27_hyperbolic_path_attention_mp_monotone",
                torch_variant="hyperbolic_path_attention_message_passing",
                trainable_curvature=False,
                structural_loss_weight=config.structural_loss_weight,
                base_config=train_dataset.model_config,
                train_dataset=train_dataset,
                validation_dataset=validation_dataset,
                model_seed=model_seed,
                config=config,
                pos_weight=pos_weight,
                structural_loss_schedule="linear",
                structural_regularizer="path_attention_monotone",
            )
        )
        runs.append(
            _run_multilabel_variant(
                variant_name="B28_hyperbolic_path_attention_mp_tree_distance",
                torch_variant="hyperbolic_path_attention_message_passing",
                trainable_curvature=False,
                structural_loss_weight=config.structural_loss_weight,
                base_config=train_dataset.model_config,
                train_dataset=train_dataset,
                validation_dataset=validation_dataset,
                model_seed=model_seed,
                config=config,
                pos_weight=pos_weight,
                structural_loss_schedule="linear",
                structural_regularizer="path_attention_tree_distance",
            )
        )
        runs.append(
            _run_multilabel_variant(
                variant_name="B29_hyperbolic_path_dual_attention_mp_separated",
                torch_variant="hyperbolic_path_dual_attention_message_passing",
                trainable_curvature=False,
                structural_loss_weight=config.structural_loss_weight,
                base_config=train_dataset.model_config,
                train_dataset=train_dataset,
                validation_dataset=validation_dataset,
                model_seed=model_seed,
                config=config,
                pos_weight=pos_weight,
                structural_loss_schedule="linear",
                structural_regularizer="path_dual_attention_separation",
            )
        )
        runs.append(
            _run_multilabel_variant(
                variant_name="B30_hyperbolic_path_dual_attention_mp_rank_separated",
                torch_variant="hyperbolic_path_dual_attention_message_passing",
                trainable_curvature=False,
                structural_loss_weight=config.structural_loss_weight,
                base_config=train_dataset.model_config,
                train_dataset=train_dataset,
                validation_dataset=validation_dataset,
                model_seed=model_seed,
                config=config,
                pos_weight=pos_weight,
                structural_loss_schedule="linear",
                structural_regularizer="path_dual_attention_separation_rank",
            )
        )
        runs.append(
            _run_multilabel_variant(
                variant_name="B31_hyperbolic_path_dual_attention_mp_soft_rank",
                torch_variant="hyperbolic_path_dual_attention_message_passing",
                trainable_curvature=False,
                structural_loss_weight=config.structural_loss_weight,
                base_config=train_dataset.model_config,
                train_dataset=train_dataset,
                validation_dataset=validation_dataset,
                model_seed=model_seed,
                config=config,
                pos_weight=pos_weight,
                structural_loss_schedule="linear",
                structural_regularizer="path_dual_attention_separation_soft_rank",
            )
        )
        runs.append(
            _run_multilabel_variant(
                variant_name="B32_lorentz_path_dual_attention_mp_soft_rank",
                torch_variant="lorentz_path_dual_attention_message_passing",
                trainable_curvature=False,
                structural_loss_weight=config.structural_loss_weight,
                base_config=train_dataset.model_config,
                train_dataset=train_dataset,
                validation_dataset=validation_dataset,
                model_seed=model_seed,
                config=config,
                pos_weight=pos_weight,
                structural_loss_schedule="linear",
                structural_regularizer="path_dual_attention_separation_soft_rank",
            )
        )
        runs.append(
            _run_multilabel_variant(
                variant_name="B34_hyperbolic_path_dual_attention_mp_adaptive_rank",
                torch_variant="hyperbolic_path_dual_attention_message_passing",
                trainable_curvature=False,
                structural_loss_weight=config.structural_loss_weight,
                base_config=train_dataset.model_config,
                train_dataset=train_dataset,
                validation_dataset=validation_dataset,
                model_seed=model_seed,
                config=config,
                pos_weight=pos_weight,
                structural_loss_schedule="linear",
                structural_regularizer="path_dual_attention_separation_adaptive_rank",
            )
        )
        runs.append(
            _run_multilabel_variant(
                variant_name="B9_lorentz_code2vec",
                torch_variant="lorentz",
                trainable_curvature=False,
                structural_loss_weight=0.0,
                base_config=train_dataset.model_config,
                train_dataset=train_dataset,
                validation_dataset=validation_dataset,
                model_seed=model_seed,
                config=config,
                pos_weight=pos_weight,
            )
        )
        runs.append(
            _run_multilabel_variant(
                variant_name="B15_lorentz_product_code2vec",
                torch_variant="lorentz_product",
                trainable_curvature=False,
                structural_loss_weight=0.0,
                base_config=train_dataset.model_config,
                train_dataset=train_dataset,
                validation_dataset=validation_dataset,
                model_seed=model_seed,
                config=config,
                pos_weight=pos_weight,
            )
        )
        runs.append(
            _run_multilabel_variant(
                variant_name="B10_factorized_product_code2vec",
                torch_variant="factorized_product",
                trainable_curvature=False,
                structural_loss_weight=0.0,
                base_config=train_dataset.model_config,
                train_dataset=train_dataset,
                validation_dataset=validation_dataset,
                model_seed=model_seed,
                config=config,
                pos_weight=pos_weight,
            )
        )
        runs.append(
            _run_multilabel_variant(
                variant_name="B11_factorized_product_struct_rank",
                torch_variant="factorized_product",
                trainable_curvature=False,
                structural_loss_weight=config.structural_loss_weight,
                base_config=train_dataset.model_config,
                train_dataset=train_dataset,
                validation_dataset=validation_dataset,
                model_seed=model_seed,
                config=config,
                pos_weight=pos_weight,
            )
        )
        runs.append(
            _run_multilabel_variant(
                variant_name="B12_factorized_product_learned_metric_rank",
                torch_variant="factorized_product_learned_metric",
                trainable_curvature=False,
                structural_loss_weight=config.structural_loss_weight,
                base_config=train_dataset.model_config,
                train_dataset=train_dataset,
                validation_dataset=validation_dataset,
                model_seed=model_seed,
                config=config,
                pos_weight=pos_weight,
            )
        )
        runs.append(
            _run_multilabel_variant(
                variant_name="B16_factorized_product_three_metric_rank",
                torch_variant="factorized_product_three_metric",
                trainable_curvature=False,
                structural_loss_weight=config.structural_loss_weight,
                base_config=train_dataset.model_config,
                train_dataset=train_dataset,
                validation_dataset=validation_dataset,
                model_seed=model_seed,
                config=config,
                pos_weight=pos_weight,
            )
        )
        runs.append(
            _run_multilabel_variant(
                variant_name="B35_code2hyp_product_frechet_adaptive",
                torch_variant="code2hyp_product_frechet",
                trainable_curvature=True,
                structural_loss_weight=config.structural_loss_weight,
                base_config=train_dataset.model_config,
                train_dataset=train_dataset,
                validation_dataset=validation_dataset,
                model_seed=model_seed,
                config=config,
                pos_weight=pos_weight,
                structural_loss_schedule="delayed_linear",
            )
        )
        runs.append(
            _run_multilabel_variant(
                variant_name="B13_factorized_product_channel_mixer_rank",
                torch_variant="factorized_product_channel_mixer",
                trainable_curvature=False,
                structural_loss_weight=config.structural_loss_weight,
                base_config=train_dataset.model_config,
                train_dataset=train_dataset,
                validation_dataset=validation_dataset,
                model_seed=model_seed,
                config=config,
                pos_weight=pos_weight,
            )
        )
        runs.append(
            _run_multilabel_variant(
                variant_name="B7_hyperbolic_attention_only",
                torch_variant="hyperbolic_attention",
                trainable_curvature=False,
                structural_loss_weight=0.0,
                base_config=train_dataset.model_config,
                train_dataset=train_dataset,
                validation_dataset=validation_dataset,
                model_seed=model_seed,
                config=config,
                pos_weight=pos_weight,
            )
        )
        runs.append(
            _run_multilabel_variant(
                variant_name="B5_euclidean_struct_loss",
                torch_variant="euclidean",
                trainable_curvature=False,
                structural_loss_weight=config.structural_loss_weight,
                base_config=train_dataset.model_config,
                train_dataset=train_dataset,
                validation_dataset=validation_dataset,
                model_seed=model_seed,
                config=config,
                pos_weight=pos_weight,
            )
        )
        runs.append(
            _run_multilabel_variant(
                variant_name="B6_euclidean_metric_code2vec",
                torch_variant="euclidean_metric",
                trainable_curvature=False,
                structural_loss_weight=0.0,
                base_config=train_dataset.model_config,
                train_dataset=train_dataset,
                validation_dataset=validation_dataset,
                model_seed=model_seed,
                config=config,
                pos_weight=pos_weight,
            )
        )
        runs.append(
            _run_multilabel_variant(
                variant_name="B14_bounded_euclidean_metric_code2vec",
                torch_variant="bounded_euclidean_metric",
                trainable_curvature=False,
                structural_loss_weight=0.0,
                base_config=train_dataset.model_config,
                train_dataset=train_dataset,
                validation_dataset=validation_dataset,
                model_seed=model_seed,
                config=config,
                pos_weight=pos_weight,
            )
        )
        runs.append(
            _run_multilabel_variant(
                variant_name="B_tree_euclidean_lca_bias",
                torch_variant="euclidean_tree",
                trainable_curvature=False,
                structural_loss_weight=0.0,
                base_config=train_dataset.model_config,
                train_dataset=train_dataset,
                validation_dataset=validation_dataset,
                model_seed=model_seed,
                config=config,
                pos_weight=pos_weight,
            )
        )

    return _build_real_pilot_result(
        train_path,
        validation_path,
        train_records,
        raw_validation_records,
        validation_records,
        train_dataset,
        config,
        pos_weight,
        runs,
    )


def write_synthetic_comparison(path: str | Path, config: SyntheticComparisonConfig) -> dict[str, Any]:
    result = run_synthetic_comparison(config)
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return result
