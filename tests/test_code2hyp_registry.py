from __future__ import annotations

from geometry_profile_research.code2hyp_experiments import RealCode2HypPilotConfig, real_variant_specs
from geometry_profile_research.code2hyp_registry import (
    format_variant_catalog,
    parse_variant_selection,
    variant_catalog,
)


def test_variant_catalog_exposes_recommended_research_variants() -> None:
    catalog = variant_catalog()

    assert catalog["B4_hyperbolic_code2vec"].status == "recommended"
    assert catalog["B29_hyperbolic_path_dual_attention_mp_separated"].status == "exploratory"
    assert catalog["B31_hyperbolic_path_dual_attention_mp_soft_rank"].status == "exploratory"
    assert catalog["B32_lorentz_path_dual_attention_mp_soft_rank"].status == "exploratory"
    assert catalog["B34_hyperbolic_path_dual_attention_mp_adaptive_rank"].status == "exploratory"
    assert catalog["B35_code2hyp_product_frechet_adaptive"].status == "candidate"
    assert catalog["B36_code2hyp_product_frechet_neighbor"].status == "candidate"
    assert catalog["B37_code2hyp_code2vec_attention_frechet"].status == "candidate"
    assert catalog["B38_code2hyp_code2vec_attention_neighbor"].status == "candidate"
    assert catalog["B39_code2vec_context_transform_baseline"].status == "baseline"
    assert catalog["B46_code2vec_context_transform_neighbor_control"].status == "diagnostic"
    assert catalog["B47_code2vec_context_transform_distance_control"].status == "diagnostic"
    assert catalog["B50_code2vec_context_transform_l1_baseline"].status == "diagnostic"
    assert catalog["B51_code2vec_context_transform_l1_distance_control"].status == "diagnostic"
    assert catalog["B40_code2hyp_context_transform_frechet"].status == "candidate"
    assert catalog["B41_code2hyp_context_transform_neighbor"].status == "candidate"
    assert catalog["B42_code2hyp_product_context_transform_frechet"].status == "candidate"
    assert catalog["B43_code2hyp_product_context_transform_neighbor"].status == "candidate"
    assert catalog["B44_code2hyp_context_transform_product_bias_frechet"].status == "candidate"
    assert catalog["B48_code2hyp_context_transform_product_bias_no_struct"].status == "diagnostic"
    assert catalog["B49_code2hyp_context_transform_product_bias_near_euclidean"].status == "diagnostic"
    assert catalog["B45_code2hyp_context_transform_product_bias_neighbor"].status == "candidate"
    assert catalog["B84_geocodepath_relation_conditioned_product_proxy"].status == "candidate"
    assert catalog["B85_geocodepath_relation_conditioned_aux_product_proxy"].status == "candidate"
    assert "balanced" in catalog["B4_hyperbolic_code2vec"].profiles
    assert "path_attention" in catalog["B31_hyperbolic_path_dual_attention_mp_soft_rank"].profiles
    assert "path_attention" in catalog["B32_lorentz_path_dual_attention_mp_soft_rank"].profiles
    assert "path_attention" in catalog["B34_hyperbolic_path_dual_attention_mp_adaptive_rank"].profiles
    assert "code2vec_replacement" in catalog["B35_code2hyp_product_frechet_adaptive"].profiles
    assert "code2vec_replacement" in catalog["B36_code2hyp_product_frechet_neighbor"].profiles
    assert "code2vec_replacement" in catalog["B37_code2hyp_code2vec_attention_frechet"].profiles
    assert "code2vec_replacement" in catalog["B38_code2hyp_code2vec_attention_neighbor"].profiles
    assert "code2vec_replacement" in catalog["B39_code2vec_context_transform_baseline"].profiles
    assert "code2vec_replacement" in catalog["B46_code2vec_context_transform_neighbor_control"].profiles
    assert "code2vec_replacement" in catalog["B47_code2vec_context_transform_distance_control"].profiles
    assert "code2vec_replacement" in catalog["B50_code2vec_context_transform_l1_baseline"].profiles
    assert "code2vec_replacement" in catalog["B51_code2vec_context_transform_l1_distance_control"].profiles
    assert "code2vec_replacement" in catalog["B40_code2hyp_context_transform_frechet"].profiles
    assert "code2vec_replacement" in catalog["B41_code2hyp_context_transform_neighbor"].profiles
    assert "code2vec_replacement" in catalog["B42_code2hyp_product_context_transform_frechet"].profiles
    assert "code2vec_replacement" in catalog["B43_code2hyp_product_context_transform_neighbor"].profiles
    assert "code2vec_replacement" in catalog["B44_code2hyp_context_transform_product_bias_frechet"].profiles
    assert "code2vec_replacement" in catalog["B48_code2hyp_context_transform_product_bias_no_struct"].profiles
    assert "code2vec_replacement" in catalog["B49_code2hyp_context_transform_product_bias_near_euclidean"].profiles
    assert "code2vec_replacement" in catalog["B45_code2hyp_context_transform_product_bias_neighbor"].profiles


def test_registry_names_are_supported_by_experiment_specs() -> None:
    available = real_variant_specs(RealCode2HypPilotConfig())

    missing = sorted(set(variant_catalog()) - set(available))

    assert missing == []


def test_parse_variant_selection_uses_profile_and_explicit_variants() -> None:
    available = real_variant_specs(RealCode2HypPilotConfig())

    assert parse_variant_selection("", "balanced", available) == (
        "B4_hyperbolic_code2vec",
        "B8_hyperbolic_frechet_code2vec",
        "B29_hyperbolic_path_dual_attention_mp_separated",
        "B31_hyperbolic_path_dual_attention_mp_soft_rank",
        "B32_lorentz_path_dual_attention_mp_soft_rank",
    )
    assert parse_variant_selection("B4_hyperbolic_code2vec", None, available) == ("B4_hyperbolic_code2vec",)
    assert parse_variant_selection("", "code2vec_replacement", available) == (
        "B35_code2hyp_product_frechet_adaptive",
        "B36_code2hyp_product_frechet_neighbor",
        "B37_code2hyp_code2vec_attention_frechet",
        "B38_code2hyp_code2vec_attention_neighbor",
        "B39_code2vec_context_transform_baseline",
        "B46_code2vec_context_transform_neighbor_control",
        "B47_code2vec_context_transform_distance_control",
        "B50_code2vec_context_transform_l1_baseline",
        "B51_code2vec_context_transform_l1_distance_control",
        "B40_code2hyp_context_transform_frechet",
        "B41_code2hyp_context_transform_neighbor",
        "B42_code2hyp_product_context_transform_frechet",
        "B43_code2hyp_product_context_transform_neighbor",
        "B44_code2hyp_context_transform_product_bias_frechet",
        "B48_code2hyp_context_transform_product_bias_no_struct",
        "B49_code2hyp_context_transform_product_bias_near_euclidean",
        "B45_code2hyp_context_transform_product_bias_neighbor",
        "B62_code2hyp_context_transform_branch_sequence_product_bias_multi_metric_frechet",
        "B66_branch_sequence_euclidean_product_l2_multi_metric_control",
        "B67_branch_sequence_euclidean_product_l1_multi_metric_control",
        "B68_branch_sequence_product_bias_near_euclidean_multi_metric",
        "B69_branch_sequence_product_bias_fixed_curvature_multi_metric",
        "B70_branch_sequence_single_hyperbolic_multi_metric_control",
        "B80_geocodepath_endpoint_geodesic_product_proxy",
        "B81_geocodepath_endpoint_lca_product_proxy",
        "B82_geocodepath_endpoint_lca_prior_product_proxy",
            "B83_geocodepath_endpoint_lca_axiom_product_proxy",
                "B84_geocodepath_relation_conditioned_product_proxy",
                "B85_geocodepath_relation_conditioned_aux_product_proxy",
                "B86_geocodepath_method_transport_aux_product_proxy",
                "B87_geocodepath_multi_metric_method_transport_aux_product_proxy",
            )


def test_b36_real_spec_fixes_neighbor_distribution_regularizer() -> None:
    specs = real_variant_specs(RealCode2HypPilotConfig())

    assert specs["B36_code2hyp_product_frechet_neighbor"]["torch_variant"] == "code2hyp_product_frechet"
    assert specs["B36_code2hyp_product_frechet_neighbor"]["trainable_curvature"] is True
    assert specs["B36_code2hyp_product_frechet_neighbor"]["structural_regularizer"] == "neighbor_distribution"
    assert specs["B36_code2hyp_product_frechet_neighbor"]["structural_loss_schedule"] == "delayed_linear"


def test_b37_real_spec_keeps_code2vec_attention_with_frechet_path_aggregation() -> None:
    specs = real_variant_specs(RealCode2HypPilotConfig())

    assert specs["B37_code2hyp_code2vec_attention_frechet"]["torch_variant"] == (
        "code2hyp_code2vec_attention_frechet"
    )
    assert specs["B37_code2hyp_code2vec_attention_frechet"]["trainable_curvature"] is True
    assert specs["B37_code2hyp_code2vec_attention_frechet"]["structural_loss_schedule"] == "delayed_linear"


def test_b38_real_spec_combines_code2vec_attention_with_neighbor_distribution_regularizer() -> None:
    specs = real_variant_specs(RealCode2HypPilotConfig())

    assert specs["B38_code2hyp_code2vec_attention_neighbor"]["torch_variant"] == (
        "code2hyp_code2vec_attention_frechet"
    )
    assert specs["B38_code2hyp_code2vec_attention_neighbor"]["trainable_curvature"] is True
    assert specs["B38_code2hyp_code2vec_attention_neighbor"]["structural_regularizer"] == "neighbor_distribution"
    assert specs["B38_code2hyp_code2vec_attention_neighbor"]["structural_loss_schedule"] == "delayed_linear"


def test_b39_b40_b41_specs_model_true_code2vec_context_transform_family() -> None:
    specs = real_variant_specs(RealCode2HypPilotConfig())

    assert specs["B39_code2vec_context_transform_baseline"]["torch_variant"] == "code2vec_context_transform"
    assert specs["B39_code2vec_context_transform_baseline"]["trainable_curvature"] is False
    assert specs["B39_code2vec_context_transform_baseline"]["structural_loss_weight"] == 0.0
    assert specs["B46_code2vec_context_transform_neighbor_control"]["torch_variant"] == "code2vec_context_transform"
    assert specs["B46_code2vec_context_transform_neighbor_control"]["trainable_curvature"] is False
    assert specs["B46_code2vec_context_transform_neighbor_control"]["structural_regularizer"] == (
        "neighbor_distribution"
    )
    assert specs["B46_code2vec_context_transform_neighbor_control"]["structural_loss_schedule"] == "delayed_linear"
    assert specs["B47_code2vec_context_transform_distance_control"]["torch_variant"] == "code2vec_context_transform"
    assert specs["B47_code2vec_context_transform_distance_control"]["trainable_curvature"] is False
    assert specs["B47_code2vec_context_transform_distance_control"]["structural_regularizer"] == "distance"
    assert specs["B47_code2vec_context_transform_distance_control"]["structural_loss_schedule"] == "delayed_linear"
    assert specs["B50_code2vec_context_transform_l1_baseline"]["torch_variant"] == "code2vec_context_transform_l1"
    assert specs["B50_code2vec_context_transform_l1_baseline"]["trainable_curvature"] is False
    assert specs["B50_code2vec_context_transform_l1_baseline"]["structural_loss_weight"] == 0.0
    assert specs["B51_code2vec_context_transform_l1_distance_control"]["torch_variant"] == (
        "code2vec_context_transform_l1"
    )
    assert specs["B51_code2vec_context_transform_l1_distance_control"]["trainable_curvature"] is False
    assert specs["B51_code2vec_context_transform_l1_distance_control"]["structural_regularizer"] == "distance"
    assert specs["B51_code2vec_context_transform_l1_distance_control"]["structural_loss_schedule"] == "delayed_linear"
    assert specs["B40_code2hyp_context_transform_frechet"]["torch_variant"] == "code2hyp_context_transform_frechet"
    assert specs["B40_code2hyp_context_transform_frechet"]["trainable_curvature"] is True
    assert specs["B40_code2hyp_context_transform_frechet"]["structural_loss_schedule"] == "delayed_linear"
    assert specs["B41_code2hyp_context_transform_neighbor"]["torch_variant"] == "code2hyp_context_transform_frechet"
    assert specs["B41_code2hyp_context_transform_neighbor"]["trainable_curvature"] is True
    assert specs["B41_code2hyp_context_transform_neighbor"]["structural_regularizer"] == "neighbor_distribution"
    assert specs["B41_code2hyp_context_transform_neighbor"]["structural_loss_schedule"] == "delayed_linear"


def test_b62_matched_control_and_relation_generalization_specs_are_available() -> None:
    specs = real_variant_specs(RealCode2HypPilotConfig())

    assert specs["B62_code2hyp_context_transform_branch_sequence_product_bias_multi_metric_frechet"][
        "structural_regularizer"
    ] == "multi_metric_distance"
    assert specs["B66_branch_sequence_euclidean_product_l2_multi_metric_control"]["torch_variant"] == (
        "code2hyp_context_transform_branch_sequence_euclidean_product_l2"
    )
    assert specs["B66_branch_sequence_euclidean_product_l2_multi_metric_control"]["structural_regularizer"] == (
        "multi_metric_distance"
    )
    assert specs["B67_branch_sequence_euclidean_product_l1_multi_metric_control"]["torch_variant"] == (
        "code2hyp_context_transform_branch_sequence_euclidean_product_l1"
    )
    assert specs["B68_branch_sequence_product_bias_near_euclidean_multi_metric"]["curvature"] == 1e-4
    assert specs["B68_branch_sequence_product_bias_near_euclidean_multi_metric"]["trainable_curvature"] is False
    assert specs["B69_branch_sequence_product_bias_fixed_curvature_multi_metric"]["curvature"] == 1.0
    assert specs["B70_branch_sequence_single_hyperbolic_multi_metric_control"]["torch_variant"] == (
        "code2hyp_context_transform_branch_sequence_single_bias_frechet"
    )
    assert specs["B71_branch_sequence_product_bias_prefix_only"]["structural_regularizer"] == "distance_prefix"
    assert specs["B72_branch_sequence_product_bias_edit_only"]["structural_regularizer"] == "distance_edit"
    assert specs["B73_branch_sequence_product_bias_jaccard_only"]["structural_regularizer"] == "distance_jaccard"
    assert specs["B74_branch_sequence_product_bias_prefix_edit"]["structural_regularizer"] == "multi_metric_prefix_edit"
    assert specs["B75_branch_sequence_product_bias_prefix_jaccard"]["structural_regularizer"] == (
        "multi_metric_prefix_jaccard"
    )
    assert specs["B76_branch_sequence_product_bias_edit_jaccard"]["structural_regularizer"] == (
        "multi_metric_edit_jaccard"
    )


def test_b80_geocodepath_endpoint_proxy_spec_is_available() -> None:
    specs = real_variant_specs(RealCode2HypPilotConfig())

    assert specs["B80_geocodepath_endpoint_geodesic_product_proxy"]["torch_variant"] == (
        "code2hyp_context_transform_endpoint_geodesic_product_bias_frechet"
    )
    assert specs["B80_geocodepath_endpoint_geodesic_product_proxy"]["trainable_curvature"] is True
    assert specs["B80_geocodepath_endpoint_geodesic_product_proxy"]["structural_regularizer"] == (
        "multi_metric_distance"
    )
    assert specs["B80_geocodepath_endpoint_geodesic_product_proxy"]["structural_loss_schedule"] == "delayed_linear"


def test_b81_geocodepath_endpoint_lca_proxy_spec_is_available() -> None:
    specs = real_variant_specs(RealCode2HypPilotConfig())

    assert specs["B81_geocodepath_endpoint_lca_product_proxy"]["torch_variant"] == (
        "code2hyp_context_transform_endpoint_lca_product_bias_frechet"
    )
    assert specs["B81_geocodepath_endpoint_lca_product_proxy"]["trainable_curvature"] is True
    assert specs["B81_geocodepath_endpoint_lca_product_proxy"]["structural_regularizer"] == "multi_metric_distance"
    assert specs["B81_geocodepath_endpoint_lca_product_proxy"]["structural_loss_schedule"] == "delayed_linear"


def test_b82_geocodepath_endpoint_lca_prior_proxy_spec_is_available() -> None:
    specs = real_variant_specs(RealCode2HypPilotConfig())

    assert specs["B82_geocodepath_endpoint_lca_prior_product_proxy"]["torch_variant"] == (
        "code2hyp_context_transform_endpoint_lca_prior_product_bias_frechet"
    )
    assert specs["B82_geocodepath_endpoint_lca_prior_product_proxy"]["trainable_curvature"] is True
    assert specs["B82_geocodepath_endpoint_lca_prior_product_proxy"]["structural_regularizer"] == (
        "multi_metric_distance"
    )
    assert specs["B82_geocodepath_endpoint_lca_prior_product_proxy"]["structural_loss_schedule"] == "delayed_linear"


def test_b83_geocodepath_endpoint_lca_axiom_proxy_spec_is_available() -> None:
    specs = real_variant_specs(RealCode2HypPilotConfig())

    assert specs["B83_geocodepath_endpoint_lca_axiom_product_proxy"]["torch_variant"] == (
        "code2hyp_context_transform_endpoint_lca_axiom_product_bias_frechet"
    )
    assert specs["B83_geocodepath_endpoint_lca_axiom_product_proxy"]["trainable_curvature"] is True
    assert specs["B83_geocodepath_endpoint_lca_axiom_product_proxy"]["structural_regularizer"] == (
        "multi_metric_lca_axiom"
    )
    assert specs["B83_geocodepath_endpoint_lca_axiom_product_proxy"]["structural_loss_schedule"] == "delayed_linear"


def test_b84_geocodepath_relation_conditioned_proxy_spec_is_available() -> None:
    specs = real_variant_specs(RealCode2HypPilotConfig())

    assert specs["B84_geocodepath_relation_conditioned_product_proxy"]["torch_variant"] == (
        "code2hyp_context_transform_relation_conditioned_product_bias_frechet"
    )
    assert specs["B84_geocodepath_relation_conditioned_product_proxy"]["trainable_curvature"] is True
    assert specs["B84_geocodepath_relation_conditioned_product_proxy"]["structural_regularizer"] == (
        "relation_conditioned_lca_axiom"
    )
    assert specs["B84_geocodepath_relation_conditioned_product_proxy"]["structural_loss_schedule"] == (
        "delayed_linear"
    )


def test_b85_geocodepath_relation_conditioned_aux_proxy_spec_is_available() -> None:
    specs = real_variant_specs(RealCode2HypPilotConfig())

    assert specs["B85_geocodepath_relation_conditioned_aux_product_proxy"]["torch_variant"] == (
        "code2hyp_context_transform_relation_conditioned_aux_product_bias_frechet"
    )
    assert specs["B85_geocodepath_relation_conditioned_aux_product_proxy"]["trainable_curvature"] is True
    assert specs["B85_geocodepath_relation_conditioned_aux_product_proxy"]["structural_regularizer"] == (
        "relation_conditioned_lca_axiom"
    )
    assert specs["B85_geocodepath_relation_conditioned_aux_product_proxy"]["structural_loss_schedule"] == (
        "delayed_linear"
    )


def test_b86_geocodepath_method_transport_proxy_spec_is_available() -> None:
    specs = real_variant_specs(RealCode2HypPilotConfig())

    assert specs["B86_geocodepath_method_transport_aux_product_proxy"]["torch_variant"] == (
        "code2hyp_context_transform_relation_conditioned_aux_product_bias_frechet"
    )
    assert specs["B86_geocodepath_method_transport_aux_product_proxy"]["trainable_curvature"] is True
    assert specs["B86_geocodepath_method_transport_aux_product_proxy"]["structural_regularizer"] == (
        "method_transport"
    )
    assert specs["B86_geocodepath_method_transport_aux_product_proxy"]["structural_loss_schedule"] == (
        "delayed_linear"
    )


def test_b87_geocodepath_multi_metric_method_transport_proxy_spec_is_available() -> None:
    specs = real_variant_specs(RealCode2HypPilotConfig())

    assert specs["B87_geocodepath_multi_metric_method_transport_aux_product_proxy"]["torch_variant"] == (
        "code2hyp_context_transform_relation_conditioned_aux_product_bias_frechet"
    )
    assert specs["B87_geocodepath_multi_metric_method_transport_aux_product_proxy"]["trainable_curvature"] is True
    assert specs["B87_geocodepath_multi_metric_method_transport_aux_product_proxy"]["structural_regularizer"] == (
        "method_transport_multi_metric"
    )
    assert specs["B87_geocodepath_multi_metric_method_transport_aux_product_proxy"]["structural_loss_schedule"] == (
        "delayed_linear"
    )


def test_b42_b43_specs_combine_code2vec_context_transform_with_product_metric_attention() -> None:
    specs = real_variant_specs(RealCode2HypPilotConfig())

    assert specs["B42_code2hyp_product_context_transform_frechet"]["torch_variant"] == (
        "code2hyp_product_context_transform_frechet"
    )
    assert specs["B42_code2hyp_product_context_transform_frechet"]["trainable_curvature"] is True
    assert specs["B42_code2hyp_product_context_transform_frechet"]["structural_loss_schedule"] == "delayed_linear"
    assert specs["B43_code2hyp_product_context_transform_neighbor"]["torch_variant"] == (
        "code2hyp_product_context_transform_frechet"
    )
    assert specs["B43_code2hyp_product_context_transform_neighbor"]["trainable_curvature"] is True
    assert specs["B43_code2hyp_product_context_transform_neighbor"]["structural_regularizer"] == (
        "neighbor_distribution"
    )
    assert specs["B43_code2hyp_product_context_transform_neighbor"]["structural_loss_schedule"] == "delayed_linear"


def test_b44_b45_specs_add_product_metric_bias_to_code2vec_attention() -> None:
    specs = real_variant_specs(RealCode2HypPilotConfig())

    assert specs["B44_code2hyp_context_transform_product_bias_frechet"]["torch_variant"] == (
        "code2hyp_context_transform_product_bias_frechet"
    )
    assert specs["B44_code2hyp_context_transform_product_bias_frechet"]["trainable_curvature"] is True
    assert specs["B44_code2hyp_context_transform_product_bias_frechet"]["structural_loss_schedule"] == (
        "delayed_linear"
    )
    assert specs["B44_code2hyp_context_transform_product_bias_frechet"]["structural_regularizer"] == "distance"
    assert specs["B48_code2hyp_context_transform_product_bias_no_struct"]["torch_variant"] == (
        "code2hyp_context_transform_product_bias_frechet"
    )
    assert specs["B48_code2hyp_context_transform_product_bias_no_struct"]["trainable_curvature"] is True
    assert specs["B48_code2hyp_context_transform_product_bias_no_struct"]["structural_loss_weight"] == 0.0
    assert specs["B49_code2hyp_context_transform_product_bias_near_euclidean"]["torch_variant"] == (
        "code2hyp_context_transform_product_bias_frechet"
    )
    assert specs["B49_code2hyp_context_transform_product_bias_near_euclidean"]["trainable_curvature"] is False
    assert specs["B49_code2hyp_context_transform_product_bias_near_euclidean"]["curvature"] == 1e-4
    assert specs["B49_code2hyp_context_transform_product_bias_near_euclidean"]["structural_regularizer"] == "distance"
    assert specs["B45_code2hyp_context_transform_product_bias_neighbor"]["torch_variant"] == (
        "code2hyp_context_transform_product_bias_frechet"
    )
    assert specs["B45_code2hyp_context_transform_product_bias_neighbor"]["trainable_curvature"] is True
    assert specs["B45_code2hyp_context_transform_product_bias_neighbor"]["structural_regularizer"] == (
        "neighbor_distribution"
    )
    assert specs["B45_code2hyp_context_transform_product_bias_neighbor"]["structural_loss_schedule"] == (
        "delayed_linear"
    )


def test_parse_variant_selection_rejects_unknown_variants_with_suggestion() -> None:
    available = real_variant_specs(RealCode2HypPilotConfig())

    try:
        parse_variant_selection("B31_hyperbolic_path_dual_attention_mp_softrank", None, available)
    except ValueError as error:
        message = str(error)
    else:
        raise AssertionError("expected ValueError")

    assert "Unknown Code2Hyp variant" in message
    assert "B31_hyperbolic_path_dual_attention_mp_soft_rank" in message


def test_format_variant_catalog_is_human_readable() -> None:
    text = format_variant_catalog(variant_catalog())

    assert "B4_hyperbolic_code2vec" in text
    assert "recommended" in text
    assert "B31_hyperbolic_path_dual_attention_mp_soft_rank" in text
    assert "B32_lorentz_path_dual_attention_mp_soft_rank" in text
    assert "B34_hyperbolic_path_dual_attention_mp_adaptive_rank" in text
    assert "B35_code2hyp_product_frechet_adaptive" in text
    assert "B36_code2hyp_product_frechet_neighbor" in text
    assert "B37_code2hyp_code2vec_attention_frechet" in text
    assert "B38_code2hyp_code2vec_attention_neighbor" in text
    assert "B39_code2vec_context_transform_baseline" in text
    assert "B46_code2vec_context_transform_neighbor_control" in text
    assert "B47_code2vec_context_transform_distance_control" in text
    assert "B50_code2vec_context_transform_l1_baseline" in text
    assert "B51_code2vec_context_transform_l1_distance_control" in text
    assert "B40_code2hyp_context_transform_frechet" in text
    assert "B41_code2hyp_context_transform_neighbor" in text
    assert "B42_code2hyp_product_context_transform_frechet" in text
    assert "B43_code2hyp_product_context_transform_neighbor" in text
    assert "B44_code2hyp_context_transform_product_bias_frechet" in text
    assert "B48_code2hyp_context_transform_product_bias_no_struct" in text
    assert "B49_code2hyp_context_transform_product_bias_near_euclidean" in text
    assert "B45_code2hyp_context_transform_product_bias_neighbor" in text


def test_format_variant_catalog_uses_natural_variant_order() -> None:
    text = format_variant_catalog(variant_catalog())

    assert text.index("B1_euclidean") < text.index("B10_factorized_product_code2vec")
    assert text.index("B4_hyperbolic_code2vec") < text.index("B4T_hyperbolic_code2vec_trainable_curvature")
