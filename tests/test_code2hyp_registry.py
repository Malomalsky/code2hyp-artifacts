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
    assert catalog["B40_code2hyp_context_transform_frechet"].status == "candidate"
    assert catalog["B41_code2hyp_context_transform_neighbor"].status == "candidate"
    assert catalog["B42_code2hyp_product_context_transform_frechet"].status == "candidate"
    assert catalog["B43_code2hyp_product_context_transform_neighbor"].status == "candidate"
    assert catalog["B44_code2hyp_context_transform_product_bias_frechet"].status == "candidate"
    assert catalog["B45_code2hyp_context_transform_product_bias_neighbor"].status == "candidate"
    assert "balanced" in catalog["B4_hyperbolic_code2vec"].profiles
    assert "path_attention" in catalog["B31_hyperbolic_path_dual_attention_mp_soft_rank"].profiles
    assert "path_attention" in catalog["B32_lorentz_path_dual_attention_mp_soft_rank"].profiles
    assert "path_attention" in catalog["B34_hyperbolic_path_dual_attention_mp_adaptive_rank"].profiles
    assert "code2vec_replacement" in catalog["B35_code2hyp_product_frechet_adaptive"].profiles
    assert "code2vec_replacement" in catalog["B36_code2hyp_product_frechet_neighbor"].profiles
    assert "code2vec_replacement" in catalog["B37_code2hyp_code2vec_attention_frechet"].profiles
    assert "code2vec_replacement" in catalog["B38_code2hyp_code2vec_attention_neighbor"].profiles
    assert "code2vec_replacement" in catalog["B39_code2vec_context_transform_baseline"].profiles
    assert "code2vec_replacement" in catalog["B40_code2hyp_context_transform_frechet"].profiles
    assert "code2vec_replacement" in catalog["B41_code2hyp_context_transform_neighbor"].profiles
    assert "code2vec_replacement" in catalog["B42_code2hyp_product_context_transform_frechet"].profiles
    assert "code2vec_replacement" in catalog["B43_code2hyp_product_context_transform_neighbor"].profiles
    assert "code2vec_replacement" in catalog["B44_code2hyp_context_transform_product_bias_frechet"].profiles
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
        "B40_code2hyp_context_transform_frechet",
        "B41_code2hyp_context_transform_neighbor",
        "B42_code2hyp_product_context_transform_frechet",
        "B43_code2hyp_product_context_transform_neighbor",
        "B44_code2hyp_context_transform_product_bias_frechet",
        "B45_code2hyp_context_transform_product_bias_neighbor",
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
    assert specs["B40_code2hyp_context_transform_frechet"]["torch_variant"] == "code2hyp_context_transform_frechet"
    assert specs["B40_code2hyp_context_transform_frechet"]["trainable_curvature"] is True
    assert specs["B40_code2hyp_context_transform_frechet"]["structural_loss_schedule"] == "delayed_linear"
    assert specs["B41_code2hyp_context_transform_neighbor"]["torch_variant"] == "code2hyp_context_transform_frechet"
    assert specs["B41_code2hyp_context_transform_neighbor"]["trainable_curvature"] is True
    assert specs["B41_code2hyp_context_transform_neighbor"]["structural_regularizer"] == "neighbor_distribution"
    assert specs["B41_code2hyp_context_transform_neighbor"]["structural_loss_schedule"] == "delayed_linear"


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
    assert "B40_code2hyp_context_transform_frechet" in text
    assert "B41_code2hyp_context_transform_neighbor" in text
    assert "B42_code2hyp_product_context_transform_frechet" in text
    assert "B43_code2hyp_product_context_transform_neighbor" in text
    assert "B44_code2hyp_context_transform_product_bias_frechet" in text
    assert "B45_code2hyp_context_transform_product_bias_neighbor" in text


def test_format_variant_catalog_uses_natural_variant_order() -> None:
    text = format_variant_catalog(variant_catalog())

    assert text.index("B1_euclidean") < text.index("B10_factorized_product_code2vec")
    assert text.index("B4_hyperbolic_code2vec") < text.index("B4T_hyperbolic_code2vec_trainable_curvature")
