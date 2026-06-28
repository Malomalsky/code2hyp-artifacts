from __future__ import annotations

from dataclasses import dataclass
from difflib import get_close_matches
import re
from typing import Mapping


@dataclass(frozen=True)
class Code2HypVariantMetadata:
    name: str
    family: str
    status: str
    profiles: tuple[str, ...]
    summary: str


def variant_catalog() -> dict[str, Code2HypVariantMetadata]:
    entries = (
        Code2HypVariantMetadata(
            "B1_euclidean",
            "baseline",
            "baseline",
            ("core",),
            "Plain Euclidean code2vec-style control.",
        ),
        Code2HypVariantMetadata(
            "B2_product_fixed_curvature",
            "product_geometry",
            "diagnostic",
            ("core",),
            "Fixed-curvature Euclidean/hyperbolic product control.",
        ),
        Code2HypVariantMetadata(
            "B3_product",
            "product_geometry",
            "diagnostic",
            ("core",),
            "Product control with trainable curvature.",
        ),
        Code2HypVariantMetadata(
            "B4_hyperbolic_code2vec",
            "full_context_hyperbolic",
            "recommended",
            ("core", "balanced"),
            "Full-context Poincare code2vec with hyperbolic aggregation.",
        ),
        Code2HypVariantMetadata(
            "B4T_hyperbolic_code2vec_trainable_curvature",
            "full_context_hyperbolic",
            "diagnostic",
            ("core",),
            "B4 with trainable curvature.",
        ),
        Code2HypVariantMetadata(
            "B8_hyperbolic_frechet_code2vec",
            "full_context_hyperbolic",
            "recommended",
            ("core", "balanced"),
            "Intrinsic Fréchet/Karcher aggregation control for B4.",
        ),
        Code2HypVariantMetadata(
            "B17_hyperbolic_path_mp_code2vec",
            "path_message_passing",
            "diagnostic",
            ("structural",),
            "Hyperbolic AST-path message passing without structural rank pressure.",
        ),
        Code2HypVariantMetadata(
            "B18_hyperbolic_path_mp_struct_rank",
            "path_message_passing",
            "diagnostic",
            ("structural",),
            "B17 plus fixed structural-rank regularization.",
        ),
        Code2HypVariantMetadata(
            "B19_hyperbolic_path_mp_rank_annealed",
            "path_message_passing",
            "diagnostic",
            ("structural",),
            "B17 plus linear structural-rank schedule.",
        ),
        Code2HypVariantMetadata(
            "B20_hyperbolic_path_mp_rank_delayed",
            "path_message_passing",
            "diagnostic",
            ("structural",),
            "B17 plus delayed linear structural-rank schedule.",
        ),
        Code2HypVariantMetadata(
            "B21_hyperbolic_path_mp_rank_cosine",
            "path_message_passing",
            "diagnostic",
            ("structural",),
            "B17 plus cosine structural-rank schedule.",
        ),
        Code2HypVariantMetadata(
            "B22_hyperbolic_path_mp_rank_warmup_decay",
            "path_message_passing",
            "diagnostic",
            ("structural",),
            "B17 plus warmup-decay structural-rank schedule.",
        ),
        Code2HypVariantMetadata(
            "B23_hyperbolic_path_attention_mp_code2vec",
            "path_attention",
            "diagnostic",
            ("path_attention",),
            "Learned attention over updated AST-path nodes.",
        ),
        Code2HypVariantMetadata(
            "B24_hyperbolic_path_attention_mp_rank_annealed",
            "path_attention",
            "diagnostic",
            ("path_attention",),
            "B23 plus linear structural-rank schedule.",
        ),
        Code2HypVariantMetadata(
            "B25_hyperbolic_path_depth_attention_mp_code2vec",
            "path_attention",
            "diagnostic",
            ("path_attention",),
            "B23 plus learned root-to-leaf depth bias.",
        ),
        Code2HypVariantMetadata(
            "B26_hyperbolic_path_depth_attention_mp_rank_annealed",
            "path_attention",
            "diagnostic",
            ("path_attention",),
            "B25 plus linear structural-rank schedule.",
        ),
        Code2HypVariantMetadata(
            "B27_hyperbolic_path_attention_mp_monotone",
            "path_attention",
            "negative_control",
            ("path_attention",),
            "B23 plus bidirectional monotone attention-profile penalty.",
        ),
        Code2HypVariantMetadata(
            "B28_hyperbolic_path_attention_mp_tree_distance",
            "path_attention",
            "negative_control",
            ("path_attention",),
            "B23 plus soft tree-distance calibration of node attention.",
        ),
        Code2HypVariantMetadata(
            "B29_hyperbolic_path_dual_attention_mp_separated",
            "path_attention",
            "exploratory",
            ("balanced", "path_attention", "structural"),
            "Dual root/detail AST-path attention with local separation.",
        ),
        Code2HypVariantMetadata(
            "B30_hyperbolic_path_dual_attention_mp_rank_separated",
            "path_attention",
            "negative_control",
            ("path_attention",),
            "B29 plus hard global structural-rank regularization.",
        ),
        Code2HypVariantMetadata(
            "B31_hyperbolic_path_dual_attention_mp_soft_rank",
            "path_attention",
            "exploratory",
            ("balanced", "path_attention"),
            "B29 plus soft global structural-rank regularization.",
        ),
        Code2HypVariantMetadata(
            "B32_lorentz_path_dual_attention_mp_soft_rank",
            "path_attention",
            "exploratory",
            ("balanced", "path_attention"),
            "Lorentz-coordinate control for B31-style dual AST-path attention.",
        ),
        Code2HypVariantMetadata(
            "B34_hyperbolic_path_dual_attention_mp_adaptive_rank",
            "path_attention",
            "exploratory",
            ("path_attention",),
            "B29 with separation-gated adaptive global structural-rank pressure.",
        ),
        Code2HypVariantMetadata(
            "B9_lorentz_code2vec",
            "coordinate_control",
            "diagnostic",
            ("core",),
            "Lorentz-coordinate hyperbolic code2vec control.",
        ),
        Code2HypVariantMetadata(
            "B15_lorentz_product_code2vec",
            "product_geometry",
            "negative_control",
            ("structural",),
            "Lorentz AST-path product-space control.",
        ),
        Code2HypVariantMetadata(
            "B10_factorized_product_code2vec",
            "product_geometry",
            "diagnostic",
            ("structural",),
            "Factorized lexical/path product-space control.",
        ),
        Code2HypVariantMetadata(
            "B11_factorized_product_struct_rank",
            "product_geometry",
            "diagnostic",
            ("structural",),
            "B10 plus structural-rank regularization.",
        ),
        Code2HypVariantMetadata(
            "B12_factorized_product_learned_metric_rank",
            "product_geometry",
            "negative_control",
            ("structural",),
            "B11 plus learned product-metric weights.",
        ),
        Code2HypVariantMetadata(
            "B16_factorized_product_three_metric_rank",
            "product_geometry",
            "negative_control",
            ("structural",),
            "Factorized start/path/end three-metric product control.",
        ),
        Code2HypVariantMetadata(
            "B35_code2hyp_product_frechet_adaptive",
            "product_geometry",
            "candidate",
            ("structural", "code2vec_replacement"),
            "Code2vec-compatible product manifold with adaptive curvature, learned start/path/end metric weights, and intrinsic path Frechet aggregation.",
        ),
        Code2HypVariantMetadata(
            "B36_code2hyp_product_frechet_neighbor",
            "product_geometry",
            "candidate",
            ("structural", "code2vec_replacement"),
            "B35 architecture with local AST-neighborhood distribution regularization.",
        ),
        Code2HypVariantMetadata(
            "B37_code2hyp_code2vec_attention_frechet",
            "product_geometry",
            "candidate",
            ("structural", "code2vec_replacement"),
            "Minimal code2vec-faithful replacement: original dot-product path-context attention plus hyperbolic AST-path Frechet aggregation.",
        ),
        Code2HypVariantMetadata(
            "B38_code2hyp_code2vec_attention_neighbor",
            "product_geometry",
            "candidate",
            ("structural", "code2vec_replacement"),
            "B37 faithful code2vec attention with local AST-neighborhood distribution regularization.",
        ),
        Code2HypVariantMetadata(
            "B39_code2vec_context_transform_baseline",
            "baseline",
            "baseline",
            ("core", "code2vec_replacement"),
            "Euclidean code2vec baseline with tanh context transform before attention.",
        ),
        Code2HypVariantMetadata(
            "B46_code2vec_context_transform_neighbor_control",
            "euclidean_control",
            "diagnostic",
            ("structural", "code2vec_replacement"),
            "B39 Euclidean context-transform control with the same delayed local AST-neighborhood distribution regularizer used by B36.",
        ),
        Code2HypVariantMetadata(
            "B47_code2vec_context_transform_distance_control",
            "euclidean_control",
            "diagnostic",
            ("structural", "code2vec_replacement"),
            "B39 Euclidean context-transform control with delayed scale-invariant AST-path distance regularization.",
        ),
        Code2HypVariantMetadata(
            "B50_code2vec_context_transform_l1_baseline",
            "l1_control",
            "diagnostic",
            ("structural", "code2vec_replacement"),
            "B39 architecture evaluated with L1 structural distances; no structural objective.",
        ),
        Code2HypVariantMetadata(
            "B51_code2vec_context_transform_l1_distance_control",
            "l1_control",
            "diagnostic",
            ("structural", "code2vec_replacement"),
            "B39 architecture with L1 structural distances and delayed scale-invariant AST-path distance regularization.",
        ),
        Code2HypVariantMetadata(
            "B40_code2hyp_context_transform_frechet",
            "product_geometry",
            "candidate",
            ("structural", "code2vec_replacement"),
            "Code2vec context-transform attention with hyperbolic AST-path Frechet aggregation.",
        ),
        Code2HypVariantMetadata(
            "B41_code2hyp_context_transform_neighbor",
            "product_geometry",
            "candidate",
            ("structural", "code2vec_replacement"),
            "B40 context-transform Code2Hyp model with local AST-neighborhood distribution regularization.",
        ),
        Code2HypVariantMetadata(
            "B42_code2hyp_product_context_transform_frechet",
            "product_geometry",
            "candidate",
            ("structural", "code2vec_replacement"),
            "Code2vec context-transform vector with product-metric R x H x R attention and hyperbolic AST-path Frechet diagnostics.",
        ),
        Code2HypVariantMetadata(
            "B43_code2hyp_product_context_transform_neighbor",
            "product_geometry",
            "candidate",
            ("structural", "code2vec_replacement"),
            "B42 product-context-transform Code2Hyp model with local AST-neighborhood distribution regularization.",
        ),
        Code2HypVariantMetadata(
            "B44_code2hyp_context_transform_product_bias_frechet",
            "product_geometry",
            "candidate",
            ("structural", "code2vec_replacement"),
            "Code2vec context-transform attention with a trainable hyperbolic product-metric structural bias.",
        ),
        Code2HypVariantMetadata(
            "B48_code2hyp_context_transform_product_bias_no_struct",
            "product_geometry",
            "diagnostic",
            ("structural", "code2vec_replacement"),
            "B44 architecture without auxiliary structural regularization for Euclidean/hyperbolic objective-factorial control.",
        ),
        Code2HypVariantMetadata(
            "B49_code2hyp_context_transform_product_bias_near_euclidean",
            "product_geometry",
            "diagnostic",
            ("structural", "code2vec_replacement"),
            "B44 architecture and distance regularizer with fixed near-zero curvature c=1e-4 for same-code-path near-Euclidean control.",
        ),
        Code2HypVariantMetadata(
            "B45_code2hyp_context_transform_product_bias_neighbor",
            "product_geometry",
            "candidate",
            ("structural", "code2vec_replacement"),
            "B44 structurally biased code2vec attention with local AST-neighborhood distribution regularization.",
        ),
        Code2HypVariantMetadata(
            "B62_code2hyp_context_transform_branch_sequence_product_bias_multi_metric_frechet",
            "relation_geometry",
            "candidate",
            ("structural", "code2vec_replacement", "relation_geometry"),
            "Branch-sequence H x H product model trained against prefix-trie, edit, and Jaccard AST-path relations.",
        ),
        Code2HypVariantMetadata(
            "B66_branch_sequence_euclidean_product_l2_multi_metric_control",
            "matched_euclidean_control",
            "diagnostic",
            ("structural", "code2vec_replacement", "relation_geometry"),
            "Branch-sequence scaffold matched to B62, but with E x E structural factors and L2 product distances.",
        ),
        Code2HypVariantMetadata(
            "B67_branch_sequence_euclidean_product_l1_multi_metric_control",
            "matched_euclidean_control",
            "diagnostic",
            ("structural", "code2vec_replacement", "relation_geometry"),
            "Branch-sequence scaffold matched to B62, but with E x E structural factors and L1 product distances.",
        ),
        Code2HypVariantMetadata(
            "B68_branch_sequence_product_bias_near_euclidean_multi_metric",
            "curvature_control",
            "diagnostic",
            ("structural", "code2vec_replacement", "relation_geometry"),
            "B62 architecture with fixed near-Euclidean curvature c=1e-4.",
        ),
        Code2HypVariantMetadata(
            "B69_branch_sequence_product_bias_fixed_curvature_multi_metric",
            "curvature_control",
            "diagnostic",
            ("structural", "code2vec_replacement", "relation_geometry"),
            "B62 architecture with fixed unit curvature instead of trainable curvature.",
        ),
        Code2HypVariantMetadata(
            "B70_branch_sequence_single_hyperbolic_multi_metric_control",
            "product_geometry_control",
            "diagnostic",
            ("structural", "code2vec_replacement", "relation_geometry"),
            "Branch-sequence scaffold with a single matched-dimensional Poincare factor instead of H x H.",
        ),
        Code2HypVariantMetadata(
            "B80_geocodepath_endpoint_geodesic_product_proxy",
            "geocodepath_proxy",
            "candidate",
            ("structural", "code2vec_replacement", "relation_geometry", "geocodepath"),
            "Endpoint-geodesic proxy that represents an oriented path context as an ordered pair of Poincare endpoint factors.",
        ),
        Code2HypVariantMetadata(
            "B81_geocodepath_endpoint_lca_product_proxy",
            "geocodepath_proxy",
            "candidate",
            ("structural", "code2vec_replacement", "relation_geometry", "geocodepath"),
            "Endpoint-geodesic proxy with an additional latent LCA/pivot Poincare factor for the internal turn of the AST path.",
        ),
        Code2HypVariantMetadata(
            "B82_geocodepath_endpoint_lca_prior_product_proxy",
            "geocodepath_proxy",
            "candidate",
            ("structural", "code2vec_replacement", "relation_geometry", "geocodepath"),
            "Endpoint-LCA product proxy with a trainable center prior on the latent pivot factor.",
        ),
        Code2HypVariantMetadata(
            "B83_geocodepath_endpoint_lca_axiom_product_proxy",
            "geocodepath_proxy",
            "candidate",
            ("structural", "code2vec_replacement", "relation_geometry", "geocodepath"),
            "Endpoint-LCA product proxy with shared node coordinates and a combined multi-metric plus LCA/Gromov axiom regularizer.",
        ),
        Code2HypVariantMetadata(
            "B84_geocodepath_relation_conditioned_product_proxy",
            "geocodepath_proxy",
            "candidate",
            ("structural", "code2vec_replacement", "relation_geometry", "geocodepath"),
            "Relation-conditioned product proxy that assigns prefix-trie and endpoint-LCA relations to separate hyperbolic factors.",
        ),
        Code2HypVariantMetadata(
            "B85_geocodepath_relation_conditioned_aux_product_proxy",
            "geocodepath_proxy",
            "candidate",
            ("structural", "code2vec_replacement", "relation_geometry", "geocodepath"),
            "Auxiliary relation-conditioned proxy that keeps branch-sequence factors in task attention and trains endpoint-LCA factors as structural auxiliaries.",
        ),
        Code2HypVariantMetadata(
            "B86_geocodepath_method_transport_aux_product_proxy",
            "geocodepath_proxy",
            "candidate",
            ("structural", "code2vec_replacement", "relation_geometry", "geocodepath"),
            "B85 backbone trained with method-level optimal-transport consistency over path-context distributions.",
        ),
        Code2HypVariantMetadata(
            "B87_geocodepath_multi_metric_method_transport_aux_product_proxy",
            "geocodepath_proxy",
            "candidate",
            ("structural", "code2vec_replacement", "relation_geometry", "geocodepath"),
            "B85 backbone trained with method-level optimal-transport consistency over prefix, edit, and path-bigram relations.",
        ),
        Code2HypVariantMetadata(
            "B71_branch_sequence_product_bias_prefix_only",
            "relation_generalization",
            "diagnostic",
            ("structural", "relation_geometry"),
            "B62 architecture trained only against prefix-trie distances for leave-one-relation-out analysis.",
        ),
        Code2HypVariantMetadata(
            "B72_branch_sequence_product_bias_edit_only",
            "relation_generalization",
            "diagnostic",
            ("structural", "relation_geometry"),
            "B62 architecture trained only against edit distances for leave-one-relation-out analysis.",
        ),
        Code2HypVariantMetadata(
            "B73_branch_sequence_product_bias_jaccard_only",
            "relation_generalization",
            "diagnostic",
            ("structural", "relation_geometry"),
            "B62 architecture trained only against path-bigram Jaccard distances for leave-one-relation-out analysis.",
        ),
        Code2HypVariantMetadata(
            "B74_branch_sequence_product_bias_prefix_edit",
            "relation_generalization",
            "diagnostic",
            ("structural", "relation_geometry"),
            "B62 architecture trained on prefix-trie and edit distances, holding Jaccard out.",
        ),
        Code2HypVariantMetadata(
            "B75_branch_sequence_product_bias_prefix_jaccard",
            "relation_generalization",
            "diagnostic",
            ("structural", "relation_geometry"),
            "B62 architecture trained on prefix-trie and Jaccard distances, holding edit distance out.",
        ),
        Code2HypVariantMetadata(
            "B76_branch_sequence_product_bias_edit_jaccard",
            "relation_generalization",
            "diagnostic",
            ("structural", "relation_geometry"),
            "B62 architecture trained on edit and Jaccard distances, holding prefix-trie distance out.",
        ),
        Code2HypVariantMetadata(
            "B13_factorized_product_channel_mixer_rank",
            "product_geometry",
            "negative_control",
            ("structural",),
            "B11-style product geometry plus shallow channel mixer.",
        ),
        Code2HypVariantMetadata(
            "B7_hyperbolic_attention_only",
            "aggregation_control",
            "negative_control",
            ("core",),
            "Hyperbolic attention score without hyperbolic aggregation.",
        ),
        Code2HypVariantMetadata(
            "B5_euclidean_struct_loss",
            "baseline",
            "diagnostic",
            ("core",),
            "Euclidean baseline with structural loss.",
        ),
        Code2HypVariantMetadata(
            "B6_euclidean_metric_code2vec",
            "euclidean_control",
            "baseline",
            ("core",),
            "Euclidean distance-based code2vec control.",
        ),
        Code2HypVariantMetadata(
            "B14_bounded_euclidean_metric_code2vec",
            "euclidean_control",
            "baseline",
            ("core",),
            "Bounded Euclidean metric-code2vec control.",
        ),
        Code2HypVariantMetadata(
            "B_tree_euclidean_lca_bias",
            "tree_control",
            "baseline",
            ("core", "structural"),
            "Euclidean explicit LCA/tree-distance attention-bias control.",
        ),
    )
    return {entry.name: entry for entry in entries}


def available_profiles(catalog: Mapping[str, Code2HypVariantMetadata] | None = None) -> tuple[str, ...]:
    entries = catalog or variant_catalog()
    return tuple(sorted({profile for entry in entries.values() for profile in entry.profiles}))


def parse_variant_selection(
    raw_variants: str,
    profile: str | None,
    available_specs: Mapping[str, object],
) -> tuple[str, ...] | None:
    requested = tuple(variant.strip() for variant in raw_variants.split(",") if variant.strip())
    if requested and profile:
        raise ValueError("--variants and --variant-profile are mutually exclusive")

    catalog = variant_catalog()
    if profile:
        if profile not in available_profiles(catalog):
            raise ValueError(f"Unknown Code2Hyp variant profile: {profile}")
        requested = tuple(
            name
            for name, metadata in catalog.items()
            if profile in metadata.profiles
        )

    if not requested:
        return None

    unknown = [variant for variant in requested if variant not in available_specs]
    if unknown:
        raise ValueError(_unknown_variant_message(unknown[0], available_specs))
    return requested


def format_variant_catalog(catalog: Mapping[str, Code2HypVariantMetadata] | None = None) -> str:
    entries = catalog or variant_catalog()
    lines = ["Code2Hyp variants:", ""]
    for name in sorted(entries, key=_variant_sort_key):
        metadata = entries[name]
        profiles = ", ".join(metadata.profiles)
        lines.append(f"- {metadata.name}")
        lines.append(f"  family: {metadata.family}; status: {metadata.status}; profiles: {profiles}")
        lines.append(f"  summary: {metadata.summary}")
    lines.append("")
    lines.append(f"Profiles: {', '.join(available_profiles(entries))}")
    return "\n".join(lines)


def _unknown_variant_message(variant: str, available_specs: Mapping[str, object]) -> str:
    available = sorted(available_specs)
    suggestions = get_close_matches(variant, available, n=3)
    suffix = f" Did you mean: {', '.join(suggestions)}?" if suggestions else ""
    return f"Unknown Code2Hyp variant: {variant}.{suffix}"


def _variant_sort_key(name: str) -> tuple[int, int, str, str]:
    match = re.match(r"B(\d+)([A-Za-z]*)", name)
    if match:
        suffix = match.group(2)
        return (int(match.group(1)), 0 if not suffix else 1, suffix, name)
    return (10_000, 0, "", name)
