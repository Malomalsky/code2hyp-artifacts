from __future__ import annotations

from scripts.run_code2hyp_java_small_pilot import build_parser


def test_runner_accepts_variant_profile_flag() -> None:
    args = build_parser().parse_args(["--variant-profile", "balanced"])

    assert args.variant_profile == "balanced"


def test_runner_accepts_test_evaluation_split() -> None:
    args = build_parser().parse_args(["--eval-split", "test"])

    assert args.eval_split == "test"


def test_runner_accepts_list_variants_flag() -> None:
    args = build_parser().parse_args(["--list-variants"])

    assert args.list_variants is True


def test_runner_accepts_neighbor_distribution_regularizer() -> None:
    args = build_parser().parse_args(["--structural-regularizer", "neighbor_distribution"])

    assert args.structural_regularizer == "neighbor_distribution"


def test_runner_accepts_relation_specific_regularizer() -> None:
    args = build_parser().parse_args(["--structural-regularizer", "multi_metric_prefix_edit"])

    assert args.structural_regularizer == "multi_metric_prefix_edit"


def test_runner_accepts_lca_axiom_regularizer() -> None:
    args = build_parser().parse_args(["--structural-regularizer", "multi_metric_lca_axiom"])

    assert args.structural_regularizer == "multi_metric_lca_axiom"


def test_runner_accepts_relation_conditioned_lca_axiom_regularizer() -> None:
    args = build_parser().parse_args(["--structural-regularizer", "relation_conditioned_lca_axiom"])

    assert args.structural_regularizer == "relation_conditioned_lca_axiom"


def test_runner_accepts_context_sample_seed() -> None:
    args = build_parser().parse_args(["--context-sample-seed", "17"])

    assert args.context_sample_seed == 17
