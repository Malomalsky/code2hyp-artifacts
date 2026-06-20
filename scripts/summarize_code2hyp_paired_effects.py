from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from geometry_profile_research.code2hyp_reporting import paired_metric_comparison


DEFAULT_RIGHT_VARIANTS = (
    "B1_euclidean",
    "B3_product",
    "B8_hyperbolic_frechet_code2vec",
    "B17_hyperbolic_path_mp_code2vec",
    "B18_hyperbolic_path_mp_struct_rank",
    "B19_hyperbolic_path_mp_rank_annealed",
    "B20_hyperbolic_path_mp_rank_delayed",
    "B21_hyperbolic_path_mp_rank_cosine",
    "B22_hyperbolic_path_mp_rank_warmup_decay",
    "B23_hyperbolic_path_attention_mp_code2vec",
    "B24_hyperbolic_path_attention_mp_rank_annealed",
    "B25_hyperbolic_path_depth_attention_mp_code2vec",
    "B26_hyperbolic_path_depth_attention_mp_rank_annealed",
    "B27_hyperbolic_path_attention_mp_monotone",
    "B28_hyperbolic_path_attention_mp_tree_distance",
    "B29_hyperbolic_path_dual_attention_mp_separated",
    "B30_hyperbolic_path_dual_attention_mp_rank_separated",
    "B31_hyperbolic_path_dual_attention_mp_soft_rank",
    "B32_lorentz_path_dual_attention_mp_soft_rank",
    "B34_hyperbolic_path_dual_attention_mp_adaptive_rank",
    "B9_lorentz_code2vec",
    "B15_lorentz_product_code2vec",
    "B10_factorized_product_code2vec",
    "B11_factorized_product_struct_rank",
    "B12_factorized_product_learned_metric_rank",
    "B16_factorized_product_three_metric_rank",
    "B13_factorized_product_channel_mixer_rank",
    "B7_hyperbolic_attention_only",
    "B6_euclidean_metric_code2vec",
    "B14_bounded_euclidean_metric_code2vec",
    "B_tree_euclidean_lca_bias",
)


def build_paired_effects_markdown(
    input_path: Path,
    left_variant: str,
    right_variants: tuple[str, ...],
    metric_key: str,
) -> str:
    result = json.loads(input_path.read_text(encoding="utf-8"))
    lines = [
        "| Left | Right | Metric | n | Mean delta | 95% bootstrap CI | sign-test p | Direction | Evidence status |",
        "|---|---|---|---:|---:|---:|---:|---|---|",
    ]
    for right_variant in right_variants:
        comparison = paired_metric_comparison(result, left_variant, right_variant, metric_key=metric_key)
        positive = int(comparison["positive_deltas"])
        negative = int(comparison["negative_deltas"])
        zero = int(comparison["zero_deltas"])
        lines.append(
            "| "
            f"{left_variant} | "
            f"{right_variant} | "
            f"{metric_key} | "
            f"{comparison['n']} | "
            f"{comparison['mean_delta']:+.4f} | "
            f"[{comparison['bootstrap_ci_low']:+.4f}, {comparison['bootstrap_ci_high']:+.4f}] | "
            f"{comparison['sign_test_p_two_sided']:.4f} | "
            f"+/{positive} -/{negative} 0/{zero} | "
            f"{comparison['evidence_status']} |"
        )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize paired Code2Hyp variant effects from a pilot JSON.")
    parser.add_argument("input", type=Path)
    parser.add_argument("--left-variant", default="B4_hyperbolic_code2vec")
    parser.add_argument("--right-variants", default=",".join(DEFAULT_RIGHT_VARIANTS))
    parser.add_argument("--metric-key", default="validation_f1")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    right_variants = tuple(variant.strip() for variant in args.right_variants.split(",") if variant.strip())
    markdown = build_paired_effects_markdown(
        args.input,
        left_variant=args.left_variant,
        right_variants=right_variants,
        metric_key=args.metric_key,
    )
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(markdown + "\n", encoding="utf-8")
        print(f"Wrote {args.output}")
    else:
        print(markdown)


if __name__ == "__main__":
    main()
