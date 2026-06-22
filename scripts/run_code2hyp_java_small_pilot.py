from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from geometry_profile_research.code2hyp_experiments import (
    RealCode2HypPilotConfig,
    real_variant_specs,
    run_real_code2hyp_pilot,
)
from geometry_profile_research.code2hyp_registry import (
    available_profiles,
    format_variant_catalog,
    parse_variant_selection,
    variant_catalog,
)
from geometry_profile_research.code2hyp_real_dataset import (
    Code2SeqPreprocessedInventory,
    inspect_archive_status,
    java_small_preprocessed_spec,
    write_dataset_manifest,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a real-data Code2Hyp pilot on the code2seq Java-small preprocessed corpus.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data/code2seq_java_small"),
        help="Directory containing the downloaded/extracted Java-small .c2s/.c2v preprocessed corpus.",
    )
    parser.add_argument(
        "--eval-split",
        choices=("val", "test"),
        default="val",
        help=(
            "Evaluation split. Use val for model selection and test only for "
            "final benchmark comparisons with code2vec/code2seq-style papers."
        ),
    )
    parser.add_argument("--train-limit", type=int, default=2048)
    parser.add_argument("--val-limit", type=int, default=512)
    parser.add_argument("--max-contexts", type=int, default=100)
    parser.add_argument("--max-path-length", type=int, default=8)
    parser.add_argument("--token-dim", type=int, default=32)
    parser.add_argument("--structural-dim", type=int, default=32)
    parser.add_argument("--curvature", type=float, default=1.0)
    parser.add_argument("--path-encoder", choices=("mean", "gru"), default="mean")
    parser.add_argument("--representation-transform", choices=("identity", "tanh"), default="identity")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.003)
    parser.add_argument("--structural-loss-weight", type=float, default=0.05)
    parser.add_argument(
        "--structural-regularizer",
        choices=("distance", "rank", "neighbor_distribution"),
        default="distance",
    )
    parser.add_argument(
        "--lexical-ablation",
        choices=("original", "obfuscated", "record_obfuscated", "structural_only"),
        default="original",
        help=(
            "Endpoint-token control: original keeps identifiers, obfuscated applies a "
            "stable global token renaming, record_obfuscated preserves only within-record "
            "token equality, structural_only masks endpoint tokens."
        ),
    )
    parser.add_argument("--no-positive-weighting", action="store_true")
    parser.add_argument("--max-positive-weight", type=float, default=20.0)
    parser.add_argument(
        "--list-variants",
        action="store_true",
        help="Print the supported Code2Hyp variant catalog and exit.",
    )
    parser.add_argument(
        "--variant-profile",
        choices=available_profiles(),
        default=None,
        help=(
            "Use a curated variant profile instead of listing variants manually. "
            "Use --list-variants to inspect profile membership."
        ),
    )
    parser.add_argument(
        "--model-seeds",
        type=str,
        default="101,202,303",
        help="Comma-separated model seeds, for example: 101 or 101,202,303.",
    )
    parser.add_argument(
        "--sample-seed",
        type=int,
        default=None,
        help=(
            "Reservoir-sample train/eval records with this seed instead of using "
            "the first N records. Leave unset to reproduce earlier first-N pilots."
        ),
    )
    parser.add_argument(
        "--context-sample-seed",
        type=int,
        default=None,
        help=(
            "Sample max-contexts path contexts per training record with this seed. "
            "Leave unset to keep the first max-contexts contexts for compatibility."
        ),
    )
    parser.add_argument(
        "--structural-eval-limit",
        type=int,
        default=512,
        help=(
            "Maximum known-target records used for structural diagnostics. Prediction "
            "metrics are still computed on the full known-target evaluation subset."
        ),
    )
    parser.add_argument(
        "--structural-eval-seed",
        type=int,
        default=314159,
        help="Seed for the structural-diagnostic subset when --structural-eval-limit is active.",
    )
    parser.add_argument(
        "--variants",
        type=str,
        default="",
        help=(
            "Optional comma-separated variant names for a focused pilot, for example: "
            "B4_hyperbolic_code2vec,B8_hyperbolic_frechet_code2vec,"
            "B17_hyperbolic_path_mp_code2vec,B18_hyperbolic_path_mp_struct_rank,"
            "B19_hyperbolic_path_mp_rank_annealed,B20_hyperbolic_path_mp_rank_delayed,"
            "B21_hyperbolic_path_mp_rank_cosine,B22_hyperbolic_path_mp_rank_warmup_decay,"
            "B23_hyperbolic_path_attention_mp_code2vec,B24_hyperbolic_path_attention_mp_rank_annealed,"
            "B25_hyperbolic_path_depth_attention_mp_code2vec,"
            "B26_hyperbolic_path_depth_attention_mp_rank_annealed,"
            "B27_hyperbolic_path_attention_mp_monotone,"
            "B28_hyperbolic_path_attention_mp_tree_distance,"
            "B29_hyperbolic_path_dual_attention_mp_separated,"
            "B30_hyperbolic_path_dual_attention_mp_rank_separated,"
            "B31_hyperbolic_path_dual_attention_mp_soft_rank,"
            "B32_lorentz_path_dual_attention_mp_soft_rank,"
            "B34_hyperbolic_path_dual_attention_mp_adaptive_rank,"
            "B35_code2hyp_product_frechet_adaptive,"
            "B36_code2hyp_product_frechet_neighbor,"
            "B37_code2hyp_code2vec_attention_frechet,"
            "B38_code2hyp_code2vec_attention_neighbor,"
            "B39_code2vec_context_transform_baseline,"
            "B46_code2vec_context_transform_neighbor_control,"
            "B40_code2hyp_context_transform_frechet,"
            "B41_code2hyp_context_transform_neighbor,"
            "B42_code2hyp_product_context_transform_frechet,"
            "B43_code2hyp_product_context_transform_neighbor,"
            "B44_code2hyp_context_transform_product_bias_frechet,"
            "B48_code2hyp_context_transform_product_bias_no_struct,"
            "B49_code2hyp_context_transform_product_bias_near_euclidean,"
            "B45_code2hyp_context_transform_product_bias_neighbor,"
            "B47_code2vec_context_transform_distance_control"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/code2hyp_java_small_real_pilot.json"),
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.list_variants:
        print(format_variant_catalog(variant_catalog()))
        return

    spec = java_small_preprocessed_spec()
    archive_status = inspect_archive_status(args.data_root / spec.archive_name, spec.expected_bytes)
    write_dataset_manifest(args.data_root / "DATASET_MANIFEST.md", spec)

    inventory = Code2SeqPreprocessedInventory.from_directory(args.data_root)
    if not inventory.has_all_required_splits:
        raise SystemExit(
            "Missing required train/val/test .c2s/.c2v splits under "
            f"{args.data_root}. Current split paths: {inventory.split_paths}"
        )
    model_seeds = tuple(int(seed) for seed in args.model_seeds.split(",") if seed.strip())
    if not model_seeds:
        raise SystemExit("--model-seeds must contain at least one integer seed")

    pilot_config = RealCode2HypPilotConfig(
        train_limit=args.train_limit,
        val_limit=args.val_limit,
        max_contexts=args.max_contexts,
        max_path_length=args.max_path_length,
        token_dim=args.token_dim,
        structural_dim=args.structural_dim,
        curvature=args.curvature,
        path_encoder=args.path_encoder,
        representation_transform=args.representation_transform,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        structural_loss_weight=args.structural_loss_weight,
        structural_regularizer=args.structural_regularizer,
        lexical_ablation=args.lexical_ablation,
        use_positive_weighting=not args.no_positive_weighting,
        max_positive_weight=args.max_positive_weight,
        model_seeds=model_seeds,
        sample_seed=args.sample_seed,
        context_sample_seed=args.context_sample_seed,
        structural_eval_limit=args.structural_eval_limit,
        structural_eval_seed=args.structural_eval_seed,
    )
    try:
        variant_filter = parse_variant_selection(
            args.variants,
            args.variant_profile,
            real_variant_specs(pilot_config),
        )
    except ValueError as error:
        raise SystemExit(str(error)) from error
    pilot_config = RealCode2HypPilotConfig(
        train_limit=pilot_config.train_limit,
        val_limit=pilot_config.val_limit,
        max_contexts=pilot_config.max_contexts,
        max_path_length=pilot_config.max_path_length,
        token_dim=pilot_config.token_dim,
        structural_dim=pilot_config.structural_dim,
        curvature=pilot_config.curvature,
        path_encoder=pilot_config.path_encoder,
        representation_transform=pilot_config.representation_transform,
        epochs=pilot_config.epochs,
        batch_size=pilot_config.batch_size,
        learning_rate=pilot_config.learning_rate,
        structural_loss_weight=pilot_config.structural_loss_weight,
        structural_regularizer=pilot_config.structural_regularizer,
        lexical_ablation=pilot_config.lexical_ablation,
        use_positive_weighting=pilot_config.use_positive_weighting,
        max_positive_weight=pilot_config.max_positive_weight,
        model_seeds=pilot_config.model_seeds,
        variant_filter=variant_filter,
        sample_seed=pilot_config.sample_seed,
        structural_eval_limit=pilot_config.structural_eval_limit,
        structural_eval_seed=pilot_config.structural_eval_seed,
    )

    evaluation_path = inventory.split_paths[args.eval_split]
    result = run_real_code2hyp_pilot(
        inventory.split_paths["train"],
        evaluation_path,
        pilot_config,
    )
    result["evaluation"] = {
        "split": args.eval_split,
        "path": str(evaluation_path),
        "claim_boundary": (
            "Use eval_split=val for model selection. Use eval_split=test only "
            "after variants and hyperparameters are fixed."
        ),
    }
    result["archive_status"] = {
        "path": str(archive_status.path),
        "exists": archive_status.exists,
        "bytes": archive_status.bytes,
        "size_matches": archive_status.size_matches,
    }
    result["inventory"] = {
        "root": str(inventory.root),
        "split_paths": {split: str(path) for split, path in inventory.split_paths.items()},
        "split_line_counts": inventory.split_line_counts,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
