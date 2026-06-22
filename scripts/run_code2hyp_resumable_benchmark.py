from __future__ import annotations

import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from geometry_profile_research.code2hyp_experiments import (
    RealCode2HypPilotConfig,
    real_variant_specs,
    run_real_code2hyp_pilot,
)
from geometry_profile_research.code2hyp_registry import (
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
from scripts.run_code2hyp_java_small_pilot import build_parser


def _run_key(run: dict[str, Any]) -> tuple[str, int]:
    return str(run["variant"]), int(run["model_seed"])


def completed_run_keys(result: dict[str, Any]) -> set[tuple[str, int]]:
    return {_run_key(run) for run in result.get("runs", [])}


def merge_single_run_result(accumulated: dict[str, Any] | None, single_result: dict[str, Any]) -> dict[str, Any]:
    if len(single_result.get("runs", [])) != 1:
        raise ValueError("single_result must contain exactly one run")

    if accumulated is None:
        merged = deepcopy(single_result)
        merged["runs"] = []
        merged["resumable_benchmark"] = {
            "status": "in_progress",
            "completed_runs": 0,
        }
    else:
        merged = deepcopy(accumulated)

    existing = completed_run_keys(merged)
    run = deepcopy(single_result["runs"][0])
    key = _run_key(run)
    if key not in existing:
        merged.setdefault("runs", []).append(run)
    merged["runs"] = sorted(merged["runs"], key=lambda item: (int(item["model_seed"]), str(item["variant"])))
    merged["resumable_benchmark"] = {
        "status": "in_progress",
        "completed_runs": len(merged["runs"]),
    }
    return merged


def mark_complete(
    result: dict[str, Any],
    expected_runs: int,
    model_seeds: tuple[int, ...] | None = None,
    variant_filter: tuple[str, ...] | None = None,
    eval_split: str | None = None,
) -> dict[str, Any]:
    completed = len(result.get("runs", []))
    status = "complete" if completed == expected_runs else "partial"
    marked = deepcopy(result)
    if model_seeds is not None:
        marked.setdefault("training", {})["model_seeds"] = list(model_seeds)
    if variant_filter is not None:
        marked.setdefault("training", {})["variant_filter"] = list(variant_filter)
    if eval_split is not None:
        marked.setdefault("evaluation", {})["split"] = eval_split
    marked["resumable_benchmark"] = {
        "status": status,
        "completed_runs": completed,
        "expected_runs": expected_runs,
    }
    return marked


def write_json_atomic(path: Path, result: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(path)


def _load_existing(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _make_config(args, model_seeds: tuple[int, ...], variant_filter: tuple[str, ...]) -> RealCode2HypPilotConfig:
    return RealCode2HypPilotConfig(
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
        variant_filter=variant_filter,
        sample_seed=args.sample_seed,
        context_sample_seed=args.context_sample_seed,
        structural_eval_limit=args.structural_eval_limit,
        structural_eval_seed=args.structural_eval_seed,
    )


def main() -> None:
    parser = build_parser()
    parser.description = (
        "Run a resumable Code2Hyp benchmark. The output JSON is flushed after "
        "each seed/variant pair, so long experiments can be safely resumed."
    )
    args = parser.parse_args()

    if args.list_variants:
        print(format_variant_catalog(variant_catalog()))
        return

    spec = java_small_preprocessed_spec()
    inspect_archive_status(args.data_root / spec.archive_name, spec.expected_bytes)
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

    probe_config = _make_config(args, model_seeds=model_seeds, variant_filter=())
    try:
        variant_filter = parse_variant_selection(
            args.variants,
            args.variant_profile,
            real_variant_specs(probe_config),
        )
    except ValueError as error:
        raise SystemExit(str(error)) from error
    if not variant_filter:
        raise SystemExit("resumable benchmark requires --variants or --variant-profile")

    evaluation_path = inventory.split_paths[args.eval_split]
    accumulated = _load_existing(args.output)
    completed = completed_run_keys(accumulated or {})
    expected_runs = len(model_seeds) * len(variant_filter)

    for model_seed in model_seeds:
        for variant_name in variant_filter:
            key = (variant_name, model_seed)
            if key in completed:
                print(f"skip completed variant={variant_name} seed={model_seed}", flush=True)
                continue
            print(f"run variant={variant_name} seed={model_seed}", flush=True)
            single_config = _make_config(
                args,
                model_seeds=(model_seed,),
                variant_filter=(variant_name,),
            )
            single_result = run_real_code2hyp_pilot(
                inventory.split_paths["train"],
                evaluation_path,
                single_config,
            )
            accumulated = merge_single_run_result(accumulated, single_result)
            write_json_atomic(args.output, accumulated)
            completed = completed_run_keys(accumulated)
            print(f"wrote {args.output} completed={len(completed)}/{expected_runs}", flush=True)

    if accumulated is None:
        raise SystemExit("no runs were executed or loaded")
    accumulated = mark_complete(
        accumulated,
        expected_runs=expected_runs,
        model_seeds=model_seeds,
        variant_filter=variant_filter,
        eval_split=args.eval_split,
    )
    write_json_atomic(args.output, accumulated)
    print(f"final status={accumulated['resumable_benchmark']['status']} output={args.output}")


if __name__ == "__main__":
    main()
