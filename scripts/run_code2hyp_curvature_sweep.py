from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from geometry_profile_research.code2hyp_experiments import (
    RealCode2HypPilotConfig,
    run_real_code2hyp_pilot,
)
from geometry_profile_research.code2hyp_real_dataset import Code2SeqPreprocessedInventory
from geometry_profile_research.code2hyp_reporting import summarize_pilot_runs
from geometry_profile_research.code2hyp_training import STRUCTURAL_DISTANCE_TARGETS


STRUCTURAL_REGULARIZER_CHOICES = (
    "distance",
    "rank",
    "neighbor_distribution",
    *STRUCTURAL_DISTANCE_TARGETS.keys(),
)


def parse_curvature_grid(raw: str) -> tuple[float, ...]:
    values = tuple(float(part.strip()) for part in raw.split(",") if part.strip())
    if not values:
        raise ValueError("curvature grid must contain at least one value")
    if any(value <= 0.0 for value in values):
        raise ValueError("all curvature values must be positive")
    return values


def curvature_output_path(output_dir: Path, curvature: float) -> Path:
    token = f"{curvature:g}".replace(".", "p").replace("-", "m")
    return output_dir / f"curvature_c{token}.json"


def _model_seeds(raw: str) -> tuple[int, ...]:
    seeds = tuple(int(seed.strip()) for seed in raw.split(",") if seed.strip())
    if not seeds:
        raise ValueError("model seed list must contain at least one integer")
    return seeds


def run_curvature_sweep(args: argparse.Namespace) -> dict[str, Any]:
    inventory = Code2SeqPreprocessedInventory.from_directory(args.data_root)
    if not inventory.has_all_required_splits:
        raise SystemExit(
            "Missing required train/val/test .c2s/.c2v splits under "
            f"{args.data_root}. Current split paths: {inventory.split_paths}"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    curvature_values = parse_curvature_grid(args.curvatures)
    model_seeds = _model_seeds(args.model_seeds)
    sweep_runs = []
    for curvature in curvature_values:
        config = RealCode2HypPilotConfig(
            train_limit=args.train_limit,
            val_limit=args.val_limit,
            max_contexts=args.max_contexts,
            max_path_length=args.max_path_length,
            token_dim=args.token_dim,
            structural_dim=args.structural_dim,
            curvature=curvature,
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
        )
        result = run_real_code2hyp_pilot(inventory.split_paths["train"], inventory.split_paths["val"], config)
        output_path = curvature_output_path(args.output_dir, curvature)
        output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        sweep_runs.append(
            {
                "curvature": curvature,
                "output_path": str(output_path),
                "summaries": summarize_pilot_runs(result),
            }
        )
        print(f"Wrote {output_path}")

    aggregate = {
        "experiment": "code2hyp_curvature_sweep",
        "curvatures": list(curvature_values),
        "output_dir": str(args.output_dir),
        "runs": sweep_runs,
        "claim_boundary": (
            "This sweep is an exploratory sensitivity analysis. It can show "
            "whether fixed curvature scale matters, but confirmatory claims "
            "require a preregistered split and paired uncertainty estimates."
        ),
    }
    args.aggregate_output.parent.mkdir(parents=True, exist_ok=True)
    args.aggregate_output.write_text(json.dumps(aggregate, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {args.aggregate_output}")
    return aggregate


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Code2Hyp fixed-curvature sensitivity sweep.")
    parser.add_argument("--data-root", type=Path, default=Path("data/code2seq_java_small"))
    parser.add_argument("--curvatures", type=str, default="0.1,0.3,1.0,3.0")
    parser.add_argument("--train-limit", type=int, default=1024)
    parser.add_argument("--val-limit", type=int, default=256)
    parser.add_argument("--max-contexts", type=int, default=30)
    parser.add_argument("--max-path-length", type=int, default=8)
    parser.add_argument("--token-dim", type=int, default=32)
    parser.add_argument("--structural-dim", type=int, default=32)
    parser.add_argument("--path-encoder", choices=("mean", "gru"), default="gru")
    parser.add_argument("--representation-transform", choices=("identity", "tanh"), default="identity")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.003)
    parser.add_argument("--structural-loss-weight", type=float, default=0.05)
    parser.add_argument(
        "--structural-regularizer",
        choices=STRUCTURAL_REGULARIZER_CHOICES,
        default="distance",
    )
    parser.add_argument(
        "--lexical-ablation",
        choices=("original", "obfuscated", "record_obfuscated", "structural_only"),
        default="original",
    )
    parser.add_argument("--no-positive-weighting", action="store_true")
    parser.add_argument("--max-positive-weight", type=float, default=20.0)
    parser.add_argument("--model-seeds", type=str, default="101")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/code2hyp_curvature_sweep"))
    parser.add_argument(
        "--aggregate-output",
        type=Path,
        default=Path("outputs/code2hyp_curvature_sweep_summary.json"),
    )
    return parser.parse_args(argv)


def main() -> None:
    run_curvature_sweep(parse_args())


if __name__ == "__main__":
    main()
