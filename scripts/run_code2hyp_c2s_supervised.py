from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from geometry_profile_research.code2hyp_supervised import train_supervised_c2s_variants


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run code2vec-compatible Code2Hyp supervised `.c2s` experiment.")
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("outputs/code2hyp_supervised_c2s.json"))
    parser.add_argument(
        "--variant",
        action="append",
        choices=("euclidean", "poincare_near_zero", "poincare"),
        dest="variants",
        default=None,
        help="Variant to run. Repeat flag for multiple variants. Defaults to all three.",
    )
    parser.add_argument("--train-limit", type=int, default=1024)
    parser.add_argument("--validation-limit", type=int, default=512)
    parser.add_argument("--max-contexts", type=int, default=64)
    parser.add_argument("--max-path-length", type=int, default=16)
    parser.add_argument("--token-dim", type=int, default=32)
    parser.add_argument("--structural-dim", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=20260623)
    parser.add_argument("--pos-weight-max", type=float, default=20.0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    variants = tuple(args.variants) if args.variants else ("euclidean", "poincare_near_zero", "poincare")
    payload = train_supervised_c2s_variants(
        train_path=args.train,
        validation_path=args.validation,
        output_path=args.output,
        variants=variants,
        train_limit=args.train_limit,
        validation_limit=args.validation_limit,
        max_contexts=args.max_contexts,
        max_path_length=args.max_path_length,
        token_dim=args.token_dim,
        structural_dim=args.structural_dim,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
        pos_weight_max=args.pos_weight_max,
    )
    print(f"wrote {args.output} | train={payload['train_record_count']} validation={payload['validation_record_count_closed_vocab']}")
    for run in payload["runs"]:
        print(
            f"{run['variant']}: F1={run['validation_f1']:.4f} "
            f"P={run['validation_precision']:.4f} R={run['validation_recall']:.4f}"
        )


if __name__ == "__main__":
    main()
