from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_VARIANTS = (
    "B46_code2vec_context_transform_neighbor_control",
    "B44_code2hyp_context_transform_product_bias_frechet",
)


def parse_int_list(raw: str) -> tuple[int, ...]:
    values = tuple(int(item.strip()) for item in raw.split(",") if item.strip())
    if not values:
        raise argparse.ArgumentTypeError("list must contain at least one integer")
    if any(value <= 0 for value in values):
        raise argparse.ArgumentTypeError("all dimensions must be positive")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a structural-dimension and memory-proxy ablation for Code2Hyp. "
            "The script delegates each dimension to the resumable benchmark runner "
            "and annotates the resulting JSON with dimension/memory metadata."
        ),
    )
    parser.add_argument("--dims", type=parse_int_list, default=parse_int_list("4,8,16,32"))
    parser.add_argument("--token-dim", type=int, default=32)
    parser.add_argument("--variants", default=",".join(DEFAULT_VARIANTS))
    parser.add_argument("--model-seeds", default="101,202,303")
    parser.add_argument("--eval-split", choices=("val", "test"), default="test")
    parser.add_argument("--train-limit", type=int, default=10000)
    parser.add_argument("--val-limit", type=int, default=4096)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-contexts", type=int, default=30)
    parser.add_argument("--max-path-length", type=int, default=8)
    parser.add_argument("--path-encoder", choices=("mean", "gru"), default="gru")
    parser.add_argument("--representation-transform", choices=("identity", "tanh"), default="identity")
    parser.add_argument("--max-positive-weight", type=float, default=7.0)
    parser.add_argument("--output-prefix", default="code2hyp_dimension_memory_ablation")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def output_path(output_prefix: str, structural_dim: int) -> Path:
    return PROJECT_ROOT / "outputs" / f"{output_prefix}_structdim{structural_dim}.json"


def run_dimension(args: argparse.Namespace, structural_dim: int) -> Path:
    path = output_path(args.output_prefix, structural_dim)
    command = [
        sys.executable,
        "scripts/run_code2hyp_resumable_benchmark.py",
        "--eval-split",
        args.eval_split,
        "--train-limit",
        str(args.train_limit),
        "--val-limit",
        str(args.val_limit),
        "--max-contexts",
        str(args.max_contexts),
        "--max-path-length",
        str(args.max_path_length),
        "--token-dim",
        str(args.token_dim),
        "--structural-dim",
        str(structural_dim),
        "--path-encoder",
        args.path_encoder,
        "--representation-transform",
        args.representation_transform,
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--model-seeds",
        args.model_seeds,
        "--max-positive-weight",
        str(args.max_positive_weight),
        "--variants",
        args.variants,
        "--output",
        str(path),
    ]
    print(" ".join(command), flush=True)
    if not args.dry_run:
        subprocess.run(command, cwd=PROJECT_ROOT, check=True)
        annotate_result(path, args, structural_dim)
    return path


def annotate_result(path: Path, args: argparse.Namespace, structural_dim: int) -> None:
    result = json.loads(path.read_text(encoding="utf-8"))
    representation_dim = 2 * args.token_dim + structural_dim
    result["dimension_memory_ablation"] = {
        "status": "dimension_sweep_member",
        "token_dim_fixed": args.token_dim,
        "structural_dim": structural_dim,
        "representation_dim": representation_dim,
        "float32_parameter_bytes": 4,
        "activation_memory_proxy": (
            "max_contexts * representation_dim * 4 bytes per example for one dense "
            "context-representation tensor; this is not peak VRAM."
        ),
        "activation_proxy_bytes_per_example": args.max_contexts * representation_dim * 4,
        "claim_boundary": (
            "Parameter memory and activation proxy are engineering proxies. "
            "Do not report them as measured peak RAM/VRAM."
        ),
    }
    result.setdefault("training", {})["token_dim"] = args.token_dim
    result.setdefault("training", {})["structural_dim"] = structural_dim
    result.setdefault("training", {})["representation_dim"] = representation_dim
    result.setdefault("training", {})["max_contexts"] = args.max_contexts
    result.setdefault("training", {})["max_path_length"] = args.max_path_length
    result.setdefault("training", {})["dimension_ablation_output_prefix"] = args.output_prefix
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_index(args: argparse.Namespace, paths: list[Path]) -> Path:
    index_path = PROJECT_ROOT / "outputs" / f"{args.output_prefix}_index.json"
    index = {
        "experiment": "code2hyp structural-dimension memory-proxy ablation",
        "token_dim_fixed": args.token_dim,
        "structural_dims": list(args.dims),
        "variants": [variant.strip() for variant in args.variants.split(",") if variant.strip()],
        "model_seeds": [int(seed.strip()) for seed in args.model_seeds.split(",") if seed.strip()],
        "eval_split": args.eval_split,
        "train_limit": args.train_limit,
        "val_limit": args.val_limit,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "max_contexts": args.max_contexts,
        "max_path_length": args.max_path_length,
        "paths": [str(path.relative_to(PROJECT_ROOT)) for path in paths],
        "claim_boundary": (
            "This experiment tests dimension efficiency of structural representations. "
            "It does not measure production peak memory unless an external profiler is added."
        ),
    }
    index_path.write_text(json.dumps(index, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return index_path


def main() -> None:
    args = parse_args()
    paths = [run_dimension(args, structural_dim) for structural_dim in args.dims]
    index_path = write_index(args, paths)
    print(f"Wrote {index_path}", flush=True)


if __name__ == "__main__":
    main()
