from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Any, Sequence

import torch
from torch.nn import functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_raw_ast_fgw_benchmark import _retrieval_overlap, _spearman  # noqa: E402


DEFAULT_FEATURES = (
    "lca_only_sinkhorn",
    "endpoint_product_sinkhorn",
    "lca_product_sinkhorn",
    "feature_ot",
    "centroid",
)


def _load_pairs(path: Path, feature_names: Sequence[str], target: str) -> tuple[list[dict[str, Any]], int | None]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    pairs = payload.get("pairs")
    if not isinstance(pairs, list) or not pairs:
        raise ValueError(f"{path} does not contain non-empty 'pairs'")
    for pair in pairs:
        if target not in pair:
            raise ValueError(f"pair is missing target field {target!r}")
        for feature_name in feature_names:
            if feature_name not in pair:
                raise ValueError(f"pair is missing feature field {feature_name!r}")
    method_count = payload.get("method_count")
    return pairs, int(method_count) if method_count is not None else None


def _split_indices(count: int, train_fraction: float, seed: int) -> tuple[list[int], list[int]]:
    if count < 3:
        raise ValueError("at least three pairs are required for a train/test split")
    if not (0.0 < train_fraction < 1.0):
        raise ValueError("train_fraction must be between 0 and 1")
    indices = list(range(count))
    random.Random(seed).shuffle(indices)
    split_at = int(round(count * train_fraction))
    split_at = max(1, min(count - 1, split_at))
    return indices[:split_at], indices[split_at:]


def _metrics(prediction: Sequence[float], target: Sequence[float]) -> dict[str, float]:
    if len(prediction) != len(target):
        raise ValueError("prediction and target lengths differ")
    errors = [pred - gold for pred, gold in zip(prediction, target)]
    mae = sum(abs(error) for error in errors) / len(errors)
    rmse = math.sqrt(sum(error * error for error in errors) / len(errors))
    return {
        "count": float(len(prediction)),
        "spearman": _spearman(prediction, target),
        "mae": float(mae),
        "rmse": float(rmse),
    }


def _matrix_from_pairs(pairs: Sequence[dict[str, Any]], values: Sequence[float], method_count: int) -> list[list[float]]:
    matrix = [[0.0 for _ in range(method_count)] for _ in range(method_count)]
    for pair, value in zip(pairs, values):
        left = int(pair["left"])
        right = int(pair["right"])
        matrix[left][right] = float(value)
        matrix[right][left] = float(value)
    return matrix


def fit_pair_distance_student(
    pair_json: Path,
    *,
    feature_names: Sequence[str] = DEFAULT_FEATURES,
    target: str = "teacher_fgw",
    train_fraction: float = 0.5,
    split_seed: int = 20260623,
    epochs: int = 1500,
    learning_rate: float = 0.03,
    l2_penalty: float = 1e-4,
) -> dict[str, Any]:
    """Fit a non-negative linear distance student from pair-level diagnostics."""

    if epochs <= 0:
        raise ValueError("epochs must be positive")
    pairs, method_count = _load_pairs(pair_json, feature_names, target)
    train_indices, test_indices = _split_indices(len(pairs), train_fraction, split_seed)

    x = torch.tensor([[float(pair[name]) for name in feature_names] for pair in pairs], dtype=torch.float32)
    y = torch.tensor([float(pair[target]) for pair in pairs], dtype=torch.float32)
    train_tensor = torch.tensor(train_indices, dtype=torch.long)
    test_tensor = torch.tensor(test_indices, dtype=torch.long)

    x_mean = x[train_tensor].mean(dim=0, keepdim=True)
    x_std = torch.clamp(x[train_tensor].std(dim=0, unbiased=False, keepdim=True), min=1e-6)
    y_mean = y[train_tensor].mean()
    y_std = torch.clamp(y[train_tensor].std(unbiased=False), min=1e-6)
    x_scaled = (x - x_mean) / x_std
    y_scaled = (y - y_mean) / y_std

    raw_weights = torch.zeros((len(feature_names),), dtype=torch.float32, requires_grad=True)
    bias = torch.zeros((), dtype=torch.float32, requires_grad=True)
    optimizer = torch.optim.Adam([raw_weights, bias], lr=learning_rate)
    for _ in range(epochs):
        optimizer.zero_grad()
        weights = F.softplus(raw_weights)
        prediction = x_scaled[train_tensor] @ weights + bias
        loss = F.mse_loss(prediction, y_scaled[train_tensor]) + l2_penalty * torch.sum(weights.square())
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        weights = F.softplus(raw_weights)
        scaled_prediction = x_scaled @ weights + bias
        prediction = scaled_prediction * y_std + y_mean
    predictions = [float(value) for value in prediction]
    targets = [float(value) for value in y]
    train_predictions = [predictions[index] for index in train_indices]
    train_targets = [targets[index] for index in train_indices]
    test_predictions = [predictions[index] for index in test_indices]
    test_targets = [targets[index] for index in test_indices]

    retrieval_overlap = None
    if method_count is not None and len(pairs) == method_count * (method_count - 1) // 2:
        retrieval_overlap = _retrieval_overlap(
            {
                "fgw": _matrix_from_pairs(pairs, targets, method_count),
                "student": _matrix_from_pairs(pairs, predictions, method_count),
            },
            k=1,
        )
        retrieval_overlap.pop("fgw", None)

    return {
        "config": {
            "pair_json": str(pair_json),
            "features": list(feature_names),
            "target": target,
            "train_fraction": train_fraction,
            "split_seed": split_seed,
            "epochs": epochs,
            "learning_rate": learning_rate,
            "l2_penalty": l2_penalty,
        },
        "pair_count": len(pairs),
        "train_pair_count": len(train_indices),
        "test_pair_count": len(test_indices),
        "weights": {name: float(value.detach()) for name, value in zip(feature_names, weights)},
        "bias_scaled": float(bias.detach()),
        "train": _metrics(train_predictions, train_targets),
        "test": _metrics(test_predictions, test_targets),
        "full": _metrics(predictions, targets),
        "retrieval_overlap_at_1": retrieval_overlap,
        "claim_boundary": (
            "This is a non-negative linear distillation diagnostic over already computed pair-level "
            "distances. It does not learn AST node embeddings phi_theta and must not be presented "
            "as the final GeoCodePath model."
        ),
    }


def write_markdown_report(result: dict[str, Any], output_path: Path) -> None:
    lines = [
        "# Pair-distance student distillation",
        "",
        "A non-negative linear student is fitted on train pairs and evaluated on held-out pairs.",
        "",
        f"- Pair JSON: `{result['config']['pair_json']}`",
        f"- Target: `{result['config']['target']}`",
        f"- Train pairs: `{result['train_pair_count']}`",
        f"- Test pairs: `{result['test_pair_count']}`",
        "",
        "## Metrics",
        "",
        "| Split | Spearman | MAE | RMSE |",
        "|---|---:|---:|---:|",
    ]
    for split_name in ("train", "test", "full"):
        row = result[split_name]
        lines.append(f"| {split_name} | {row['spearman']:.6f} | {row['mae']:.6f} | {row['rmse']:.6f} |")
    lines.extend(["", "## Learned non-negative weights", "", "| Feature | Weight |", "|---|---:|"])
    for name, value in sorted(result["weights"].items(), key=lambda item: item[1], reverse=True):
        lines.append(f"| {name} | {value:.6f} |")
    if result.get("retrieval_overlap_at_1"):
        lines.extend(["", "## Retrieval overlap", "", f"- Top-1 overlap: `{result['retrieval_overlap_at_1']['student']:.6f}`"])
    lines.extend(["", "## Claim boundary", "", result["claim_boundary"], ""])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fit a non-negative pair-distance student.")
    parser.add_argument("--pair-json", type=Path, required=True)
    parser.add_argument("--feature", dest="features", action="append", default=None)
    parser.add_argument("--target", default="teacher_fgw")
    parser.add_argument("--train-fraction", type=float, default=0.5)
    parser.add_argument("--split-seed", type=int, default=20260623)
    parser.add_argument("--epochs", type=int, default=1500)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--l2-penalty", type=float, default=1e-4)
    parser.add_argument("--output", type=Path, default=Path("outputs/pair_distance_student.json"))
    parser.add_argument("--report", type=Path, default=Path("reports/pair_distance_student.md"))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = fit_pair_distance_student(
        args.pair_json,
        feature_names=tuple(args.features) if args.features else DEFAULT_FEATURES,
        target=args.target,
        train_fraction=args.train_fraction,
        split_seed=args.split_seed,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        l2_penalty=args.l2_penalty,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    write_markdown_report(result, args.report)
    print(
        json.dumps(
            {
                "train": result["train"],
                "test": result["test"],
                "weights": result["weights"],
                "retrieval_overlap_at_1": result["retrieval_overlap_at_1"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    print(f"wrote {args.output}")
    print(f"wrote {args.report}")


if __name__ == "__main__":
    main()
