from __future__ import annotations

import argparse
import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from geometry_profile_research.lca_path_measure import (  # noqa: E402
    EuclideanPathMeasure,
    PoincarePathMeasure,
    poincare_gromov_product_at_origin,
    sinkhorn_euclidean_path_measure_distance,
    sinkhorn_path_measure_distance,
)
from scripts.run_raw_ast_fgw_benchmark import _spearman, collect_raw_ast_method_spaces  # noqa: E402


@dataclass(frozen=True)
class IndexedPathMethod:
    lca_labels: Tensor
    start_labels: Tensor
    end_labels: Tensor
    lca_prefixes: Tensor
    start_prefixes: Tensor
    end_prefixes: Tensor
    lca_depths: Tensor
    start_depths: Tensor
    end_depths: Tensor
    mass: Tensor


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
    errors = [pred - gold for pred, gold in zip(prediction, target)]
    mae = sum(abs(error) for error in errors) / len(errors)
    rmse = math.sqrt(sum(error * error for error in errors) / len(errors))
    return {
        "count": float(len(prediction)),
        "spearman": _spearman(prediction, target),
        "mae": float(mae),
        "rmse": float(rmse),
    }


def _load_teacher_pairs(pair_json: Path) -> tuple[list[dict[str, Any]], int]:
    payload = json.loads(pair_json.read_text(encoding="utf-8"))
    pairs = payload.get("pairs")
    if not isinstance(pairs, list) or not pairs:
        raise ValueError("pair_json must contain non-empty pairs")
    method_count = int(payload.get("method_count", 0))
    if method_count <= 1:
        raise ValueError("pair_json must contain method_count > 1")
    for pair in pairs:
        for key in ("left", "right", "teacher_fgw"):
            if key not in pair:
                raise ValueError(f"pair is missing {key!r}")
    return pairs, method_count


def _build_label_vocab(spaces: Sequence[Any]) -> dict[str, int]:
    labels = {"<unk>": 0}
    for space in spaces:
        for label in space.tree.labels.values():
            labels.setdefault(label, len(labels))
    return labels


def _node_path_signature(tree: Any, node: int) -> str:
    ancestors = tuple(reversed(tree.ancestors(node)))
    parts = []
    for current in ancestors:
        parent = tree.parent(current)
        child_index = 0 if parent is None else tree.children_by_node[parent].index(current)
        parts.append(f"{tree.labels.get(current, '')}:{child_index}")
    return "/".join(parts)


def _build_prefix_vocab(spaces: Sequence[Any]) -> dict[str, int]:
    prefixes = {"<unk>": 0}
    for space in spaces:
        for node in space.tree.preorder():
            prefixes.setdefault(_node_path_signature(space.tree, node), len(prefixes))
    return prefixes


def _index_method(space: Any, label_to_id: dict[str, int], prefix_to_id: dict[str, int]) -> IndexedPathMethod:
    lca_labels = []
    start_labels = []
    end_labels = []
    lca_prefixes = []
    start_prefixes = []
    end_prefixes = []
    lca_depths = []
    start_depths = []
    end_depths = []
    for path in space.paths:
        lca = path.lca(space.tree)
        lca_labels.append(label_to_id.get(space.tree.labels.get(lca, "<unk>"), 0))
        start_labels.append(label_to_id.get(space.tree.labels.get(path.start, "<unk>"), 0))
        end_labels.append(label_to_id.get(space.tree.labels.get(path.end, "<unk>"), 0))
        lca_prefixes.append(prefix_to_id.get(_node_path_signature(space.tree, lca), 0))
        start_prefixes.append(prefix_to_id.get(_node_path_signature(space.tree, path.start), 0))
        end_prefixes.append(prefix_to_id.get(_node_path_signature(space.tree, path.end), 0))
        lca_depths.append(float(space.tree.depth(lca)))
        start_depths.append(float(space.tree.depth(path.start)))
        end_depths.append(float(space.tree.depth(path.end)))
    path_count = len(space.paths)
    return IndexedPathMethod(
        lca_labels=torch.tensor(lca_labels, dtype=torch.long),
        start_labels=torch.tensor(start_labels, dtype=torch.long),
        end_labels=torch.tensor(end_labels, dtype=torch.long),
        lca_prefixes=torch.tensor(lca_prefixes, dtype=torch.long),
        start_prefixes=torch.tensor(start_prefixes, dtype=torch.long),
        end_prefixes=torch.tensor(end_prefixes, dtype=torch.long),
        lca_depths=torch.tensor(lca_depths, dtype=torch.float32),
        start_depths=torch.tensor(start_depths, dtype=torch.float32),
        end_depths=torch.tensor(end_depths, dtype=torch.float32),
        mass=torch.full((path_count,), 1.0 / path_count, dtype=torch.float32),
    )


class SharedLabelDepthPoincareEncoder(nn.Module):
    """Shared phi_theta(label, depth, optional prefix) for product path measures."""

    def __init__(
        self,
        label_count: int,
        *,
        prefix_count: int = 1,
        use_prefix_encoder: bool = False,
        geometry: str = "poincare",
        node_input_mode: str = "label_depth_prefix",
        curvature: float = 1.0,
        seed: int = 0,
    ) -> None:
        super().__init__()
        if geometry not in {"poincare", "euclidean"}:
            raise ValueError(f"unknown geometry: {geometry!r}")
        if node_input_mode not in {"label_only", "label_depth", "label_depth_prefix"}:
            raise ValueError(f"unknown node_input_mode: {node_input_mode!r}")
        generator = torch.Generator().manual_seed(seed)
        self.geometry = geometry
        self.node_input_mode = node_input_mode
        self.curvature = float(curvature)
        self.use_prefix_encoder = bool(use_prefix_encoder) and node_input_mode == "label_depth_prefix"
        self.label_direction = nn.Parameter(torch.randn((label_count, 2), generator=generator) * 0.05)
        self.prefix_direction = nn.Parameter(torch.randn((prefix_count, 2), generator=generator) * 0.05)
        self.prefix_logit = nn.Parameter(torch.tensor(-0.5))
        self.radial_logit = nn.Parameter(torch.tensor(-1.0))
        self.factor_logits = nn.Parameter(torch.tensor([1.0, -1.5, -1.5], dtype=torch.float32))

    def _embed_nodes(self, labels: Tensor, prefixes: Tensor, depths: Tensor) -> Tensor:
        direction_source = self.label_direction[labels]
        if self.use_prefix_encoder:
            direction_source = direction_source + F.softplus(self.prefix_logit) * self.prefix_direction[prefixes]
        direction = F.normalize(direction_source, dim=-1)
        sqrt_c = math.sqrt(self.curvature)
        radial_scale = F.softplus(self.radial_logit)
        effective_depths = depths if self.node_input_mode in {"label_depth", "label_depth_prefix"} else torch.ones_like(depths)
        if self.geometry == "poincare":
            radius = torch.tanh(radial_scale * effective_depths).unsqueeze(-1) / sqrt_c
        else:
            radius = (radial_scale * effective_depths).unsqueeze(-1)
        return direction * radius

    def measure(self, method: IndexedPathMethod) -> PoincarePathMeasure | EuclideanPathMeasure:
        points = torch.stack(
            [
                self._embed_nodes(method.lca_labels, method.lca_prefixes, method.lca_depths),
                self._embed_nodes(method.start_labels, method.start_prefixes, method.start_depths),
                self._embed_nodes(method.end_labels, method.end_prefixes, method.end_depths),
            ],
            dim=1,
        )
        if self.geometry == "euclidean":
            return EuclideanPathMeasure(points=points, mass=method.mass)
        return PoincarePathMeasure(points=points, mass=method.mass, curvature=self.curvature)

    def path_gromov_products(self, method: IndexedPathMethod) -> Tensor:
        start = self._embed_nodes(method.start_labels, method.start_prefixes, method.start_depths)
        end = self._embed_nodes(method.end_labels, method.end_prefixes, method.end_depths)
        if self.geometry == "euclidean":
            origin = torch.zeros_like(start)
            return 0.5 * (
                torch.linalg.vector_norm(start - origin, dim=-1)
                + torch.linalg.vector_norm(end - origin, dim=-1)
                - torch.linalg.vector_norm(start - end, dim=-1)
            )
        return poincare_gromov_product_at_origin(start, end, curvature=self.curvature)

    def factor_weights(self) -> tuple[Tensor, Tensor, Tensor]:
        weights = F.softplus(self.factor_logits)
        return weights[0], weights[1], weights[2]

    def distance(self, left: IndexedPathMethod, right: IndexedPathMethod, *, epsilon: float, iterations: int) -> Tensor:
        lca_weight, start_weight, end_weight = self.factor_weights()
        if self.geometry == "euclidean":
            return sinkhorn_euclidean_path_measure_distance(
                self.measure(left),
                self.measure(right),
                epsilon=epsilon,
                iterations=iterations,
                lca_weight=lca_weight,
                start_weight=start_weight,
                end_weight=end_weight,
                unoriented=True,
            )
        return sinkhorn_path_measure_distance(
            self.measure(left),
            self.measure(right),
            epsilon=epsilon,
            iterations=iterations,
            lca_weight=lca_weight,
            start_weight=start_weight,
            end_weight=end_weight,
            unoriented=True,
        )


def fit_learned_poincare_path_student(
    sources: Sequence[Path],
    *,
    pair_json: Path,
    max_files: int | None = 20,
    max_methods: int | None = 32,
    sample_seed: int | None = None,
    max_paths_per_method: int = 16,
    min_paths_per_method: int = 4,
    teacher_relation: str = "lca_depth",
    train_fraction: float = 0.5,
    split_seed: int = 20260623,
    model_seed: int = 20260623,
    epochs: int = 40,
    batch_size: int = 32,
    learning_rate: float = 0.03,
    curvature: float = 1.0,
    epsilon: float = 0.05,
    sinkhorn_iterations: int = 40,
    gromov_loss_weight: float = 0.0,
    use_prefix_encoder: bool = False,
    geometry: str = "poincare",
    node_input_mode: str = "label_depth_prefix",
) -> dict[str, Any]:
    """Train a small shared phi_theta path-measure student against raw-AST FGW pairs."""

    pairs, pair_method_count = _load_teacher_pairs(pair_json)
    spaces = collect_raw_ast_method_spaces(
        sources,
        max_files=max_files,
        max_methods=max_methods,
        max_paths_per_method=max_paths_per_method,
        min_paths_per_method=min_paths_per_method,
        structural_relation=teacher_relation,
        sample_seed=sample_seed,
    )
    if len(spaces) != pair_method_count:
        raise ValueError(f"pair_json has {pair_method_count} methods, collected {len(spaces)} methods")
    label_to_id = _build_label_vocab(spaces)
    prefix_to_id = _build_prefix_vocab(spaces)
    indexed_methods = [_index_method(space, label_to_id, prefix_to_id) for space in spaces]
    train_indices, test_indices = _split_indices(len(pairs), train_fraction, split_seed)
    teacher = torch.tensor([float(pair["teacher_fgw"]) for pair in pairs], dtype=torch.float32)
    train_teacher = teacher[torch.tensor(train_indices, dtype=torch.long)]
    y_mean = train_teacher.mean()
    y_std = torch.clamp(train_teacher.std(unbiased=False), min=1e-6)

    torch.manual_seed(model_seed)
    model = SharedLabelDepthPoincareEncoder(
        len(label_to_id),
        prefix_count=len(prefix_to_id),
        use_prefix_encoder=use_prefix_encoder,
        geometry=geometry,
        node_input_mode=node_input_mode,
        curvature=curvature,
        seed=model_seed,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    rng = random.Random(split_seed)
    training_trace = []

    def gromov_alignment_loss(method_indices: Sequence[int]) -> Tensor:
        losses = []
        for method_index in sorted(set(method_indices)):
            method = indexed_methods[method_index]
            products = model.path_gromov_products(method)
            target = method.lca_depths
            product_std = torch.clamp(products.std(unbiased=False), min=1e-6)
            target_std = torch.clamp(target.std(unbiased=False), min=1e-6)
            losses.append(
                F.mse_loss(
                    (products - products.mean()) / product_std,
                    (target - target.mean()) / target_std,
                )
            )
        if not losses:
            return torch.zeros((), dtype=torch.float32)
        return torch.stack(losses).mean()

    for epoch in range(epochs):
        batch_indices = rng.sample(train_indices, k=min(batch_size, len(train_indices)))
        predictions = []
        targets = []
        method_indices_for_gromov = []
        optimizer.zero_grad()
        for pair_index in batch_indices:
            pair = pairs[pair_index]
            method_indices_for_gromov.extend([int(pair["left"]), int(pair["right"])])
            distance = model.distance(
                indexed_methods[int(pair["left"])],
                indexed_methods[int(pair["right"])],
                epsilon=epsilon,
                iterations=sinkhorn_iterations,
            )
            predictions.append(distance)
            targets.append((teacher[pair_index] - y_mean) / y_std)
        pred_tensor = torch.stack(predictions)
        target_tensor = torch.stack(targets)
        pred_scaled = (pred_tensor - pred_tensor.mean()) / torch.clamp(pred_tensor.std(unbiased=False), min=1e-6)
        distance_loss = F.mse_loss(pred_scaled, target_tensor)
        gromov_loss = gromov_alignment_loss(method_indices_for_gromov)
        loss = distance_loss + gromov_loss_weight * gromov_loss
        loss.backward()
        optimizer.step()
        if epoch == 0 or epoch == epochs - 1:
            training_trace.append(
                {
                    "epoch": epoch + 1,
                    "loss": float(loss.detach()),
                    "distance_loss": float(distance_loss.detach()),
                    "gromov_loss": float(gromov_loss.detach()),
                }
            )

    def predict(indices: Sequence[int]) -> list[float]:
        values = []
        with torch.no_grad():
            for pair_index in indices:
                pair = pairs[pair_index]
                value = model.distance(
                    indexed_methods[int(pair["left"])],
                    indexed_methods[int(pair["right"])],
                    epsilon=epsilon,
                    iterations=sinkhorn_iterations,
                )
                values.append(float(value))
        return values

    all_indices = list(range(len(pairs)))
    predictions = predict(all_indices)
    targets = [float(value) for value in teacher]
    train_predictions = [predictions[index] for index in train_indices]
    train_targets = [targets[index] for index in train_indices]
    test_predictions = [predictions[index] for index in test_indices]
    test_targets = [targets[index] for index in test_indices]
    lca_weight, start_weight, end_weight = model.factor_weights()

    gromov_products = []
    gromov_targets = []
    with torch.no_grad():
        for method in indexed_methods:
            gromov_products.extend(float(value) for value in model.path_gromov_products(method))
            gromov_targets.extend(float(value) for value in method.lca_depths)
    return {
        "config": {
            "sources": [str(source) for source in sources],
            "pair_json": str(pair_json),
            "teacher_relation": teacher_relation,
            "max_files": max_files,
            "max_methods": max_methods,
            "sample_seed": sample_seed,
            "max_paths_per_method": max_paths_per_method,
            "min_paths_per_method": min_paths_per_method,
            "train_fraction": train_fraction,
            "split_seed": split_seed,
            "model_seed": model_seed,
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "geometry": geometry,
            "curvature": curvature,
            "epsilon": epsilon,
            "sinkhorn_iterations": sinkhorn_iterations,
            "gromov_loss_weight": gromov_loss_weight,
            "use_prefix_encoder": use_prefix_encoder,
            "node_input_mode": node_input_mode,
        },
        "method_count": len(spaces),
        "pair_count": len(pairs),
        "label_count": len(label_to_id),
        "prefix_count": len(prefix_to_id),
        "factor_weights": {
            "lca_weight": float(lca_weight.detach()),
            "start_weight": float(start_weight.detach()),
            "end_weight": float(end_weight.detach()),
            "radial_scale": float(F.softplus(model.radial_logit).detach()),
            "prefix_strength": float(F.softplus(model.prefix_logit).detach()),
        },
        "train": _metrics(train_predictions, train_targets),
        "test": _metrics(test_predictions, test_targets),
        "full": _metrics(predictions, targets),
        "gromov_alignment": {
            "path_count": len(gromov_products),
            "spearman": _spearman(gromov_products, gromov_targets),
        },
        "training_trace": training_trace,
        "claim_boundary": (
            "This pilot learns a shared label/depth Poincare encoder and product factor weights. "
            "It is a small-scale diagnostic for phi_theta feasibility, not a final large-scale "
            "GeoCodePath benchmark."
        ),
    }


def write_markdown_report(result: dict[str, Any], output_path: Path) -> None:
    lines = [
        "# Learned Poincare path-measure student",
        "",
        "The model learns a shared AST-label/depth node encoder and compares methods as path measures.",
        "",
        f"- Methods: `{result['method_count']}`",
        f"- Pairs: `{result['pair_count']}`",
        f"- Labels: `{result['label_count']}`",
        f"- Teacher relation: `{result['config']['teacher_relation']}`",
        f"- Geometry: `{result['config'].get('geometry', 'poincare')}`",
        f"- Node input mode: `{result['config'].get('node_input_mode', 'label_depth_prefix')}`",
        "",
        "## Metrics",
        "",
        "| Split | Spearman | MAE | RMSE |",
        "|---|---:|---:|---:|",
    ]
    for split_name in ("train", "test", "full"):
        row = result[split_name]
        lines.append(f"| {split_name} | {row['spearman']:.6f} | {row['mae']:.6f} | {row['rmse']:.6f} |")
    lines.extend(["", "## Factor weights", "", "| Factor | Value |", "|---|---:|"])
    for key, value in result["factor_weights"].items():
        lines.append(f"| {key} | {value:.6f} |")
    lines.extend(["", "## Claim boundary", "", result["claim_boundary"], ""])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a learned Poincare path-measure student.")
    parser.add_argument("--source", type=Path, action="append", required=True)
    parser.add_argument("--pair-json", type=Path, required=True)
    parser.add_argument("--max-files", type=int, default=20)
    parser.add_argument("--max-methods", type=int, default=32)
    parser.add_argument("--sample-seed", type=int, default=None)
    parser.add_argument("--max-paths-per-method", type=int, default=16)
    parser.add_argument("--min-paths-per-method", type=int, default=4)
    parser.add_argument("--teacher-relation", choices=("lca_depth", "endpoint", "lca_anchored_product", "edge_jaccard", "path_length"), default="lca_depth")
    parser.add_argument("--train-fraction", type=float, default=0.5)
    parser.add_argument("--split-seed", type=int, default=20260623)
    parser.add_argument("--model-seed", type=int, default=20260623)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--geometry", choices=("poincare", "euclidean"), default="poincare")
    parser.add_argument("--curvature", type=float, default=1.0)
    parser.add_argument("--epsilon", type=float, default=0.05)
    parser.add_argument("--sinkhorn-iterations", type=int, default=40)
    parser.add_argument("--gromov-loss-weight", type=float, default=0.0)
    parser.add_argument("--use-prefix-encoder", action="store_true")
    parser.add_argument(
        "--node-input-mode",
        choices=("label_only", "label_depth", "label_depth_prefix"),
        default="label_depth_prefix",
    )
    parser.add_argument("--output", type=Path, default=Path("outputs/learned_poincare_path_student.json"))
    parser.add_argument("--report", type=Path, default=Path("reports/learned_poincare_path_student.md"))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = fit_learned_poincare_path_student(
        tuple(args.source),
        pair_json=args.pair_json,
        max_files=args.max_files,
        max_methods=args.max_methods,
        sample_seed=args.sample_seed,
        max_paths_per_method=args.max_paths_per_method,
        min_paths_per_method=args.min_paths_per_method,
        teacher_relation=args.teacher_relation,
        train_fraction=args.train_fraction,
        split_seed=args.split_seed,
        model_seed=args.model_seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        geometry=args.geometry,
        curvature=args.curvature,
        epsilon=args.epsilon,
        sinkhorn_iterations=args.sinkhorn_iterations,
        gromov_loss_weight=args.gromov_loss_weight,
        use_prefix_encoder=args.use_prefix_encoder,
        node_input_mode=args.node_input_mode,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    write_markdown_report(result, args.report)
    print(
        json.dumps(
            {
                "train": result["train"],
                "test": result["test"],
                "factor_weights": result["factor_weights"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    print(f"wrote {args.output}")
    print(f"wrote {args.report}")


if __name__ == "__main__":
    main()
