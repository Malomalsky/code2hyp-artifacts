from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from geometry_profile_research.constant_curvature import (
    ConstantCurvatureProduct,
    ProductMeasure,
    median_positive_cost_scale,
    scaled_sinkhorn_epsilon,
)
from geometry_profile_research.raw_ast import terminal_to_terminal_paths
from geometry_profile_research.raw_ast_code2hyp import RawASTCode2Hyp, build_raw_ast_token_vocab
from geometry_profile_research.raw_ast_retrieval import (
    PositiveMode,
    build_retrieval_triples,
    make_positive_tree,
)
from scripts.run_raw_ast_code2hyp_retrieval import Language, collect_retrieval_items


DEFAULT_CURVATURES = (0.0, 1e-6, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 4.0)


@dataclass(frozen=True)
class ProbeProject:
    label: str
    source: Path


def run_downstream_probe(
    *,
    projects: Sequence[ProbeProject],
    output_path: Path,
    language: Language = "python",
    curvatures: Sequence[float] = DEFAULT_CURVATURES,
    dim: int = 4,
    epochs: int = 3,
    learning_rate: float = 1e-2,
    max_files: int | None = 128,
    max_methods: int | None = 64,
    max_eval_methods: int | None = None,
    max_paths: int = 16,
    seed: int = 20260625,
    min_structural_gap: float = 0.05,
    sinkhorn_iterations: int = 30,
    sinkhorn_projection_iterations: int = 256,
    kappa: float = 0.05,
    train_scale_methods: int = 16,
    positive_mode: PositiveMode = "alpha_structural_noop",
    side_weight: float = 1.0,
    max_ball_fraction: float = 0.35,
) -> dict[str, Any]:
    """Evaluate one frozen encoder under a constant-curvature continuation.

    The encoder is trained once in tangent/Euclidean coordinates. The resulting
    path measures are then re-scored through the same ``ConstantCurvatureProduct``
    implementation for every curvature value. This isolates the evaluation
    metric/transport layer from optimization-path differences.
    """

    if not projects:
        raise ValueError("at least one project is required")
    torch.manual_seed(seed)
    rows = []
    for project in projects:
        rows.extend(
            _run_project_probe(
                project,
                language=language,
                curvatures=curvatures,
                dim=dim,
                epochs=epochs,
                learning_rate=learning_rate,
                max_files=max_files,
                max_methods=max_methods,
                max_eval_methods=max_eval_methods,
                max_paths=max_paths,
                seed=seed,
                min_structural_gap=min_structural_gap,
                sinkhorn_iterations=sinkhorn_iterations,
                sinkhorn_projection_iterations=sinkhorn_projection_iterations,
                kappa=kappa,
                train_scale_methods=train_scale_methods,
                positive_mode=positive_mode,
                side_weight=side_weight,
                max_ball_fraction=max_ball_fraction,
            )
        )
        _write_payload(output_path, _payload(projects, rows, language=language, curvatures=curvatures, dim=dim, epochs=epochs, seed=seed))
    payload = _payload(projects, rows, language=language, curvatures=curvatures, dim=dim, epochs=epochs, seed=seed)
    _write_payload(output_path, payload)
    return payload


def _run_project_probe(
    project: ProbeProject,
    *,
    language: Language,
    curvatures: Sequence[float],
    dim: int,
    epochs: int,
    learning_rate: float,
    max_files: int | None,
    max_methods: int | None,
    max_eval_methods: int | None,
    max_paths: int,
    seed: int,
    min_structural_gap: float,
    sinkhorn_iterations: int,
    sinkhorn_projection_iterations: int,
    kappa: float,
    train_scale_methods: int,
    positive_mode: PositiveMode,
    side_weight: float,
    max_ball_fraction: float,
) -> list[dict[str, Any]]:
    items = collect_retrieval_items(
        (project.source,),
        language=language,
        max_files=max_files,
        max_methods=max_methods,
        min_paths=2,
        max_paths=max_paths,
    )
    if len(items) < 2:
        raise ValueError(f"project {project.label!r} has fewer than two retrieval items")
    vocab = build_raw_ast_token_vocab(tuple(item.tree for item in items), terminal_policy="class", node_input_mode="label_only")
    model = RawASTCode2Hyp(
        vocab,
        dim=dim,
        manifold="euclidean",
        max_paths=max_paths,
        terminal_policy="class",
        node_input_mode="label_only",
        path_object_mode="lca_product",
        method_aggregation="measure",
        path_cost_orientation="directed",
        curvature=1.0,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    triples = build_retrieval_triples(items, min_structural_gap=min_structural_gap, positive_mode=positive_mode)
    history = []
    for epoch in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        loss = model.training_loss(
            triples,
            sinkhorn_epsilon=0.05,
            sinkhorn_iterations=sinkhorn_iterations,
        )
        loss["loss"].backward()
        optimizer.step()
        history.append({"epoch": epoch + 1, "loss": float(loss["loss"].detach()), "retrieval": float(loss["retrieval"].detach())})

    if max_eval_methods is not None:
        eval_items = items[:max_eval_methods]
    else:
        eval_items = items
    if len(eval_items) < 2:
        raise ValueError("evaluation requires at least two method-level items")

    encoded_items = [_encode_item_measure(model, item.tree, max_paths=max_paths) for item in eval_items]
    point_scale = _point_scale(encoded_items, max_curvature=max(curvatures), max_ball_fraction=max_ball_fraction)
    encoded_items = [_scale_measure(measure, point_scale=point_scale) for measure in encoded_items]
    rows = []
    for curvature in curvatures:
        geometry = ConstantCurvatureProduct(curvature=float(curvature), side_weight=side_weight)
        train_measures = encoded_items[: max(1, min(train_scale_methods, len(encoded_items)))]
        train_costs = [geometry.path_cost_matrix(left, right) for left in train_measures for right in train_measures]
        cost_scale = median_positive_cost_scale(train_costs)
        epsilon = scaled_sinkhorn_epsilon(cost_scale, kappa=kappa)
        metrics = _evaluate_items(
            model,
            eval_items,
            encoded_items,
            geometry,
            epsilon=epsilon,
            sinkhorn_iterations=sinkhorn_iterations,
            sinkhorn_projection_iterations=sinkhorn_projection_iterations,
            max_paths=max_paths,
            positive_mode=positive_mode,
            point_scale=point_scale,
        )
        rows.append(
            {
                "project": project.label,
                "source": str(project.source),
                "curvature": float(curvature),
                "dim": dim,
                "seed": seed,
                "item_count": len(items),
                "eval_item_count": len(eval_items),
                "vocab_size": len(vocab),
                "point_scale": point_scale,
                "cost_scale": cost_scale,
                "epsilon": epsilon,
                "training_history": history,
                **metrics,
            }
        )
    return rows


def _encode_item_measure(model: RawASTCode2Hyp, tree: Any, *, max_paths: int) -> ProductMeasure:
    raw = model.encode_method(tree, paths=terminal_to_terminal_paths(tree, max_paths=max_paths))
    side = torch.cat((raw.left_branch, raw.right_branch), dim=-1)
    return ProductMeasure(points=raw.points.detach(), mass=raw.mass.detach(), side_features=side.detach())


def _scale_measure(measure: ProductMeasure, *, point_scale: float) -> ProductMeasure:
    return ProductMeasure(points=measure.points * point_scale, mass=measure.mass, side_features=measure.side_features)


def _point_scale(measures: Sequence[ProductMeasure], *, max_curvature: float, max_ball_fraction: float) -> float:
    if max_curvature <= 0.0:
        return 1.0
    max_norm = max(float(torch.linalg.vector_norm(measure.points.reshape(-1, measure.points.shape[-1]), dim=-1).max()) for measure in measures)
    if max_norm <= 0.0:
        return 1.0
    allowed = max_ball_fraction / math.sqrt(max_curvature)
    return min(1.0, allowed / max_norm)


def _evaluate_items(
    model: RawASTCode2Hyp,
    items: Sequence[Any],
    encoded_items: Sequence[ProductMeasure],
    geometry: ConstantCurvatureProduct,
    *,
    epsilon: float,
    sinkhorn_iterations: int,
    sinkhorn_projection_iterations: int,
    max_paths: int,
    positive_mode: PositiveMode,
    point_scale: float,
) -> dict[str, Any]:
    ranks = []
    positive_distances = []
    nearest_negative_distances = []
    query_records = []
    with torch.no_grad():
        for anchor_index, anchor in enumerate(items):
            anchor_measure = encoded_items[anchor_index]
            positive_tree = make_positive_tree(anchor.tree, mode=positive_mode)
            positive_measure = _scale_measure(
                _encode_item_measure(model, positive_tree, max_paths=max_paths),
                point_scale=point_scale,
            )
            candidate_measures = [positive_measure] + [
                encoded_items[index] for index, item in enumerate(items) if item.item_id != anchor.item_id
            ]
            candidate_ids = ["__positive__"] + [item.item_id for item in items if item.item_id != anchor.item_id]
            distances = [
                float(
                    geometry.sinkhorn_divergence(
                        anchor_measure,
                        candidate,
                        epsilon=epsilon,
                        iterations=sinkhorn_iterations,
                        projection_iterations=sinkhorn_projection_iterations,
                    ).detach()
                )
                for candidate in candidate_measures
            ]
            ordered = sorted(range(len(distances)), key=lambda index: (distances[index], index))
            rank = ordered.index(0) + 1
            ranks.append(rank)
            positive_distances.append(distances[0])
            nearest_negative_index = min(range(1, len(distances)), key=lambda index: (distances[index], index))
            nearest_negative_distances.append(distances[nearest_negative_index])
            query_records.append(
                {
                    "anchor_id": anchor.item_id,
                    "rank": rank,
                    "candidate_count": len(candidate_measures),
                    "positive_distance": distances[0],
                    "nearest_negative_id": candidate_ids[nearest_negative_index],
                    "nearest_negative_distance": distances[nearest_negative_index],
                    "margin": distances[nearest_negative_index] - distances[0],
                }
            )
    total = len(ranks)
    margins = [negative - positive for positive, negative in zip(positive_distances, nearest_negative_distances)]
    return {
        "recall_at_1": sum(rank <= 1 for rank in ranks) / total,
        "recall_at_3": sum(rank <= 3 for rank in ranks) / total,
        "recall_at_5": sum(rank <= 5 for rank in ranks) / total,
        "mrr": sum(1.0 / rank for rank in ranks) / total,
        "mean_rank": sum(float(rank) for rank in ranks) / total,
        "positive_distance_mean": sum(positive_distances) / total,
        "nearest_negative_distance_mean": sum(nearest_negative_distances) / total,
        "margin_mean": sum(margins) / total,
        "margin_min": min(margins),
        "query_records": query_records,
    }


def _payload(
    projects: Sequence[ProbeProject],
    rows: Sequence[dict[str, Any]],
    *,
    language: Language,
    curvatures: Sequence[float],
    dim: int,
    epochs: int,
    seed: int,
) -> dict[str, Any]:
    return {
        "experiment": "constant_curvature_downstream_probe",
        "status": "complete" if len(rows) == len(projects) * len(curvatures) else "partial",
        "completed_runs": len(rows),
        "expected_runs": len(projects) * len(curvatures),
        "config": {
            "projects": [{"label": project.label, "source": str(project.source)} for project in projects],
            "language": language,
            "curvatures": [float(value) for value in curvatures],
            "dim": dim,
            "epochs": epochs,
            "seed": seed,
            "encoder": "single Euclidean/tangent RawASTCode2Hyp encoder per project; curvature varied only in ConstantCurvatureProduct evaluation",
            "epsilon_policy": "epsilon(c) = kappa * median positive train ground cost at curvature c",
        },
        "runs": list(rows),
    }


def _write_payload(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _parse_projects(values: Sequence[Sequence[str]]) -> tuple[ProbeProject, ...]:
    return tuple(ProbeProject(label=_safe_label(label), source=Path(source)) for label, source in values)


def _safe_label(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip()).strip("._") or "project"


def _parse_csv_floats(value: str) -> tuple[float, ...]:
    return tuple(float(part.strip()) for part in value.split(",") if part.strip())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run frozen-encoder constant-curvature downstream retrieval probe.")
    parser.add_argument("--project", action="append", nargs=2, metavar=("LABEL", "PATH"), required=True)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "outputs/constant_curvature_downstream_probe.json")
    parser.add_argument("--language", choices=("auto", "java", "python"), default="python")
    parser.add_argument("--curvatures", default="0,1e-6,1e-4,1e-3,1e-2,1e-1,1,4")
    parser.add_argument("--dim", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=1e-2)
    parser.add_argument("--max-files", type=int, default=128)
    parser.add_argument("--max-methods", type=int, default=64)
    parser.add_argument("--max-eval-methods", type=int, default=None)
    parser.add_argument("--max-paths", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260625)
    parser.add_argument("--min-structural-gap", type=float, default=0.05)
    parser.add_argument("--sinkhorn-iterations", type=int, default=30)
    parser.add_argument("--sinkhorn-projection-iterations", type=int, default=256)
    parser.add_argument("--kappa", type=float, default=0.05)
    parser.add_argument("--train-scale-methods", type=int, default=16)
    parser.add_argument("--positive-mode", choices=("alpha_rename", "structural_noop", "alpha_structural_noop"), default="alpha_structural_noop")
    parser.add_argument("--side-weight", type=float, default=1.0)
    parser.add_argument("--max-ball-fraction", type=float, default=0.35)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    payload = run_downstream_probe(
        projects=_parse_projects(args.project),
        output_path=args.output,
        language=args.language,
        curvatures=_parse_csv_floats(args.curvatures),
        dim=args.dim,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        max_files=args.max_files,
        max_methods=args.max_methods,
        max_eval_methods=args.max_eval_methods,
        max_paths=args.max_paths,
        seed=args.seed,
        min_structural_gap=args.min_structural_gap,
        sinkhorn_iterations=args.sinkhorn_iterations,
        sinkhorn_projection_iterations=args.sinkhorn_projection_iterations,
        kappa=args.kappa,
        train_scale_methods=args.train_scale_methods,
        positive_mode=args.positive_mode,
        side_weight=args.side_weight,
        max_ball_fraction=args.max_ball_fraction,
    )
    print(f"status={payload['status']} completed={payload['completed_runs']}/{payload['expected_runs']} output={args.output}")


if __name__ == "__main__":
    main()
