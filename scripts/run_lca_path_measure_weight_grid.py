from __future__ import annotations

import argparse
import json
import random
import sys
from itertools import combinations
from pathlib import Path
from statistics import mean, median
from typing import Any, Literal, Sequence

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from geometry_profile_research.gromov_wasserstein import (  # noqa: E402
    entropic_fused_gromov_wasserstein,
    sinkhorn_divergence,
)
from geometry_profile_research.lca_path_measure import (  # noqa: E402
    poincare_node_embeddings_from_tree,
    poincare_path_measure_from_paths,
    sinkhorn_path_measure_distance,
)
from scripts.run_raw_ast_fgw_benchmark import (  # noqa: E402
    _feature_cost,
    _retrieval_overlap,
    _spearman,
    _structural_space_for_relation,
    _zscore_rows,
    collect_raw_ast_method_spaces,
)


WeightTriple = tuple[float, float, float]


def _validate_weight_grid(weight_grid: Sequence[WeightTriple]) -> tuple[WeightTriple, ...]:
    if not weight_grid:
        raise ValueError("weight_grid must contain at least one weight triple")
    validated = []
    for triple in weight_grid:
        if len(triple) != 3:
            raise ValueError("each weight triple must be (lca_weight, start_weight, end_weight)")
        lca_weight, start_weight, end_weight = (float(value) for value in triple)
        if min(lca_weight, start_weight, end_weight) < 0.0:
            raise ValueError("path-object weights must be non-negative")
        if lca_weight + start_weight + end_weight <= 0.0:
            raise ValueError("at least one path-object weight must be positive")
        validated.append((lca_weight, start_weight, end_weight))
    return tuple(validated)


def _distance_summary(values: Sequence[float]) -> dict[str, float]:
    if not values:
        return {"count": 0.0, "mean": 0.0, "median": 0.0, "min": 0.0, "max": 0.0}
    return {
        "count": float(len(values)),
        "mean": float(mean(values)),
        "median": float(median(values)),
        "min": float(min(values)),
        "max": float(max(values)),
    }


def _parse_weight_spec(spec: str) -> WeightTriple:
    parts = [part.strip() for part in spec.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("weight must have format lca,start,end")
    try:
        return (float(parts[0]), float(parts[1]), float(parts[2]))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("weight values must be floats") from exc


def _default_weight_grid() -> tuple[WeightTriple, ...]:
    return (
        (1.0, 0.0, 0.0),
        (1.0, 0.25, 0.25),
        (1.0, 0.5, 0.5),
        (1.0, 1.0, 1.0),
        (2.0, 0.5, 0.5),
        (4.0, 0.5, 0.5),
        (0.5, 1.0, 1.0),
        (0.0, 1.0, 1.0),
    )


def run_lca_path_measure_weight_grid(
    sources: Sequence[Path],
    *,
    weight_grid: Sequence[WeightTriple] = _default_weight_grid(),
    max_files: int | None = 20,
    max_methods: int | None = 32,
    sample_seed: int | None = None,
    max_paths_per_method: int = 16,
    min_paths_per_method: int = 4,
    teacher_relation: str = "lca_depth",
    pair_limit: int | None = None,
    alpha: float = 0.75,
    curvature: float = 1.0,
    radial_scale: float = 0.35,
    angle_mode: Literal["branch_sector", "label_hash", "path_signature_hash", "depth_only"] = "label_hash",
    epsilon: float = 0.05,
    gw_iterations: int = 8,
    sinkhorn_iterations: int = 80,
    train_fraction: float | None = None,
    split_seed: int = 0,
) -> dict[str, Any]:
    """Grid-search relation-specific weights for product-hyperbolic path measures."""

    weight_grid = _validate_weight_grid(weight_grid)
    if train_fraction is not None and not (0.0 < train_fraction < 1.0):
        raise ValueError("train_fraction must be between 0 and 1")
    spaces = collect_raw_ast_method_spaces(
        sources,
        max_files=max_files,
        max_methods=max_methods,
        max_paths_per_method=max_paths_per_method,
        min_paths_per_method=min_paths_per_method,
        structural_relation=teacher_relation,
        sample_seed=sample_seed,
    )
    if len(spaces) < 2:
        raise ValueError("weight-grid benchmark requires at least two eligible methods")

    path_measures = []
    for space in spaces:
        node_embeddings = poincare_node_embeddings_from_tree(
            space.tree,
            curvature=curvature,
            radial_scale=radial_scale,
            angle_mode=angle_mode,
        )
        path_measures.append(
            poincare_path_measure_from_paths(
                space.tree,
                space.paths,
                node_embeddings=node_embeddings,
                curvature=curvature,
            )
        )

    centroid = _zscore_rows(torch.stack([space.centroid_features for space in spaces]))
    pair_indices = list(combinations(range(len(spaces)), 2))
    total_pair_count = len(pair_indices)
    if pair_limit is not None:
        pair_indices = pair_indices[:pair_limit]
    complete_pair_matrix = len(pair_indices) == total_pair_count

    teacher_values = []
    baseline_values = {"feature_ot": [], "centroid": []}
    weighted_values: dict[WeightTriple, list[float]] = {triple: [] for triple in weight_grid}
    teacher_matrix = [[0.0 for _ in spaces] for _ in spaces]
    weighted_matrices: dict[WeightTriple, list[list[float]]] = {
        triple: [[0.0 for _ in spaces] for _ in spaces] for triple in weight_grid
    }

    for left_index, right_index in pair_indices:
        left = spaces[left_index]
        right = spaces[right_index]
        left_measure = path_measures[left_index]
        right_measure = path_measures[right_index]
        left_teacher_space = _structural_space_for_relation(left.tree, left.paths, teacher_relation)
        right_teacher_space = _structural_space_for_relation(right.tree, right.paths, teacher_relation)
        feature_cost = _feature_cost(left.path_tokens, right.path_tokens)
        teacher = float(
            entropic_fused_gromov_wasserstein(
                left_teacher_space,
                right_teacher_space,
                feature_cost,
                alpha=alpha,
                epsilon=epsilon,
                iterations=gw_iterations,
                sinkhorn_iterations=sinkhorn_iterations,
            ).objective
        )
        feature_ot = float(
            torch.clamp(
                sinkhorn_divergence(
                    feature_cost,
                    left.endpoint_space.mass,
                    right.endpoint_space.mass,
                    left_self_cost=left.feature_self_cost,
                    right_self_cost=right.feature_self_cost,
                    epsilon=epsilon,
                    iterations=sinkhorn_iterations,
                ),
                min=0.0,
            )
        )
        centroid_distance = float(torch.linalg.norm(centroid[left_index] - centroid[right_index]))
        teacher_values.append(teacher)
        baseline_values["feature_ot"].append(feature_ot)
        baseline_values["centroid"].append(centroid_distance)
        teacher_matrix[left_index][right_index] = teacher
        teacher_matrix[right_index][left_index] = teacher

        for triple in weight_grid:
            lca_weight, start_weight, end_weight = triple
            value = float(
                torch.clamp(
                    sinkhorn_path_measure_distance(
                        left_measure,
                        right_measure,
                        epsilon=epsilon,
                        iterations=sinkhorn_iterations,
                        lca_weight=lca_weight,
                        start_weight=start_weight,
                        end_weight=end_weight,
                        unoriented=True,
                    ),
                    min=0.0,
                )
            )
            weighted_values[triple].append(value)
            weighted_matrices[triple][left_index][right_index] = value
            weighted_matrices[triple][right_index][left_index] = value

    top1_by_weight: dict[WeightTriple, float | None] = {}
    top3_by_weight: dict[WeightTriple, float | None] = {}
    if complete_pair_matrix:
        matrix_pack = {"fgw": teacher_matrix}
        matrix_pack.update({str(triple): matrix for triple, matrix in weighted_matrices.items()})
        top1 = _retrieval_overlap(matrix_pack, k=1)
        top3 = _retrieval_overlap(matrix_pack, k=3)
        for triple in weight_grid:
            top1_by_weight[triple] = top1.get(str(triple))
            top3_by_weight[triple] = top3.get(str(triple))
    else:
        for triple in weight_grid:
            top1_by_weight[triple] = None
            top3_by_weight[triple] = None

    weight_results = []
    train_indices: list[int] | None = None
    test_indices: list[int] | None = None
    if train_fraction is not None and len(pair_indices) >= 3:
        shuffled = list(range(len(pair_indices)))
        random.Random(split_seed).shuffle(shuffled)
        split_at = int(round(len(shuffled) * train_fraction))
        split_at = max(1, min(len(shuffled) - 1, split_at))
        train_indices = shuffled[:split_at]
        test_indices = shuffled[split_at:]

    for triple in weight_grid:
        lca_weight, start_weight, end_weight = triple
        distances = weighted_values[triple]
        row = {
            "lca_weight": lca_weight,
            "start_weight": start_weight,
            "end_weight": end_weight,
            "spearman_against_teacher": _spearman(distances, teacher_values),
            "top1_overlap": top1_by_weight[triple],
            "top3_overlap": top3_by_weight[triple],
            "distance_summary": _distance_summary(distances),
        }
        if train_indices is not None and test_indices is not None:
            train_distances = [distances[index] for index in train_indices]
            train_teacher = [teacher_values[index] for index in train_indices]
            test_distances = [distances[index] for index in test_indices]
            test_teacher = [teacher_values[index] for index in test_indices]
            row["train_spearman"] = _spearman(train_distances, train_teacher)
            row["test_spearman"] = _spearman(test_distances, test_teacher)
        weight_results.append(row)
    best_by_spearman = max(weight_results, key=lambda item: item["spearman_against_teacher"])
    heldout_selection = None
    if train_indices is not None and test_indices is not None:
        selected = max(weight_results, key=lambda item: item["train_spearman"])
        heldout_selection = {
            "train_fraction": train_fraction,
            "split_seed": split_seed,
            "train_pair_count": len(train_indices),
            "test_pair_count": len(test_indices),
            "selected_by_train": {
                "lca_weight": selected["lca_weight"],
                "start_weight": selected["start_weight"],
                "end_weight": selected["end_weight"],
                "train_spearman": selected["train_spearman"],
                "test_spearman": selected["test_spearman"],
                "full_spearman": selected["spearman_against_teacher"],
            },
        }

    return {
        "config": {
            "sources": [str(source) for source in sources],
            "weight_grid": [
                {"lca_weight": a, "start_weight": b, "end_weight": c} for a, b, c in weight_grid
            ],
            "max_files": max_files,
            "max_methods": max_methods,
            "sample_seed": sample_seed,
            "max_paths_per_method": max_paths_per_method,
            "min_paths_per_method": min_paths_per_method,
            "teacher_relation": teacher_relation,
            "pair_limit": pair_limit,
            "alpha": alpha,
            "curvature": curvature,
            "radial_scale": radial_scale,
            "angle_mode": angle_mode,
            "epsilon": epsilon,
            "gw_iterations": gw_iterations,
            "sinkhorn_iterations": sinkhorn_iterations,
            "train_fraction": train_fraction,
            "split_seed": split_seed,
        },
        "method_count": len(spaces),
        "pair_count": len(pair_indices),
        "total_pair_count": total_pair_count,
        "complete_pair_matrix": complete_pair_matrix,
        "teacher_summary": _distance_summary(teacher_values),
        "baseline_spearman": {
            name: _spearman(values, teacher_values) for name, values in baseline_values.items()
        },
        "weight_results": weight_results,
        "best_by_spearman": best_by_spearman,
        "heldout_selection": heldout_selection,
        "claim_boundary": {
            "teacher": "raw-AST FGW",
            "student": "deterministic product-hyperbolic path measure with relation-specific weights",
            "scope": (
                "This grid search tests whether factor weights in the LCA/start/end product ground "
                "cost can approximate a raw-AST FGW teacher. It is a geometry diagnostic, not a "
                "downstream code retrieval benchmark."
            ),
        },
    }


def write_markdown_report(result: dict[str, Any], output_path: Path) -> None:
    lines = [
        "# LCA-anchored path-measure weight-grid benchmark",
        "",
        "The teacher is raw-AST FGW. The student is a Sinkhorn distance over product-hyperbolic path objects with varied LCA/start/end weights.",
        "",
        f"- Methods: `{result['method_count']}`",
        f"- Pairs: `{result['pair_count']}` of `{result['total_pair_count']}`",
        f"- Teacher relation: `{result['config']['teacher_relation']}`",
        f"- Angle mode: `{result['config']['angle_mode']}`",
        "",
        "## Baselines",
        "",
        "| Baseline | Spearman vs teacher |",
        "|---|---:|",
    ]
    for name, value in result["baseline_spearman"].items():
        lines.append(f"| {name} | {value:.6f} |")
    lines.extend(
        [
            "",
            "## Product-weight grid",
            "",
            "| lca | start | end | Spearman vs teacher | Top-1 overlap | Top-3 overlap |",
            "|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in sorted(result["weight_results"], key=lambda item: item["spearman_against_teacher"], reverse=True):
        top1 = "n/a" if row["top1_overlap"] is None else f"{row['top1_overlap']:.6f}"
        top3 = "n/a" if row["top3_overlap"] is None else f"{row['top3_overlap']:.6f}"
        lines.append(
            "| "
            f"{row['lca_weight']:.3g} | {row['start_weight']:.3g} | {row['end_weight']:.3g} | "
            f"{row['spearman_against_teacher']:.6f} | {top1} | {top3} |"
        )
    if result.get("heldout_selection"):
        selected = result["heldout_selection"]["selected_by_train"]
        lines.extend(
            [
                "",
                "## Hold-out selection",
                "",
                f"- Train pairs: `{result['heldout_selection']['train_pair_count']}`",
                f"- Test pairs: `{result['heldout_selection']['test_pair_count']}`",
                f"- Selected weights: `lca={selected['lca_weight']}`, `start={selected['start_weight']}`, `end={selected['end_weight']}`",
                f"- Train Spearman: `{selected['train_spearman']:.6f}`",
                f"- Test Spearman: `{selected['test_spearman']:.6f}`",
                "",
            ]
        )
    best = result["best_by_spearman"]
    lines.extend(
        [
            "",
            "## Best weight triple",
            "",
            f"`lca={best['lca_weight']}`, `start={best['start_weight']}`, `end={best['end_weight']}`, Spearman `{best['spearman_against_teacher']:.6f}`.",
            "",
            "## Claim boundary",
            "",
            result["claim_boundary"]["scope"],
            "",
        ]
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Grid-search product-hyperbolic path-measure weights.")
    parser.add_argument("--source", type=Path, action="append", required=True)
    parser.add_argument("--weight", type=_parse_weight_spec, action="append", default=None)
    parser.add_argument("--max-files", type=int, default=20)
    parser.add_argument("--max-methods", type=int, default=32)
    parser.add_argument("--sample-seed", type=int, default=None)
    parser.add_argument("--max-paths-per-method", type=int, default=16)
    parser.add_argument("--min-paths-per-method", type=int, default=4)
    parser.add_argument("--teacher-relation", choices=("lca_depth", "endpoint", "lca_anchored_product", "edge_jaccard", "path_length"), default="lca_depth")
    parser.add_argument("--pair-limit", type=int, default=None)
    parser.add_argument("--alpha", type=float, default=0.75)
    parser.add_argument("--curvature", type=float, default=1.0)
    parser.add_argument("--radial-scale", type=float, default=0.35)
    parser.add_argument("--angle-mode", choices=("branch_sector", "label_hash", "path_signature_hash", "depth_only"), default="label_hash")
    parser.add_argument("--epsilon", type=float, default=0.05)
    parser.add_argument("--gw-iterations", type=int, default=8)
    parser.add_argument("--sinkhorn-iterations", type=int, default=80)
    parser.add_argument("--train-fraction", type=float, default=None)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("outputs/lca_path_measure_weight_grid.json"))
    parser.add_argument("--report", type=Path, default=Path("reports/lca_path_measure_weight_grid.md"))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = run_lca_path_measure_weight_grid(
        tuple(args.source),
        weight_grid=tuple(args.weight) if args.weight else _default_weight_grid(),
        max_files=args.max_files,
        max_methods=args.max_methods,
        sample_seed=args.sample_seed,
        max_paths_per_method=args.max_paths_per_method,
        min_paths_per_method=args.min_paths_per_method,
        teacher_relation=args.teacher_relation,
        pair_limit=args.pair_limit,
        alpha=args.alpha,
        curvature=args.curvature,
        radial_scale=args.radial_scale,
        angle_mode=args.angle_mode,
        epsilon=args.epsilon,
        gw_iterations=args.gw_iterations,
        sinkhorn_iterations=args.sinkhorn_iterations,
        train_fraction=args.train_fraction,
        split_seed=args.split_seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    write_markdown_report(result, args.report)
    print(
        json.dumps(
            {
                "method_count": result["method_count"],
                "pair_count": result["pair_count"],
                "baseline_spearman": result["baseline_spearman"],
                "best_by_spearman": result["best_by_spearman"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    print(f"wrote {args.output}")
    print(f"wrote {args.report}")


if __name__ == "__main__":
    main()
