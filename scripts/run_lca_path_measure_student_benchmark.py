from __future__ import annotations

import argparse
import json
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
    _empty_distance_matrix,
    _feature_cost,
    _retrieval_overlap,
    _spearman,
    _structural_space_for_relation,
    _summarize,
    _zscore_rows,
    collect_raw_ast_method_spaces,
)


def _teacher_relation_name(name: str) -> str:
    if name == "lca_depth":
        return "lca_depth"
    if name == "endpoint":
        return "endpoint"
    if name == "lca_anchored_product":
        return "lca_anchored_product"
    if name == "edge_jaccard":
        return "edge_jaccard"
    if name == "path_length":
        return "path_length"
    raise ValueError(f"unsupported teacher relation: {name!r}")


def _method_summary(space: Any) -> dict[str, Any]:
    return {
        "source_path": space.source_path,
        "scope_node": space.scope_node,
        "scope_name": space.scope_name,
        "scope_label": space.scope_label,
        "node_count": space.node_count,
        "leaf_count": space.leaf_count,
        "path_count": space.path_count,
    }


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


def run_lca_path_measure_student_benchmark(
    sources: Sequence[Path],
    *,
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
    angle_mode: Literal["branch_sector", "label_hash", "path_signature_hash", "depth_only"] = "branch_sector",
    epsilon: float = 0.05,
    gw_iterations: int = 8,
    sinkhorn_iterations: int = 80,
) -> dict[str, Any]:
    """Compare product-hyperbolic path-measure students with raw-AST FGW teacher."""

    teacher_relation = _teacher_relation_name(teacher_relation)
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
        raise ValueError("student benchmark requires at least two eligible methods")

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
    matrices = {
        "teacher_fgw": [[0.0 for _ in spaces] for _ in spaces],
        "lca_product_sinkhorn": [[0.0 for _ in spaces] for _ in spaces],
        "endpoint_product_sinkhorn": [[0.0 for _ in spaces] for _ in spaces],
        "lca_only_sinkhorn": [[0.0 for _ in spaces] for _ in spaces],
        "feature_ot": [[0.0 for _ in spaces] for _ in spaces],
        "centroid": [[0.0 for _ in spaces] for _ in spaces],
    }
    pair_indices = list(combinations(range(len(spaces)), 2))
    total_pair_count = len(pair_indices)
    if pair_limit is not None:
        pair_indices = pair_indices[:pair_limit]

    records: list[dict[str, Any]] = []
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
        lca_product = float(
            torch.clamp(
                sinkhorn_path_measure_distance(
                    left_measure,
                    right_measure,
                    epsilon=epsilon,
                    iterations=sinkhorn_iterations,
                    lca_weight=1.0,
                    start_weight=1.0,
                    end_weight=1.0,
                    unoriented=True,
                ),
                min=0.0,
            )
        )
        endpoint_product = float(
            torch.clamp(
                sinkhorn_path_measure_distance(
                    left_measure,
                    right_measure,
                    epsilon=epsilon,
                    iterations=sinkhorn_iterations,
                    lca_weight=0.0,
                    start_weight=1.0,
                    end_weight=1.0,
                    unoriented=True,
                ),
                min=0.0,
            )
        )
        lca_only = float(
            torch.clamp(
                sinkhorn_path_measure_distance(
                    left_measure,
                    right_measure,
                    epsilon=epsilon,
                    iterations=sinkhorn_iterations,
                    lca_weight=1.0,
                    start_weight=0.0,
                    end_weight=0.0,
                    unoriented=True,
                ),
                min=0.0,
            )
        )
        centroid_distance = float(torch.linalg.norm(centroid[left_index] - centroid[right_index]))
        distances = {
            "teacher_fgw": teacher,
            "lca_product_sinkhorn": lca_product,
            "endpoint_product_sinkhorn": endpoint_product,
            "lca_only_sinkhorn": lca_only,
            "feature_ot": feature_ot,
            "centroid": centroid_distance,
        }
        for name, value in distances.items():
            matrices[name][left_index][right_index] = value
            matrices[name][right_index][left_index] = value
        records.append({"left": left_index, "right": right_index, **distances})

    values = {name: [record[name] for record in records] for name in matrices}
    complete_pair_matrix = len(records) == total_pair_count
    overlap_input = {"fgw": matrices["teacher_fgw"], **{name: matrix for name, matrix in matrices.items() if name != "teacher_fgw"}}
    top1 = _retrieval_overlap(overlap_input, k=1) if complete_pair_matrix else None
    top3 = _retrieval_overlap(overlap_input, k=3) if complete_pair_matrix else None
    if top1:
        top1.pop("fgw", None)
    if top3:
        top3.pop("fgw", None)
    return {
        "config": {
            "sources": [str(source) for source in sources],
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
        },
        "method_count": len(spaces),
        "pair_count": len(records),
        "total_pair_count": total_pair_count,
        "complete_pair_matrix": complete_pair_matrix,
        "methods": [_method_summary(space) for space in spaces],
        "distance_summary": {name: _distance_summary(metric_values) for name, metric_values in values.items()},
        "spearman_against_teacher": {
            name: _spearman(metric_values, values["teacher_fgw"])
            for name, metric_values in values.items()
            if name != "teacher_fgw"
        },
        "retrieval_overlap_at_1": top1,
        "retrieval_overlap_at_3": top3,
        "pairs": records,
        "claim_boundary": {
            "teacher": "raw-AST FGW",
            "student": "deterministic product-hyperbolic Sinkhorn distance",
            "scope": (
                "This benchmark evaluates a scalable geometry student against a raw-AST FGW target; "
                "it is not a downstream method-name prediction result."
            ),
        },
    }


def write_markdown_report(result: dict[str, Any], output_path: Path) -> None:
    lines = [
        "# LCA-anchored path-measure student benchmark",
        "",
        "The teacher is raw-AST FGW. Student distances compare methods as measures over LCA-anchored product-hyperbolic path objects.",
        "",
        f"- Methods: `{result['method_count']}`",
        f"- Pairs: `{result['pair_count']}` of `{result['total_pair_count']}`",
        f"- Teacher relation: `{result['config']['teacher_relation']}`",
        "",
        "## Agreement with teacher",
        "",
        "| Student/control | Spearman vs teacher | Top-1 overlap | Top-3 overlap |",
        "|---|---:|---:|---:|",
    ]
    top1 = result.get("retrieval_overlap_at_1") or {}
    top3 = result.get("retrieval_overlap_at_3") or {}
    for name, value in result["spearman_against_teacher"].items():
        top1_text = f"{top1[name]:.6f}" if name in top1 else "n/a"
        top3_text = f"{top3[name]:.6f}" if name in top3 else "n/a"
        lines.append(f"| {name} | {value:.6f} | {top1_text} | {top3_text} |")
    lines.extend(
        [
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
    parser = argparse.ArgumentParser(description="Run LCA-anchored path-measure student benchmark.")
    parser.add_argument("--source", type=Path, action="append", required=True)
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
    parser.add_argument("--angle-mode", choices=("branch_sector", "label_hash", "path_signature_hash", "depth_only"), default="branch_sector")
    parser.add_argument("--epsilon", type=float, default=0.05)
    parser.add_argument("--gw-iterations", type=int, default=8)
    parser.add_argument("--sinkhorn-iterations", type=int, default=80)
    parser.add_argument("--output", type=Path, default=Path("outputs/lca_path_measure_student_benchmark.json"))
    parser.add_argument("--report", type=Path, default=Path("reports/lca_path_measure_student_benchmark.md"))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = run_lca_path_measure_student_benchmark(
        tuple(args.source),
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
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    write_markdown_report(result, args.report)
    print(
        json.dumps(
            {
                "method_count": result["method_count"],
                "pair_count": result["pair_count"],
                "spearman_against_teacher": result["spearman_against_teacher"],
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
