from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from geometry_profile_research.raw_ast_axiom_probe import load_axiom_probe_dataset, run_axiom_probe


def _parse_ints(value: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def _parse_strings(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def _summaries(results: list[dict[str, int | float]]) -> list[dict[str, int | float]]:
    grouped: dict[tuple[str, int], list[dict[str, int | float]]] = defaultdict(list)
    for result in results:
        grouped[(str(result["geometry"]), int(result["dim"]))].append(result)
    rows: list[dict[str, int | float]] = []
    for (geometry, dim), items in sorted(grouped.items()):
        length_spearman = [float(item["eval_length_spearman"]) for item in items]
        lca_spearman = [float(item["eval_lca_depth_spearman"]) for item in items]
        length_stress = [float(item["eval_length_stress"]) for item in items]
        lca_stress = [float(item["eval_lca_depth_stress"]) for item in items]
        lca_radial = [float(item["eval_lca_radial_depth_spearman"]) for item in items]
        additivity = [float(item["eval_additivity_residual_mean"]) for item in items]
        rows.append(
            {
                "geometry": geometry,
                "dim": dim,
                "seeds": len(items),
                "eval_length_spearman_mean": mean(length_spearman),
                "eval_length_spearman_sd": stdev(length_spearman) if len(length_spearman) > 1 else 0.0,
                "eval_lca_depth_spearman_mean": mean(lca_spearman),
                "eval_lca_depth_spearman_sd": stdev(lca_spearman) if len(lca_spearman) > 1 else 0.0,
                "eval_lca_radial_depth_spearman_mean": mean(lca_radial),
                "eval_lca_radial_depth_spearman_sd": stdev(lca_radial) if len(lca_radial) > 1 else 0.0,
                "eval_length_stress_mean": mean(length_stress),
                "eval_lca_depth_stress_mean": mean(lca_stress),
                "eval_additivity_residual_mean": mean(additivity),
            }
        )
    return rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run a raw-AST node-embedding axiom probe: direct edge lengths are "
            "trained, held-out AST paths test length, LCA/Gromov and additivity."
        ),
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("outputs/java_raw_ast_order_relations_validation_callable_sample_100.jsonl"),
    )
    parser.add_argument("--output", type=Path, default=Path("outputs/raw_ast_axiom_probe_results.json"))
    parser.add_argument("--dims", type=str, default="2,4,8")
    parser.add_argument("--seeds", type=str, default="101,202,303")
    parser.add_argument("--geometries", type=str, default="poincare,euclidean")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--curvature", type=float, default=1.0)
    parser.add_argument("--edge-weight", type=float, default=1.0)
    parser.add_argument("--length-weight", type=float, default=1.0)
    parser.add_argument("--lca-weight", type=float, default=0.5)
    parser.add_argument("--depth-weight", type=float, default=0.5)
    parser.add_argument("--additivity-weight", type=float, default=0.5)
    parser.add_argument("--max-scopes", type=int, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    dims = _parse_ints(args.dims)
    seeds = _parse_ints(args.seeds)
    geometries = _parse_strings(args.geometries)
    dataset = load_axiom_probe_dataset(args.input, max_scopes=args.max_scopes)
    results: list[dict[str, int | float]] = []
    for geometry in geometries:
        for dim in dims:
            for seed in seeds:
                result = run_axiom_probe(
                    dataset,
                    geometry=geometry,  # type: ignore[arg-type]
                    dim=dim,
                    seed=seed,
                    epochs=args.epochs,
                    learning_rate=args.learning_rate,
                    curvature=args.curvature,
                    edge_weight=args.edge_weight,
                    length_weight=args.length_weight,
                    lca_weight=args.lca_weight,
                    depth_weight=args.depth_weight,
                    additivity_weight=args.additivity_weight,
                )
                results.append(result.as_dict())
                print(
                    f"{geometry} dim={dim} seed={seed}: "
                    f"rho_len={result.eval_length_spearman:.4f}, "
                    f"rho_lca={result.eval_lca_depth_spearman:.4f}, "
                    f"stress_len={result.eval_length_stress:.4f}"
                )

    payload = {
        "input": str(args.input),
        "node_count": dataset.node_count,
        "train_edge_count": int(dataset.train_edge_left.numel()),
        "train_path_count": int(dataset.train_path_start.numel()),
        "eval_path_count": int(dataset.eval_path_start.numel()),
        "config": {
            "dims": list(dims),
            "seeds": list(seeds),
            "geometries": list(geometries),
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "curvature": args.curvature,
            "edge_weight": args.edge_weight,
            "length_weight": args.length_weight,
            "lca_weight": args.lca_weight,
            "depth_weight": args.depth_weight,
            "additivity_weight": args.additivity_weight,
            "max_scopes": args.max_scopes,
        },
        "runs": results,
        "summary": _summaries(results),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
