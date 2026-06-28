from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def summarize_factor_query_deltas(
    input_path: Path,
    *,
    bootstrap_samples: int = 1000,
    seed: int = 20260624,
) -> dict[str, Any]:
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    runs = [dict(run) for run in payload.get("runs", [])]
    for run in runs:
        run["query_records"] = _load_query_records(run)
    path_object_deltas = _path_object_deltas(runs, bootstrap_samples=bootstrap_samples, seed=seed)
    aggregation_deltas = _aggregation_deltas(runs, bootstrap_samples=bootstrap_samples, seed=seed)
    curvature_deltas = _curvature_deltas(runs, bootstrap_samples=bootstrap_samples, seed=seed)
    orientation_deltas = _orientation_deltas(runs, bootstrap_samples=bootstrap_samples, seed=seed)
    full_model_deltas = _full_model_deltas(runs, bootstrap_samples=bootstrap_samples, seed=seed)
    return {
        "input": str(input_path),
        "bootstrap_samples": bootstrap_samples,
        "seed": seed,
        "path_object_deltas_aggregated": _aggregate_rows(path_object_deltas, bootstrap_samples=bootstrap_samples, seed=seed),
        "aggregation_deltas_aggregated": _aggregate_rows(aggregation_deltas, bootstrap_samples=bootstrap_samples, seed=seed),
        "curvature_deltas_aggregated": _aggregate_rows(curvature_deltas, bootstrap_samples=bootstrap_samples, seed=seed),
        "orientation_deltas_aggregated": _aggregate_rows(orientation_deltas, bootstrap_samples=bootstrap_samples, seed=seed),
        "full_model_deltas_aggregated": _aggregate_rows(full_model_deltas, bootstrap_samples=bootstrap_samples, seed=seed),
        "path_object_deltas": path_object_deltas,
        "aggregation_deltas": aggregation_deltas,
        "curvature_deltas": curvature_deltas,
        "orientation_deltas": orientation_deltas,
        "full_model_deltas": full_model_deltas,
    }


def format_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Raw-AST Code2Hyp query-level paired deltas",
        "",
        f"Input: `{summary['input']}`",
        "",
        f"Bootstrap samples: `{summary['bootstrap_samples']}`.",
    ]
    _append_table(lines, "Aggregated query-level path-object deltas", summary["path_object_deltas_aggregated"], show_seed_count=True)
    _append_table(lines, "Aggregated query-level aggregation deltas", summary["aggregation_deltas_aggregated"], show_seed_count=True)
    _append_table(lines, "Aggregated query-level curvature deltas", summary["curvature_deltas_aggregated"], show_seed_count=True)
    _append_table(lines, "Aggregated query-level orientation deltas", summary["orientation_deltas_aggregated"], show_seed_count=True)
    _append_table(lines, "Aggregated query-level full-model deltas", summary["full_model_deltas_aggregated"], show_seed_count=True)
    _append_table(lines, "Seed-level query-level path-object deltas", summary["path_object_deltas"])
    _append_table(lines, "Seed-level query-level aggregation deltas", summary["aggregation_deltas"])
    _append_table(lines, "Seed-level query-level curvature deltas", summary["curvature_deltas"])
    _append_table(lines, "Seed-level query-level orientation deltas", summary["orientation_deltas"])
    _append_table(lines, "Seed-level query-level full-model deltas", summary["full_model_deltas"])
    return "\n".join(lines) + "\n"


def _append_table(lines: list[str], title: str, rows: Sequence[dict[str, Any]], *, show_seed_count: bool = False) -> None:
    seed_column = " | Seeds" if show_seed_count else ""
    seed_rule = "|---:" if show_seed_count else ""
    lines.extend(
        [
            "",
            f"## {title}",
            "",
            f"| Project | Node input | Cell | Contrast{seed_column} | n queries | Delta RR | 95% CI | Positive queries | Delta margin |",
            f"|---|---|---|---{seed_rule}|---:|---:|---:|---:|---:|",
        ]
    )
    if not rows:
        empty_seed = " | 0" if show_seed_count else ""
        lines.append(f"| - | - | - | -{empty_seed} | 0 | - | - | - | - |")
        return
    for row in rows:
        seed_value = f" | {row.get('n_seeds', 1)}" if show_seed_count else ""
        template = (
            "| {project} | {node_input_mode} | {cell} | {contrast}"
            + seed_value
            + " | {n_queries} | {mean_delta_reciprocal_rank:+.4f} | "
            "[{ci_low_delta_reciprocal_rank:+.4f}, {ci_high_delta_reciprocal_rank:+.4f}] | "
            "{positive_queries}/{n_queries} | {mean_delta_margin:+.4f} |"
        )
        lines.append(template.format(**row))


def _load_query_records(run: dict[str, Any]) -> list[dict[str, Any]]:
    if run.get("query_records"):
        return list(run["query_records"])
    output_path = run.get("output_path")
    if not output_path:
        return []
    path = Path(str(output_path))
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    return list(payload.get("query_records", []))


def _path_object_deltas(runs: Sequence[dict[str, Any]], *, bootstrap_samples: int, seed: int) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], dict[str, dict[str, Any]]] = defaultdict(dict)
    for run in runs:
        key = (
            run["project"],
            run["node_input_mode"],
            run["method_aggregation"],
            _orientation(run),
            run["geometry"],
            float(run["curvature"]),
            int(run["dim"]),
            int(run["seed"]),
        )
        grouped[key][run["path_object_mode"]] = run
    rows = []
    for key, variants in sorted(grouped.items()):
        single = variants.get("single_point")
        lca = variants.get("lca_product")
        if single is None or lca is None:
            continue
        rows.append(_paired_query_row(key, "LCA-product - single point", lca, single, bootstrap_samples=bootstrap_samples, seed=seed))
    return rows


def _aggregation_deltas(runs: Sequence[dict[str, Any]], *, bootstrap_samples: int, seed: int) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], dict[str, dict[str, Any]]] = defaultdict(dict)
    for run in runs:
        key = (
            run["project"],
            run["node_input_mode"],
            run["path_object_mode"],
            _orientation(run),
            run["geometry"],
            float(run["curvature"]),
            int(run["dim"]),
            int(run["seed"]),
        )
        grouped[key][run["method_aggregation"]] = run
    rows = []
    for key, variants in sorted(grouped.items()):
        centroid = variants.get("centroid")
        measure = variants.get("measure")
        if centroid is None or measure is None:
            continue
        rows.append(_paired_query_row(key, "measure - centroid", measure, centroid, bootstrap_samples=bootstrap_samples, seed=seed))
    return rows


def _curvature_deltas(runs: Sequence[dict[str, Any]], *, bootstrap_samples: int, seed: int) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], dict[tuple[str, float], dict[str, Any]]] = defaultdict(dict)
    for run in runs:
        key = (
            run["project"],
            run["node_input_mode"],
            run["path_object_mode"],
            run["method_aggregation"],
            _orientation(run),
            int(run["dim"]),
            int(run["seed"]),
        )
        grouped[key][(run["geometry"], float(run["curvature"]))] = run
    rows = []
    for key, variants in sorted(grouped.items()):
        euclidean = variants.get(("euclidean", 1.0))
        if euclidean is None:
            continue
        for (geometry, curvature), poincare in sorted(variants.items()):
            if geometry == "poincare":
                rows.append(
                    _paired_query_row(
                        key,
                        f"Poincare c={curvature:g} - Euclidean",
                        poincare,
                        euclidean,
                        bootstrap_samples=bootstrap_samples,
                        seed=seed,
                    )
                )
    return rows


def _orientation_deltas(runs: Sequence[dict[str, Any]], *, bootstrap_samples: int, seed: int) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], dict[str, dict[str, Any]]] = defaultdict(dict)
    for run in runs:
        key = (
            run["project"],
            run["node_input_mode"],
            run["path_object_mode"],
            run["method_aggregation"],
            run["geometry"],
            float(run["curvature"]),
            int(run["dim"]),
            int(run["seed"]),
        )
        grouped[key][_orientation(run)] = run
    rows = []
    for key, variants in sorted(grouped.items()):
        directed = variants.get("directed")
        unoriented = variants.get("unoriented")
        if directed is None or unoriented is None:
            continue
        rows.append(_paired_query_row(key, "unoriented - directed", unoriented, directed, bootstrap_samples=bootstrap_samples, seed=seed))
    return rows


def _full_model_deltas(runs: Sequence[dict[str, Any]], *, bootstrap_samples: int, seed: int) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], dict[tuple[str, float, str, str], dict[str, Any]]] = defaultdict(dict)
    for run in runs:
        key = (run["project"], run["node_input_mode"], _orientation(run), int(run["dim"]), int(run["seed"]))
        variant = (run["geometry"], float(run["curvature"]), run["path_object_mode"], run["method_aggregation"])
        grouped[key][variant] = run
    rows = []
    for key, variants in sorted(grouped.items()):
        baseline = variants.get(("euclidean", 1.0, "single_point", "centroid"))
        full = variants.get(("poincare", 1.0, "lca_product", "measure"))
        if baseline is None or full is None:
            continue
        rows.append(_paired_query_row(key, "Poincare LCA-product measure - Euclidean single-point centroid", full, baseline, bootstrap_samples=bootstrap_samples, seed=seed))
    return rows


def _paired_query_row(
    key: tuple[Any, ...],
    contrast: str,
    left: dict[str, Any],
    right: dict[str, Any],
    *,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    left_by_anchor = {str(record["anchor_id"]): record for record in left.get("query_records", [])}
    right_by_anchor = {str(record["anchor_id"]): record for record in right.get("query_records", [])}
    anchors = sorted(set(left_by_anchor) & set(right_by_anchor))
    deltas = [
        {
            "delta_reciprocal_rank": _reciprocal_rank(left_by_anchor[anchor]) - _reciprocal_rank(right_by_anchor[anchor]),
            "delta_margin": float(left_by_anchor[anchor].get("margin", 0.0)) - float(right_by_anchor[anchor].get("margin", 0.0)),
            "delta_positive_distance": float(left_by_anchor[anchor].get("positive_distance", 0.0))
            - float(right_by_anchor[anchor].get("positive_distance", 0.0)),
        }
        for anchor in anchors
    ]
    rr_values = [item["delta_reciprocal_rank"] for item in deltas]
    margin_values = [item["delta_margin"] for item in deltas]
    ci_low, ci_high = _bootstrap_ci(rr_values, bootstrap_samples=bootstrap_samples, seed=seed)
    return {
        "project": str(key[0]),
        "node_input_mode": str(key[1]),
        "cell": _cell_label(left),
        "contrast": contrast,
        "seed": int(key[-1]),
        "n_queries": len(deltas),
        "mean_delta_reciprocal_rank": _mean(rr_values),
        "ci_low_delta_reciprocal_rank": ci_low,
        "ci_high_delta_reciprocal_rank": ci_high,
        "positive_queries": sum(value > 0.0 for value in rr_values),
        "mean_delta_margin": _mean(margin_values),
        "delta_reciprocal_rank_values": rr_values,
        "delta_margin_values": margin_values,
    }


def _cell_label(run: dict[str, Any]) -> str:
    geometry = str(run.get("geometry", "?"))
    curvature = float(run.get("curvature", 0.0))
    if geometry == "poincare":
        geometry = f"poincare c={curvature:g}"
    return (
        f"path={run.get('path_object_mode', '?')}; "
        f"method={run.get('method_aggregation', '?')}; "
        f"path_cost={_orientation(run)}; "
        f"geometry={geometry}; "
        f"d={run.get('dim', '?')}; "
        f"seed={run.get('seed', '?')}"
    )


def _aggregate_rows(rows: Sequence[dict[str, Any]], *, bootstrap_samples: int, seed: int) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        cell = str(row["cell"])
        cell_without_seed = "; ".join(part for part in cell.split("; ") if not part.startswith("seed="))
        key = (str(row["project"]), str(row["node_input_mode"]), cell_without_seed, str(row["contrast"]))
        grouped[key].append(row)

    aggregated = []
    for (project, node_input_mode, cell, contrast), items in sorted(grouped.items()):
        rr_values = [value for item in items for value in item.get("delta_reciprocal_rank_values", [])]
        margin_values = [value for item in items for value in item.get("delta_margin_values", [])]
        ci_low, ci_high = _bootstrap_ci(rr_values, bootstrap_samples=bootstrap_samples, seed=seed)
        aggregated.append(
            {
                "project": project,
                "node_input_mode": node_input_mode,
                "cell": cell,
                "contrast": contrast,
                "n_seeds": len(items),
                "n_queries": len(rr_values),
                "mean_delta_reciprocal_rank": _mean(rr_values),
                "ci_low_delta_reciprocal_rank": ci_low,
                "ci_high_delta_reciprocal_rank": ci_high,
                "positive_queries": sum(value > 0.0 for value in rr_values),
                "mean_delta_margin": _mean(margin_values),
            }
        )
    return aggregated


def _orientation(run: dict[str, Any]) -> str:
    return str(run.get("path_cost_orientation") or run.get("config", {}).get("path_cost_orientation") or "directed")


def _reciprocal_rank(record: dict[str, Any]) -> float:
    rank = max(1, int(record.get("rank", 1)))
    return 1.0 / rank


def _bootstrap_ci(values: Sequence[float], *, bootstrap_samples: int, seed: int) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    if bootstrap_samples <= 0:
        mean = _mean(values)
        return mean, mean
    rng = random.Random(seed)
    n = len(values)
    estimates = []
    for _ in range(bootstrap_samples):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        estimates.append(_mean(sample))
    estimates.sort()
    low_index = min(len(estimates) - 1, max(0, int(0.025 * len(estimates))))
    high_index = min(len(estimates) - 1, max(0, int(0.975 * len(estimates))))
    return estimates[low_index], estimates[high_index]


def _mean(values: Iterable[float]) -> float:
    items = list(values)
    if not items:
        return 0.0
    return sum(items) / len(items)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize query-level paired deltas for raw-AST factor matrices.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--json-output", type=Path)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260624)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = summarize_factor_query_deltas(args.input, bootstrap_samples=args.bootstrap_samples, seed=args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(format_markdown(summary), encoding="utf-8")
    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
