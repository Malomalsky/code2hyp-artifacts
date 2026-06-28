from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence


METRICS = ("recall_at_1", "ndcg_at_3", "mrr", "margin_mean")


def summarize_factor_matrix(input_path: Path) -> dict[str, Any]:
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    runs = payload.get("runs", [])
    if not runs:
        raise ValueError(f"no runs found in {input_path}")
    return {
        "input": str(input_path),
        "status": payload.get("status"),
        "completed_runs": payload.get("completed_runs"),
        "expected_runs": payload.get("expected_runs"),
        "means": _means(runs),
        "curvature_deltas": _curvature_deltas(runs),
        "path_object_deltas": _path_object_deltas(runs),
        "aggregation_deltas": _aggregation_deltas(runs),
        "orientation_deltas": _orientation_deltas(runs),
        "full_model_deltas": _full_model_deltas(runs),
    }


def _means(runs: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str, str, str, float, int], list[dict[str, Any]]] = defaultdict(list)
    for run in runs:
        grouped[_variant_key(run)].append(run)

    rows: list[dict[str, Any]] = []
    for (
        project,
        node_input_mode,
        path_object_mode,
        method_aggregation,
        path_cost_orientation,
        geometry,
        curvature,
        dim,
    ), items in sorted(grouped.items()):
        row: dict[str, Any] = {
            "project": project,
            "node_input_mode": node_input_mode,
            "path_object_mode": path_object_mode,
            "method_aggregation": method_aggregation,
            "path_cost_orientation": path_cost_orientation,
            "geometry": geometry,
            "curvature": curvature,
            "dim": dim,
            "n": len(items),
        }
        for metric in METRICS:
            row[f"mean_{metric}"] = _mean(float(item[metric]) for item in items if metric in item)
        if geometry == "poincare":
            diagnostics = [item.get("geometry_diagnostics", {}) for item in items]
            row["mean_sqrt_curvature_norm_mean"] = _mean(
                float(item.get("sqrt_curvature_norm_mean", 0.0)) for item in diagnostics
            )
            row["max_sqrt_curvature_norm_max"] = max(
                float(item.get("sqrt_curvature_norm_max", 0.0)) for item in diagnostics
            )
        rows.append(row)
    return rows


def _curvature_deltas(runs: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str, str, int, int], dict[tuple[str, float], dict[str, Any]]] = defaultdict(dict)
    for run in runs:
        grouped[
            (
                str(run["project"]),
                str(run["node_input_mode"]),
                str(run["path_object_mode"]),
                str(run["method_aggregation"]),
                _orientation(run),
                int(run["dim"]),
                int(run["seed"]),
            )
        ][(str(run["geometry"]), float(run["curvature"]))] = run

    deltas: dict[tuple[str, str, str, str, int, float], list[dict[str, float]]] = defaultdict(list)
    for (project, node_input_mode, path_object_mode, method_aggregation, path_cost_orientation, dim, seed), variants in grouped.items():
        baseline = variants.get(("euclidean", 1.0))
        if baseline is None:
            continue
        for (geometry, curvature), run in variants.items():
            if geometry != "poincare":
                continue
            deltas[(project, node_input_mode, path_object_mode, method_aggregation, path_cost_orientation, dim, curvature)].append(
                _delta_record(run, baseline, seed)
            )
    return _delta_rows(
        deltas,
        ("project", "node_input_mode", "path_object_mode", "method_aggregation", "path_cost_orientation", "dim", "curvature"),
    )


def _path_object_deltas(runs: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str, str, float, int, int], dict[str, dict[str, Any]]] = defaultdict(dict)
    for run in runs:
        grouped[
            (
                str(run["project"]),
                str(run["node_input_mode"]),
                str(run["method_aggregation"]),
                _orientation(run),
                str(run["geometry"]),
                float(run["curvature"]),
                int(run["dim"]),
                int(run["seed"]),
            )
        ][str(run["path_object_mode"])] = run

    deltas: dict[tuple[str, str, str, str, float, int], list[dict[str, float]]] = defaultdict(list)
    for (project, node_input_mode, method_aggregation, path_cost_orientation, geometry, curvature, dim, seed), variants in grouped.items():
        single = variants.get("single_point")
        lca_product = variants.get("lca_product")
        if single is None or lca_product is None:
            continue
        deltas[(project, node_input_mode, method_aggregation, path_cost_orientation, geometry, curvature, dim)].append(
            _delta_record(lca_product, single, seed)
        )
    return _delta_rows(
        deltas,
        ("project", "node_input_mode", "method_aggregation", "path_cost_orientation", "geometry", "curvature", "dim"),
    )


def _aggregation_deltas(runs: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str, str, float, int, int], dict[str, dict[str, Any]]] = defaultdict(dict)
    for run in runs:
        grouped[
            (
                str(run["project"]),
                str(run["node_input_mode"]),
                str(run["path_object_mode"]),
                _orientation(run),
                str(run["geometry"]),
                float(run["curvature"]),
                int(run["dim"]),
                int(run["seed"]),
            )
        ][str(run["method_aggregation"])] = run

    deltas: dict[tuple[str, str, str, str, float, int], list[dict[str, float]]] = defaultdict(list)
    for (project, node_input_mode, path_object_mode, path_cost_orientation, geometry, curvature, dim, seed), variants in grouped.items():
        centroid = variants.get("centroid")
        measure = variants.get("measure")
        if centroid is None or measure is None:
            continue
        deltas[(project, node_input_mode, path_object_mode, path_cost_orientation, geometry, curvature, dim)].append(
            _delta_record(measure, centroid, seed)
        )
    return _delta_rows(
        deltas,
        ("project", "node_input_mode", "path_object_mode", "path_cost_orientation", "geometry", "curvature", "dim"),
    )


def _orientation_deltas(runs: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str, str, float, int, int], dict[str, dict[str, Any]]] = defaultdict(dict)
    for run in runs:
        grouped[
            (
                str(run["project"]),
                str(run["node_input_mode"]),
                str(run["path_object_mode"]),
                str(run["method_aggregation"]),
                str(run["geometry"]),
                float(run["curvature"]),
                int(run["dim"]),
                int(run["seed"]),
            )
        ][_orientation(run)] = run

    deltas: dict[tuple[str, str, str, str, str, float, int], list[dict[str, float]]] = defaultdict(list)
    for (project, node_input_mode, path_object_mode, method_aggregation, geometry, curvature, dim, seed), variants in grouped.items():
        directed = variants.get("directed")
        unoriented = variants.get("unoriented")
        if directed is None or unoriented is None:
            continue
        deltas[(project, node_input_mode, path_object_mode, method_aggregation, geometry, curvature, dim)].append(
            _delta_record(unoriented, directed, seed)
        )
    return _delta_rows(
        deltas,
        ("project", "node_input_mode", "path_object_mode", "method_aggregation", "geometry", "curvature", "dim"),
    )


def _full_model_deltas(runs: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, int, int], dict[tuple[str, float, str, str], dict[str, Any]]] = defaultdict(dict)
    for run in runs:
        grouped[
            (
                str(run["project"]),
                str(run["node_input_mode"]),
                _orientation(run),
                int(run["dim"]),
                int(run["seed"]),
            )
        ][
            (
                str(run["geometry"]),
                float(run["curvature"]),
                str(run["path_object_mode"]),
                str(run["method_aggregation"]),
            )
        ] = run

    deltas: dict[tuple[str, str, int], list[dict[str, float]]] = defaultdict(list)
    for (project, node_input_mode, path_cost_orientation, dim, seed), variants in grouped.items():
        baseline = variants.get(("euclidean", 1.0, "single_point", "centroid"))
        full = variants.get(("poincare", 1.0, "lca_product", "measure"))
        if baseline is None or full is None:
            continue
        deltas[(project, node_input_mode, path_cost_orientation, dim)].append(_delta_record(full, baseline, seed))
    return _delta_rows(deltas, ("project", "node_input_mode", "path_cost_orientation", "dim"))


def format_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Raw-AST Code2Hyp factor-matrix summary",
        "",
        f"Input: `{summary['input']}`",
        "",
        f"Status: `{summary.get('status')}`; completed `{summary.get('completed_runs')}/{summary.get('expected_runs')}`.",
        "",
        "## Mean metrics",
        "",
        "| Project | Node input | Path object | Method | Path cost | Geometry | c | d | n | R@1 | NDCG@3 | MRR | Margin |",
        "|---|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary["means"]:
        lines.append(
            "| {project} | {node_input_mode} | {path_object_mode} | {method_aggregation} | {path_cost_orientation} | {geometry} | "
            "{curvature:g} | {dim} | {n} | {mean_recall_at_1:.4f} | {mean_ndcg_at_3:.4f} | "
            "{mean_mrr:.4f} | {mean_margin_mean:.4f} |".format(**row)
        )
    _append_delta_table(
        lines,
        title="Curvature deltas: Poincare minus matched Euclidean",
        rows=summary["curvature_deltas"],
        columns=("project", "path_object_mode", "method_aggregation", "path_cost_orientation", "curvature"),
    )
    _append_delta_table(
        lines,
        title="Path-object deltas: LCA-product minus single point",
        rows=summary["path_object_deltas"],
        columns=("project", "method_aggregation", "path_cost_orientation", "geometry", "curvature"),
    )
    _append_delta_table(
        lines,
        title="Aggregation deltas: measure minus centroid",
        rows=summary["aggregation_deltas"],
        columns=("project", "path_object_mode", "path_cost_orientation", "geometry", "curvature"),
    )
    _append_delta_table(
        lines,
        title="Orientation deltas: unoriented minus directed",
        rows=summary["orientation_deltas"],
        columns=("project", "path_object_mode", "method_aggregation", "geometry", "curvature"),
    )
    _append_delta_table(
        lines,
        title="Full Code2Hyp-v1 deltas: Poincare LCA-product measure minus Euclidean single-point centroid",
        rows=summary["full_model_deltas"],
        columns=("project", "path_cost_orientation"),
    )
    return "\n".join(lines) + "\n"


def _append_delta_table(lines: list[str], *, title: str, rows: Sequence[dict[str, Any]], columns: Sequence[str]) -> None:
    lines.extend(
        [
            "",
            f"## {title}",
            "",
            "| " + " | ".join(_label(column) for column in columns) + " | n | Delta R@1 | Delta NDCG@3 | Delta MRR | Positive MRR seeds |",
            "|"
            + "|".join("---" for _ in columns)
            + "|---:|---:|---:|---:|---:|",
        ]
    )
    if not rows:
        lines.append("| " + " | ".join("-" for _ in columns) + " | 0 | - | - | - | - |")
        return
    for row in rows:
        prefix = " | ".join(_format_cell(row.get(column)) for column in columns)
        lines.append(
            f"| {prefix} | {row['n']} | {row['mean_delta_recall_at_1']:+.4f} | "
            f"{row['mean_delta_ndcg_at_3']:+.4f} | {row['mean_delta_mrr']:+.4f} | {row['positive_mrr']}/{row['n']} |"
        )


def _delta_record(left: dict[str, Any], right: dict[str, Any], seed: int) -> dict[str, float]:
    return {"seed": float(seed), **{metric: float(left[metric]) - float(right[metric]) for metric in METRICS}}


def _delta_rows(
    deltas: dict[tuple[Any, ...], list[dict[str, float]]],
    key_names: Sequence[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for key, items in sorted(deltas.items()):
        row: dict[str, Any] = dict(zip(key_names, key))
        row["n"] = len(items)
        for metric in METRICS:
            values = [float(item[metric]) for item in items]
            row[f"mean_delta_{metric}"] = _mean(values)
            row[f"positive_{metric}"] = sum(value > 0.0 for value in values)
            row[f"per_seed_delta_{metric}"] = values
        rows.append(row)
    return rows


def _variant_key(run: dict[str, Any]) -> tuple[str, str, str, str, str, str, float, int]:
    return (
        str(run["project"]),
        str(run["node_input_mode"]),
        str(run["path_object_mode"]),
        str(run["method_aggregation"]),
        _orientation(run),
        str(run["geometry"]),
        float(run["curvature"]),
        int(run["dim"]),
    )


def _orientation(run: dict[str, Any]) -> str:
    return str(run.get("path_cost_orientation") or run.get("config", {}).get("path_cost_orientation") or "directed")


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def _label(value: str) -> str:
    return value.replace("_", " ").title()


def _format_cell(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize raw-AST Code2Hyp factor-matrix outputs.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--json-output", type=Path)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = summarize_factor_matrix(args.input)
    markdown = format_markdown(summary)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(markdown, encoding="utf-8")
    else:
        sys.stdout.write(markdown)
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
