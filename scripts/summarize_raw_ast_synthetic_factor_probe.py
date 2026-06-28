from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any, Iterable


def summarize_synthetic_factor_probe(input_path: Path) -> dict[str, Any]:
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    rows = list(payload.get("rows", []))
    return {
        "input": str(input_path),
        "row_count": len(rows),
        "means": _mean_rows(rows),
        "lca_product_deltas": _lca_product_deltas(rows),
        "curvature_deltas": _curvature_deltas(rows),
        "full_model_deltas": _full_model_deltas(rows),
    }


def format_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Raw-AST synthetic factor probe",
        "",
        f"Input rows: {summary['row_count']}",
        "",
        "## LCA-product path-object effect",
        "",
        _delta_overview(summary["lca_product_deltas"], "single_point - lca_product"),
        "",
        "## Poincare curvature effect",
        "",
        _delta_overview(summary["curvature_deltas"], "euclidean - poincare"),
        "",
        "## Full canonical contrast",
        "",
        _delta_overview(summary["full_model_deltas"], "euclidean single_point - poincare lca_product"),
        "",
        "## Mean path stress",
        "",
        "| case | dim | geometry | c | path object | stress | Spearman rho |",
        "|---|---:|---|---:|---|---:|---:|",
    ]
    for row in summary["means"]:
        lines.append(
            "| {case} | {dim} | {geometry} | {curvature:g} | {path_object_mode} | {mean_path_stress:.4f} | {mean_path_spearman:.4f} |".format(
                **row
            )
        )
    return "\n".join(lines) + "\n"


def _delta_overview(rows: list[dict[str, Any]], label: str) -> str:
    if not rows:
        return f"No matched rows for `{label}`."
    values = [float(row["delta_path_stress"]) for row in rows]
    positive = sum(value > 0.0 for value in values)
    mean = statistics.mean(values)
    median = statistics.median(values)
    return (
        f"`{label}`: positive stress reduction in {positive}/{len(values)} matched contrasts; "
        f"mean delta {mean:.4f}, median delta {median:.4f}."
    )


def _mean_rows(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int, str, float, str], list[dict[str, Any]]] = {}
    for row in rows:
        key = (str(row["case"]), int(row["dim"]), str(row["geometry"]), float(row["curvature"]), str(row["path_object_mode"]))
        grouped.setdefault(key, []).append(row)
    means = []
    for (case, dim, geometry, curvature, path_object_mode), items in sorted(grouped.items()):
        means.append(
            {
                "case": case,
                "dim": dim,
                "geometry": geometry,
                "curvature": curvature,
                "path_object_mode": path_object_mode,
                "mean_path_stress": statistics.mean(float(item["path_stress"]) for item in items),
                "mean_path_spearman": statistics.mean(float(item["path_spearman"]) for item in items),
                "n": len(items),
            }
        )
    return means


def _lca_product_deltas(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int, str, float], dict[str, dict[str, Any]]] = {}
    for row in rows:
        key = (str(row["case"]), int(row["dim"]), str(row["geometry"]), float(row["curvature"]))
        grouped.setdefault(key, {})[str(row["path_object_mode"])] = row
    deltas = []
    for (case, dim, geometry, curvature), variants in sorted(grouped.items()):
        single = variants.get("single_point")
        lca = variants.get("lca_product")
        if single is None or lca is None:
            continue
        deltas.append(_delta_row(case, dim, geometry, curvature, "single_point - lca_product", single, lca))
    return deltas


def _curvature_deltas(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int, str], dict[str, list[dict[str, Any]]]] = {}
    for row in rows:
        key = (str(row["case"]), int(row["dim"]), str(row["path_object_mode"]))
        grouped.setdefault(key, {}).setdefault(str(row["geometry"]), []).append(row)
    deltas = []
    for (case, dim, path_object_mode), variants in sorted(grouped.items()):
        euclidean_rows = variants.get("euclidean", [])
        poincare_rows = variants.get("poincare", [])
        if not euclidean_rows or not poincare_rows:
            continue
        euclidean = euclidean_rows[0]
        for poincare in sorted(poincare_rows, key=lambda item: float(item["curvature"])):
            deltas.append(
                _delta_row(case, dim, path_object_mode, float(poincare["curvature"]), "euclidean - poincare", euclidean, poincare)
            )
    return deltas


def _full_model_deltas(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int], dict[tuple[str, float, str], dict[str, Any]]] = {}
    for row in rows:
        key = (str(row["case"]), int(row["dim"]))
        variant = (str(row["geometry"]), float(row["curvature"]), str(row["path_object_mode"]))
        grouped.setdefault(key, {})[variant] = row
    deltas = []
    for (case, dim), variants in sorted(grouped.items()):
        baseline = variants.get(("euclidean", 0.0, "single_point"))
        if baseline is None:
            continue
        for (geometry, curvature, path_object_mode), canonical in sorted(variants.items()):
            if geometry == "poincare" and path_object_mode == "lca_product":
                deltas.append(
                    _delta_row(
                        case,
                        dim,
                        "full",
                        curvature,
                        "euclidean single_point - poincare lca_product",
                        baseline,
                        canonical,
                    )
                )
    return deltas


def _delta_row(
    case: str,
    dim: int,
    factor: str,
    curvature: float,
    contrast: str,
    baseline: dict[str, Any],
    improved: dict[str, Any],
) -> dict[str, Any]:
    return {
        "case": case,
        "dim": dim,
        "factor": factor,
        "curvature": curvature,
        "contrast": contrast,
        "baseline_path_stress": float(baseline["path_stress"]),
        "improved_path_stress": float(improved["path_stress"]),
        "delta_path_stress": float(baseline["path_stress"]) - float(improved["path_stress"]),
        "baseline_path_spearman": float(baseline["path_spearman"]),
        "improved_path_spearman": float(improved["path_spearman"]),
        "delta_path_spearman": float(improved["path_spearman"]) - float(baseline["path_spearman"]),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize synthetic raw-AST Code2Hyp factor probes.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--json-output", type=Path)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = summarize_synthetic_factor_probe(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(format_markdown(summary), encoding="utf-8")
    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
