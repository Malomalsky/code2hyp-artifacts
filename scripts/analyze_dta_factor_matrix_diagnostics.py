from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


CANONICAL_CELL = {"geometry": "H_1", "path_object_mode": "lca_product", "method_aggregation": "measure"}
NEAR_ZERO_CELL = {"geometry": "H_1e-4", "path_object_mode": "lca_product", "method_aggregation": "measure"}
EUCLIDEAN_CELL = {"geometry": "E", "path_object_mode": "lca_product", "method_aggregation": "measure"}
DEFAULT_COST_MODE = "unnormalized_combined"
DELTA_TOLERANCE = 1e-4

DIAGNOSTIC_CONTRASTS = (
    {
        "contrast": "H1_minus_H1e-4",
        "label": "active H_1 - near-zero H_1e-4",
        "left": CANONICAL_CELL,
        "right": NEAR_ZERO_CELL,
    },
    {
        "contrast": "H1_minus_E",
        "label": "active H_1 - Euclidean",
        "left": CANONICAL_CELL,
        "right": EUCLIDEAN_CELL,
    },
    {
        "contrast": "H1e-4_minus_E",
        "label": "near-zero H_1e-4 - Euclidean",
        "left": NEAR_ZERO_CELL,
        "right": EUCLIDEAN_CELL,
    },
    {
        "contrast": "measure_minus_centroid",
        "label": "measure - centroid",
        "left": CANONICAL_CELL,
        "right": {"geometry": "H_1", "path_object_mode": "lca_product", "method_aggregation": "centroid"},
    },
    {
        "contrast": "lca_product_minus_single_point",
        "label": "LCA-product - single-point",
        "left": CANONICAL_CELL,
        "right": {"geometry": "H_1", "path_object_mode": "single_point", "method_aggregation": "measure"},
    },
)


def analyze_factor_matrix_diagnostics(input_paths: Path | Sequence[Path]) -> dict[str, Any]:
    paths = (input_paths,) if isinstance(input_paths, Path) else tuple(input_paths)
    if not paths:
        raise ValueError("at least one input path is required")
    runs = _load_runs(paths)
    cell_summaries = _cell_summaries(runs)
    contrast_rows = _contrast_rows(runs)
    flags = _diagnostic_flags(cell_summaries, contrast_rows)
    return {
        "inputs": [str(path) for path in paths],
        "run_count": len(runs),
        "seed_count": len({run["seed"] for run in runs}),
        "encoder_policies": sorted({str(run.get("encoder_policy")) for run in runs}),
        "cell_summaries": cell_summaries,
        "diagnostic_contrasts": contrast_rows,
        "diagnostic_flags": flags,
        "interpretation": _interpretation(flags, contrast_rows, cell_summaries),
    }


def _load_runs(paths: Sequence[Path]) -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        config = payload.get("config", {})
        payload_seed = config.get("seed")
        payload_encoder_policy = config.get("encoder_policy")
        for run in payload.get("runs", []):
            row = dict(run)
            row.setdefault("seed", payload_seed)
            row.setdefault("source_file", str(path))
            row.setdefault("encoder_policy", payload_encoder_policy)
            runs.append(row)
    return runs


def _cell_summaries(runs: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for run in runs:
        grouped[_cell_key(run)].append(run)
    rows: list[dict[str, Any]] = []
    for key, group in sorted(grouped.items()):
        cost_mode, geometry, path_object_mode, method_aggregation = key
        rows.append(
            {
                "cell_id": f"{cost_mode}__{geometry}__{path_object_mode}__{method_aggregation}",
                "cost_mode": cost_mode,
                "geometry": geometry,
                "path_object_mode": path_object_mode,
                "method_aggregation": method_aggregation,
                "seed_count": len({run["seed"] for run in group}),
                "mean_mrr": _mean(_float_values(group, "mrr")),
                "mean_recall_at_1": _mean(_float_values(group, "recall_at_1")),
                "mean_side_cost_share": _mean([_cost_value(run, "side_cost_share") for run in group]),
                "mean_point_cost_share": _mean([_cost_value(run, "point_cost_share") for run in group]),
                "mean_total_distance_side_expected_cost_spearman": _mean(
                    [_retrieval_value(run, "total_distance_side_expected_cost_spearman") for run in group]
                ),
                "mean_total_distance_point_expected_cost_spearman": _mean(
                    [_retrieval_value(run, "total_distance_point_expected_cost_spearman") for run in group]
                ),
                "mean_transport_entropy": _mean([_retrieval_value(run, "transport_entropy_mean") for run in group]),
                "mean_radius_fraction_median": _mean([_radius_fraction(run, "median") for run in group]),
                "mean_radius_fraction_max": _mean([_radius_fraction(run, "max") for run in group]),
            }
        )
    return rows


def _contrast_rows(runs: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key = {(run["seed"], *_cell_key(run)): run for run in runs}
    rows: list[dict[str, Any]] = []
    cost_modes = sorted({_cost_mode(run) for run in runs})
    for cost_mode in cost_modes:
        for spec in DIAGNOSTIC_CONTRASTS:
            deltas = []
            pairs = []
            for seed in sorted({run["seed"] for run in runs}):
                left = by_key.get((seed, cost_mode, *_spec_key(spec["left"])))
                right = by_key.get((seed, cost_mode, *_spec_key(spec["right"])))
                if left is None or right is None:
                    continue
                delta = float(left.get("mrr", 0.0)) - float(right.get("mrr", 0.0))
                deltas.append(delta)
                pairs.append({"seed": seed, "left_mrr": float(left.get("mrr", 0.0)), "right_mrr": float(right.get("mrr", 0.0)), "delta_mrr": delta})
            rows.append(
                {
                    "contrast": spec["contrast"],
                    "label": spec["label"],
                    "cost_mode": cost_mode,
                    "seed_count": len(pairs),
                    "mean_delta_mrr": _mean(deltas),
                    "median_delta_mrr": _median(deltas),
                    "positive_seeds": sum(1 for value in deltas if value > 0.0),
                    "seed_deltas": pairs,
                    "encoder_confounded": _is_encoder_confounded(runs),
                }
            )
    return rows


def _diagnostic_flags(cell_summaries: Sequence[dict[str, Any]], contrast_rows: Sequence[dict[str, Any]]) -> dict[str, bool]:
    flag_cost_mode = _flag_cost_mode(cell_summaries)
    contrasts = {row["contrast"]: row for row in contrast_rows if row.get("cost_mode") == flag_cost_mode}
    canonical = _find_cell(cell_summaries, {**CANONICAL_CELL, "cost_mode": flag_cost_mode})
    near_zero = _find_cell(cell_summaries, {**NEAR_ZERO_CELL, "cost_mode": flag_cost_mode})
    return {
        "flag_cost_mode": flag_cost_mode,
        "active_curvature_underperforms_near_zero": _is_negative(_mean_delta(contrasts, "H1_minus_H1e-4")),
        "active_curvature_underperforms_euclidean": _is_negative(_mean_delta(contrasts, "H1_minus_E")),
        "measure_underperforms_centroid": _is_negative(_mean_delta(contrasts, "measure_minus_centroid")),
        "lca_product_underperforms_single_point": _is_negative(_mean_delta(contrasts, "lca_product_minus_single_point")),
        "side_cost_dominates": float(canonical.get("mean_side_cost_share", 0.0)) > 0.5,
        "active_curvature_radius_active": float(canonical.get("mean_radius_fraction_median", 0.0)) > 0.2,
        "near_zero_radius_near_center": float(near_zero.get("mean_radius_fraction_median", 1.0)) < 0.05,
        "geometry_confounded_aggregation_comparison": any(row["contrast"] == "measure_minus_centroid" and row["encoder_confounded"] for row in contrast_rows),
    }


def _interpretation(
    flags: dict[str, bool],
    contrast_rows: Sequence[dict[str, Any]],
    cell_summaries: Sequence[dict[str, Any]],
) -> list[str]:
    flag_cost_mode = str(flags.get("flag_cost_mode", DEFAULT_COST_MODE))
    contrasts = {row["contrast"]: row for row in contrast_rows if row.get("cost_mode") == flag_cost_mode}
    canonical = _find_cell(cell_summaries, {**CANONICAL_CELL, "cost_mode": flag_cost_mode})
    near_zero = _find_cell(cell_summaries, {**NEAR_ZERO_CELL, "cost_mode": flag_cost_mode})
    lines = []
    if flags["active_curvature_underperforms_near_zero"]:
        row = contrasts["H1_minus_H1e-4"]
        lines.append(
            f"Under `{flag_cost_mode}`, the active-curvature cell underperforms the near-zero curvature control "
            f"(mean delta MRR {row['mean_delta_mrr']:+.4f})."
        )
    elif abs(_mean_delta(contrasts, "H1_minus_H1e-4")) <= DELTA_TOLERANCE:
        lines.append("The active-curvature and near-zero curvature cells are tied within numerical tolerance.")
    if flags["active_curvature_underperforms_euclidean"]:
        row = contrasts["H1_minus_E"]
        lines.append(
            f"Under `{flag_cost_mode}`, the active-curvature cell underperforms the Euclidean matched cell "
            f"(mean delta MRR {row['mean_delta_mrr']:+.4f})."
        )
    elif abs(_mean_delta(contrasts, "H1_minus_E")) <= DELTA_TOLERANCE:
        lines.append("The active-curvature and Euclidean matched cells are tied within numerical tolerance.")
    if flags["active_curvature_radius_active"] and flags["near_zero_radius_near_center"]:
        radius_prefix = (
            "The negative active-curvature result"
            if flags["active_curvature_underperforms_near_zero"] or flags["active_curvature_underperforms_euclidean"]
            else "The near-tie active-curvature result"
        )
        lines.append(
            f"{radius_prefix} is not explained by all points collapsing to the origin: "
            f"H_1 median radius fraction is {canonical.get('mean_radius_fraction_median', 0.0):.4f}, "
            f"whereas H_1e-4 is {near_zero.get('mean_radius_fraction_median', 0.0):.4f}."
        )
    if flags["side_cost_dominates"]:
        lines.append(
            f"Under `{flag_cost_mode}`, the canonical product cost is dominated by side features "
            f"(mean side-cost share {canonical.get('mean_side_cost_share', 0.0):.4f}); this is the next methodological target."
        )
    if flags["geometry_confounded_aggregation_comparison"]:
        lines.append(
            "The measure-vs-centroid contrast is encoder-confounded under geometry_aware policy; a shared-encoder control is required before attributing the delta to transport aggregation alone."
        )
    if not lines:
        lines.append("No diagnostic failure mode was triggered by the configured thresholds.")
    return lines


def format_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# DTA factor-matrix diagnostic audit",
        "",
        "Inputs:",
        *[f"- `{path}`" for path in summary["inputs"]],
        f"Runs: {summary['run_count']}; seeds: {summary['seed_count']}.",
        f"Encoder policies: {', '.join(summary['encoder_policies'])}.",
        "",
        "## Canonical cell summaries",
        "",
        "| Cost mode | Cell | Seeds | MRR | R@1 | Side-cost share | Point-cost share | rho(total, side) | rho(total, point) | Transport entropy | Median radius fraction | Max radius fraction |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary["cell_summaries"]:
        lines.append(
            f"| `{row['cost_mode']}` | `{row['cell_id']}` | {row['seed_count']} | {row['mean_mrr']:.4f} | {row['mean_recall_at_1']:.4f} | "
            f"{row['mean_side_cost_share']:.4f} | {row['mean_point_cost_share']:.4f} | "
            f"{row['mean_total_distance_side_expected_cost_spearman']:.4f} | "
            f"{row['mean_total_distance_point_expected_cost_spearman']:.4f} | "
            f"{row['mean_transport_entropy']:.4f} | "
            f"{row['mean_radius_fraction_median']:.4f} | {row['mean_radius_fraction_max']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Diagnostic contrasts",
            "",
            "| Cost mode | Contrast | Seeds | Positive seeds | Mean delta MRR | Median delta MRR | Encoder-confounded |",
            "|---|---|---:|---:|---:|---:|---|",
        ]
    )
    for row in summary["diagnostic_contrasts"]:
        lines.append(
            f"| `{row['cost_mode']}` | {row['label']} | {row['seed_count']} | {row['positive_seeds']}/{row['seed_count']} | "
            f"{row['mean_delta_mrr']:+.4f} | {row['median_delta_mrr']:+.4f} | {str(row['encoder_confounded']).lower()} |"
        )
    lines.extend(["", "## Diagnostic flags", ""])
    for name, value in summary["diagnostic_flags"].items():
        lines.append(f"- `{name}`: {str(value).lower()}")
    lines.extend(["", "## Interpretation", ""])
    lines.extend(summary["interpretation"])
    lines.append("")
    return "\n".join(lines)


def _cell_key(run: dict[str, Any]) -> tuple[str, str, str, str]:
    return (_cost_mode(run), str(run["geometry"]), str(run["path_object_mode"]), str(run["method_aggregation"]))


def _spec_key(spec: dict[str, str]) -> tuple[str, str, str]:
    return (spec["geometry"], spec["path_object_mode"], spec["method_aggregation"])


def _cost_mode(run: dict[str, Any]) -> str:
    return str(run.get("cost_mode") or DEFAULT_COST_MODE)


def _flag_cost_mode(rows: Sequence[dict[str, Any]]) -> str:
    cost_modes = {str(row.get("cost_mode") or DEFAULT_COST_MODE) for row in rows}
    if DEFAULT_COST_MODE in cost_modes:
        return DEFAULT_COST_MODE
    if "train_normalized_combined" in cost_modes:
        return "train_normalized_combined"
    return sorted(cost_modes)[0] if cost_modes else DEFAULT_COST_MODE


def _find_cell(rows: Sequence[dict[str, Any]], spec: dict[str, str]) -> dict[str, Any]:
    for row in rows:
        if all(row[key] == value for key, value in spec.items()):
            return row
    return {}


def _mean_delta(rows: dict[str, dict[str, Any]], contrast: str) -> float:
    return float(rows.get(contrast, {}).get("mean_delta_mrr", 0.0))


def _is_negative(value: float) -> bool:
    return value < -DELTA_TOLERANCE


def _float_values(rows: Sequence[dict[str, Any]], key: str) -> list[float]:
    return [float(row[key]) for row in rows if key in row and row[key] is not None]


def _cost_value(run: dict[str, Any], key: str) -> float:
    return float(run.get("cost_component_diagnostics", {}).get(key, 0.0))


def _retrieval_value(run: dict[str, Any], key: str) -> float:
    return float(run.get("retrieval_diagnostics", {}).get(key, 0.0))


def _radius_fraction(run: dict[str, Any], statistic: str) -> float:
    curvature = float(run.get("curvature", 0.0))
    if curvature <= 0.0:
        return 0.0
    fractions = run.get("embedding_norm_diagnostics", {}).get("curvature_radius_fractions", {})
    payload = _lookup_radius_payload(fractions, curvature)
    if not payload:
        return 0.0
    key = f"scaled_radius_fraction_{statistic}"
    return float(payload.get(key, 0.0))


def _lookup_radius_payload(fractions: dict[str, Any], curvature: float) -> dict[str, Any]:
    for key, value in fractions.items():
        try:
            if abs(float(key) - curvature) <= 1e-12:
                return dict(value)
        except ValueError:
            continue
    return {}


def _is_encoder_confounded(runs: Sequence[dict[str, Any]]) -> bool:
    return any(run.get("encoder_policy") == "geometry_aware" for run in runs)


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _median(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return 0.5 * (ordered[middle - 1] + ordered[middle])


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit diagnostic failure modes in a DTA Code2Hyp factor matrix.")
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--json-output", type=Path)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = analyze_factor_matrix_diagnostics(tuple(args.input))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(format_markdown(summary), encoding="utf-8")
    if args.json_output:
        _write_json(args.json_output, summary)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
