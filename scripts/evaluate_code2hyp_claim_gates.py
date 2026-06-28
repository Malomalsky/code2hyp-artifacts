from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def evaluate_claim_gates(
    input_path: Path,
    *,
    min_projects: int = 2,
    practical_threshold: float = 0.01,
) -> dict[str, Any]:
    """Evaluate pre-declared Code2Hyp claim gates from query-level deltas."""

    payload = json.loads(input_path.read_text(encoding="utf-8"))
    gates = [
        _gate(
            name="H1_LCA_product_path_object",
            rows=payload.get("path_object_deltas_aggregated", []),
            predicate=lambda row: row.get("contrast") == "LCA-product - single point",
            min_projects=min_projects,
            practical_threshold=practical_threshold,
            claim="LCA-product path objects improve over single-point path pooling.",
        ),
        _gate(
            name="H2_measure_over_paths",
            rows=payload.get("aggregation_deltas_aggregated", []),
            predicate=lambda row: row.get("contrast") == "measure - centroid",
            min_projects=min_projects,
            practical_threshold=practical_threshold,
            claim="Keeping a method as a path measure improves over centroid collapse.",
        ),
        _gate(
            name="H4_poincare_parameterization_auxiliary",
            rows=payload.get("curvature_deltas_aggregated", []),
            predicate=lambda row: (
                row.get("contrast") == "Poincare c=1 - Euclidean"
                and "path=lca_product" in str(row.get("cell", ""))
                and "method=measure" in str(row.get("cell", ""))
            ),
            min_projects=min_projects,
            practical_threshold=practical_threshold,
            claim=(
                "Matched Poincare parameterization improves over Euclidean in the canonical "
                "LCA-product + measure cell; interpret as negative-curvature evidence only "
                "with the near-zero-curvature control."
            ),
        ),
        _gate(
            name="full_model_contrast",
            rows=payload.get("full_model_deltas_aggregated", []),
            predicate=lambda row: "Poincare LCA-product measure" in str(row.get("contrast", "")),
            min_projects=min_projects,
            practical_threshold=practical_threshold,
            claim="Full Code2Hyp-v1 improves over Euclidean single-point centroid.",
        ),
        _gate(
            name="orientation_gate",
            rows=payload.get("orientation_deltas_aggregated", []),
            predicate=lambda row: row.get("contrast") == "unoriented - directed",
            min_projects=min_projects,
            practical_threshold=practical_threshold,
            claim=(
                "Orientation control determines the task readout: ordered LCA-anchored paths "
                "or quotient distance under endpoint reversal."
            ),
            two_sided_direction=True,
        ),
    ]
    return {
        "input": str(input_path),
        "min_projects": min_projects,
        "practical_threshold": practical_threshold,
        "gates": gates,
        "overall_claim_status": _overall_claim_status(gates),
    }


def format_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Code2Hyp claim-gate evaluation",
        "",
        f"Input: `{payload['input']}`",
        "",
        f"Minimum projects: `{payload['min_projects']}`.",
        f"Practical threshold for mean delta RR: `{payload['practical_threshold']}`.",
        "",
        f"Overall claim status: `{payload['overall_claim_status']}`.",
        "",
        "| Gate | Decision | Supported projects | Supported cells | Claim |",
        "|---|---|---:|---:|---|",
    ]
    for gate in payload["gates"]:
        lines.append(
            "| {name} | {decision} | {supported_project_count} | {supported_cell_count} | {claim} |".format(**gate)
        )
    lines.extend(["", "## Details", ""])
    for gate in payload["gates"]:
        lines.extend(
            [
                f"### {gate['name']}",
                "",
                f"Decision: `{gate['decision']}`.",
                "",
                "| Project | Cell | Delta RR | 95% CI | Supported |",
                "|---|---|---:|---:|---|",
            ]
        )
        if not gate["rows"]:
            lines.append("| - | - | - | - | no_rows |")
        for row in gate["rows"]:
            lines.append(
                "| {project} | {cell} | {mean_delta_reciprocal_rank:+.4f} | "
                "[{ci_low_delta_reciprocal_rank:+.4f}, {ci_high_delta_reciprocal_rank:+.4f}] | "
                "{support_label} |".format(**row)
            )
        lines.append("")
    return "\n".join(lines)


def _gate(
    *,
    name: str,
    rows: Sequence[dict[str, Any]],
    predicate: Any,
    min_projects: int,
    practical_threshold: float,
    claim: str,
    two_sided_direction: bool = False,
) -> dict[str, Any]:
    selected = [dict(row) for row in rows if predicate(row)]
    enriched = [_support_row(row, practical_threshold=practical_threshold, two_sided_direction=two_sided_direction) for row in selected]
    supported = [row for row in enriched if row["supported"]]
    supported_projects = sorted({str(row["project"]) for row in supported})
    if not enriched:
        decision = "not_evaluable"
    elif len(supported_projects) >= min_projects:
        decision = "supported"
    elif supported:
        decision = "partial_support"
    else:
        decision = "not_supported"
    return {
        "name": name,
        "claim": claim,
        "decision": decision,
        "supported_project_count": len(supported_projects),
        "supported_cell_count": len(supported),
        "supported_projects": supported_projects,
        "rows": enriched,
    }


def _support_row(row: dict[str, Any], *, practical_threshold: float, two_sided_direction: bool) -> dict[str, Any]:
    low = float(row.get("ci_low_delta_reciprocal_rank", 0.0))
    high = float(row.get("ci_high_delta_reciprocal_rank", 0.0))
    mean_delta = float(row.get("mean_delta_reciprocal_rank", 0.0))
    if two_sided_direction:
        supported = (high < 0.0 or low > 0.0) and abs(mean_delta) >= practical_threshold
    else:
        supported = low > 0.0 and mean_delta >= practical_threshold
    enriched = dict(row)
    enriched["supported"] = supported
    enriched["support_label"] = "yes" if supported else "no"
    return enriched


def _overall_claim_status(gates: Sequence[dict[str, Any]]) -> str:
    by_name = {gate["name"]: gate for gate in gates}
    h1 = by_name.get("H1_LCA_product_path_object", {}).get("decision")
    h2 = by_name.get("H2_measure_over_paths", {}).get("decision")
    full = by_name.get("full_model_contrast", {}).get("decision")
    h4 = by_name.get("H4_poincare_parameterization_auxiliary", {}).get("decision")
    orientation = by_name.get("orientation_gate", {}).get("decision")
    if h1 == h2 == full == "supported" and orientation in {"supported", "partial_support", "not_evaluable"}:
        if h4 == "supported":
            return "representation_claims_supported_with_auxiliary_poincare_signal"
        return "representation_claims_supported_under_matched_geometry_controls"
    if h1 == "supported" and h2 == "supported":
        return "core_product_measure_claim_supported_but_system_validation_incomplete"
    return "continue_experiments_before_claiming_main_result"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate pre-declared Code2Hyp claim gates from query-level deltas.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--json-output", type=Path)
    parser.add_argument("--min-projects", type=int, default=2)
    parser.add_argument("--practical-threshold", type=float, default=0.01)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    payload = evaluate_claim_gates(
        args.input,
        min_projects=args.min_projects,
        practical_threshold=args.practical_threshold,
    )
    markdown = format_markdown(payload)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(markdown, encoding="utf-8")
    else:
        print(markdown)
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()
