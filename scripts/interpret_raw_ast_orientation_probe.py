from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.summarize_raw_ast_factor_query_deltas import summarize_factor_query_deltas


def interpret_orientation_probe(
    input_path: Path,
    *,
    bootstrap_samples: int = 1000,
    seed: int = 20260624,
    practical_threshold: float = 0.01,
) -> dict[str, Any]:
    """Interpret directed-vs-unoriented orientation controls.

    The paired contrast is defined by the summarizer as ``unoriented -
    directed``. Positive confidence intervals support using a quotient-distance
    readout. Negative confidence intervals support retaining ordered left/right
    endpoint roles in the readout. Intervals crossing zero are treated as
    inconclusive. This diagnostic chooses a task readout, not the universally
    correct mathematical object.
    """

    summary = summarize_factor_query_deltas(input_path, bootstrap_samples=bootstrap_samples, seed=seed)
    rows = list(summary.get("orientation_deltas_aggregated", []))
    decisions = [_decision(row, practical_threshold=practical_threshold) for row in rows]
    counts: dict[str, int] = {}
    for row in decisions:
        counts[row["decision"]] = counts.get(row["decision"], 0) + 1
    return {
        "input": str(input_path),
        "bootstrap_samples": bootstrap_samples,
        "seed": seed,
        "practical_threshold": practical_threshold,
        "contrast": "unoriented - directed",
        "decision_counts": counts,
        "decisions": decisions,
        "overall": _overall_decision(decisions),
    }


def format_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Raw-AST orientation probe interpretation",
        "",
        f"Input: `{payload['input']}`",
        "",
        "Contrast: `unoriented - directed`.",
        "",
        "Interpretation rule:",
        "",
        "- CI entirely below zero: ordered directed readout is supported.",
        "- CI entirely above zero: unoriented quotient readout is supported.",
        "- CI crosses zero: orientation effect is inconclusive for that cell.",
        "",
        f"Overall decision: `{payload['overall']}`.",
        "",
        "## Cell decisions",
        "",
        "| Project | Node input | Cell | n seeds | n queries | Delta RR | 95% CI | Decision |",
        "|---|---|---|---:|---:|---:|---:|---|",
    ]
    for row in payload["decisions"]:
        lines.append(
            "| {project} | {node_input_mode} | {cell} | {n_seeds} | {n_queries} | "
            "{mean_delta_reciprocal_rank:+.4f} | "
            "[{ci_low_delta_reciprocal_rank:+.4f}, {ci_high_delta_reciprocal_rank:+.4f}] | "
            "{decision} |".format(**row)
        )
    if not payload["decisions"]:
        lines.append("| - | - | - | 0 | 0 | - | - | no_complete_orientation_pairs |")
    return "\n".join(lines) + "\n"


def _decision(row: dict[str, Any], *, practical_threshold: float) -> dict[str, Any]:
    low = float(row["ci_low_delta_reciprocal_rank"])
    high = float(row["ci_high_delta_reciprocal_rank"])
    mean_delta = float(row["mean_delta_reciprocal_rank"])
    enriched = dict(row)
    if high < 0.0 and abs(mean_delta) >= practical_threshold:
        enriched["decision"] = "ordered_directed_supported"
    elif low > 0.0 and abs(mean_delta) >= practical_threshold:
        enriched["decision"] = "unoriented_quotient_supported"
    elif high < 0.0:
        enriched["decision"] = "ordered_directed_small_effect"
    elif low > 0.0:
        enriched["decision"] = "unoriented_quotient_small_effect"
    else:
        enriched["decision"] = "inconclusive"
    return enriched


def _overall_decision(decisions: Sequence[dict[str, Any]]) -> str:
    if not decisions:
        return "no_complete_orientation_pairs"
    strong = [str(row["decision"]) for row in decisions if not str(row["decision"]).endswith("small_effect")]
    directed = sum(value == "ordered_directed_supported" for value in strong)
    quotient = sum(value == "unoriented_quotient_supported" for value in strong)
    inconclusive = sum(value == "inconclusive" for value in strong)
    if directed > 0 and quotient == 0 and directed >= inconclusive:
        return "ordered_directed_readout"
    if quotient > 0 and directed == 0 and quotient >= inconclusive:
        return "unoriented_quotient_readout"
    if directed == 0 and quotient == 0:
        return "orientation_not_resolved"
    return "mixed_orientation_evidence"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Interpret raw-AST Code2Hyp directed/unoriented orientation controls.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--json-output", type=Path)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260624)
    parser.add_argument("--practical-threshold", type=float, default=0.01)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    payload = interpret_orientation_probe(
        args.input,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
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
