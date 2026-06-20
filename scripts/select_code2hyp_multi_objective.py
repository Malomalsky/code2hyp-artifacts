from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from geometry_profile_research.code2hyp_reporting import multi_objective_variant_selection


def build_multi_objective_markdown(
    result: dict[str, Any],
    f1_weight: float = 0.5,
    spearman_weight: float = 0.5,
    variant_filter: tuple[str, ...] | None = None,
) -> str:
    selection = multi_objective_variant_selection(
        result,
        objectives=(
            ("validation_f1_mean", "max", f1_weight),
            ("validation_structural_spearman_mean", "max", spearman_weight),
        ),
        variant_filter=variant_filter,
    )
    lines = [
        "# Multi-objective Code2Hyp selection",
        "",
        "Objective:",
        "",
        f"`score = {f1_weight:.3f} * normalized(F1) + "
        f"{spearman_weight:.3f} * normalized(AST-distance Spearman)`",
        "",
        f"Best variant: `{selection['best']['variant']}`",
        "",
        "| Rank | Variant | Score | F1 mean | Spearman mean | Pareto | Note |",
        "|---:|---|---:|---:|---:|---|---|",
    ]
    for rank, row in enumerate(selection["ranked"], start=1):
        note = "best" if rank == 1 else ""
        if row["pareto_frontier"]:
            note = f"{note}; pareto".strip("; ")
        lines.append(
            "| "
            f"{rank} | "
            f"{row['variant']} | "
            f"{float(row['multi_objective_score']):.4f} | "
            f"{float(row['validation_f1_mean']):.4f} | "
            f"{float(row['validation_structural_spearman_mean']):+.4f} | "
            f"{'yes' if row['pareto_frontier'] else 'no'} | "
            f"{note} |"
        )
    lines.extend(
        [
            "",
            "Interpretation boundary:",
            "",
            "This is a selection diagnostic over already completed validation runs. "
            "It does not create a new trained model and must not be reported as a "
            "confirmatory statistical result without a preregistered larger run.",
        ]
    )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a multi-objective Code2Hyp selection report.")
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--f1-weight", type=float, default=0.5)
    parser.add_argument("--spearman-weight", type=float, default=0.5)
    parser.add_argument(
        "--variants",
        type=str,
        default="",
        help="Optional comma-separated variant filter.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    result = json.loads(args.input.read_text(encoding="utf-8"))
    variant_filter = tuple(variant.strip() for variant in args.variants.split(",") if variant.strip()) or None
    markdown = build_multi_objective_markdown(
        result,
        f1_weight=args.f1_weight,
        spearman_weight=args.spearman_weight,
        variant_filter=variant_filter,
    )
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(markdown + "\n", encoding="utf-8")
        print(f"Wrote {args.output}")
    else:
        print(markdown)


if __name__ == "__main__":
    main()
