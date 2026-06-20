from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from geometry_profile_research.code2hyp_reporting import summarize_pilot_runs


def build_structural_retrieval_markdown(inputs: tuple[tuple[str, Path], ...]) -> str:
    lines = [
        "# Code2Hyp structural retrieval diagnostics",
        "",
        "| Regime | Variant | n | F1 | Spearman | Overlap@1 | Overlap@3 |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    best_overlap_1: tuple[str, str, float] | None = None
    best_overlap_3: tuple[str, str, float] | None = None
    for regime, path in inputs:
        result = json.loads(path.read_text(encoding="utf-8"))
        for row in summarize_pilot_runs(result):
            variant = str(row["variant"])
            overlap_1 = float(row.get("validation_structural_neighbor_overlap_at_1_mean", 0.0))
            overlap_3 = float(row.get("validation_structural_neighbor_overlap_at_3_mean", 0.0))
            lines.append(
                "| "
                f"{regime} | "
                f"{variant} | "
                f"{int(row['n'])} | "
                f"{float(row.get('validation_f1_mean', 0.0)):.4f} | "
                f"{float(row.get('validation_structural_spearman_mean', 0.0)):+.4f} | "
                f"{overlap_1:.4f} | "
                f"{overlap_3:.4f} |"
            )
            if best_overlap_1 is None or overlap_1 > best_overlap_1[2]:
                best_overlap_1 = (regime, variant, overlap_1)
            if best_overlap_3 is None or overlap_3 > best_overlap_3[2]:
                best_overlap_3 = (regime, variant, overlap_3)
    lines.extend(
        [
            "",
            "Interpretation boundary:",
            "",
            "Overlap@k is a local structural diagnostic: for each AST path-context "
            "it checks whether the k nearest neighbors in the learned geometry fall "
            "inside the tie-tolerant set of k nearest AST neighbors. It complements "
            "global Spearman correlation and does not replace task F1.",
        ]
    )
    if best_overlap_1 is not None:
        lines.append(
            f"Best Overlap@1 variant: `{best_overlap_1[1]}` "
            f"({best_overlap_1[0]}, {best_overlap_1[2]:.4f})."
        )
    if best_overlap_3 is not None:
        lines.append(
            f"Best Overlap@3 variant: `{best_overlap_3[1]}` "
            f"({best_overlap_3[0]}, {best_overlap_3[2]:.4f})."
        )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize Code2Hyp structural retrieval diagnostics.")
    parser.add_argument(
        "--input",
        action="append",
        nargs=2,
        metavar=("REGIME", "PATH"),
        required=True,
        help="Regime label and pilot JSON path. Can be repeated.",
    )
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    inputs = tuple((label, Path(path)) for label, path in args.input)
    markdown = build_structural_retrieval_markdown(inputs)
    if args.output is None:
        print(markdown)
        return
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(markdown + "\n", encoding="utf-8")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
