#!/usr/bin/env python3
"""Summarize Code2Hyp multiview sensitivity to AST path sampling policy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


VARIANTS = (
    "code2hyp_multiview_selected",
    "code2hyp_multiview_no_lca_selected",
    "token_ast_selected",
)


def summarize(inputs: list[tuple[str, Path]]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for label, path in inputs:
        payload = json.loads(path.read_text(encoding="utf-8"))
        lca_view = payload.get("lca_view", "path_signature_plus_tokens")
        summaries = {(row["dataset"], row["variant"]): row for row in payload["cell_summaries"]}
        for dataset in sorted({dataset for dataset, _variant in summaries}):
            multiview = summaries[(dataset, "code2hyp_multiview_selected")]
            no_lca = summaries[(dataset, "code2hyp_multiview_no_lca_selected")]
            token_ast = summaries[(dataset, "token_ast_selected")]
            selected_weights = multiview.get("selected_weights_by_seed", {})
            lca_weights = [
                float(weights.get(lca_view, 0.0))
                for weights in selected_weights.values()
            ]
            rows.append(
                {
                    "label": label,
                    "path": str(path),
                    "dataset": dataset,
                    "path_selection_policy": payload.get("path_selection_policy", "preorder_first"),
                    "lca_view": lca_view,
                    "lca_selection_margin": payload.get("lca_selection_margin"),
                    "max_paths": int(payload.get("max_paths", 0)),
                    "multiview_mrr": float(multiview["mrr"]),
                    "no_lca_mrr": float(no_lca["mrr"]),
                    "token_ast_mrr": float(token_ast["mrr"]),
                    "multiview_minus_no_lca_mrr": float(multiview["mrr"] - no_lca["mrr"]),
                    "mean_selected_lca_weight": sum(lca_weights) / len(lca_weights) if lca_weights else 0.0,
                    "multiview_recall_at_5": float(multiview["recall_at_5"]),
                    "no_lca_recall_at_5": float(no_lca["recall_at_5"]),
                }
            )
    return {"inputs": [{"label": label, "path": str(path)} for label, path in inputs], "rows": rows}


def format_markdown(result: dict[str, Any]) -> str:
    lines = [
        "# Code2Hyp AST path-sampling sensitivity",
        "",
        "This diagnostic checks whether the LCA-path view in the final multiview kernel depends on the terminal-pair sampling policy.",
        "",
        "| Run | Dataset | Policy | LCA view | K | C2H-MV MRR | MV-noLCA MRR | Tok+AST MRR | C2H-MV - noLCA | Mean LCA weight |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in result["rows"]:
        lines.append(
            f"| {row['label']} | {row['dataset']} | {row['path_selection_policy']} | {row['lca_view']} | {row['max_paths']} | "
            f"{row['multiview_mrr']:.4f} | {row['no_lca_mrr']:.4f} | {row['token_ast_mrr']:.4f} | "
            f"{row['multiview_minus_no_lca_mrr']:+.4f} | {row['mean_selected_lca_weight']:.4f} |"
        )
    lines.extend(
        [
            "",
            "Interpretation: the earlier lexicalized LCA view is sensitive to terminal-pair sampling. This diagnostic motivates the final protocol, which separates the clean LCA-path signature from raw token evidence, uses LCA-depth-stratified sampling, and evaluates the LCA view inside a nested train-selected multiview grid.",
            "",
        ]
    )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", nargs=2, action="append", metavar=("LABEL", "PATH"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--json-output", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = summarize([(label, Path(path)) for label, path in args.input])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(format_markdown(result), encoding="utf-8")
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
