from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


APPROXIMATION_LABELS = {
    "centroid": "Centroid proxy",
    "gw_structure": "Structural GW",
    "ot_feature": "Feature OT",
}


RELATION_LABELS = {
    "endpoint": "Endpoint",
    "lca_depth": "LCA depth",
    "lca_anchored_product": "LCA product",
    "edge_jaccard": "Edge Jaccard",
    "path_length": "Path length",
    "multi_endpoint_lca_edge": "Endpoint+\nLCA+edge",
    "multi_endpoint_lca_edge_length": "Endpoint+\nLCA+edge+length",
}


def _relation_from_payload(path: Path, payload: dict[str, Any]) -> str:
    relation = payload.get("config", {}).get("structural_relation")
    if relation:
        return str(relation)
    stem = path.stem
    for known in RELATION_LABELS:
        if known in stem:
            return known
    if "alpha0p75" in stem and "32methods" in stem:
        return "endpoint"
    return stem


def _rows_from_result(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    relation = _relation_from_payload(path, payload)
    rows: list[dict[str, Any]] = []
    spearman = payload.get("spearman_against_fgw", {})
    top1 = payload.get("retrieval_overlap_at_1") or {}
    for approximation in ("centroid", "gw_structure", "ot_feature"):
        rows.append(
            {
                "relation": relation,
                "relation_label": RELATION_LABELS.get(relation, relation),
                "approximation": approximation,
                "approximation_label": APPROXIMATION_LABELS[approximation],
                "spearman_vs_fgw": float(spearman.get(approximation, 0.0)),
                "top1_overlap": float(top1.get(approximation, 0.0)),
                "method_count": int(payload.get("method_count", 0)),
                "pair_count": int(payload.get("pair_count", 0)),
                "source": str(path),
            }
        )
    return rows


def _write_summary_csv(rows: Sequence[dict[str, Any]], csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = (
        "relation",
        "relation_label",
        "approximation",
        "approximation_label",
        "spearman_vs_fgw",
        "top1_overlap",
        "method_count",
        "pair_count",
        "source",
    )
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_relation_ablation(input_paths: Sequence[Path], output_stem: Path) -> dict[str, Path]:
    rows: list[dict[str, Any]] = []
    for path in input_paths:
        rows.extend(_rows_from_result(path))
    if not rows:
        raise ValueError("at least one relation-ablation result is required")

    df = pd.DataFrame(rows)
    relation_order = [
        label
        for relation, label in RELATION_LABELS.items()
        if label in set(df["relation_label"])
    ]
    approximation_order = [APPROXIMATION_LABELS[key] for key in ("gw_structure", "ot_feature", "centroid")]

    sns.set_theme(style="whitegrid", context="paper", font_scale=1.05)
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.8), sharey=False)
    palette = {
        "Structural GW": "#2f5597",
        "Feature OT": "#70ad47",
        "Centroid proxy": "#a5a5a5",
    }

    sns.barplot(
        data=df,
        x="relation_label",
        y="spearman_vs_fgw",
        hue="approximation_label",
        order=relation_order,
        hue_order=approximation_order,
        palette=palette,
        ax=axes[0],
    )
    axes[0].set_title("Rank agreement with FGW")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("Spearman correlation")
    axes[0].set_ylim(0.0, 1.0)
    axes[0].tick_params(axis="x", rotation=18)

    sns.barplot(
        data=df,
        x="relation_label",
        y="top1_overlap",
        hue="approximation_label",
        order=relation_order,
        hue_order=approximation_order,
        palette=palette,
        ax=axes[1],
    )
    axes[1].set_title("Nearest-neighbor agreement with FGW")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("Top-1 overlap")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].tick_params(axis="x", rotation=18)
    axes[1].get_legend().remove()

    handles, labels = axes[0].get_legend_handles_labels()
    axes[0].legend_.remove()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0.10, 1, 1))

    output_stem.parent.mkdir(parents=True, exist_ok=True)
    png_path = output_stem.with_suffix(".png")
    pdf_path = output_stem.with_suffix(".pdf")
    csv_path = output_stem.with_name(output_stem.name + "_summary.csv")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    _write_summary_csv(rows, csv_path)
    return {"png": png_path, "pdf": pdf_path, "csv": csv_path}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot raw-AST FGW relation-ablation results.")
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--output-stem", type=Path, default=Path("figures/raw_ast_fgw_relation_ablation"))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    written = plot_relation_ablation(tuple(args.input), args.output_stem)
    print(json.dumps({key: str(value) for key, value in written.items()}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
