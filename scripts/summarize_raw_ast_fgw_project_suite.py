from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean, median, stdev
from typing import Any, Iterable, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


RELATION_LABELS = {
    "endpoint": "Endpoint",
    "lca_depth": "LCA depth",
    "lca_anchored_product": "LCA product",
    "edge_jaccard": "Edge Jaccard",
    "path_length": "Path length",
    "multi_endpoint_lca_edge": "Endpoint+LCA+edge",
    "multi_endpoint_lca_edge_length": "Endpoint+LCA+edge+length",
}


def _project_from_sources(sources: Sequence[str]) -> str:
    if not sources:
        return "unknown"
    path = Path(str(sources[0]).rstrip("/"))
    if path.name:
        return path.name
    return "unknown"


def _relation_from_payload(path: Path, payload: dict[str, Any]) -> str:
    relation = payload.get("config", {}).get("structural_relation")
    if relation:
        return str(relation)
    stem = path.stem
    for known_relation in RELATION_LABELS:
        if known_relation in stem:
            return known_relation
    return stem


def _float(payload: dict[str, Any], key: str, fallback: float = 0.0) -> float:
    value = payload.get(key, fallback)
    if value is None:
        return fallback
    return float(value)


def _row_from_result(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    config = payload.get("config", {})
    spearman = payload.get("spearman_against_fgw", {})
    top1 = payload.get("retrieval_overlap_at_1") or {}
    top3 = payload.get("retrieval_overlap_at_3") or {}
    relation = _relation_from_payload(path, payload)
    project = _project_from_sources(tuple(config.get("sources", ())))
    gw = _float(spearman, "gw_structure")
    feature = _float(spearman, "ot_feature")
    centroid = _float(spearman, "centroid")
    return {
        "project": project,
        "relation": relation,
        "relation_label": RELATION_LABELS.get(relation, relation),
        "alpha": _float(config, "alpha"),
        "method_count": int(payload.get("method_count", 0)),
        "pair_count": int(payload.get("pair_count", 0)),
        "complete_pair_matrix": bool(payload.get("complete_pair_matrix", False)),
        "gw_structure_spearman": gw,
        "ot_feature_spearman": feature,
        "centroid_spearman": centroid,
        "gw_minus_feature_spearman": gw - feature,
        "gw_minus_centroid_spearman": gw - centroid,
        "gw_structure_top1": _float(top1, "gw_structure"),
        "ot_feature_top1": _float(top1, "ot_feature"),
        "centroid_top1": _float(top1, "centroid"),
        "gw_structure_top3": _float(top3, "gw_structure"),
        "ot_feature_top3": _float(top3, "ot_feature"),
        "centroid_top3": _float(top3, "centroid"),
        "source": str(path),
    }


def _bootstrap_mean_ci(
    values: Sequence[float],
    *,
    iterations: int = 2000,
    seed: int = 1729,
    alpha: float = 0.05,
) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    if len(values) == 1:
        return float(values[0]), float(values[0])
    rng = random.Random(seed)
    estimates: list[float] = []
    n_values = len(values)
    for _ in range(iterations):
        sample = [values[rng.randrange(n_values)] for _ in range(n_values)]
        estimates.append(float(mean(sample)))
    estimates.sort()
    lower_index = max(0, min(len(estimates) - 1, round((alpha / 2) * (len(estimates) - 1))))
    upper_index = max(0, min(len(estimates) - 1, round((1 - alpha / 2) * (len(estimates) - 1))))
    return estimates[lower_index], estimates[upper_index]


def _summarize(values: Sequence[float], *, seed: int) -> dict[str, float]:
    if not values:
        return {
            "mean": 0.0,
            "median": 0.0,
            "std": 0.0,
            "min": 0.0,
            "max": 0.0,
            "ci95_low": 0.0,
            "ci95_high": 0.0,
        }
    ci_low, ci_high = _bootstrap_mean_ci(values, seed=seed)
    return {
        "mean": float(mean(values)),
        "median": float(median(values)),
        "std": float(stdev(values)) if len(values) > 1 else 0.0,
        "min": float(min(values)),
        "max": float(max(values)),
        "ci95_low": float(ci_low),
        "ci95_high": float(ci_high),
    }


def _relation_summaries(rows: Sequence[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["relation"]].append(row)
    summaries: dict[str, dict[str, Any]] = {}
    metrics = (
        "gw_structure_spearman",
        "ot_feature_spearman",
        "centroid_spearman",
        "gw_minus_feature_spearman",
        "gw_minus_centroid_spearman",
        "gw_structure_top1",
        "ot_feature_top1",
        "centroid_top1",
    )
    for relation, relation_rows in sorted(grouped.items()):
        summary: dict[str, Any] = {
            "relation": relation,
            "relation_label": RELATION_LABELS.get(relation, relation),
            "project_count": len({str(row["project"]) for row in relation_rows}),
            "run_count": len(relation_rows),
            "method_count_mean": float(mean(float(row["method_count"]) for row in relation_rows)),
            "pair_count_mean": float(mean(float(row["pair_count"]) for row in relation_rows)),
        }
        for metric_index, metric in enumerate(metrics):
            metric_summary = _summarize(
                [float(row[metric]) for row in relation_rows],
                seed=1729 + metric_index,
            )
            for statistic, value in metric_summary.items():
                summary[f"{metric}_{statistic}"] = value
        summaries[relation] = summary
    return summaries


def _write_csv(rows: Iterable[dict[str, Any]], output_path: Path, fieldnames: Sequence[str]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _write_markdown(rows: Sequence[dict[str, Any]], summaries: dict[str, dict[str, Any]], output_path: Path) -> None:
    lines = [
        "# Raw-AST FGW project-suite summary",
        "",
        "The suite evaluates how different raw-AST path relations approximate the same FGW teacher target across Java projects.",
        "Project-level rows are treated as independent robustness checks; confidence intervals are bootstrap intervals over project runs, not over method pairs.",
        "",
        f"- Runs: `{len(rows)}`",
        f"- Projects: `{len({row['project'] for row in rows})}`",
        f"- Relations: `{len(summaries)}`",
        "",
        "## Relation-level summary",
        "",
        "| Relation | Projects | GW Spearman mean | 95% CI | Feature OT mean | GW - feature | Top-1 GW | Top-1 feature |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for relation, summary in summaries.items():
        lines.append(
            "| "
            f"{summary['relation_label']} | "
            f"{summary['project_count']} | "
            f"{summary['gw_structure_spearman_mean']:.3f} | "
            f"[{summary['gw_structure_spearman_ci95_low']:.3f}, {summary['gw_structure_spearman_ci95_high']:.3f}] | "
            f"{summary['ot_feature_spearman_mean']:.3f} | "
            f"{summary['gw_minus_feature_spearman_mean']:.3f} | "
            f"{summary['gw_structure_top1_mean']:.3f} | "
            f"{summary['ot_feature_top1_mean']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation guardrails",
            "",
            "- FGW is used as a geometry teacher target, not as a downstream functional-quality metric.",
            "- A strong relation is one whose structure-only GW distances align with its own FGW target across projects.",
            "- Cross-relation and project-level variation must be reported; a relation should not be described as universally superior unless it is stable across these controls.",
            "",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def _plot(rows: Sequence[dict[str, Any]], output_stem: Path) -> tuple[Path, Path]:
    df = pd.DataFrame(rows)
    relation_order = [
        label
        for relation, label in RELATION_LABELS.items()
        if label in set(df["relation_label"])
    ]
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.05)
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.6), sharex=False)
    palette = {
        "gw_structure_spearman": "#2f5597",
        "ot_feature_spearman": "#70ad47",
        "centroid_spearman": "#a5a5a5",
    }
    long_df = df.melt(
        id_vars=("project", "relation_label"),
        value_vars=("gw_structure_spearman", "ot_feature_spearman", "centroid_spearman"),
        var_name="approximation",
        value_name="spearman_vs_fgw",
    )
    long_df["approximation_label"] = long_df["approximation"].map(
        {
            "gw_structure_spearman": "Structural GW",
            "ot_feature_spearman": "Feature OT",
            "centroid_spearman": "Centroid proxy",
        }
    )
    sns.pointplot(
        data=long_df,
        x="relation_label",
        y="spearman_vs_fgw",
        hue="approximation_label",
        order=relation_order,
        errorbar=("ci", 95),
        dodge=0.32,
        markers="o",
        palette={
            "Structural GW": palette["gw_structure_spearman"],
            "Feature OT": palette["ot_feature_spearman"],
            "Centroid proxy": palette["centroid_spearman"],
        },
        ax=axes[0],
    )
    axes[0].set_title("Agreement with the relation-specific FGW teacher")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("Spearman correlation")
    axes[0].set_ylim(-0.1, 1.05)
    axes[0].tick_params(axis="x", rotation=18)
    axes[0].legend_.remove()

    sns.stripplot(
        data=df,
        x="relation_label",
        y="gw_minus_feature_spearman",
        order=relation_order,
        color="#2f5597",
        size=6,
        jitter=0.16,
        ax=axes[1],
    )
    sns.pointplot(
        data=df,
        x="relation_label",
        y="gw_minus_feature_spearman",
        order=relation_order,
        errorbar=("ci", 95),
        color="#1f1f1f",
        markers="_",
        linestyles="none",
        ax=axes[1],
    )
    axes[1].axhline(0.0, color="#7f7f7f", linewidth=1.0, linestyle=":")
    axes[1].set_title("Structure-only advantage over feature OT")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("Spearman difference")
    axes[1].tick_params(axis="x", rotation=18)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0.12, 1, 1))

    png_path = output_stem.with_suffix(".png")
    pdf_path = output_stem.with_suffix(".pdf")
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return png_path, pdf_path


def summarize_project_suite(input_paths: Sequence[Path], output_stem: Path) -> dict[str, Path]:
    rows = [_row_from_result(path) for path in input_paths]
    if not rows:
        raise ValueError("at least one raw-AST FGW result is required")
    rows = sorted(rows, key=lambda row: (str(row["relation"]), str(row["project"]), str(row["source"])))
    summaries = _relation_summaries(rows)

    rows_csv = output_stem.with_name(output_stem.name + "_rows.csv")
    summary_csv = output_stem.with_name(output_stem.name + "_relations.csv")
    json_path = output_stem.with_suffix(".json")
    md_path = output_stem.with_suffix(".md")
    png_path, pdf_path = _plot(rows, output_stem)

    _write_csv(rows, rows_csv, tuple(rows[0].keys()))
    _write_csv(summaries.values(), summary_csv, tuple(next(iter(summaries.values())).keys()))
    json_path.write_text(
        json.dumps({"rows": rows, "relations": summaries}, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    _write_markdown(rows, summaries, md_path)
    return {
        "rows_csv": rows_csv,
        "relations_csv": summary_csv,
        "json": json_path,
        "md": md_path,
        "png": png_path,
        "pdf": pdf_path,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize a multi-project raw-AST FGW benchmark suite.")
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--output-stem", type=Path, default=Path("figures/raw_ast_fgw_project_suite"))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    written = summarize_project_suite(tuple(args.input), args.output_stem)
    print(json.dumps({key: str(value) for key, value in written.items()}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
