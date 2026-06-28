#!/usr/bin/env python3
"""Build reproducibility figures for the Code2Hyp artifact package.

The figures are generated only from the final confirmatory benchmark JSON.
No values are copied into the script by hand.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib-cache"))
os.environ.setdefault("XDG_CACHE_HOME", str(PROJECT_ROOT / ".matplotlib-cache"))

import pandas as pd
from PIL import Image
from plotnine import (
    aes,
    coord_flip,
    element_blank,
    element_line,
    element_rect,
    element_text,
    facet_grid,
    facet_wrap,
    geom_col,
    geom_errorbar,
    geom_errorbarh,
    geom_hline,
    geom_label,
    geom_line,
    geom_point,
    geom_rect,
    geom_segment,
    geom_text,
    geom_vline,
    ggplot,
    labs,
    position_dodge,
    scale_color_manual,
    scale_fill_manual,
    scale_linetype_manual,
    scale_x_continuous,
    scale_y_continuous,
    theme,
    theme_bw,
)

OUT_DIR = PROJECT_ROOT / "figures"
GRAYSCALE_DIR = PROJECT_ROOT / "build" / "grayscale_previews"
RESULT_JSON = PROJECT_ROOT / "outputs" / "final_confirmatory_representation_benchmark_2026-06-28.json"
SIMPLE_JSON = PROJECT_ROOT / "outputs" / "task_retrieval_simple_baselines_2026-06-28.json"
HYBRID_JSON = PROJECT_ROOT / "outputs" / "code2hyp_hybrid_task_retrieval_lca_kernel_nested_tokenast_margin001_2026-06-28.json"
HYBRID_CONTRAST_JSON = PROJECT_ROOT / "outputs" / "code2hyp_hybrid_task_level_contrasts_lca_kernel_nested_tokenast_margin001_2026-06-28.json"

FIGURE_STEMS = (
    "figure01_code2hyp_architecture",
    "figure02_main_results",
    "figure03_geometry_diagnostics",
    "figure04_distance_levels",
)

PALETTE = {
    "single-point + centroid": "#6B6B6B",
    "single-point + measure": "#0072B2",
    "LCA-product + centroid": "#D55E00",
    "LCA-product + measure": "#009E73",
}


def load_result() -> dict:
    if not RESULT_JSON.exists():
        raise FileNotFoundError(f"Missing benchmark JSON: {RESULT_JSON}")
    return json.loads(RESULT_JSON.read_text(encoding="utf-8"))


def load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing JSON artifact: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def base_theme():
    return (
        theme_bw(base_size=9, base_family="Times New Roman")
        + theme(
            figure_size=(7.0, 3.8),
            panel_grid_minor=element_blank(),
            panel_grid_major=element_line(color="#D9D9D9", size=0.35),
            plot_title=element_text(weight="bold", size=10),
            axis_title=element_text(size=8.5),
            axis_text=element_text(size=7.5),
            legend_title=element_blank(),
            legend_position="bottom",
            strip_background=element_rect(fill="#F2F2F2", color="#BDBDBD"),
            strip_text=element_text(size=8, weight="bold"),
        )
    )


def save_plot(plot, stem: str, *, width: float = 7.0, height: float = 3.8) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plot.save(OUT_DIR / f"{stem}.png", width=width, height=height, units="in", dpi=450, verbose=False)
    plot.save(OUT_DIR / f"{stem}.pdf", width=width, height=height, units="in", dpi=450, verbose=False)


def build_grayscale_previews() -> None:
    GRAYSCALE_DIR.mkdir(parents=True, exist_ok=True)
    for stem in FIGURE_STEMS:
        src = OUT_DIR / f"{stem}.png"
        if src.exists():
            Image.open(src).convert("L").save(GRAYSCALE_DIR / src.name)


def representation_label(path_object_mode: str, aggregation: str) -> str:
    path = "LCA-product" if path_object_mode == "lca_product" else "single-point"
    agg = "measure" if aggregation == "measure" else "centroid"
    return f"{path} + {agg}"


def dataset_label(dataset: str) -> str:
    return {
        "bugnet_python": "BugNet Python",
        "dta_zenodo": "DTA Zenodo",
    }.get(dataset, dataset)


def build_architecture_figure() -> None:
    boxes = pd.DataFrame(
        [
            {"x1": 0.02, "x2": 0.16, "y1": 0.42, "y2": 0.62, "label": "Python\nsource", "fill": "source"},
            {"x1": 0.21, "x2": 0.37, "y1": 0.42, "y2": 0.62, "label": "Raw AST\npaths", "fill": "ast"},
            {"x1": 0.43, "x2": 0.61, "y1": 0.42, "y2": 0.62, "label": "LCA-path object\n(LCA, start, end)", "fill": "proposed"},
            {"x1": 0.67, "x2": 0.84, "y1": 0.66, "y2": 0.86, "label": "Path measure\ntransport", "fill": "measure"},
            {"x1": 0.67, "x2": 0.84, "y1": 0.18, "y2": 0.38, "label": "Path, token,\nAST views", "fill": "views"},
            {"x1": 0.88, "x2": 0.99, "y1": 0.18, "y2": 0.38, "label": "Selected\nkernel", "fill": "retrieval"},
        ]
    )
    boxes["cx"] = (boxes["x1"] + boxes["x2"]) / 2
    boxes["cy"] = (boxes["y1"] + boxes["y2"]) / 2
    edges = pd.DataFrame(
        [
            {"x": 0.16, "xend": 0.21, "y": 0.52, "yend": 0.52},
            {"x": 0.37, "xend": 0.43, "y": 0.52, "yend": 0.52},
            {"x": 0.61, "xend": 0.67, "y": 0.54, "yend": 0.76},
            {"x": 0.61, "xend": 0.67, "y": 0.50, "yend": 0.28},
            {"x": 0.84, "xend": 0.88, "y": 0.28, "yend": 0.28},
        ]
    )
    notes = pd.DataFrame(
        [
            {"x": 0.76, "y": 0.93, "label": "Measure branch gives structural explanations by transport alignments."},
            {"x": 0.73, "y": 0.07, "label": "Retrieval branch uses a validation-selected mixture of LCA-path, token and AST views."},
        ]
    )
    plot = (
        ggplot()
        + geom_segment(edges, aes("x", "y", xend="xend", yend="yend"), color="#3A3A3A", size=0.55)
        + geom_rect(boxes, aes(xmin="x1", xmax="x2", ymin="y1", ymax="y2", fill="fill"), color="#3A3A3A", size=0.45)
        + geom_text(boxes, aes("cx", "cy", label="label"), size=8.0)
        + geom_text(notes, aes("x", "y", label="label"), size=7.3)
        + scale_fill_manual(
            values={
                "source": "#F4F4F4",
                "ast": "#E8F1F7",
                "proposed": "#FFF0E6",
                "measure": "#F6E8F7",
                "views": "#FFF6D9",
                "retrieval": "#EAF5EA",
            },
            guide=None,
        )
        + scale_x_continuous(limits=(0, 1), expand=(0, 0))
        + scale_y_continuous(limits=(0, 1), expand=(0, 0))
        + labs(title="Code2Hyp LCA-path and multiview retrieval pipeline")
        + theme_void_for_diagram()
    )
    save_plot(plot, "figure01_code2hyp_architecture", width=7.0, height=2.8)


def theme_void_for_diagram():
    return (
        theme_bw(base_size=9, base_family="Times New Roman")
        + theme(
            panel_grid=element_blank(),
            panel_border=element_blank(),
            axis_title=element_blank(),
            axis_text=element_blank(),
            axis_ticks=element_blank(),
            legend_position="none",
            plot_title=element_text(weight="bold", size=10),
        )
    )


def build_main_results_figure(result: dict) -> None:
    simple = load_json(SIMPLE_JSON)
    hybrid = load_json(HYBRID_JSON)
    rows: list[dict] = []
    for cell in simple["cell_summaries"]:
        if cell["baseline"] in {"random_expected", "token_bag", "ast_node_bag"}:
            method = {
                "random_expected": "Random",
                "token_bag": "Token",
                "ast_node_bag": "AST",
            }[cell["baseline"]]
            for metric, label in (("mrr", "MRR"), ("recall_at_5", "Recall@5")):
                rows.append(
                    {
                        "Dataset": dataset_label(cell["dataset"]),
                        "Method": method,
                        "Metric": label,
                        "value": cell[metric],
                    }
                )
    for cell in hybrid["cell_summaries"]:
        if cell["variant"] in {"code2hyp_multiview_no_lca_selected", "code2hyp_multiview_selected"}:
            method = "MV-noLCA" if cell["variant"] == "code2hyp_multiview_no_lca_selected" else "C2H-MV"
            for metric, label in (("mrr", "MRR"), ("recall_at_5", "Recall@5")):
                rows.append(
                    {
                        "Dataset": dataset_label(cell["dataset"]),
                        "Method": method,
                        "Metric": label,
                        "value": cell[metric],
                    }
                )
    df = pd.DataFrame(rows)
    method_order = [
        "Random",
        "AST",
        "Token",
        "MV-noLCA",
        "C2H-MV",
    ]
    df["Method"] = pd.Categorical(df["Method"], method_order)
    df["Metric"] = pd.Categorical(df["Metric"], ["MRR", "Recall@5"])
    method_palette = {
        "Random": "#9E9E9E",
        "AST": "#D55E00",
        "Token": "#0072B2",
        "MV-noLCA": "#CC79A7",
        "C2H-MV": "#009E73",
    }
    plot = (
        ggplot(df, aes("Method", "value", fill="Method"))
        + geom_col(width=0.72)
        + coord_flip()
        + facet_grid("Metric ~ Dataset")
        + scale_fill_manual(values=method_palette)
        + scale_y_continuous(limits=(0, 1.0))
        + labs(
            title="Task-level retrieval quality with and without the LCA-path view",
            x="",
            y="Metric value",
        )
        + base_theme()
        + theme(legend_position="none", figure_size=(7.0, 4.4))
    )
    save_plot(plot, "figure02_main_results", width=7.0, height=4.4)


def build_old_main_results_figure(result: dict) -> None:
    rows: list[dict] = []
    for cell in result["cell_summaries"]:
        rep = representation_label(cell["path_object_mode"], cell["method_aggregation"])
        for metric, label in (("mrr", "MRR"), ("recall_at_1", "Recall@1"), ("recall_at_5", "Recall@5")):
            lo, hi = cell["bootstrap_ci"][metric]
            rows.append(
                {
                    "Dataset": dataset_label(cell["dataset"]),
                    "Representation": rep,
                    "Metric": label,
                    "value": cell[metric],
                    "lo": lo,
                    "hi": hi,
                }
            )
    df = pd.DataFrame(rows)
    df["Representation"] = pd.Categorical(df["Representation"], list(PALETTE))
    df["Metric"] = pd.Categorical(df["Metric"], ["MRR", "Recall@1", "Recall@5"])
    plot = (
        ggplot(df, aes("Representation", "value", color="Representation"))
        + geom_point(position=position_dodge(width=0.45), size=2.1)
        + geom_errorbar(aes(ymin="lo", ymax="hi"), position=position_dodge(width=0.45), width=0.18, size=0.45)
        + coord_flip()
        + facet_grid("Metric ~ Dataset", scales="free_x")
        + scale_color_manual(values=PALETTE)
        + scale_y_continuous(limits=(0, 0.85))
        + labs(
            title="Retrieval quality by representation",
            x="",
            y="Metric value with bootstrap 95% CI",
        )
        + base_theme()
        + theme(legend_position="none", figure_size=(7.0, 4.7))
    )
    save_plot(plot, "figure02_main_results", width=7.0, height=4.7)


def build_effects_figure(result: dict) -> None:
    result = load_json(HYBRID_CONTRAST_JSON)
    rows: list[dict] = []
    label_map = {
        "Code2Hyp multiview selected - token bag": "multiview selected\nminus token bag",
        "Code2Hyp multiview selected - AST node bag": "multiview selected\nminus AST node bag",
        "Code2Hyp multiview selected - LCA path signature": "multiview selected\nminus path signature",
        "Code2Hyp multiview selected - multiview without LCA path view": "multiview selected\nminus no-LCA multiview",
        "Code2Hyp multiview selected - token+AST selected": "multiview selected\nminus token+AST selected",
    }
    for contrast in result["contrasts"]:
        lo, hi = contrast["task_bootstrap_ci"]["delta_mrr"]
        rows.append(
            {
                "Dataset": dataset_label(contrast["dataset"]),
                "Contrast": label_map.get(contrast["label"], contrast["label"]),
                "delta": contrast["delta_mrr"],
                "lo": lo,
                "hi": hi,
            }
        )
    df = pd.DataFrame(rows)
    order = [
        "multiview selected\nminus no-LCA multiview",
        "multiview selected\nminus token+AST selected",
        "multiview selected\nminus path signature",
        "multiview selected\nminus AST node bag",
        "multiview selected\nminus token bag",
    ]
    df["Contrast"] = pd.Categorical(df["Contrast"], order)
    plot = (
        ggplot(df, aes("delta", "Contrast", color="Dataset"))
        + geom_vline(xintercept=0, linetype="dashed", color="#444444", size=0.45)
        + geom_errorbarh(aes(xmin="lo", xmax="hi"), height=0.18, size=0.45)
        + geom_point(size=2.1)
        + facet_wrap("~ Dataset", ncol=1)
        + scale_color_manual(values={"BugNet Python": "#0072B2", "DTA Zenodo": "#D55E00"})
        + labs(
            title="Task-level paired effect of the multiview Code2Hyp kernel",
            x="Delta MRR with task-bootstrap 95% CI",
            y="",
        )
        + base_theme()
        + theme(legend_position="none", figure_size=(7.0, 4.2))
    )
    save_plot(plot, "figure03_geometry_diagnostics", width=7.0, height=4.2)


def build_rank_figure(result: dict) -> None:
    simple = load_json(SIMPLE_JSON)
    hybrid = load_json(HYBRID_JSON)
    rows = []
    for row in simple["query_rows"]:
        if row["baseline"] in {"token_bag", "ast_node_bag"}:
            rows.append(
                {
                    "Dataset": dataset_label(row["dataset"]),
                    "Representation": "Token bag" if row["baseline"] == "token_bag" else "AST node bag",
                    "rank": min(int(row["rank"]), 20),
                }
            )
    for row in hybrid["query_rows"]:
        if row["variant"] == "code2hyp_multiview_selected":
            rows.append(
                {
                    "Dataset": dataset_label(row["dataset"]),
                    "Representation": "Code2Hyp multiview",
                    "rank": min(int(row["rank"]), 20),
                }
            )
    df = pd.DataFrame(rows)
    grouped = []
    for (dataset, rep, rank), block in df.groupby(["Dataset", "Representation", "rank"], observed=True):
        grouped.append({"Dataset": dataset, "Representation": rep, "rank": rank, "count": len(block)})
    counts = pd.DataFrame(grouped)
    completed = []
    for (dataset, rep), block in counts.groupby(["Dataset", "Representation"], observed=True):
        total = int(block["count"].sum())
        for rank in range(1, 21):
            count = int(block.loc[block["rank"] <= rank, "count"].sum())
            completed.append({"Dataset": dataset, "Representation": rep, "rank": rank, "share": count / total})
    cdf = pd.DataFrame(completed)
    rank_palette = {
        "AST node bag": "#D55E00",
        "Token bag": "#0072B2",
        "Code2Hyp multiview": "#009E73",
    }
    cdf["Representation"] = pd.Categorical(cdf["Representation"], list(rank_palette))
    plot = (
        ggplot(cdf, aes("rank", "share", color="Representation", linetype="Representation"))
        + geom_line(size=0.75)
        + geom_point(size=0.9)
        + facet_wrap("~ Dataset")
        + scale_color_manual(values=rank_palette)
        + scale_linetype_manual(values=["dashdot", "dotted", "solid"])
        + scale_x_continuous(breaks=[1, 5, 10, 15, 20], limits=(1, 20))
        + scale_y_continuous(limits=(0, 1), breaks=[0.0, 0.25, 0.5, 0.75, 1.0])
        + labs(
            title="Cumulative rank of the first relevant program",
            x="Rank threshold",
            y="Share of queries with relevant method within threshold",
        )
        + base_theme()
        + theme(figure_size=(7.0, 3.9))
    )
    save_plot(plot, "figure04_distance_levels", width=7.0, height=3.9)


def build_summary_table(result: dict) -> pd.DataFrame:
    rows = []
    for cell in result["cell_summaries"]:
        rows.append(
            {
                "dataset": dataset_label(cell["dataset"]),
                "representation": representation_label(cell["path_object_mode"], cell["method_aggregation"]),
                "queries": cell["query_count"],
                "mrr": cell["mrr"],
                "mrr_lo": cell["bootstrap_ci"]["mrr"][0],
                "mrr_hi": cell["bootstrap_ci"]["mrr"][1],
                "r1": cell["recall_at_1"],
                "r5": cell["recall_at_5"],
                "mean_rank": cell["mean_rank"],
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grayscale-preview", action="store_true", help="also write grayscale preview PNGs")
    args = parser.parse_args()

    result = load_result()
    build_architecture_figure()
    build_main_results_figure(result)
    build_effects_figure(result)
    build_rank_figure(result)
    if args.grayscale_preview:
        build_grayscale_previews()

    print(f"Wrote figures to {OUT_DIR}")
    print(build_summary_table(result).to_string(index=False, float_format=lambda value: f"{value:.4f}"))


if __name__ == "__main__":
    main()
