from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib-cache"))
os.environ.setdefault("XDG_CACHE_HOME", str(PROJECT_ROOT / ".matplotlib-cache"))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from plotnine import (
    aes,
    coord_flip,
    element_text,
    facet_wrap,
    geom_errorbar,
    geom_hline,
    geom_point,
    geom_text,
    ggplot,
    labs,
    scale_color_manual,
    theme,
    theme_bw,
)


CONTRAST_ORDER = (
    "LCA-product measure - single-point measure",
    "LCA-product centroid - single-point centroid",
    "LCA-product measure - LCA-product centroid",
    "single-point measure - single-point centroid",
)


def plot_confirmatory_benchmark(*, input_path: Path, output_prefix: Path) -> tuple[Path, Path]:
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    frame = _contrast_frame(payload)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    png_path = output_prefix.with_suffix(".png")
    pdf_path = output_prefix.with_suffix(".pdf")
    plot = _contrast_plot(frame)
    plot.save(png_path, dpi=300, verbose=False)
    plot.save(pdf_path, verbose=False)
    return png_path, pdf_path


def _contrast_frame(payload: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for row in payload.get("paired_contrasts", []):
        ci_low, ci_high = row["bootstrap_ci"]["delta_mrr"]
        rows.append(
            {
                "dataset": _dataset_label(row["dataset"]),
                "contrast": row["label"],
                "contrast_ordered": row["label"],
                "delta_mrr": float(row["delta_mrr"]),
                "ci_low": float(ci_low),
                "ci_high": float(ci_high),
                "n_label": f"n={int(row['paired_query_count'])}",
                "direction": "positive" if float(row["delta_mrr"]) >= 0.0 else "negative",
            }
        )
    if not rows:
        raise ValueError("confirmatory benchmark plot requires at least one paired contrast")
    frame = pd.DataFrame(rows)
    frame["contrast_ordered"] = pd.Categorical(frame["contrast_ordered"], categories=CONTRAST_ORDER, ordered=True)
    return frame


def _contrast_plot(frame: pd.DataFrame) -> ggplot:
    return (
        ggplot(frame, aes(x="contrast_ordered", y="delta_mrr", color="direction"))
        + geom_hline(yintercept=0.0, color="#222222", size=0.45)
        + geom_errorbar(aes(ymin="ci_low", ymax="ci_high"), width=0.18, size=0.7)
        + geom_point(size=2.5)
        + geom_text(aes(label="n_label"), nudge_y=0.008, size=7.0, color="#222222", va="bottom")
        + facet_wrap("~ dataset", ncol=1)
        + coord_flip()
        + scale_color_manual(values={"positive": "#0072B2", "negative": "#D55E00"})
        + labs(
            title="Confirmatory Code2Hyp representation contrasts",
            subtitle="Paired-query MRR deltas with deterministic 95% bootstrap confidence intervals",
            x="Representation contrast",
            y="Delta MRR",
            color="Direction",
        )
        + theme_bw(base_size=8.5)
        + theme(
            figure_size=(8.0, 5.2),
            legend_position="none",
            axis_text_y=element_text(size=7.2),
            strip_text=element_text(weight="bold"),
            plot_title=element_text(weight="bold"),
        )
    )


def _dataset_label(value: str) -> str:
    labels = {
        "bugnet_python": "BugNet Python",
        "dta_zenodo": "DTA Zenodo",
    }
    return labels.get(value, value)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot final Code2Hyp confirmatory benchmark paired contrasts.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    for output in plot_confirmatory_benchmark(input_path=args.input, output_prefix=args.output_prefix):
        print(f"Wrote {output}")


if __name__ == "__main__":
    main()
