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
from plotnine import aes, element_text, facet_wrap, geom_col, ggplot, labs, scale_fill_manual, scale_y_continuous, theme, theme_bw


METRIC_LABELS = {
    "validation_f1": "Oracle-top-k F1",
    "validation_fixed_top3_f1": "Fixed-top-3 F1",
    "validation_precision": "Precision",
    "validation_recall": "Recall",
}
METRIC_ORDER = ("Oracle-top-k F1", "Fixed-top-3 F1", "Precision", "Recall")
VARIANT_COLORS = {
    "euclidean": "#4D4D4D",
    "poincare_near_zero": "#7A9E9F",
    "poincare": "#006D77",
}


def build_supervised_metric_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for run in payload.get("runs", []):
        variant = str(run["variant"])
        for metric_key, metric_label in METRIC_LABELS.items():
            if metric_key not in run:
                continue
            rows.append(
                {
                    "variant": variant,
                    "metric": metric_label,
                    "value": float(run[metric_key]),
                    "geometry": str(run.get("geometry", "")),
                    "curvature": float(run.get("curvature", 1.0)),
                    "parameter_count": int(run.get("parameter_count", 0)),
                }
            )
    return rows


def plot_supervised_results(*, input_path: Path, output_prefix: Path) -> None:
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    rows = build_supervised_metric_rows(payload)
    if not rows:
        raise ValueError("no supported supervised metrics found")
    frame = pd.DataFrame(rows)
    variant_order = [str(run["variant"]) for run in payload.get("runs", [])]
    frame["variant"] = pd.Categorical(frame["variant"], categories=variant_order, ordered=True)
    frame["metric"] = pd.Categorical(frame["metric"], categories=METRIC_ORDER, ordered=True)

    plot = (
        ggplot(frame, aes(x="variant", y="value", fill="variant"))
        + geom_col(width=0.68, color="#262626", size=0.25)
        + facet_wrap("~ metric", scales="free_y", ncol=2)
        + scale_fill_manual(values={variant: VARIANT_COLORS.get(variant, "#4C78A8") for variant in variant_order})
        + scale_y_continuous(expand=(0.02, 0.04))
        + labs(
            title="Code2Hyp supervised method-name subtoken prediction",
            x="Matched model/control",
            y="Metric value",
            fill="Variant",
        )
        + theme_bw(base_size=9)
        + theme(
            figure_size=(7.2, 5.0),
            legend_position="bottom",
            axis_text_x=element_text(rotation=25, ha="right"),
        )
    )
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    plot.save(output_prefix.with_suffix(".png"), dpi=300, verbose=False)
    plot.save(output_prefix.with_suffix(".pdf"), verbose=False)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot code2vec-compatible Code2Hyp supervised JSON using plotnine.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, default=PROJECT_ROOT / "figures/code2hyp_supervised_c2s")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    plot_supervised_results(input_path=args.input, output_prefix=args.output_prefix)
    print(f"Wrote {args.output_prefix.with_suffix('.png')}")
    print(f"Wrote {args.output_prefix.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
