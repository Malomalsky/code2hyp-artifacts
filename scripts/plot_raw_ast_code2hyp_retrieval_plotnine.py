from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib-cache"))
os.environ.setdefault("XDG_CACHE_HOME", str(PROJECT_ROOT / ".matplotlib-cache"))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from plotnine import (
    aes,
    element_text,
    facet_wrap,
    geom_col,
    ggplot,
    labs,
    scale_fill_manual,
    scale_y_continuous,
    theme,
    theme_bw,
)


DEFAULT_INPUTS = (
    ("Euclidean", PROJECT_ROOT / "outputs/raw_ast_code2hyp_python_project_euclidean.json"),
    ("Poincare", PROJECT_ROOT / "outputs/raw_ast_code2hyp_python_project_poincare.json"),
    ("Poincare near-zero", PROJECT_ROOT / "outputs/raw_ast_code2hyp_python_project_poincare_nearzero.json"),
)
DEFAULT_OUTPUT_PREFIX = PROJECT_ROOT / "figures/raw_ast_code2hyp_retrieval_plotnine"

METRIC_LABELS = {
    "mrr": "MRR",
    "recall_at_1": "Recall@1",
    "margin_mean": "Mean margin",
    "margin_min": "Minimum margin",
}
METRIC_ORDER = (
    "MRR",
    "Recall@1",
    "Mean margin",
    "Minimum margin",
)
VARIANT_COLORS = {
    "Euclidean": "#5B5B5B",
    "Poincare": "#006D77",
    "Poincare near-zero": "#A23E48",
}


def build_metric_rows(named_payloads: Sequence[tuple[str, dict[str, Any]]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for variant, payload in named_payloads:
        config = payload.get("config", {})
        metrics = payload.get("metrics", {})
        for metric_key, metric_label in METRIC_LABELS.items():
            if metric_key not in metrics:
                continue
            rows.append(
                {
                    "variant": variant,
                    "metric": metric_label,
                    "value": float(metrics[metric_key]),
                    "language": str(config.get("language", "")),
                    "geometry": str(config.get("geometry", "")),
                    "curvature": float(config.get("curvature", 1.0)),
                    "dim": int(config.get("dim", 0)),
                    "item_count": int(payload.get("item_count", 0)),
                    "vocab_size": int(payload.get("vocab_size", 0)),
                }
            )
    return rows


def plot_retrieval_results(
    *,
    inputs: Sequence[tuple[str, Path]],
    output_prefix: Path,
) -> None:
    payloads = tuple((label, _load_json(path)) for label, path in inputs)
    rows = build_metric_rows(payloads)
    if not rows:
        raise ValueError("no supported retrieval metrics found in input JSON files")
    frame = pd.DataFrame(rows)
    frame["metric"] = pd.Categorical(frame["metric"], categories=METRIC_ORDER, ordered=True)
    frame["variant"] = pd.Categorical(frame["variant"], categories=[label for label, _ in inputs], ordered=True)

    plot = (
        ggplot(frame, aes(x="variant", y="value", fill="variant"))
        + geom_col(width=0.68, color="#262626", size=0.25)
        + facet_wrap("~ metric", scales="free_y", ncol=2)
        + scale_fill_manual(values={key: VARIANT_COLORS.get(key, "#4C78A8") for key, _ in inputs})
        + scale_y_continuous(expand=(0.02, 0.04))
        + labs(
            title="Raw-AST Code2Hyp structural retrieval diagnostics",
            x="Model/control",
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


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot raw-AST Code2Hyp retrieval JSON files using plotnine.")
    parser.add_argument(
        "--input",
        action="append",
        nargs=2,
        metavar=("LABEL", "PATH"),
        help="Variant label and retrieval JSON path. Can be supplied multiple times.",
    )
    parser.add_argument("--output-prefix", type=Path, default=DEFAULT_OUTPUT_PREFIX)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    inputs = tuple((label, Path(path)) for label, path in args.input) if args.input else DEFAULT_INPUTS
    plot_retrieval_results(inputs=inputs, output_prefix=args.output_prefix)
    print(f"Wrote {args.output_prefix.with_suffix('.png')}")
    print(f"Wrote {args.output_prefix.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
