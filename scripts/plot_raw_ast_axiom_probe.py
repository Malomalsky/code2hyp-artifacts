from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


METRIC_LABELS = {
    "eval_length_spearman_mean": "Path length rank correlation",
    "eval_lca_depth_spearman_mean": "Gromov product vs. LCA depth",
    "eval_lca_radial_depth_spearman_mean": "Radial LCA depth correlation",
}

METRIC_SD_KEYS = {
    "eval_length_spearman_mean": "eval_length_spearman_sd",
    "eval_lca_depth_spearman_mean": "eval_lca_depth_spearman_sd",
    "eval_lca_radial_depth_spearman_mean": "eval_lca_radial_depth_spearman_sd",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot raw-AST axiom-probe geometry diagnostics.")
    parser.add_argument("--input", type=Path, default=Path("outputs/raw_ast_axiom_probe_100files_dims2_4_8_3seeds_depth.json"))
    parser.add_argument("--output-prefix", type=Path, default=Path("figures/raw_ast_axiom_probe_geometry"))
    return parser


def _summary_frame(input_path: Path) -> pd.DataFrame:
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    rows: list[dict[str, float | int | str]] = []
    for item in payload["summary"]:
        for key, label in METRIC_LABELS.items():
            rows.append(
                {
                    "Geometry": str(item["geometry"]).capitalize(),
                    "Dimension": int(item["dim"]),
                    "Metric": label,
                    "Value": float(item[key]),
                    "SD": float(item.get(METRIC_SD_KEYS[key], 0.0)),
                }
            )
    return pd.DataFrame(rows)


def plot_axiom_probe(input_path: Path, output_prefix: Path) -> None:
    frame = _summary_frame(input_path)
    sns.set_theme(style="ticks", context="paper", font_scale=1.0)
    palette = {"Poincare": "#0072B2", "Euclidean": "#D55E00"}

    fig, axes = plt.subplots(1, 3, figsize=(9.2, 2.8), sharex=True)
    for ax, metric in zip(axes, METRIC_LABELS.values(), strict=True):
        subset = frame[frame["Metric"] == metric]
        for geometry, marker in (("Euclidean", "o"), ("Poincare", "X")):
            series = subset[subset["Geometry"] == geometry].sort_values("Dimension")
            ax.errorbar(
                series["Dimension"],
                series["Value"],
                yerr=series["SD"],
                label=geometry,
                marker=marker,
                color=palette[geometry],
                linewidth=1.8,
                markersize=5.5,
                capsize=3,
            )
        ax.set_title(metric, fontsize=9)
        ax.set_xlabel("Embedding dimension")
        ax.set_ylabel("Spearman rho")
        ax.set_xticks(sorted(frame["Dimension"].unique()))
        ax.set_ylim(-0.02, max(0.55, float(frame["Value"].max()) + 0.05))
        ax.grid(axis="y", color="#d9d9d9", linewidth=0.6)
        sns.despine(ax=ax)

    handles, labels = axes[0].get_legend_handles_labels()
    for ax in axes:
        legend = ax.get_legend()
        if legend is not None:
            legend.remove()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0.0, 0.08, 1.0, 1.0))

    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_prefix.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(output_prefix.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = build_parser().parse_args()
    plot_axiom_probe(args.input, args.output_prefix)
    print(f"wrote {args.output_prefix.with_suffix('.png')}")
    print(f"wrote {args.output_prefix.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
