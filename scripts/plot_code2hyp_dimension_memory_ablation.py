from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot Code2Hyp structural-dimension memory ablation summary.")
    parser.add_argument("--input", type=Path, default=Path("reports/code2hyp_dimension_memory_ablation_pilot_5k_summary.csv"))
    parser.add_argument("--output-prefix", type=Path, default=Path("figures/code2hyp_dimension_memory_ablation_pilot_5k"))
    return parser.parse_args()


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def plot_dimension_ablation(input_path: Path, output_prefix: Path) -> tuple[Path, Path]:
    df = pd.read_csv(input_path)
    df = df.sort_values(["variant_label", "structural_dim"])

    sns.set_theme(style="ticks", context="paper", font_scale=1.05)
    palette = {
        "Code2Hyp B44": "#0072B2",
        "Euclidean B46": "#D55E00",
    }
    markers = {
        "Code2Hyp B44": "o",
        "Euclidean B46": "s",
    }

    panels = [
        ("validation_f1", "Target-subtoken F1", "higher is better"),
        ("validation_structural_spearman", "AST-distance Spearman", "higher is better"),
        ("validation_structural_normalized_stress", "Normalized stress", "lower is better"),
        ("parameter_memory_mib_float32", "Parameter memory proxy (MiB)", "float32 weights only"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(7.1, 5.2), constrained_layout=True)
    for ax, (metric, ylabel, subtitle) in zip(axes.ravel(), panels, strict=True):
        if metric == "parameter_memory_mib_float32":
            memory = (
                df.groupby("structural_dim", as_index=False)["parameter_memory_mib_float32"]
                .mean()
                .sort_values("structural_dim")
            )
            ax.plot(
                memory["structural_dim"],
                memory["parameter_memory_mib_float32"],
                color="#333333",
                marker="D",
                linewidth=1.8,
                markersize=4.5,
            )
        else:
            for label, group in df.groupby("variant_label", sort=False):
                x = group["structural_dim"].to_numpy()
                y = group[f"{metric}_mean"].to_numpy()
                yerr = group[f"{metric}_sd"].to_numpy()
                ax.errorbar(
                    x,
                    y,
                    yerr=yerr,
                    label=label,
                    marker=markers.get(label, "o"),
                    linewidth=1.8,
                    markersize=4.5,
                    capsize=3,
                    color=palette.get(label),
                )
        ax.set_xscale("log", base=2)
        ax.set_xticks(sorted(df["structural_dim"].unique()))
        ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
        ax.set_xlabel("Structural channel dimension")
        ax.set_ylabel(ylabel)
        ax.set_title(subtitle, fontsize=9)
        ax.grid(True, axis="y", color="#DDDDDD", linewidth=0.6)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.04))
    fig.suptitle("Structural AST-path dimension sweep", fontsize=11, y=1.08)

    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    png_path = output_prefix.with_suffix(".png")
    pdf_path = output_prefix.with_suffix(".pdf")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return png_path, pdf_path


def main() -> None:
    args = parse_args()
    png_path, pdf_path = plot_dimension_ablation(_resolve(args.input), _resolve(args.output_prefix))
    print(f"Wrote {png_path}")
    print(f"Wrote {pdf_path}")


if __name__ == "__main__":
    main()
