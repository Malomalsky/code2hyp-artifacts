from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib-cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


OKABE_ITO = {
    "blue": "#0072B2",
    "orange": "#E69F00",
    "green": "#009E73",
    "vermillion": "#D55E00",
}


def _read_summary(path: Path) -> list[dict[str, float]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = []
        for record in csv.DictReader(handle):
            rows.append(
                {
                    "alpha": float(record["ollivier_idleness"]),
                    "mean": float(record["ollivier_mean_mean"]),
                    "mean_std": float(record["ollivier_mean_std"]),
                    "negative": float(record["ollivier_negative_mass_mean"]),
                    "near_zero": float(record["ollivier_near_zero_mass_mean"]),
                    "positive": float(record["ollivier_positive_mass_mean"]),
                }
            )
    return sorted(rows, key=lambda item: item["alpha"])


def plot_ollivier_sensitivity(summary_csv: Path, output_stem: Path) -> None:
    rows = _read_summary(summary_csv)
    alphas = [row["alpha"] for row in rows]

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 9,
            "legend.fontsize": 7,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.1))
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.24, top=0.88, wspace=0.28)

    axes[0].errorbar(
        alphas,
        [row["mean"] for row in rows],
        yerr=[row["mean_std"] for row in rows],
        color=OKABE_ITO["blue"],
        marker="o",
        linewidth=1.5,
        capsize=3,
    )
    axes[0].axhline(0.0, color="#666666", linewidth=0.8, linestyle="--")
    axes[0].set_title("A. Mean Ollivier-Ricci curvature")
    axes[0].set_xlabel("Idleness parameter")
    axes[0].set_ylabel("Mean curvature")
    axes[0].spines["top"].set_visible(False)
    axes[0].spines["right"].set_visible(False)

    mass_series = [
        ("Negative", "negative", OKABE_ITO["vermillion"], "o"),
        ("Near-zero", "near_zero", OKABE_ITO["orange"], "s"),
        ("Positive", "positive", OKABE_ITO["green"], "^"),
    ]
    for label, key, color, marker in mass_series:
        axes[1].plot(
            alphas,
            [row[key] for row in rows],
            label=label,
            color=color,
            marker=marker,
            linewidth=1.5,
        )
    axes[1].set_title("B. Edge curvature regimes")
    axes[1].set_xlabel("Idleness parameter")
    axes[1].set_ylabel("Mean edge fraction")
    axes[1].set_ylim(-0.03, 1.03)
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.75, 0.02),
        ncol=3,
    )
    axes[1].spines["top"].set_visible(False)
    axes[1].spines["right"].set_visible(False)

    for axis in axes:
        axis.set_xticks(alphas)
        axis.grid(axis="y", color="#DDDDDD", linewidth=0.5)

    output_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_stem.with_suffix(".pdf"))
    fig.savefig(output_stem.with_suffix(".png"), dpi=300)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot Ollivier-Ricci idleness sensitivity from summary CSV."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("reports/ollivier_idleness_sensitivity.csv"),
    )
    parser.add_argument(
        "--output-stem",
        type=Path,
        default=Path("figures/ollivier_idleness_sensitivity"),
    )
    args = parser.parse_args()
    plot_ollivier_sensitivity(args.input, args.output_stem)
    print(f"wrote {args.output_stem.with_suffix('.pdf')}")
    print(f"wrote {args.output_stem.with_suffix('.png')}")


if __name__ == "__main__":
    main()
