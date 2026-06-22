from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]

METRIC_KEYS = (
    "validation_f1",
    "validation_structural_spearman",
    "validation_structural_normalized_stress",
    "validation_structural_neighbor_overlap_at_3",
)

VARIANT_LABELS = {
    "B36_code2hyp_product_frechet_neighbor": "Code2Hyp B36",
    "B44_code2hyp_context_transform_product_bias_frechet": "Code2Hyp B44",
    "B46_code2vec_context_transform_neighbor_control": "Euclidean B46",
    "B39_code2vec_context_transform_baseline": "Euclidean B39",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize a Code2Hyp structural-dimension memory ablation.")
    parser.add_argument("inputs", nargs="*", type=Path)
    parser.add_argument("--glob", default="outputs/code2hyp_dimension_memory_ablation_structdim*.json")
    parser.add_argument("--csv-output", type=Path, default=Path("reports/code2hyp_dimension_memory_ablation_summary.csv"))
    parser.add_argument("--md-output", type=Path, default=Path("reports/code2hyp_dimension_memory_ablation_summary.md"))
    parser.add_argument("--spearman-threshold", type=float, default=0.70)
    parser.add_argument("--stress-threshold", type=float, default=0.20)
    return parser.parse_args()


def load_results(paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        result = json.loads(path.read_text(encoding="utf-8"))
        training = result.get("training", {})
        ablation = result.get("dimension_memory_ablation", {})
        structural_dim = int(training.get("structural_dim", ablation.get("structural_dim")))
        token_dim = int(training.get("token_dim", ablation.get("token_dim_fixed", 32)))
        representation_dim = int(training.get("representation_dim", 2 * token_dim + structural_dim))
        activation_proxy_bytes = int(ablation.get("activation_proxy_bytes_per_example", 0))
        for run in result.get("runs", []):
            parameter_count = int(run["parameter_count"])
            row = {
                "source": str(path),
                "variant": run["variant"],
                "variant_label": VARIANT_LABELS.get(run["variant"], run["variant"]),
                "model_seed": int(run["model_seed"]),
                "token_dim": token_dim,
                "structural_dim": structural_dim,
                "representation_dim": representation_dim,
                "parameter_count": parameter_count,
                "parameter_memory_mib_float32": parameter_count * 4 / (1024 * 1024),
                "activation_proxy_kib_per_example_float32": activation_proxy_bytes / 1024,
            }
            for key in METRIC_KEYS:
                row[key] = float(run[key])
            rows.append(row)
    return rows


def _mean_sd(values: list[float]) -> tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    if len(values) == 1:
        return values[0], 0.0
    return mean(values), stdev(values)


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(int(row["structural_dim"]), str(row["variant"]))].append(row)

    summary: list[dict[str, Any]] = []
    for (structural_dim, variant), items in sorted(grouped.items()):
        out: dict[str, Any] = {
            "structural_dim": structural_dim,
            "variant": variant,
            "variant_label": VARIANT_LABELS.get(variant, variant),
            "n_seeds": len(items),
            "token_dim": int(items[0]["token_dim"]),
            "representation_dim": int(items[0]["representation_dim"]),
            "parameter_count": int(round(mean(float(item["parameter_count"]) for item in items))),
            "parameter_memory_mib_float32": mean(float(item["parameter_memory_mib_float32"]) for item in items),
            "activation_proxy_kib_per_example_float32": mean(
                float(item["activation_proxy_kib_per_example_float32"]) for item in items
            ),
        }
        for key in METRIC_KEYS:
            metric_mean, metric_sd = _mean_sd([float(item[key]) for item in items])
            out[f"{key}_mean"] = metric_mean
            out[f"{key}_sd"] = metric_sd
        summary.append(out)
    return summary


def write_csv(summary: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(summary[0].keys()) if summary else []
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary)


def _fmt_mean_sd(item: dict[str, Any], key: str) -> str:
    return f"{item[f'{key}_mean']:.4f} ± {item[f'{key}_sd']:.4f}"


def _find_min_dim(summary: list[dict[str, Any]], variant: str, metric: str, threshold: float, direction: str) -> dict[str, Any] | None:
    candidates = [item for item in summary if item["variant"] == variant]
    candidates.sort(key=lambda item: int(item["structural_dim"]))
    for item in candidates:
        value = float(item[f"{metric}_mean"])
        if direction == "ge" and value >= threshold:
            return item
        if direction == "le" and value <= threshold:
            return item
    return None


def build_markdown(summary: list[dict[str, Any]], spearman_threshold: float, stress_threshold: float) -> str:
    lines = [
        "# Code2Hyp structural-dimension and memory-proxy ablation",
        "",
        "## Scope",
        "",
        "This report tests dimension efficiency, not production peak memory. Parameter memory is estimated as",
        "`parameter_count * 4` bytes for float32 weights. The activation proxy is a single dense",
        "`max_contexts * representation_dim` context-representation tensor per example. It is useful for",
        "architecture comparison, but it must not be reported as measured RAM or VRAM.",
        "",
        "## Summary table",
        "",
        "| structural dim | variant | params | param MiB | repr dim | F1 | AST Spearman | stress | Overlap@3 | n |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in summary:
        lines.append(
            "| "
            f"{item['structural_dim']} | "
            f"{item['variant_label']} | "
            f"{item['parameter_count']} | "
            f"{item['parameter_memory_mib_float32']:.2f} | "
            f"{item['representation_dim']} | "
            f"{_fmt_mean_sd(item, 'validation_f1')} | "
            f"{_fmt_mean_sd(item, 'validation_structural_spearman')} | "
            f"{_fmt_mean_sd(item, 'validation_structural_normalized_stress')} | "
            f"{_fmt_mean_sd(item, 'validation_structural_neighbor_overlap_at_3')} | "
            f"{item['n_seeds']} |"
        )

    lines.extend(["", "## Threshold analysis", ""])
    for variant in sorted({str(item["variant"]) for item in summary}):
        label = VARIANT_LABELS.get(variant, variant)
        spearman_hit = _find_min_dim(summary, variant, "validation_structural_spearman", spearman_threshold, "ge")
        stress_hit = _find_min_dim(summary, variant, "validation_structural_normalized_stress", stress_threshold, "le")
        if spearman_hit is None:
            lines.append(f"- {label}: does not reach AST Spearman >= {spearman_threshold:.2f} in the tested grid.")
        else:
            lines.append(
                f"- {label}: first reaches AST Spearman >= {spearman_threshold:.2f} at structural_dim="
                f"{spearman_hit['structural_dim']} ({spearman_hit['parameter_memory_mib_float32']:.2f} MiB float32 params)."
            )
        if stress_hit is None:
            lines.append(f"- {label}: does not reach normalized stress <= {stress_threshold:.2f} in the tested grid.")
        else:
            lines.append(
                f"- {label}: first reaches normalized stress <= {stress_threshold:.2f} at structural_dim="
                f"{stress_hit['structural_dim']} ({stress_hit['parameter_memory_mib_float32']:.2f} MiB float32 params)."
            )

    lines.extend(
        [
            "",
            "## Claim boundary",
            "",
            "A valid claim is dimension-efficiency only if a lower-dimensional hyperbolic variant reaches",
            "the same structural-fidelity threshold as a higher-dimensional Euclidean control under the same",
            "data budget, seeds, optimizer, and evaluation split. If this condition is not met, the correct",
            "claim is only that hyperbolic geometry improves structural fidelity at the tested dimensions.",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    input_paths = args.inputs
    if not input_paths:
        input_paths = sorted(PROJECT_ROOT.glob(args.glob))
    input_paths = [path if path.is_absolute() else PROJECT_ROOT / path for path in input_paths]
    if not input_paths:
        raise SystemExit("no input JSON files found")

    rows = load_results(input_paths)
    if not rows:
        raise SystemExit("input files contain no runs")
    summary = summarize(rows)
    write_csv(summary, args.csv_output if args.csv_output.is_absolute() else PROJECT_ROOT / args.csv_output)
    markdown = build_markdown(summary, args.spearman_threshold, args.stress_threshold)
    md_output = args.md_output if args.md_output.is_absolute() else PROJECT_ROOT / args.md_output
    md_output.parent.mkdir(parents=True, exist_ok=True)
    md_output.write_text(markdown + "\n", encoding="utf-8")
    print(f"Wrote {args.csv_output}")
    print(f"Wrote {args.md_output}")


if __name__ == "__main__":
    main()
