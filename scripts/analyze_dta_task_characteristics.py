from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from statistics import fmean
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

MPL_CACHE_DIR = PROJECT_ROOT / ".matplotlib-cache"
MPL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE_DIR))
os.environ.setdefault("XDG_CACHE_HOME", str(MPL_CACHE_DIR))

from geometry_profile_research.analysis import geometry_profile_for_ast_source
from geometry_profile_research.dta import load_dta_records, stratified_validation_test_split
from geometry_profile_research.geometry_features import geometry_feature_sets
from geometry_profile_research.task_characterization import (
    benjamini_hochberg_q_values,
    bootstrap_spearman_ci,
    leave_one_out_spearman,
    spearman_correlation,
    spearman_permutation_p_value,
    summarize_numeric_features_by_label,
)


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _length_only_map_deltas(task_summary_path: Path) -> dict[str, float]:
    deltas: dict[str, float] = {}
    for row in _read_csv_rows(task_summary_path):
        if row.get("feature_set") != "length_only" or row.get("metric") != "map":
            continue
        deltas[str(row["task_label"])] = float(row["mean_delta_mean"])
    return deltas


def _test_feature_summaries(
    dataset_dir: Path,
    *,
    validation_per_task: int,
    test_per_task: int,
    split_seed: int,
) -> list[dict[str, Any]]:
    split = stratified_validation_test_split(
        load_dta_records(dataset_dir),
        validation_per_task=validation_per_task,
        test_per_task=test_per_task,
        seed=split_seed,
    )
    labels: list[int] = []
    vectors: list[dict[str, float]] = []
    syntax_errors = 0
    for record in split["test"]:
        try:
            profile = geometry_profile_for_ast_source(record.code)
        except SyntaxError:
            syntax_errors += 1
            continue
        feature_sets = geometry_feature_sets(profile)
        vector = dict(feature_sets["all"])
        vector["length_scale"] = (
            feature_sets["length_only"]["log_node_count"]
            + feature_sets["length_only"]["log_edge_count"]
        )
        labels.append(record.task_id)
        vectors.append(vector)

    rows = summarize_numeric_features_by_label(labels, vectors)
    for row in rows:
        row["split_seed"] = split_seed
        row["syntax_errors"] = syntax_errors
    return rows


def _aggregate_task_characteristics(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["task_label"])].append(row)

    aggregate_rows: list[dict[str, Any]] = []
    for task_label, task_rows in sorted(grouped.items(), key=lambda item: int(item[0])):
        numeric_keys = sorted(
            key
            for key in {key for row in task_rows for key in row}
            if key not in {"task_label", "split_seed"}
            and all(isinstance(row.get(key), (int, float)) for row in task_rows if key in row)
        )
        aggregate: dict[str, Any] = {
            "task_label": task_label,
            "n_splits": len(task_rows),
            "split_seeds": ",".join(str(row["split_seed"]) for row in task_rows),
        }
        for key in numeric_keys:
            aggregate[key] = fmean(float(row[key]) for row in task_rows)
        aggregate_rows.append(aggregate)
    return aggregate_rows


def _correlate_with_length_delta(
    task_characteristics: list[dict[str, Any]],
    length_deltas: dict[str, float],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    aligned = [row for row in task_characteristics if str(row["task_label"]) in length_deltas]
    if len(aligned) < 3:
        return rows
    y = [length_deltas[str(row["task_label"])] for row in aligned]
    candidate_features = sorted(
        key
        for key in {key for row in aligned for key in row}
        if key.endswith("_mean") or key.endswith("_std")
    )
    for feature in candidate_features:
        x = [float(row.get(feature, 0.0)) for row in aligned]
        ci_low, ci_high = bootstrap_spearman_ci(x, y, iterations=5000, seed=17)
        loo = leave_one_out_spearman(
            x,
            y,
            labels=[f"task-{int(row['task_label']):02d}" for row in aligned],
        )
        rows.append(
            {
                "target": "length_only_map_delta",
                "feature": feature,
                "spearman_rho": spearman_correlation(x, y),
                "spearman_ci95_low": ci_low,
                "spearman_ci95_high": ci_high,
                "permutation_p_two_sided": spearman_permutation_p_value(
                    x,
                    y,
                    iterations=20_000,
                    seed=17,
                ),
                "loo_rho_min": loo["rho_min"],
                "loo_rho_max": loo["rho_max"],
                "loo_omit_label_at_min": loo["omit_label_at_min"],
                "loo_omit_label_at_max": loo["omit_label_at_max"],
                "n_tasks": len(aligned),
            }
        )
    q_values = benjamini_hochberg_q_values(
        [float(row["permutation_p_two_sided"]) for row in rows]
    )
    for row, q_value in zip(rows, q_values):
        row["bh_q_value"] = q_value
    return sorted(rows, key=lambda row: abs(float(row["spearman_rho"])), reverse=True)


def _save_characteristic_figure(
    path: Path,
    task_characteristics: list[dict[str, Any]],
    length_deltas: dict[str, float],
    correlation_rows: list[dict[str, Any]],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    aligned = [row for row in task_characteristics if str(row["task_label"]) in length_deltas]
    if not aligned:
        return
    panels = [
        ("branching_entropy_mean", "Branching entropy mean"),
        ("length_scale_mean", "Length scale mean"),
    ]
    correlations_by_feature = {
        str(row["feature"]): row
        for row in correlation_rows
    }
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    y = [length_deltas[str(row["task_label"])] for row in aligned]
    for ax, (feature, title) in zip(axes, panels):
        x = [float(row.get(feature, 0.0)) for row in aligned]
        correlation = correlations_by_feature.get(feature, {})
        rho = float(correlation.get("spearman_rho", spearman_correlation(x, y)))
        p_value = float(correlation.get("permutation_p_two_sided", 1.0))
        q_value = float(correlation.get("bh_q_value", 1.0))
        ax.scatter(x, y, color="#2f5d8c")
        for row, x_value, y_value in zip(aligned, x, y):
            ax.annotate(f"task-{int(row['task_label']):02d}", (x_value, y_value), fontsize=7, xytext=(4, 3), textcoords="offset points")
        ax.axhline(0.0, color="#333333", linewidth=1)
        ax.set_xlabel(title)
        ax.set_title(f"{title}\nrho={rho:.2f}, p={p_value:.3f}, q={q_value:.3f}")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("length_only MAP delta")
    fig.suptitle("Task characteristics associated with length_only retrieval gain")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".png"), dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def run_task_characterization(
    dataset_dir: Path,
    *,
    validation_per_task: int,
    test_per_task: int,
    split_seeds: list[int],
    task_summary_path: Path,
) -> dict[str, Any]:
    split_rows: list[dict[str, Any]] = []
    for split_seed in split_seeds:
        split_rows.extend(
            _test_feature_summaries(
                dataset_dir,
                validation_per_task=validation_per_task,
                test_per_task=test_per_task,
                split_seed=split_seed,
            )
        )
    aggregate_rows = _aggregate_task_characteristics(split_rows)
    correlation_rows = _correlate_with_length_delta(
        aggregate_rows,
        _length_only_map_deltas(task_summary_path),
    )
    return {
        "split_rows": split_rows,
        "aggregate_rows": aggregate_rows,
        "correlation_rows": correlation_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze DTA task-level AST characteristics against feature-set MAP deltas."
    )
    parser.add_argument("--dataset-dir", type=Path, default=Path("data/dta_zenodo_7799972/extracted"))
    parser.add_argument("--validation-per-task", type=int, default=20)
    parser.add_argument("--test-per-task", type=int, default=50)
    parser.add_argument("--split-seeds", nargs="+", type=int, default=[101, 202, 303])
    parser.add_argument("--task-summary", type=Path, default=Path("reports/multisplit_task_summary.csv"))
    parser.add_argument("--output-characteristics", type=Path, default=Path("reports/task_characteristics.csv"))
    parser.add_argument("--output-correlations", type=Path, default=Path("reports/task_characteristic_correlations.csv"))
    parser.add_argument("--output-figure", type=Path, default=Path("figures/fig08_task_characteristics_vs_delta.png"))
    args = parser.parse_args()

    payload = run_task_characterization(
        args.dataset_dir,
        validation_per_task=args.validation_per_task,
        test_per_task=args.test_per_task,
        split_seeds=args.split_seeds,
        task_summary_path=args.task_summary,
    )
    _write_csv(args.output_characteristics, payload["aggregate_rows"])
    _write_csv(args.output_correlations, payload["correlation_rows"])
    _save_characteristic_figure(
        args.output_figure,
        payload["aggregate_rows"],
        _length_only_map_deltas(args.task_summary),
        payload["correlation_rows"],
    )
    print(
        json.dumps(
            {
                "split_rows": len(payload["split_rows"]),
                "task_characteristics": len(payload["aggregate_rows"]),
                "correlations": len(payload["correlation_rows"]),
                "output_characteristics": str(args.output_characteristics),
                "output_correlations": str(args.output_correlations),
                "output_figure": str(args.output_figure),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
