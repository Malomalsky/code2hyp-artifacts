from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

MPL_CACHE_DIR = PROJECT_ROOT / ".matplotlib-cache"
MPL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE_DIR))
os.environ.setdefault("XDG_CACHE_HOME", str(MPL_CACHE_DIR))

from geometry_profile_research.experiment_artifacts import (
    extract_delta_rows,
    extract_inventory_row,
    extract_metric_rows,
    extract_task_delta_rows,
    extract_task_metric_rows,
    summarize_delta_rows,
    summarize_task_delta_rows,
)


def _load_payloads(outputs_dir: Path) -> list[tuple[Path, dict[str, Any]]]:
    payloads: list[tuple[Path, dict[str, Any]]] = []
    for path in sorted(outputs_dir.glob("*.json")):
        payloads.append((path, json.loads(path.read_text(encoding="utf-8"))))
    return payloads


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


def _metric_value(row: dict[str, Any], key: str) -> float:
    value = row.get(key, "")
    return float(value) if value != "" else 0.0


def _summary_value(row: dict[str, Any], key: str) -> float:
    value = row.get(key, "")
    return float(value) if value != "" else 0.0


def _find_rows(
    metric_rows: list[dict[str, Any]],
    *,
    file_contains: str,
    scope: str | None = None,
) -> list[dict[str, Any]]:
    rows = [
        row
        for row in metric_rows
        if file_contains in str(row.get("experiment_file", ""))
    ]
    if scope is not None:
        rows = [row for row in rows if row.get("scope") == scope]
    return rows


def _save_figure(fig, output_base: Path) -> None:
    output_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_base.with_suffix(".png"), dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(output_base.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")


def _plot_markov_baselines(metric_rows: list[dict[str, Any]], figures_dir: Path) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = _find_rows(metric_rows, file_contains="dta_markov_baselines", scope="overall")
    if not rows:
        return []

    labels = [str(row["method"]).replace("_", "\n") for row in rows]
    map_values = [_metric_value(row, "map") for row in rows]
    recall_values = [_metric_value(row, "recall@10") for row in rows]

    fig, ax = plt.subplots(figsize=(11, 5))
    x = list(range(len(rows)))
    width = 0.38
    ax.bar([i - width / 2 for i in x], map_values, width=width, label="MAP", color="#2f5d8c")
    ax.bar([i + width / 2 for i in x], recall_values, width=width, label="Recall@10", color="#9b6b2f")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Metric value")
    ax.set_title("Baseline distance definitions on DTA retrieval")
    ax.set_ylim(0, max(map_values + recall_values) * 1.15)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    output = figures_dir / "fig01_markov_baseline_comparison"
    _save_figure(fig, output)
    plt.close(fig)
    return [str(output.with_suffix(".png")), str(output.with_suffix(".pdf"))]


def _plot_weight_sweep(metric_rows: list[dict[str, Any]], figures_dir: Path) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = _find_rows(
        metric_rows,
        file_contains="dta_weight_sweep_transition_count_limit50_seed13",
        scope="overall",
    )
    if not rows:
        return []
    rows = sorted(rows, key=lambda row: float(str(row["method"]).replace("weight_", "")))

    weights = [float(str(row["method"]).replace("weight_", "")) for row in rows]
    map_values = [_metric_value(row, "map") for row in rows]
    recall_values = [_metric_value(row, "recall@10") for row in rows]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(weights, map_values, marker="o", label="MAP", color="#2f5d8c")
    ax.plot(weights, recall_values, marker="s", label="Recall@10", color="#9b6b2f")
    ax.invert_xaxis()
    ax.set_xlabel("Markov distance weight")
    ax.set_ylabel("Metric value")
    ax.set_title("Validation of Markov/geometry mixing weight")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    output = figures_dir / "fig02_transition_count_weight_sweep"
    _save_figure(fig, output)
    plt.close(fig)
    return [str(output.with_suffix(".png")), str(output.with_suffix(".pdf"))]


def _plot_feature_ablation(metric_rows: list[dict[str, Any]], figures_dir: Path) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = _find_rows(
        metric_rows,
        file_contains="dta_feature_ablation_transition_count_limit50_seed13_w090",
        scope="overall",
    )
    if not rows:
        return []

    order = [
        "M2_transition_count_jsd",
        "M4_markov_length_only",
        "M4_markov_size_depth",
        "M4_markov_branching",
        "M4_markov_metric_distortion",
        "M4_markov_shape",
        "M4_markov_all",
    ]
    by_method = {row["method"]: row for row in rows}
    rows = [by_method[method] for method in order if method in by_method]
    labels = [str(row["method"]).replace("M4_markov_", "+").replace("M2_transition_count_jsd", "M2\ntransition-count") for row in rows]
    map_values = [_metric_value(row, "map") for row in rows]
    baseline_value = map_values[0]

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["#555555"] + ["#2f5d8c"] * (len(rows) - 1)
    ax.bar(labels, map_values, color=colors)
    ax.axhline(baseline_value, color="#333333", linestyle="--", linewidth=1)
    ax.set_ylabel("MAP (zoomed y-axis)")
    ax.set_title("Feature ablation against transition-count JSD baseline")
    ax.set_ylim(min(map_values) * 0.98, max(map_values) * 1.01)
    for label, value in zip(labels, map_values):
        ax.text(label, value + 0.0005, f"{value:.4f}", ha="center", va="bottom", fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    output = figures_dir / "fig03_feature_ablation_map"
    _save_figure(fig, output)
    plt.close(fig)
    return [str(output.with_suffix(".png")), str(output.with_suffix(".pdf"))]


def _plot_confirmatory_delta(delta_rows: list[dict[str, Any]], figures_dir: Path) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = [
        row
        for row in delta_rows
        if "dta_confirmatory_split" in str(row.get("experiment_file", ""))
        and row.get("comparison") == "candidate_minus_baseline"
        and row.get("metric") in {"map", "recall@10", "mrr", "top1_accuracy"}
    ]
    if not rows:
        return []
    metric_order = ["top1_accuracy", "mrr", "map", "recall@10"]
    rows = sorted(rows, key=lambda row: metric_order.index(str(row["metric"])))
    labels = [str(row["metric"]) for row in rows]
    deltas = [float(row["mean_delta"]) for row in rows]
    low = [float(row["ci95_low"]) for row in rows]
    high = [float(row["ci95_high"]) for row in rows]
    p_values = [float(row["p_one_sided"]) for row in rows]
    lower_errors = [delta - lo for delta, lo in zip(deltas, low)]
    upper_errors = [hi - delta for delta, hi in zip(deltas, high)]
    colors = ["#2f5d8c" if p_value <= 0.05 else "#9a9a9a" for p_value in p_values]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(labels, deltas, color=colors)
    ax.errorbar(labels, deltas, yerr=[lower_errors, upper_errors], fmt="none", ecolor="#222222", capsize=4)
    ax.axhline(0.0, color="#333333", linewidth=1)
    for label, delta, high_bound, p_value in zip(labels, deltas, high, p_values):
        marker = "*" if p_value <= 0.05 else "n.s."
        ax.text(label, high_bound + 0.00035, f"{marker}\np={p_value:.3g}", ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("Candidate minus baseline")
    ax.set_title("Confirmatory test effect sizes with bootstrap CI and permutation p-values")
    ax.set_ylim(0.0, max(high) * 1.25)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    output = figures_dir / "fig04_confirmatory_delta_ci"
    _save_figure(fig, output)
    plt.close(fig)
    return [str(output.with_suffix(".png")), str(output.with_suffix(".pdf"))]


def _plot_confirmatory_feature_sweep(delta_rows: list[dict[str, Any]], figures_dir: Path) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = summarize_delta_rows(
        delta_rows,
        file_contains="dta_confirmatory_feature_sweep",
        metrics={"map", "recall@10"},
    )
    if not rows:
        return []

    feature_order = ["length_only", "size_depth", "branching", "metric_distortion", "shape", "all"]
    metrics = ["map", "recall@10"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=False)
    for ax, metric in zip(axes, metrics):
        metric_rows = [
            row
            for row in rows
            if row.get("metric") == metric and row.get("feature_set") in feature_order
        ]
        metric_rows = sorted(metric_rows, key=lambda row: feature_order.index(str(row["feature_set"])))
        labels = [str(row["feature_set"]) for row in metric_rows]
        deltas = [_summary_value(row, "mean_delta_mean") for row in metric_rows]
        low = [_summary_value(row, "mean_delta_min") for row in metric_rows]
        high = [_summary_value(row, "mean_delta_max") for row in metric_rows]
        errors = [
            [delta - lo for delta, lo in zip(deltas, low)],
            [hi - delta for delta, hi in zip(deltas, high)],
        ]
        colors = []
        for row in metric_rows:
            n_splits = int(row.get("n_splits", 0))
            significant_splits = int(row.get("significant_splits", 0))
            if significant_splits == n_splits and n_splits > 0:
                colors.append("#2f5d8c")
            elif significant_splits > 0:
                colors.append("#6e8fb1")
            else:
                colors.append("#9a9a9a")
        ax.bar(labels, deltas, color=colors)
        ax.errorbar(labels, deltas, yerr=errors, fmt="none", ecolor="#222222", capsize=4)
        ax.axhline(0.0, color="#333333", linewidth=1)
        ax.set_title(f"{metric} delta, mean over split seeds")
        ax.set_ylabel("Candidate minus baseline")
        ax.tick_params(axis="x", rotation=25)
        ax.grid(axis="y", alpha=0.25)
        y_min = min(0.0, min(low)) if low else 0.0
        y_max = max(0.0, max(high)) if high else 0.0
        pad = max((y_max - y_min) * 0.2, 0.0001)
        ax.set_ylim(y_min - pad, y_max + pad)
        for label, high_bound, row in zip(labels, high, metric_rows):
            marker = f"{row['significant_splits']}/{row['n_splits']}"
            ax.text(label, high_bound + pad * 0.2, marker, ha="center", va="bottom", fontsize=8)
    fig.suptitle("Confirmatory feature-set controls across split seeds")
    fig.tight_layout()
    output = figures_dir / "fig05_confirmatory_feature_sweep"
    _save_figure(fig, output)
    plt.close(fig)
    return [str(output.with_suffix(".png")), str(output.with_suffix(".pdf"))]


def _plot_confirmatory_residual_sweep(delta_rows: list[dict[str, Any]], figures_dir: Path) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = summarize_delta_rows(
        delta_rows,
        file_contains="dta_confirmatory_residual_sweep",
        metrics={"map", "recall@10"},
    )
    if not rows:
        return []

    feature_order = [
        "residual_size_depth",
        "residual_branching",
        "residual_metric_distortion",
        "residual_shape",
        "residual_all_nonlength",
    ]
    metrics = ["map", "recall@10"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=False)
    for ax, metric in zip(axes, metrics):
        metric_rows = [
            row
            for row in rows
            if row.get("metric") == metric and row.get("feature_set") in feature_order
        ]
        metric_rows = sorted(metric_rows, key=lambda row: feature_order.index(str(row["feature_set"])))
        labels = [str(row["feature_set"]).replace("residual_", "") for row in metric_rows]
        deltas = [_summary_value(row, "mean_delta_mean") for row in metric_rows]
        low = [_summary_value(row, "mean_delta_min") for row in metric_rows]
        high = [_summary_value(row, "mean_delta_max") for row in metric_rows]
        errors = [
            [delta - lo for delta, lo in zip(deltas, low)],
            [hi - delta for delta, hi in zip(deltas, high)],
        ]
        colors = []
        for row in metric_rows:
            n_splits = int(row.get("n_splits", 0))
            significant_splits = int(row.get("significant_splits", 0))
            if significant_splits == n_splits and n_splits > 0:
                colors.append("#2f5d8c")
            elif significant_splits > 0:
                colors.append("#6e8fb1")
            else:
                colors.append("#9a9a9a")
        ax.bar(labels, deltas, color=colors)
        ax.errorbar(labels, deltas, yerr=errors, fmt="none", ecolor="#222222", capsize=4)
        ax.axhline(0.0, color="#333333", linewidth=1)
        ax.set_title(f"{metric} delta, mean over split seeds")
        ax.set_ylabel("Candidate minus baseline")
        ax.tick_params(axis="x", rotation=25)
        ax.grid(axis="y", alpha=0.25)
        y_min = min(0.0, min(low)) if low else 0.0
        y_max = max(0.0, max(high)) if high else 0.0
        pad = max((y_max - y_min) * 0.2, 0.0001)
        ax.set_ylim(y_min - pad, y_max + pad)
        for label, high_bound, row in zip(labels, high, metric_rows):
            marker = f"{row['significant_splits']}/{row['n_splits']}"
            ax.text(label, high_bound + pad * 0.2, marker, ha="center", va="bottom", fontsize=8)
    fig.suptitle("Residual feature controls after regressing out length_only, across split seeds")
    fig.tight_layout()
    output = figures_dir / "fig06_confirmatory_residual_sweep"
    _save_figure(fig, output)
    plt.close(fig)
    return [str(output.with_suffix(".png")), str(output.with_suffix(".pdf"))]


def _plot_task_level_map_delta(task_summary_rows: list[dict[str, Any]], figures_dir: Path) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    rows = [
        row
        for row in task_summary_rows
        if row.get("metric") == "map"
    ]
    if not rows:
        return []

    feature_order = ["length_only", "size_depth", "all", "shape", "metric_distortion", "branching"]
    task_labels = sorted({str(row["task_label"]) for row in rows}, key=lambda value: int(value))
    by_key = {
        (str(row["feature_set"]), str(row["task_label"])): _summary_value(row, "mean_delta_mean")
        for row in rows
    }
    matrix = [
        [by_key.get((feature_set, task_label), 0.0) for task_label in task_labels]
        for feature_set in feature_order
        if any((feature_set, task_label) in by_key for task_label in task_labels)
    ]
    labels = [
        feature_set
        for feature_set in feature_order
        if any((feature_set, task_label) in by_key for task_label in task_labels)
    ]
    if not matrix:
        return []

    flat_values = [value for row in matrix for value in row]
    value_limit = max(abs(min(flat_values)), abs(max(flat_values)), 0.001)
    fig, ax = plt.subplots(figsize=(11, 4.8))
    image = ax.imshow(
        matrix,
        cmap="RdBu_r",
        norm=TwoSlopeNorm(vmin=-value_limit, vcenter=0.0, vmax=value_limit),
        aspect="auto",
    )
    ax.set_xticks(range(len(task_labels)))
    ax.set_xticklabels([f"task-{int(label):02d}" for label in task_labels], rotation=45, ha="right")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_title("Task-level MAP delta over transition-count JSD, mean across split seeds")
    for row_index, row_values in enumerate(matrix):
        for col_index, value in enumerate(row_values):
            ax.text(
                col_index,
                row_index,
                f"{value:+.3f}",
                ha="center",
                va="center",
                fontsize=7,
                color="white" if abs(value) > value_limit * 0.55 else "#111111",
            )
    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label("Candidate minus baseline MAP")
    fig.tight_layout()
    output = figures_dir / "fig07_task_level_map_delta"
    _save_figure(fig, output)
    plt.close(fig)
    return [str(output.with_suffix(".png")), str(output.with_suffix(".pdf"))]


def _markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = []
    for row in rows:
        values = []
        for column in columns:
            value = row.get(column, "")
            if isinstance(value, float):
                values.append(f"{value:.4f}")
            else:
                values.append(str(value))
        body.append("| " + " | ".join(values) + " |")
    return "\n".join([header, sep, *body])


def _write_registry_markdown(
    path: Path,
    *,
    inventory_rows: list[dict[str, Any]],
    metric_rows: list[dict[str, Any]],
    delta_rows: list[dict[str, Any]],
    feature_summary_rows: list[dict[str, Any]],
    residual_summary_rows: list[dict[str, Any]],
    task_summary_rows: list[dict[str, Any]],
    figure_paths: list[str],
) -> None:
    confirmatory_metrics = _find_rows(metric_rows, file_contains="dta_confirmatory_split", scope="test")
    confirmatory_deltas = [
        row for row in delta_rows
        if "dta_confirmatory_split" in str(row.get("experiment_file", ""))
    ]
    baseline_rows = _find_rows(metric_rows, file_contains="dta_markov_baselines", scope="overall")
    feature_sweep_deltas = [
        row for row in delta_rows
        if "dta_confirmatory_feature_sweep" in str(row.get("experiment_file", ""))
        and row.get("metric") in {"map", "recall@10"}
    ]
    residual_sweep_deltas = [
        row for row in delta_rows
        if "dta_confirmatory_residual_sweep" in str(row.get("experiment_file", ""))
        and row.get("metric") in {"map", "recall@10"}
    ]

    content = [
        "# Experiment Registry",
        "",
        "This file is generated from `outputs/*.json` by `scripts/build_experiment_artifacts.py`.",
        "",
        "## Full Experiment Inventory",
        "",
        _markdown_table(
            inventory_rows,
            [
                "experiment_file",
                "experiment_type",
                "role",
                "article_use",
                "records",
                "baseline_kind",
                "markov_weight",
                "geometry_weight",
                "feature_set",
                "sample_seed",
            ],
        ),
        "",
        "## Confirmatory Result",
        "",
        _markdown_table(
            confirmatory_metrics,
            ["experiment_file", "method", "top1_accuracy", "mrr", "map", "recall@10"],
        ),
        "",
        "## Confirmatory Paired Deltas",
        "",
        _markdown_table(
            confirmatory_deltas,
            ["experiment_file", "feature_set", "metric", "mean_delta", "ci95_low", "ci95_high", "p_one_sided"],
        ),
        "",
        "## Confirmatory Feature-Set Controls",
        "",
        _markdown_table(
            feature_sweep_deltas,
            ["feature_set", "metric", "mean_delta", "ci95_low", "ci95_high", "p_one_sided"],
        ),
        "",
        "## Multi-Split Feature-Set Summary",
        "",
        _markdown_table(
            feature_summary_rows,
            [
                "feature_set",
                "metric",
                "n_splits",
                "split_seeds",
                "mean_delta_mean",
                "mean_delta_std",
                "mean_delta_min",
                "mean_delta_max",
                "significant_splits",
            ],
        ),
        "",
        "## Confirmatory Residual Controls After Length",
        "",
        _markdown_table(
            residual_sweep_deltas,
            ["feature_set", "metric", "mean_delta", "ci95_low", "ci95_high", "p_one_sided"],
        ),
        "",
        "## Multi-Split Residual Summary After Length",
        "",
        _markdown_table(
            residual_summary_rows,
            [
                "feature_set",
                "metric",
                "n_splits",
                "split_seeds",
                "mean_delta_mean",
                "mean_delta_std",
                "mean_delta_min",
                "mean_delta_max",
                "significant_splits",
            ],
        ),
        "",
        "## Multi-Split Task-Level MAP Delta",
        "",
        _markdown_table(
            [
                row for row in task_summary_rows
                if row.get("metric") == "map"
            ],
            [
                "feature_set",
                "task_label",
                "metric",
                "n_splits",
                "mean_delta_mean",
                "mean_delta_min",
                "mean_delta_max",
            ],
        ),
        "",
        "## Baseline Audit",
        "",
        _markdown_table(
            baseline_rows,
            ["method", "top1_accuracy", "mrr", "map", "recall@10"],
        ),
        "",
        "## Generated Figures",
        "",
        *[f"- `{figure_path}`" for figure_path in figure_paths],
        "",
        "## Figure Interpretation",
        "",
        "- `fig01`: transition-count JSD is the strongest non-geometric baseline and must be treated as the main comparator.",
        "- `fig02`: the validation sweep selects a mostly Markov distance with a small geometry component, not a geometry-only model.",
        "- `fig03`: geometry features add signal over the transition-count baseline; the effect size is modest.",
        "- `fig04`: the confirmatory test supports a stable MAP and Recall@10 improvement, while Top1 and MRR remain weaker claims.",
        "- `fig05`: the feature-set control summarizes mean deltas across confirmatory split seeds and checks whether the gain is explained by AST length/scale or by richer shape/distortion features.",
        "- `fig06`: residual controls summarize mean deltas across confirmatory split seeds after regressing out length_only.",
        "- `fig07`: task-level MAP deltas show whether the effect is broad across DTA tasks or concentrated in a few task groups.",
        "",
        "## Interpretation Rule",
        "",
        (
            "Exploratory sweeps and ablations are used for method design. "
            "Article claims should be based on confirmatory validation/test "
            "splits and interpreted together with the feature-set control. "
            "The current strongest additional signal is length_only, not the "
            "full geometry profile."
        ),
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(content), encoding="utf-8")


def build_artifacts(outputs_dir: Path, reports_dir: Path, figures_dir: Path) -> dict[str, Any]:
    payloads = _load_payloads(outputs_dir)
    inventory_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    delta_rows: list[dict[str, Any]] = []
    task_metric_rows: list[dict[str, Any]] = []
    task_delta_rows: list[dict[str, Any]] = []
    for path, payload in payloads:
        inventory_rows.append(extract_inventory_row(path.name, payload))
        metric_rows.extend(extract_metric_rows(path.name, payload))
        delta_rows.extend(extract_delta_rows(path.name, payload))
        task_metric_rows.extend(extract_task_metric_rows(path.name, payload))
        task_delta_rows.extend(extract_task_delta_rows(path.name, payload))

    _write_csv(reports_dir / "experiment_inventory.csv", inventory_rows)
    _write_csv(reports_dir / "experiment_metrics.csv", metric_rows)
    _write_csv(reports_dir / "experiment_deltas.csv", delta_rows)
    _write_csv(reports_dir / "experiment_task_metrics.csv", task_metric_rows)
    _write_csv(reports_dir / "experiment_task_deltas.csv", task_delta_rows)
    feature_summary_rows = summarize_delta_rows(
        delta_rows,
        file_contains="dta_confirmatory_feature_sweep",
        metrics={"map", "recall@10"},
    )
    residual_summary_rows = summarize_delta_rows(
        delta_rows,
        file_contains="dta_confirmatory_residual_sweep",
        metrics={"map", "recall@10"},
    )
    _write_csv(reports_dir / "multisplit_feature_summary.csv", feature_summary_rows)
    _write_csv(reports_dir / "multisplit_residual_summary.csv", residual_summary_rows)
    task_summary_rows = summarize_task_delta_rows(
        task_delta_rows,
        file_contains="dta_confirmatory_feature_sweep",
        metrics={"map", "recall@10"},
    )
    _write_csv(reports_dir / "multisplit_task_summary.csv", task_summary_rows)

    figure_paths: list[str] = []
    figure_paths.extend(_plot_markov_baselines(metric_rows, figures_dir))
    figure_paths.extend(_plot_weight_sweep(metric_rows, figures_dir))
    figure_paths.extend(_plot_feature_ablation(metric_rows, figures_dir))
    figure_paths.extend(_plot_confirmatory_delta(delta_rows, figures_dir))
    figure_paths.extend(_plot_confirmatory_feature_sweep(delta_rows, figures_dir))
    figure_paths.extend(_plot_confirmatory_residual_sweep(delta_rows, figures_dir))
    figure_paths.extend(_plot_task_level_map_delta(task_summary_rows, figures_dir))

    _write_registry_markdown(
        reports_dir / "experiment_registry.md",
        inventory_rows=inventory_rows,
        metric_rows=metric_rows,
        delta_rows=delta_rows,
        feature_summary_rows=feature_summary_rows,
        residual_summary_rows=residual_summary_rows,
        task_summary_rows=task_summary_rows,
        figure_paths=figure_paths,
    )

    return {
        "json_files": len(payloads),
        "inventory_rows": len(inventory_rows),
        "metric_rows": len(metric_rows),
        "delta_rows": len(delta_rows),
        "task_metric_rows": len(task_metric_rows),
        "task_delta_rows": len(task_delta_rows),
        "feature_summary_rows": len(feature_summary_rows),
        "residual_summary_rows": len(residual_summary_rows),
        "task_summary_rows": len(task_summary_rows),
        "figures": figure_paths,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build article-ready experiment tables and figures from JSON outputs."
    )
    parser.add_argument("--outputs-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--reports-dir", type=Path, default=Path("reports"))
    parser.add_argument("--figures-dir", type=Path, default=Path("figures"))
    args = parser.parse_args()

    summary = build_artifacts(args.outputs_dir, args.reports_dir, args.figures_dir)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
