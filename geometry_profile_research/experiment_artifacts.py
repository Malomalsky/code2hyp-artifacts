from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from statistics import stdev
from typing import Any


METRIC_KEYS = ("top1_accuracy", "mrr", "map", "recall@10")


def classify_experiment(filename: str, payload: dict[str, Any]) -> str:
    """Classify an experiment JSON payload for registry generation."""
    name = Path(filename).name
    if "feature_set_results" in payload:
        protocol_status = payload.get("protocol", {}).get("status", "")
        if "residual" in name or "residual" in protocol_status:
            return "confirmatory_residual_sweep"
        return "confirmatory_feature_sweep"
    if "test_results" in payload:
        return "confirmatory_split"
    if "method_summary" in payload:
        return "multiseed_ablation"
    if "results" in payload:
        return "weight_sweep"
    if "methods" in payload and "markov_baselines" in name:
        return "markov_baseline_audit"
    if "methods" in payload:
        return "method_comparison"
    if "overall_geometry" in payload:
        return "ast_geometry_pilot"
    if "profile" in payload:
        return "file_tree_smoke_test"
    return "unknown"


def _baseline_kind(filename: str, payload: dict[str, Any]) -> Any:
    parameters = payload.get("parameters", {})
    if parameters.get("baseline_kind", "") != "":
        return parameters["baseline_kind"]

    name = Path(filename).name
    experiment_type = classify_experiment(filename, payload)
    if "markov_baselines" in name:
        return "multiple"
    if "transition_count" in name:
        return "transition_count"
    if experiment_type in {"method_comparison", "weight_sweep", "multiseed_ablation"}:
        return "flat_markov_jsd"
    return ""


def _markov_weight(payload: dict[str, Any]) -> Any:
    parameters = payload.get("parameters", {})
    return parameters.get(
        "markov_weight",
        parameters.get(
            "selected_markov_weight",
            parameters.get("combined_markov_weight", ""),
        ),
    )


def _geometry_weight(payload: dict[str, Any]) -> Any:
    parameters = payload.get("parameters", {})
    return parameters.get(
        "feature_weight",
        parameters.get(
            "selected_geometry_weight",
            parameters.get("combined_geometry_weight", ""),
        ),
    )


def _record_count(payload: dict[str, Any]) -> Any:
    records = payload.get("records") or {}
    for key in ("test", "valid", "profiled", "loaded"):
        if key in records:
            return records[key]

    runs = payload.get("runs", [])
    run_total = 0
    for run in runs:
        run_records = run.get("records", {})
        if "valid" in run_records:
            run_total += int(run_records["valid"])
        elif "test" in run_records:
            run_total += int(run_records["test"])
    if run_total:
        return run_total

    dataset = payload.get("dataset", {})
    limit_per_task = dataset.get("limit_per_task")
    if limit_per_task != "" and limit_per_task is not None:
        task_count = records.get("task_count", "")
        if task_count != "":
            return int(limit_per_task) * int(task_count)
        return limit_per_task
    return ""


def _base_row(filename: str, payload: dict[str, Any]) -> dict[str, Any]:
    dataset = payload.get("dataset", {})
    return {
        "experiment_file": Path(filename).name,
        "experiment_type": classify_experiment(filename, payload),
        "baseline_kind": _baseline_kind(filename, payload),
        "markov_weight": _markov_weight(payload),
        "geometry_weight": _geometry_weight(payload),
        "feature_set": payload.get("parameters", {}).get("feature_set_name", ""),
        "limit_per_task": dataset.get("limit_per_task", ""),
        "sample_seed": dataset.get("sample_seed", dataset.get("split_seed", "")),
        "validation_per_task": dataset.get("validation_per_task", ""),
        "test_per_task": dataset.get("test_per_task", ""),
        "valid_records": _record_count(payload),
    }


def _experiment_role(experiment_type: str) -> str:
    if experiment_type in {
        "confirmatory_split",
        "confirmatory_feature_sweep",
        "confirmatory_residual_sweep",
    }:
        return "confirmatory"
    if experiment_type in {"ast_geometry_pilot", "file_tree_smoke_test"}:
        return "smoke_test"
    return "exploratory"


def _article_use(experiment_type: str) -> str:
    if experiment_type == "confirmatory_split":
        return "primary evidence"
    if experiment_type == "confirmatory_feature_sweep":
        return "feature-set control"
    if experiment_type == "confirmatory_residual_sweep":
        return "residual feature control"
    if experiment_type == "markov_baseline_audit":
        return "baseline selection"
    if experiment_type == "weight_sweep":
        return "hyperparameter selection"
    if experiment_type == "method_comparison":
        return "ablation / method design"
    if experiment_type == "multiseed_ablation":
        return "robustness check"
    if experiment_type == "ast_geometry_pilot":
        return "pipeline sanity check"
    if experiment_type == "file_tree_smoke_test":
        return "auxiliary sanity check"
    return "supporting material"


def extract_inventory_row(filename: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Summarize an experiment JSON payload for the article registry."""
    base = _base_row(filename, payload)
    experiment_type = base["experiment_type"]
    dataset = payload.get("dataset", {})
    return {
        "experiment_file": base["experiment_file"],
        "experiment_type": experiment_type,
        "role": _experiment_role(experiment_type),
        "article_use": _article_use(experiment_type),
        "records": _record_count(payload),
        "dataset_path": dataset.get("path", payload.get("input", "")),
        "limit_per_task": dataset.get("limit_per_task", ""),
        "validation_per_task": dataset.get("validation_per_task", ""),
        "test_per_task": dataset.get("test_per_task", ""),
        "sample_seed": dataset.get("sample_seed", dataset.get("split_seed", "")),
        "seeds": ",".join(str(seed) for seed in dataset.get("seeds", [])),
        "baseline_kind": base["baseline_kind"],
        "markov_weight": base["markov_weight"],
        "geometry_weight": base["geometry_weight"],
        "feature_set": base["feature_set"],
    }


def _metric_row(
    filename: str,
    payload: dict[str, Any],
    *,
    scope: str,
    method: str,
    metrics: dict[str, Any],
) -> dict[str, Any]:
    row = _base_row(filename, payload)
    row.update(
        {
            "scope": scope,
            "method": method,
        }
    )
    for key in METRIC_KEYS:
        row[key] = metrics.get(key, "")
    return row


def extract_metric_rows(filename: str, payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract one row per method/configuration from an experiment payload."""
    rows: list[dict[str, Any]] = []

    if "test_results" in payload:
        test_results = payload["test_results"]
        rows.append(
            _metric_row(
                filename,
                payload,
                scope="test",
                method="baseline",
                metrics=test_results["baseline"],
            )
        )
        rows.append(
            _metric_row(
                filename,
                payload,
                scope="test",
                method="candidate",
                metrics=test_results["candidate"],
            )
        )
        for validation_row in payload.get("validation_results", []):
            rows.append(
                _metric_row(
                    filename,
                    payload,
                    scope="validation",
                    method=f"weight_{validation_row.get('markov_weight')}",
                    metrics=validation_row.get("metrics", {}),
                )
            )
        return rows

    if "feature_set_results" in payload:
        baseline = payload.get("baseline", {}).get("test", {})
        rows.append(
            _metric_row(
                filename,
                payload,
                scope="test",
                method="baseline",
                metrics=baseline,
            )
        )
        for feature_set_name, result in payload["feature_set_results"].items():
            row = _metric_row(
                filename,
                payload,
                scope="test",
                method=f"candidate_{feature_set_name}",
                metrics=result.get("test_results", {}).get("candidate", {}),
            )
            row["feature_set"] = feature_set_name
            row["markov_weight"] = result.get("selected_markov_weight", "")
            row["geometry_weight"] = result.get("selected_geometry_weight", "")
            rows.append(row)
        return rows

    if "methods" in payload:
        for method, method_payload in payload["methods"].items():
            metrics = method_payload.get("metrics", method_payload)
            rows.append(
                _metric_row(
                    filename,
                    payload,
                    scope="overall",
                    method=method,
                    metrics=metrics,
                )
            )
        return rows

    if "results" in payload:
        for result in payload["results"]:
            rows.append(
                _metric_row(
                    filename,
                    payload,
                    scope="overall",
                    method=f"weight_{result.get('markov_weight')}",
                    metrics=result.get("metrics", {}),
                )
            )
        return rows

    if "method_summary" in payload:
        for method, metric_summary in payload["method_summary"].items():
            row = _base_row(filename, payload)
            row.update({"scope": "multiseed_mean", "method": method})
            for key in METRIC_KEYS:
                row[key] = metric_summary.get(key, {}).get("mean", "")
                row[f"{key}_std"] = metric_summary.get(key, {}).get("std", "")
            rows.append(row)
    return rows


def _delta_rows_from_test_results(filename: str, payload: dict[str, Any]) -> list[dict[str, Any]]:
    if "feature_set_results" in payload:
        rows = []
        for feature_set_name, feature_result in payload["feature_set_results"].items():
            tests = feature_result.get("test_results", {}).get(
                "paired_tests_candidate_minus_baseline",
                {},
            )
            for metric, result in tests.items():
                ci = result.get("bootstrap_ci95", ["", ""])
                row = _base_row(filename, payload)
                row.update(
                    {
                        "scope": "test",
                        "comparison": f"candidate_{feature_set_name}_minus_baseline",
                        "feature_set": feature_set_name,
                        "metric": metric,
                        "mean_delta": result.get("mean_delta", ""),
                        "ci95_low": ci[0] if len(ci) > 0 else "",
                        "ci95_high": ci[1] if len(ci) > 1 else "",
                        "p_one_sided": result.get("permutation_p_one_sided", ""),
                    }
                )
                rows.append(row)
        return rows

    tests = payload.get("test_results", {}).get("paired_tests_candidate_minus_baseline", {})
    rows = []
    for metric, result in tests.items():
        ci = result.get("bootstrap_ci95", ["", ""])
        row = _base_row(filename, payload)
        row.update(
            {
                "scope": "test",
                "comparison": "candidate_minus_baseline",
                "metric": metric,
                "mean_delta": result.get("mean_delta", ""),
                "ci95_low": ci[0] if len(ci) > 0 else "",
                "ci95_high": ci[1] if len(ci) > 1 else "",
                "p_one_sided": result.get("permutation_p_one_sided", ""),
            }
        )
        rows.append(row)
    return rows


def extract_delta_rows(filename: str, payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract paired-comparison deltas from experiment payloads."""
    rows = _delta_rows_from_test_results(filename, payload)

    for comparison_key in ("paired_tests_M4_minus_M2", "paired_tests_M_minus_M2"):
        tests = payload.get(comparison_key, {})
        for metric, result in tests.items():
            ci = result.get("bootstrap_ci95", ["", ""])
            row = _base_row(filename, payload)
            row.update(
                {
                    "scope": "overall",
                    "comparison": comparison_key,
                    "metric": metric,
                    "mean_delta": result.get("mean_delta", ""),
                    "ci95_low": ci[0] if len(ci) > 0 else "",
                    "ci95_high": ci[1] if len(ci) > 1 else "",
                    "p_one_sided": result.get("permutation_p_one_sided", ""),
                }
            )
            rows.append(row)

    methods = payload.get("methods", {})
    for method, method_payload in methods.items():
        tests = method_payload.get("paired_tests_M_minus_M2")
        if not tests:
            continue
        for metric, result in tests.items():
            ci = result.get("bootstrap_ci95", ["", ""])
            row = _base_row(filename, payload)
            row.update(
                {
                    "scope": "overall",
                    "comparison": f"{method}_minus_M2",
                    "metric": metric,
                    "mean_delta": result.get("mean_delta", ""),
                    "ci95_low": ci[0] if len(ci) > 0 else "",
                    "ci95_high": ci[1] if len(ci) > 1 else "",
                    "p_one_sided": result.get("permutation_p_one_sided", ""),
                }
            )
            rows.append(row)

    delta_summary = payload.get("delta_summary_against_M2", {})
    for method, metric_summary in delta_summary.items():
        for metric, summary in metric_summary.items():
            row = _base_row(filename, payload)
            row.update(
                {
                    "scope": "multiseed_mean",
                    "comparison": f"{method}_minus_M2",
                    "metric": metric,
                    "mean_delta": summary.get("mean", ""),
                    "ci95_low": "",
                    "ci95_high": "",
                    "p_one_sided": "",
                    "delta_std": summary.get("std", ""),
                    "delta_min": summary.get("min", ""),
                    "delta_max": summary.get("max", ""),
                }
            )
            rows.append(row)

    return rows


def extract_task_metric_rows(filename: str, payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract task-level retrieval metrics when an experiment stores them."""
    rows: list[dict[str, Any]] = []
    baseline_by_task = payload.get("baseline", {}).get("test_by_task", {})
    for task_label, metrics in baseline_by_task.items():
        row = _base_row(filename, payload)
        row.update(
            {
                "scope": "test_by_task",
                "task_label": str(task_label),
                "method": "baseline",
                "feature_set": "",
            }
        )
        for key in METRIC_KEYS:
            row[key] = metrics.get(key, "")
        rows.append(row)

    for feature_set_name, result in payload.get("feature_set_results", {}).items():
        candidate_by_task = result.get("test_results", {}).get("candidate_by_task", {})
        for task_label, metrics in candidate_by_task.items():
            row = _base_row(filename, payload)
            row.update(
                {
                    "scope": "test_by_task",
                    "task_label": str(task_label),
                    "method": f"candidate_{feature_set_name}",
                    "feature_set": feature_set_name,
                    "markov_weight": result.get("selected_markov_weight", ""),
                    "geometry_weight": result.get("selected_geometry_weight", ""),
                }
            )
            for key in METRIC_KEYS:
                row[key] = metrics.get(key, "")
            rows.append(row)
    return rows


def extract_task_delta_rows(filename: str, payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract descriptive task-level candidate-minus-baseline deltas."""
    rows: list[dict[str, Any]] = []
    baseline_by_task = payload.get("baseline", {}).get("test_by_task", {})
    if not baseline_by_task:
        return rows

    for feature_set_name, result in payload.get("feature_set_results", {}).items():
        candidate_by_task = result.get("test_results", {}).get("candidate_by_task", {})
        for task_label, baseline_metrics in baseline_by_task.items():
            candidate_metrics = candidate_by_task.get(task_label)
            if candidate_metrics is None:
                continue
            for metric in METRIC_KEYS:
                baseline_value = _try_float(baseline_metrics.get(metric))
                candidate_value = _try_float(candidate_metrics.get(metric))
                if baseline_value is None or candidate_value is None:
                    continue
                row = _base_row(filename, payload)
                row.update(
                    {
                        "scope": "test_by_task",
                        "task_label": str(task_label),
                        "comparison": f"candidate_{feature_set_name}_minus_baseline",
                        "feature_set": feature_set_name,
                        "metric": metric,
                        "mean_delta": candidate_value - baseline_value,
                    }
                )
                rows.append(row)
    return rows


def _try_float(value: Any) -> float | None:
    if value == "" or value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def summarize_delta_rows(
    delta_rows: list[dict[str, Any]],
    *,
    file_contains: str,
    metrics: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Aggregate paired deltas across independent confirmatory split seeds."""
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in delta_rows:
        if file_contains not in str(row.get("experiment_file", "")):
            continue
        metric = str(row.get("metric", ""))
        if metrics is not None and metric not in metrics:
            continue
        delta = _try_float(row.get("mean_delta"))
        if delta is None:
            continue
        feature_set = str(row.get("feature_set", ""))
        grouped[(feature_set, metric)].append(row)

    summary_rows: list[dict[str, Any]] = []
    for (feature_set, metric), rows in sorted(grouped.items()):
        deltas = [float(row["mean_delta"]) for row in rows]
        p_values = [_try_float(row.get("p_one_sided")) for row in rows]
        seeds = sorted({str(row.get("sample_seed", "")) for row in rows if row.get("sample_seed", "") != ""})
        mean_delta = sum(deltas) / len(deltas)
        summary_rows.append(
            {
                "feature_set": feature_set,
                "metric": metric,
                "n_splits": len(deltas),
                "split_seeds": ",".join(seeds),
                "mean_delta_mean": mean_delta,
                "mean_delta_std": stdev(deltas) if len(deltas) > 1 else 0.0,
                "mean_delta_min": min(deltas),
                "mean_delta_max": max(deltas),
                "significant_splits": sum(1 for p_value in p_values if p_value is not None and p_value <= 0.05),
            }
        )
    return summary_rows


def summarize_task_delta_rows(
    task_delta_rows: list[dict[str, Any]],
    *,
    file_contains: str,
    metrics: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Aggregate task-level deltas across confirmatory split seeds."""
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in task_delta_rows:
        if file_contains not in str(row.get("experiment_file", "")):
            continue
        metric = str(row.get("metric", ""))
        if metrics is not None and metric not in metrics:
            continue
        delta = _try_float(row.get("mean_delta"))
        if delta is None:
            continue
        feature_set = str(row.get("feature_set", ""))
        task_label = str(row.get("task_label", ""))
        grouped[(feature_set, task_label, metric)].append(row)

    summary_rows: list[dict[str, Any]] = []
    for (feature_set, task_label, metric), rows in sorted(grouped.items()):
        deltas = [float(row["mean_delta"]) for row in rows]
        seeds = sorted({str(row.get("sample_seed", "")) for row in rows if row.get("sample_seed", "") != ""})
        summary_rows.append(
            {
                "feature_set": feature_set,
                "task_label": task_label,
                "metric": metric,
                "n_splits": len(deltas),
                "split_seeds": ",".join(seeds),
                "mean_delta_mean": sum(deltas) / len(deltas),
                "mean_delta_std": stdev(deltas) if len(deltas) > 1 else 0.0,
                "mean_delta_min": min(deltas),
                "mean_delta_max": max(deltas),
            }
        )
    return summary_rows
