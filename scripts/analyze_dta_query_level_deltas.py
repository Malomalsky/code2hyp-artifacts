from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
from collections import defaultdict, deque
from pathlib import Path
from statistics import fmean, pstdev
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

MPL_CACHE_DIR = PROJECT_ROOT / ".matplotlib-cache"
MPL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE_DIR))
os.environ.setdefault("XDG_CACHE_HOME", str(MPL_CACHE_DIR))

from geometry_profile_research.analysis import geometry_profile_for_ast_source
from geometry_profile_research.ast_features import ast_root_paths
from geometry_profile_research.dta import (
    DtaRecord,
    load_dta_records,
    stratified_validation_test_split,
)
from geometry_profile_research.geometry_features import geometry_feature_sets
from geometry_profile_research.graphs import build_file_tree_graph
from geometry_profile_research.retrieval import (
    apply_zscore_scaler,
    combine_distance_matrices,
    euclidean_sparse_distance,
    fit_zscore_scaler,
    jensen_shannon_divergence,
    median_nonzero_distance,
    pairwise_distances,
    per_query_retrieval_records,
)
from geometry_profile_research.task_characterization import spearman_correlation
from scripts.run_dta_confirmatory_feature_sweep import _select_weight
from scripts.run_dta_confirmatory_split import _baseline_vector, _feature_vector_from_sets


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


def _load_feature_cache(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _write_feature_cache(path: Path, cache: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")


def _graph_edge_count(nodes: Sequence[str], graph: Any) -> int:
    return sum(len(graph.neighbors(node)) for node in nodes) // 2


def _farthest_node(graph: Any, start: str) -> tuple[str, int]:
    seen = {start}
    queue = deque([(start, 0)])
    farthest = (start, 0)
    while queue:
        node, distance = queue.popleft()
        if distance > farthest[1]:
            farthest = (node, distance)
        for neighbor in graph.neighbors(node):
            if neighbor in seen:
                continue
            seen.add(neighbor)
            queue.append((neighbor, distance + 1))
    return farthest


def _tree_diameter(nodes: Sequence[str], graph: Any) -> int:
    if not nodes:
        return 0
    first_endpoint, _ = _farthest_node(graph, nodes[0])
    _second_endpoint, diameter = _farthest_node(graph, first_endpoint)
    return diameter


def _branching_entropy(child_counts: Sequence[int]) -> float:
    if not child_counts:
        return 0.0
    frequencies: dict[int, int] = {}
    for count in child_counts:
        frequencies[count] = frequencies.get(count, 0) + 1
    total = len(child_counts)
    return -sum(
        (frequency / total) * math.log2(frequency / total)
        for frequency in frequencies.values()
        if frequency > 0
    )


def _fast_feature_sets_for_ast_source(code: str) -> dict[str, dict[str, float]]:
    """Compute non-distortion AST geometry features without metric embeddings."""
    unique_paths = sorted({path for path in ast_root_paths(code) if path})
    graph = build_file_tree_graph(unique_paths)
    nodes = sorted(graph.nodes)
    child_counts = [
        sum(
            1
            for neighbor in graph.neighbors(node)
            if graph.depth(neighbor) > graph.depth(node)
        )
        for node in nodes
    ]
    internal_child_counts = [count for count in child_counts if count > 0]
    leaf_count = sum(1 for count in child_counts if count == 0)
    length_only = {
        "log_node_count": math.log1p(len(nodes)),
        "log_edge_count": math.log1p(_graph_edge_count(nodes, graph)),
    }
    size_depth = {
        **length_only,
        "max_depth": float(max((graph.depth(node) for node in nodes), default=0)),
        "diameter": float(_tree_diameter(nodes, graph)),
    }
    branching = {
        "leaf_fraction": leaf_count / len(nodes) if nodes else 0.0,
        "mean_branching_factor": (
            sum(internal_child_counts) / len(internal_child_counts)
            if internal_child_counts
            else 0.0
        ),
        "max_branching_factor": float(max(child_counts, default=0)),
        "branching_entropy": _branching_entropy(child_counts),
    }
    return {
        "length_only": length_only,
        "size_depth": size_depth,
        "branching": branching,
        "all": {**length_only, **size_depth, **branching},
    }


def _profile_records_with_metadata(
    records: Sequence[DtaRecord],
    *,
    baseline_kind: str,
    feature_cache: dict[str, Any],
    require_metric_distortion: bool,
) -> tuple[
    list[DtaRecord],
    list[int],
    list[dict[str, float]],
    list[dict[str, dict[str, float]]],
    list[dict[str, float]],
    list[dict[str, str]],
]:
    kept_records: list[DtaRecord] = []
    labels: list[int] = []
    baseline_vectors: list[dict[str, float]] = []
    feature_sets_by_record: list[dict[str, dict[str, float]]] = []
    flat_features: list[dict[str, float]] = []
    errors: list[dict[str, str]] = []

    for record in records:
        try:
            full_key = f"{record.record_id}:full"
            fast_key = f"{record.record_id}:fast"
            cached = (
                feature_cache.get(full_key)
                if require_metric_distortion
                else feature_cache.get(full_key) or feature_cache.get(fast_key)
            )
            if cached is None:
                if require_metric_distortion:
                    profile = geometry_profile_for_ast_source(record.code)
                    feature_sets = geometry_feature_sets(profile)
                    cache_key = full_key
                else:
                    feature_sets = _fast_feature_sets_for_ast_source(record.code)
                    cache_key = fast_key
                all_features = dict(feature_sets["all"])
                all_features["length_scale"] = (
                    feature_sets["length_only"]["log_node_count"]
                    + feature_sets["length_only"]["log_edge_count"]
                )
                cached = {
                    "feature_sets": feature_sets,
                    "flat_features": all_features,
                }
                feature_cache[cache_key] = cached
            feature_sets = cached["feature_sets"]
            all_features = cached["flat_features"]
            kept_records.append(record)
            labels.append(record.task_id)
            baseline_vectors.append(_baseline_vector(record, baseline_kind))
            feature_sets_by_record.append(feature_sets)
            flat_features.append(all_features)
        except SyntaxError as exc:
            errors.append(
                {
                    "record_id": record.record_id,
                    "source_file": record.source_file,
                    "error": str(exc),
                }
            )

    return kept_records, labels, baseline_vectors, feature_sets_by_record, flat_features, errors


def _metric_delta_rows(
    *,
    split_seed: int,
    feature_set_name: str,
    selected_markov_weight: float,
    test_records: Sequence[DtaRecord],
    test_features: Sequence[Mapping[str, float]],
    baseline_records: Sequence[Mapping[str, float | int | str]],
    candidate_records: Sequence[Mapping[str, float | int | str]],
) -> list[dict[str, Any]]:
    baseline_by_index = {
        int(record["query_index"]): record
        for record in baseline_records
    }
    rows: list[dict[str, Any]] = []
    for candidate in candidate_records:
        query_index = int(candidate["query_index"])
        baseline = baseline_by_index[query_index]
        source_record = test_records[query_index]
        query_features = test_features[query_index]
        row: dict[str, Any] = {
            "split_seed": split_seed,
            "feature_set": feature_set_name,
            "selected_markov_weight": selected_markov_weight,
            "selected_geometry_weight": 1.0 - selected_markov_weight,
            "query_index": query_index,
            "record_id": source_record.record_id,
            "source_file": source_record.source_file,
            "task_label": f"task-{source_record.task_id:02d}",
            "baseline_top1": float(baseline["top1"]),
            "candidate_top1": float(candidate["top1"]),
            "top1_delta": float(candidate["top1"]) - float(baseline["top1"]),
            "baseline_mrr": float(baseline["reciprocal_rank"]),
            "candidate_mrr": float(candidate["reciprocal_rank"]),
            "mrr_delta": float(candidate["reciprocal_rank"]) - float(baseline["reciprocal_rank"]),
            "baseline_map": float(baseline["average_precision"]),
            "candidate_map": float(candidate["average_precision"]),
            "map_delta": float(candidate["average_precision"]) - float(baseline["average_precision"]),
        }
        for key in ("recall@1", "recall@3", "recall@5", "recall@10"):
            safe_key = key.replace("@", "_at_")
            row[f"baseline_{safe_key}"] = float(baseline[key])
            row[f"candidate_{safe_key}"] = float(candidate[key])
            row[f"{safe_key}_delta"] = float(candidate[key]) - float(baseline[key])
        for key, value in query_features.items():
            row[key] = float(value)
        rows.append(row)
    return rows


def _mean(values: Sequence[float]) -> float:
    return fmean(values) if values else 0.0


def _cluster_bootstrap_ci(
    rows: Sequence[Mapping[str, Any]],
    *,
    value_key: str,
    cluster_key: str,
    iterations: int = 1000,
    seed: int = 29,
) -> tuple[float, float]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row[cluster_key])].append(row)
    clusters = list(grouped)
    if not clusters:
        return (0.0, 0.0)

    rng = random.Random(seed)
    estimates: list[float] = []
    for _ in range(iterations):
        sampled_values: list[float] = []
        for _ in clusters:
            cluster = rng.choice(clusters)
            sampled_values.extend(float(row[value_key]) for row in grouped[cluster])
        estimates.append(_mean(sampled_values))
    estimates.sort()
    return (
        estimates[int(0.025 * len(estimates))],
        estimates[min(len(estimates) - 1, int(0.975 * len(estimates)))],
    )


def _cluster_bootstrap_spearman_ci(
    rows: Sequence[Mapping[str, Any]],
    *,
    x_key: str,
    y_key: str,
    cluster_key: str,
    iterations: int = 800,
    seed: int = 31,
) -> tuple[float, float]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row[cluster_key])].append(row)
    clusters = list(grouped)
    if len(clusters) < 3:
        return (0.0, 0.0)

    rng = random.Random(seed)
    estimates: list[float] = []
    for _ in range(iterations):
        sampled_rows: list[Mapping[str, Any]] = []
        for _ in clusters:
            sampled_rows.extend(grouped[rng.choice(clusters)])
        x = [float(row[x_key]) for row in sampled_rows]
        y = [float(row[y_key]) for row in sampled_rows]
        estimates.append(spearman_correlation(x, y))
    estimates.sort()
    return (
        estimates[int(0.025 * len(estimates))],
        estimates[min(len(estimates) - 1, int(0.975 * len(estimates)))],
    )


def _summary_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["feature_set"])].append(row)

    summary: list[dict[str, Any]] = []
    for feature_set, group in sorted(grouped.items()):
        cluster_rows = []
        by_cluster: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for row in group:
            by_cluster[f"{row['split_seed']}:{row['task_label']}"].append(row)
        for cluster, cluster_group in by_cluster.items():
            cluster_rows.append(
                {
                    "cluster": cluster,
                    "map_delta": _mean([float(row["map_delta"]) for row in cluster_group]),
                }
            )
        ci_low, ci_high = _cluster_bootstrap_ci(
            group,
            value_key="map_delta",
            cluster_key="cluster_id",
        )
        summary.append(
            {
                "feature_set": feature_set,
                "n_queries": len(group),
                "n_seed_task_clusters": len(by_cluster),
                "mean_map_delta": _mean([float(row["map_delta"]) for row in group]),
                "cluster_mean_map_delta": _mean([float(row["map_delta"]) for row in cluster_rows]),
                "cluster_bootstrap_ci95_low": ci_low,
                "cluster_bootstrap_ci95_high": ci_high,
                "mean_recall_at_10_delta": _mean(
                    [float(row["recall_at_10_delta"]) for row in group]
                ),
                "mean_top1_delta": _mean([float(row["top1_delta"]) for row in group]),
                "positive_map_delta_fraction": _mean(
                    [1.0 if float(row["map_delta"]) > 0.0 else 0.0 for row in group]
                ),
                "negative_map_delta_fraction": _mean(
                    [1.0 if float(row["map_delta"]) < 0.0 else 0.0 for row in group]
                ),
            }
        )
    return summary


def _correlation_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    predictor_keys = [
        "length_scale",
        "log_node_count",
        "log_edge_count",
        "max_depth",
        "diameter",
        "leaf_fraction",
        "mean_branching_factor",
        "max_branching_factor",
        "branching_entropy",
        "euclidean_stress",
        "hyperbolic_stress",
        "geometry_advantage",
    ]
    rows_by_feature_set: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        rows_by_feature_set[str(row["feature_set"])].append(row)

    correlations: list[dict[str, Any]] = []
    for feature_set, group in sorted(rows_by_feature_set.items()):
        y = [float(row["map_delta"]) for row in group]
        for predictor in predictor_keys:
            if predictor not in group[0]:
                continue
            x = [float(row[predictor]) for row in group]
            ci_low, ci_high = _cluster_bootstrap_spearman_ci(
                group,
                x_key=predictor,
                y_key="map_delta",
                cluster_key="cluster_id",
            )
            correlations.append(
                {
                    "feature_set": feature_set,
                    "target": "map_delta",
                    "predictor": predictor,
                    "n_queries": len(group),
                    "n_seed_task_clusters": len({row["cluster_id"] for row in group}),
                    "spearman_rho_query_level_descriptive": spearman_correlation(x, y),
                    "cluster_bootstrap_ci95_low": ci_low,
                    "cluster_bootstrap_ci95_high": ci_high,
                    "interpretation": "descriptive; query rows are clustered by split seed and task",
                }
            )
    return sorted(
        correlations,
        key=lambda row: (
            row["feature_set"],
            -abs(float(row["spearman_rho_query_level_descriptive"])),
        ),
    )


def _quantile_bins(values: Sequence[float], bins: int) -> list[float]:
    sorted_values = sorted(values)
    if not sorted_values:
        return []
    cutpoints = []
    for index in range(1, bins):
        position = int(index * len(sorted_values) / bins)
        cutpoints.append(sorted_values[min(len(sorted_values) - 1, position)])
    return cutpoints


def _bin_index(value: float, cutpoints: Sequence[float]) -> int:
    for index, cutpoint in enumerate(cutpoints):
        if value <= cutpoint:
            return index + 1
    return len(cutpoints) + 1


def _decile_rows(rows: Sequence[Mapping[str, Any]], *, bins: int = 5) -> list[dict[str, Any]]:
    features = ["length_scale", "branching_entropy", "geometry_advantage"]
    output: list[dict[str, Any]] = []
    for feature_set in sorted({str(row["feature_set"]) for row in rows}):
        group = [row for row in rows if row["feature_set"] == feature_set]
        for feature in features:
            if feature not in group[0]:
                continue
            cutpoints = _quantile_bins([float(row[feature]) for row in group], bins)
            grouped: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
            for row in group:
                grouped[_bin_index(float(row[feature]), cutpoints)].append(row)
            for bin_number, bin_rows in sorted(grouped.items()):
                output.append(
                    {
                        "feature_set": feature_set,
                        "bin_feature": feature,
                        "bin_number": bin_number,
                        "n_queries": len(bin_rows),
                        "feature_min": min(float(row[feature]) for row in bin_rows),
                        "feature_max": max(float(row[feature]) for row in bin_rows),
                        "mean_map_delta": _mean([float(row["map_delta"]) for row in bin_rows]),
                        "sd_map_delta": pstdev([float(row["map_delta"]) for row in bin_rows])
                        if len(bin_rows) > 1
                        else 0.0,
                        "positive_map_delta_fraction": _mean(
                            [1.0 if float(row["map_delta"]) > 0.0 else 0.0 for row in bin_rows]
                        ),
                    }
                )
    return output


def _save_query_delta_figure(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    preferred = ["length_only", "size_depth", "shape", "metric_distortion", "all"]
    present = [feature_set for feature_set in preferred if any(row["feature_set"] == feature_set for row in rows)]
    feature_sets = present[:3]
    if not feature_sets:
        return
    fig, axes = plt.subplots(1, len(feature_sets), figsize=(15, 4.8), sharey=True)
    if len(feature_sets) == 1:
        axes = [axes]
    for ax, feature_set in zip(axes, feature_sets):
        group = [row for row in rows if row["feature_set"] == feature_set]
        if not group:
            continue
        x = [float(row["length_scale"]) for row in group]
        y = [float(row["map_delta"]) for row in group]
        ax.scatter(x, y, s=10, alpha=0.25, color="#2f5d8c", linewidths=0)
        cutpoints = _quantile_bins(x, 6)
        grouped: dict[int, list[tuple[float, float]]] = defaultdict(list)
        for x_value, y_value in zip(x, y):
            grouped[_bin_index(x_value, cutpoints)].append((x_value, y_value))
        binned_x = [_mean([pair[0] for pair in grouped[index]]) for index in sorted(grouped)]
        binned_y = [_mean([pair[1] for pair in grouped[index]]) for index in sorted(grouped)]
        ax.plot(binned_x, binned_y, color="#9b3d2f", marker="o", linewidth=2)
        ax.axhline(0.0, color="#333333", linewidth=1)
        ax.set_title(feature_set)
        ax.set_xlabel("query AST length scale")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("candidate minus baseline MAP")
    fig.suptitle("Query-level retrieval gains are heterogeneous, not a single uniform effect")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".png"), dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def run_query_level_analysis(
    dataset_dir: Path,
    *,
    validation_per_task: int,
    test_per_task: int,
    split_seeds: Sequence[int],
    baseline_kind: str,
    feature_set_names: Sequence[str],
    weights: Sequence[float],
    feature_cache: dict[str, Any] | None = None,
    feature_cache_path: Path | None = None,
) -> dict[str, Any]:
    query_rows: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    feature_cache = feature_cache if feature_cache is not None else {}
    require_metric_distortion = any(
        feature_set_name in {"metric_distortion", "all"}
        for feature_set_name in feature_set_names
    )

    for split_seed in split_seeds:
        split = stratified_validation_test_split(
            load_dta_records(dataset_dir),
            validation_per_task=validation_per_task,
            test_per_task=test_per_task,
            seed=split_seed,
        )
        (
            validation_records,
            validation_labels,
            validation_baseline_vectors,
            validation_feature_sets,
            _validation_flat_features,
            validation_errors,
        ) = _profile_records_with_metadata(
            split["validation"],
            baseline_kind=baseline_kind,
            feature_cache=feature_cache,
            require_metric_distortion=require_metric_distortion,
        )
        (
            test_records,
            test_labels,
            test_baseline_vectors,
            test_feature_sets,
            test_flat_features,
            test_errors,
        ) = _profile_records_with_metadata(
            split["test"],
            baseline_kind=baseline_kind,
            feature_cache=feature_cache,
            require_metric_distortion=require_metric_distortion,
        )
        if feature_cache_path is not None:
            _write_feature_cache(feature_cache_path, feature_cache)

        validation_baseline_distances = pairwise_distances(
            validation_baseline_vectors,
            jensen_shannon_divergence,
        )
        test_baseline_distances = pairwise_distances(
            test_baseline_vectors,
            jensen_shannon_divergence,
        )
        baseline_scale = median_nonzero_distance(validation_baseline_distances)
        baseline_records = per_query_retrieval_records(
            test_labels,
            test_baseline_distances,
            k_values=(1, 3, 5, 10),
        )

        diagnostics.append(
            {
                "split_seed": split_seed,
                "validation_records": len(validation_records),
                "test_records": len(test_records),
                "validation_syntax_errors": len(validation_errors),
                "test_syntax_errors": len(test_errors),
                "baseline_scale": baseline_scale,
                "feature_cache_entries": len(feature_cache),
            }
        )

        for feature_set_name in feature_set_names:
            validation_feature_vectors = [
                _feature_vector_from_sets(feature_sets, feature_set_name)
                for feature_sets in validation_feature_sets
            ]
            test_feature_vectors = [
                _feature_vector_from_sets(feature_sets, feature_set_name)
                for feature_sets in test_feature_sets
            ]
            scaler = fit_zscore_scaler(validation_feature_vectors)
            scaled_validation_features = apply_zscore_scaler(validation_feature_vectors, scaler)
            scaled_test_features = apply_zscore_scaler(test_feature_vectors, scaler)
            validation_feature_distances = pairwise_distances(
                scaled_validation_features,
                euclidean_sparse_distance,
            )
            test_feature_distances = pairwise_distances(
                scaled_test_features,
                euclidean_sparse_distance,
            )
            feature_scale = median_nonzero_distance(validation_feature_distances)
            selected_weight, _validation_results = _select_weight(
                validation_labels,
                validation_baseline_distances,
                validation_feature_distances,
                weights=list(weights),
                baseline_scale=baseline_scale,
                feature_scale=feature_scale,
            )
            candidate_distances = combine_distance_matrices(
                test_baseline_distances,
                test_feature_distances,
                left_weight=selected_weight,
                left_scale=baseline_scale,
                right_scale=feature_scale,
            )
            candidate_records = per_query_retrieval_records(
                test_labels,
                candidate_distances,
                k_values=(1, 3, 5, 10),
            )
            rows = _metric_delta_rows(
                split_seed=split_seed,
                feature_set_name=feature_set_name,
                selected_markov_weight=selected_weight,
                test_records=test_records,
                test_features=test_flat_features,
                baseline_records=baseline_records,
                candidate_records=candidate_records,
            )
            for row in rows:
                row["cluster_id"] = f"{row['split_seed']}:{row['task_label']}"
                row["feature_scale"] = feature_scale
            query_rows.extend(rows)

    return {
        "query_rows": query_rows,
        "summary_rows": _summary_rows(query_rows),
        "correlation_rows": _correlation_rows(query_rows),
        "decile_rows": _decile_rows(query_rows),
        "diagnostics": diagnostics,
        "feature_cache": feature_cache,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build query-level diagnostics for DTA AST geometry retrieval deltas."
    )
    parser.add_argument("--dataset-dir", type=Path, default=Path("data/dta_zenodo_7799972/extracted"))
    parser.add_argument("--validation-per-task", type=int, default=20)
    parser.add_argument("--test-per-task", type=int, default=50)
    parser.add_argument("--split-seeds", nargs="+", type=int, default=[101, 202, 303])
    parser.add_argument(
        "--baseline-kind",
        choices=["transition_count", "flat_markov"],
        default="transition_count",
    )
    parser.add_argument(
        "--feature-sets",
        nargs="+",
        default=["length_only", "size_depth", "branching", "shape"],
    )
    parser.add_argument(
        "--weights",
        nargs="+",
        type=float,
        default=[1.0, 0.99, 0.98, 0.97, 0.95, 0.9, 0.85, 0.8],
    )
    parser.add_argument("--output-query-rows", type=Path, default=Path("reports/query_level_deltas.csv"))
    parser.add_argument("--output-summary", type=Path, default=Path("reports/query_level_summary.csv"))
    parser.add_argument(
        "--output-correlations",
        type=Path,
        default=Path("reports/query_level_feature_correlations.csv"),
    )
    parser.add_argument("--output-deciles", type=Path, default=Path("reports/query_level_decile_summary.csv"))
    parser.add_argument("--output-diagnostics", type=Path, default=Path("reports/query_level_diagnostics.json"))
    parser.add_argument("--figure-output", type=Path, default=Path("figures/fig09_query_level_delta_profile"))
    parser.add_argument("--feature-cache", type=Path, default=Path("outputs/query_level_feature_cache.json"))
    args = parser.parse_args()

    feature_cache = _load_feature_cache(args.feature_cache)
    result = run_query_level_analysis(
        args.dataset_dir,
        validation_per_task=args.validation_per_task,
        test_per_task=args.test_per_task,
        split_seeds=args.split_seeds,
        baseline_kind=args.baseline_kind,
        feature_set_names=args.feature_sets,
        weights=args.weights,
        feature_cache=feature_cache,
        feature_cache_path=args.feature_cache,
    )
    _write_csv(args.output_query_rows, result["query_rows"])
    _write_csv(args.output_summary, result["summary_rows"])
    _write_csv(args.output_correlations, result["correlation_rows"])
    _write_csv(args.output_deciles, result["decile_rows"])
    args.output_diagnostics.parent.mkdir(parents=True, exist_ok=True)
    args.output_diagnostics.write_text(
        json.dumps(result["diagnostics"], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _write_feature_cache(args.feature_cache, result["feature_cache"])
    _save_query_delta_figure(args.figure_output, result["query_rows"])

    print(f"wrote {args.output_query_rows}")
    print(f"wrote {args.output_summary}")
    print(f"wrote {args.output_correlations}")
    print(f"wrote {args.output_deciles}")
    print(f"wrote {args.feature_cache}")
    print(f"wrote {args.figure_output.with_suffix('.png')}")


if __name__ == "__main__":
    main()
