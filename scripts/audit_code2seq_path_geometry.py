from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median
from typing import Iterable

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from geometry_profile_research.code2hyp_data import RawCode2VecContext, parse_code2vec_line
from geometry_profile_research.code2hyp_real_dataset import Code2SeqPreprocessedInventory


@dataclass(frozen=True)
class ContextView:
    split: str
    record_index: int
    context_index: int
    start_token: str
    end_token: str
    full_path: tuple[str, ...]
    truncated_path: tuple[str, ...]


def _choose_records(path: Path, limit: int | None, seed: int | None) -> list[tuple[int, str]]:
    """Return `(original_line_index, line)` pairs.

    If `seed` is None, use the first `limit` lines. If a seed is provided, use
    reservoir sampling so the audit can match the confirmatory validation
    protocol instead of silently inheriting file-order bias.
    """
    if limit is None:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            return [(index, line) for index, line in enumerate(handle)]
    if limit < 0:
        raise ValueError("limit must be non-negative")
    if seed is None:
        records: list[tuple[int, str]] = []
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            for index, line in enumerate(handle):
                records.append((index, line))
                if len(records) >= limit:
                    break
        return records

    rng = random.Random(seed)
    reservoir: list[tuple[int, str]] = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for index, line in enumerate(handle):
            item = (index, line)
            if index < limit:
                reservoir.append(item)
                continue
            replacement_index = rng.randint(0, index)
            if replacement_index < limit:
                reservoir[replacement_index] = item
    return reservoir


def _iter_context_views(
    split: str,
    path: Path,
    record_limit: int | None,
    sample_seed: int | None,
    max_contexts: int,
    max_path_length: int,
) -> Iterable[ContextView]:
    for original_index, line in _choose_records(path, record_limit, sample_seed):
        try:
            record = parse_code2vec_line(line)
        except ValueError:
            continue
        for context_index, context in enumerate(record.contexts[:max_contexts]):
            yield ContextView(
                split=split,
                record_index=original_index,
                context_index=context_index,
                start_token=context.start_token,
                end_token=context.end_token,
                full_path=context.ast_path,
                truncated_path=context.ast_path[:max_path_length],
            )


def _prefix_distance(left: tuple[str, ...], right: tuple[str, ...]) -> int:
    lcp = 0
    for left_token, right_token in zip(left, right):
        if left_token != right_token:
            break
        lcp += 1
    return len(left) + len(right) - 2 * lcp


def _edit_distance(left: tuple[str, ...], right: tuple[str, ...]) -> int:
    if left == right:
        return 0
    if not left:
        return len(right)
    if not right:
        return len(left)
    previous = list(range(len(right) + 1))
    for row_index, left_token in enumerate(left, start=1):
        current = [row_index]
        for column_index, right_token in enumerate(right, start=1):
            substitution_cost = 0 if left_token == right_token else 1
            current.append(
                min(
                    previous[column_index] + 1,
                    current[column_index - 1] + 1,
                    previous[column_index - 1] + substitution_cost,
                )
            )
        previous = current
    return previous[-1]


def _ngrams(path: tuple[str, ...], n: int = 2) -> set[tuple[str, ...]]:
    if len(path) < n:
        return {path} if path else set()
    return {path[index : index + n] for index in range(len(path) - n + 1)}


def _jaccard_bigram_distance(left: tuple[str, ...], right: tuple[str, ...]) -> float:
    left_ngrams = _ngrams(left, n=2)
    right_ngrams = _ngrams(right, n=2)
    union = left_ngrams | right_ngrams
    if not union:
        return 0.0
    return 1.0 - (len(left_ngrams & right_ngrams) / len(union))


def _distance_by_metric(metric: str, left: tuple[str, ...], right: tuple[str, ...]) -> float:
    if metric == "prefix_tree":
        return float(_prefix_distance(left, right))
    if metric == "edit":
        return float(_edit_distance(left, right))
    if metric == "jaccard_bigrams":
        return float(_jaccard_bigram_distance(left, right))
    raise ValueError(f"unknown path metric: {metric}")


def _four_point_delta_for_metric(
    metric: str,
    a: tuple[str, ...],
    b: tuple[str, ...],
    c: tuple[str, ...],
    d: tuple[str, ...],
) -> float:
    sums = sorted(
        (
            _distance_by_metric(metric, a, b) + _distance_by_metric(metric, c, d),
            _distance_by_metric(metric, a, c) + _distance_by_metric(metric, b, d),
            _distance_by_metric(metric, a, d) + _distance_by_metric(metric, b, c),
        )
    )
    return (sums[2] - sums[1]) / 2.0


def _pair_count(count: int) -> int:
    return count * (count - 1) // 2


def _quantile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    position = (len(sorted_values) - 1) * q
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return float(sorted_values[lower])
    weight = position - lower
    return float(sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight)


def _entropy(counter: Counter[int]) -> float:
    total = sum(counter.values())
    if total == 0:
        return 0.0
    return -sum((count / total) * math.log2(count / total) for count in counter.values())


def _average_ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        average_rank = (start + end - 1) / 2.0
        for position in range(start, end):
            ranks[order[position]] = average_rank
        start = end
    return ranks


def _spearman(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        raise ValueError("rank correlation inputs must have the same length")
    if len(left) < 2:
        return 0.0
    left_ranks = _average_ranks(left)
    right_ranks = _average_ranks(right)
    left_mean = mean(left_ranks)
    right_mean = mean(right_ranks)
    left_centered = [value - left_mean for value in left_ranks]
    right_centered = [value - right_mean for value in right_ranks]
    numerator = sum(l_value * r_value for l_value, r_value in zip(left_centered, right_centered))
    left_norm = math.sqrt(sum(value * value for value in left_centered))
    right_norm = math.sqrt(sum(value * value for value in right_centered))
    if left_norm <= 1e-12 or right_norm <= 1e-12:
        return 0.0
    return numerator / (left_norm * right_norm)


def audit_contexts(contexts: list[ContextView], k: int, hyperbolicity_samples: int, seed: int) -> dict[str, object]:
    context_count = len(contexts)
    full_counter = Counter(context.full_path for context in contexts)
    truncated_counter = Counter(context.truncated_path for context in contexts)
    truncated_to_full: dict[tuple[str, ...], Counter[tuple[str, ...]]] = defaultdict(Counter)
    for context in contexts:
        truncated_to_full[context.truncated_path][context.full_path] += 1

    total_context_pairs = _pair_count(context_count)
    same_truncated_pairs = sum(_pair_count(count) for count in truncated_counter.values())
    same_full_pairs = sum(_pair_count(count) for count in full_counter.values())
    ambiguous_truncated_pair_count = sum(
        _pair_count(sum(fulls.values())) - sum(_pair_count(count) for count in fulls.values())
        for fulls in truncated_to_full.values()
    )

    ambiguous_classes = {key: fulls for key, fulls in truncated_to_full.items() if len(fulls) > 1}
    contexts_in_ambiguous_classes = sum(sum(fulls.values()) for fulls in ambiguous_classes.values())
    full_variants_per_truncated = [len(fulls) for fulls in truncated_to_full.values()]
    context_count_per_truncated = list(truncated_counter.values())

    by_record: dict[tuple[str, int], list[ContextView]] = defaultdict(list)
    for context in contexts:
        by_record[(context.split, context.record_index)].append(context)

    path_metrics = ("prefix_tree", "edit", "jaccard_bigrams")
    metric_distance_values: dict[str, list[float]] = {metric: [] for metric in path_metrics}
    metric_distance_counters: dict[str, Counter[float]] = {metric: Counter() for metric in path_metrics}
    distance_counter: Counter[int] = Counter()
    record_pair_count = 0
    zero_distance_pairs = 0
    ambiguous_zero_pairs = 0
    tie_sizes_at_k: list[int] = []
    unique_distance_levels_per_record: list[int] = []
    pair_counts_per_record: list[int] = []

    for record_contexts in by_record.values():
        if len(record_contexts) <= 1:
            continue
        pair_counts_per_record.append(_pair_count(len(record_contexts)))
        record_distances: list[int] = []
        for left_index, left in enumerate(record_contexts):
            anchor_distances: list[int] = []
            for right_index, right in enumerate(record_contexts):
                if left_index == right_index:
                    continue
                distance = _prefix_distance(left.truncated_path, right.truncated_path)
                anchor_distances.append(distance)
            if anchor_distances:
                effective_k = min(k, len(anchor_distances))
                kth = sorted(anchor_distances)[effective_k - 1]
                tie_sizes_at_k.append(sum(1 for distance in anchor_distances if distance <= kth))

        for left_index in range(len(record_contexts)):
            for right_index in range(left_index + 1, len(record_contexts)):
                left = record_contexts[left_index]
                right = record_contexts[right_index]
                distances = {
                    metric: _distance_by_metric(metric, left.truncated_path, right.truncated_path)
                    for metric in path_metrics
                }
                distance = int(distances["prefix_tree"])
                record_distances.append(distance)
                distance_counter[distance] += 1
                for metric, metric_distance in distances.items():
                    metric_distance_values[metric].append(float(metric_distance))
                    metric_distance_counters[metric][round(float(metric_distance), 8)] += 1
                record_pair_count += 1
                if distance == 0:
                    zero_distance_pairs += 1
                    if left.full_path != right.full_path:
                        ambiguous_zero_pairs += 1
        if record_distances:
            unique_distance_levels_per_record.append(len(set(record_distances)))

    unique_truncated_paths = list(truncated_counter)
    rng = random.Random(seed)
    hyperbolicity: dict[str, dict[str, list[float]]] = {
        metric: {"delta": [], "normalized_delta": []} for metric in path_metrics
    }
    if len(unique_truncated_paths) >= 4:
        diameter_candidates = unique_truncated_paths[: min(len(unique_truncated_paths), 400)]
        diameters = {
            metric: max(
                _distance_by_metric(metric, left, right)
                for index, left in enumerate(diameter_candidates)
                for right in diameter_candidates[index + 1 :]
            )
            for metric in path_metrics
        }
        for _ in range(hyperbolicity_samples):
            quartet = rng.sample(unique_truncated_paths, 4)
            for metric in path_metrics:
                delta = _four_point_delta_for_metric(metric, *quartet)
                hyperbolicity[metric]["delta"].append(delta)
                diameter = diameters[metric]
                hyperbolicity[metric]["normalized_delta"].append(delta / diameter if diameter else 0.0)

    top_distance_levels = [
        {"distance": distance, "pairs": count, "share": count / record_pair_count if record_pair_count else 0.0}
        for distance, count in distance_counter.most_common(12)
    ]
    metric_level_summaries = {
        metric: [
            {"distance": distance, "pairs": count, "share": count / record_pair_count if record_pair_count else 0.0}
            for distance, count in metric_distance_counters[metric].most_common(12)
        ]
        for metric in path_metrics
    }
    cross_metric_spearman = {
        f"{left_metric}_vs_{right_metric}": _spearman(
            metric_distance_values[left_metric],
            metric_distance_values[right_metric],
        )
        for left_index, left_metric in enumerate(path_metrics)
        for right_metric in path_metrics[left_index + 1 :]
    }
    hyperbolicity_summary = {
        metric: {
            "samples": len(values["delta"]),
            "delta_mean": mean(values["delta"]) if values["delta"] else 0.0,
            "delta_max": max(values["delta"]) if values["delta"] else 0.0,
            "delta_over_diameter_mean": mean(values["normalized_delta"]) if values["normalized_delta"] else 0.0,
            "delta_over_diameter_max": max(values["normalized_delta"]) if values["normalized_delta"] else 0.0,
        }
        for metric, values in hyperbolicity.items()
    }

    return {
        "contexts": context_count,
        "records_with_at_least_one_context": len(by_record),
        "unique_full_paths": len(full_counter),
        "unique_truncated_paths": len(truncated_counter),
        "full_path_collision_rate_pairwise": same_full_pairs / total_context_pairs if total_context_pairs else 0.0,
        "truncated_path_collision_rate_pairwise": same_truncated_pairs / total_context_pairs if total_context_pairs else 0.0,
        "ambiguous_truncated_pair_rate": (
            ambiguous_truncated_pair_count / total_context_pairs if total_context_pairs else 0.0
        ),
        "truncated_classes": len(truncated_to_full),
        "ambiguous_truncated_classes": len(ambiguous_classes),
        "ambiguous_truncated_class_rate": len(ambiguous_classes) / len(truncated_to_full) if truncated_to_full else 0.0,
        "contexts_in_ambiguous_truncated_classes": contexts_in_ambiguous_classes,
        "contexts_in_ambiguous_truncated_classes_rate": (
            contexts_in_ambiguous_classes / context_count if context_count else 0.0
        ),
        "mean_full_variants_per_truncated_path": mean(full_variants_per_truncated)
        if full_variants_per_truncated
        else 0.0,
        "p95_full_variants_per_truncated_path": _quantile([float(value) for value in full_variants_per_truncated], 0.95),
        "max_full_variants_per_truncated_path": max(full_variants_per_truncated) if full_variants_per_truncated else 0,
        "mean_contexts_per_truncated_path": mean(context_count_per_truncated) if context_count_per_truncated else 0.0,
        "p95_contexts_per_truncated_path": _quantile([float(value) for value in context_count_per_truncated], 0.95),
        "max_contexts_per_truncated_path": max(context_count_per_truncated) if context_count_per_truncated else 0,
        "within_record_pairs": record_pair_count,
        "within_record_zero_distance_pairs": zero_distance_pairs,
        "within_record_zero_distance_pair_rate": zero_distance_pairs / record_pair_count if record_pair_count else 0.0,
        "within_record_ambiguous_zero_pairs": ambiguous_zero_pairs,
        "within_record_ambiguous_zero_pair_rate": ambiguous_zero_pairs / record_pair_count if record_pair_count else 0.0,
        "prefix_distance_unique_levels_global": len(distance_counter),
        "prefix_distance_entropy_bits": _entropy(distance_counter),
        "metric_unique_levels_global": {
            metric: len(metric_distance_counters[metric])
            for metric in path_metrics
        },
        "metric_entropy_bits": {
            metric: _entropy(metric_distance_counters[metric])
            for metric in path_metrics
        },
        "cross_metric_spearman": cross_metric_spearman,
        "top_prefix_distance_levels": top_distance_levels,
        "top_metric_distance_levels": metric_level_summaries,
        "mean_unique_distance_levels_per_record": mean(unique_distance_levels_per_record)
        if unique_distance_levels_per_record
        else 0.0,
        "median_unique_distance_levels_per_record": median(unique_distance_levels_per_record)
        if unique_distance_levels_per_record
        else 0.0,
        "mean_pair_count_per_record": mean(pair_counts_per_record) if pair_counts_per_record else 0.0,
        "tie_k": k,
        "mean_tie_set_size_at_k": mean(tie_sizes_at_k) if tie_sizes_at_k else 0.0,
        "median_tie_set_size_at_k": median(tie_sizes_at_k) if tie_sizes_at_k else 0.0,
        "p95_tie_set_size_at_k": _quantile([float(value) for value in tie_sizes_at_k], 0.95),
        "max_tie_set_size_at_k": max(tie_sizes_at_k) if tie_sizes_at_k else 0,
        "mean_tie_expansion_ratio_at_k": mean([value / k for value in tie_sizes_at_k]) if tie_sizes_at_k else 0.0,
        "four_point_hyperbolicity_by_metric": hyperbolicity_summary,
        "four_point_hyperbolicity_samples": hyperbolicity_summary["prefix_tree"]["samples"],
        "four_point_delta_mean": hyperbolicity_summary["prefix_tree"]["delta_mean"],
        "four_point_delta_max": hyperbolicity_summary["prefix_tree"]["delta_max"],
        "four_point_delta_over_diameter_mean": hyperbolicity_summary["prefix_tree"]["delta_over_diameter_mean"],
        "four_point_delta_over_diameter_max": hyperbolicity_summary["prefix_tree"]["delta_over_diameter_max"],
    }


def _write_markdown(
    output_path: Path,
    config: dict[str, object],
    split_summaries: dict[str, dict[str, object]],
    combined_summary: dict[str, object],
) -> None:
    def fmt(value: object) -> str:
        if isinstance(value, float):
            return f"{value:.6g}"
        return str(value)

    lines = [
        "# AST-label path geometry audit",
        "",
        "This audit describes the structural object available in the official",
        "preprocessed code2seq Java-small corpus: serialized terminal-to-terminal",
        "AST-label paths. It does not claim access to original AST node IDs.",
        "",
        "## Configuration",
        "",
    ]
    for key, value in config.items():
        lines.append(f"- {key}: `{value}`")

    lines.extend(
        [
            "",
            "## Split-level summary",
            "",
            "| Split | Contexts | Unique full paths | Unique truncated paths | Ambiguous truncated classes | Contexts in ambiguous classes | Within-record zero-distance pairs | Mean tie-set size@k | Mean four-point delta |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for split, summary in split_summaries.items():
        lines.append(
            "| "
            f"{split} | "
            f"{summary['contexts']} | "
            f"{summary['unique_full_paths']} | "
            f"{summary['unique_truncated_paths']} | "
            f"{summary['ambiguous_truncated_classes']} | "
            f"{fmt(summary['contexts_in_ambiguous_truncated_classes_rate'])} | "
            f"{fmt(summary['within_record_zero_distance_pair_rate'])} | "
            f"{fmt(summary['mean_tie_set_size_at_k'])} | "
            f"{fmt(summary['four_point_delta_mean'])} |"
        )

    lines.extend(["", "## Combined summary", ""])
    important_keys = [
        "contexts",
        "records_with_at_least_one_context",
        "unique_full_paths",
        "unique_truncated_paths",
        "full_path_collision_rate_pairwise",
        "truncated_path_collision_rate_pairwise",
        "ambiguous_truncated_class_rate",
        "contexts_in_ambiguous_truncated_classes_rate",
        "within_record_zero_distance_pair_rate",
        "within_record_ambiguous_zero_pair_rate",
        "prefix_distance_unique_levels_global",
        "prefix_distance_entropy_bits",
        "metric_unique_levels_global",
        "metric_entropy_bits",
        "cross_metric_spearman",
        "mean_unique_distance_levels_per_record",
        "mean_tie_set_size_at_k",
        "p95_tie_set_size_at_k",
        "mean_tie_expansion_ratio_at_k",
        "four_point_hyperbolicity_by_metric",
        "four_point_delta_mean",
        "four_point_delta_max",
        "four_point_delta_over_diameter_max",
    ]
    for key in important_keys:
        lines.append(f"- {key}: `{fmt(combined_summary.get(key, 'n/a'))}`")

    lines.extend(["", "## Most frequent within-record prefix distances", ""])
    lines.append("| Distance | Pair count | Share |")
    lines.append("|---:|---:|---:|")
    for row in combined_summary["top_prefix_distance_levels"]:  # type: ignore[index]
        lines.append(f"| {row['distance']} | {row['pairs']} | {fmt(row['share'])} |")

    lines.extend(["", "## Cross-metric geometry", ""])
    lines.append("| Metric | Unique levels | Entropy, bits | Mean four-point delta | Max normalized delta |")
    lines.append("|---|---:|---:|---:|---:|")
    metric_unique_levels = combined_summary["metric_unique_levels_global"]  # type: ignore[index]
    metric_entropy = combined_summary["metric_entropy_bits"]  # type: ignore[index]
    metric_hyperbolicity = combined_summary["four_point_hyperbolicity_by_metric"]  # type: ignore[index]
    for metric in ("prefix_tree", "edit", "jaccard_bigrams"):
        hyperbolicity = metric_hyperbolicity[metric]
        lines.append(
            "| "
            f"{metric} | "
            f"{metric_unique_levels[metric]} | "
            f"{fmt(metric_entropy[metric])} | "
            f"{fmt(hyperbolicity['delta_mean'])} | "
            f"{fmt(hyperbolicity['delta_over_diameter_max'])} |"
        )

    lines.extend(["", "### Spearman correlation between path metrics", ""])
    lines.append("| Metric pair | Spearman rho |")
    lines.append("|---|---:|")
    for pair, rho in combined_summary["cross_metric_spearman"].items():  # type: ignore[union-attr]
        lines.append(f"| {pair} | {fmt(rho)} |")

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The current structural target is best described as `truncated AST-label",
            "prefix-trie distance`. The sampled four-point hyperbolicity is zero,",
            "as expected for a tree metric on the prefix trie. This supports the",
            "mathematical interpretation of B44 as preserving prefix-trie geometry,",
            "not necessarily the full original AST path geometry.",
            "",
            "The cross-metric block checks whether the chosen structural proxy",
            "is interchangeable with edit and n-gram-overlap relations. Low",
            "cross-metric correlations or different hyperbolicity profiles mean",
            "that model claims must name the preserved relation explicitly.",
            "",
            "If ambiguous truncated classes or within-record zero-distance pairs are",
            "non-negligible, the metric must be treated as a proxy or pseudometric",
            "over the original AST paths. A stronger follow-up needs node IDs and",
            "full untruncated paths to evaluate endpoint, edge-overlap, Hausdorff,",
            "and edit-distance metrics independently.",
            "",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit code2seq AST-label path geometry.")
    parser.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "data/code2seq_java_small")
    parser.add_argument("--splits", default="train,val,test")
    parser.add_argument("--record-limit", type=int, default=25000)
    parser.add_argument("--sample-seed", type=int, default=777)
    parser.add_argument("--max-contexts", type=int, default=30)
    parser.add_argument("--max-path-length", type=int, default=8)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--hyperbolicity-samples", type=int, default=2000)
    parser.add_argument("--output-json", type=Path, default=PROJECT_ROOT / "outputs/code2seq_path_geometry_audit_25k_seed777.json")
    parser.add_argument("--output-md", type=Path, default=PROJECT_ROOT / "reports/code2seq_path_geometry_audit_25k_seed777.md")
    parser.add_argument("--distance-csv", type=Path, default=PROJECT_ROOT / "reports/code2seq_path_geometry_audit_distance_levels_25k_seed777.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    inventory = Code2SeqPreprocessedInventory.from_directory(args.data_root)
    requested_splits = [split.strip() for split in args.splits.split(",") if split.strip()]
    split_summaries: dict[str, dict[str, object]] = {}
    all_contexts: list[ContextView] = []
    for offset, split in enumerate(requested_splits):
        if split not in inventory.split_paths:
            raise ValueError(f"split {split!r} not found under {args.data_root}")
        contexts = list(
            _iter_context_views(
                split=split,
                path=inventory.split_paths[split],
                record_limit=args.record_limit,
                sample_seed=args.sample_seed + offset if args.sample_seed is not None else None,
                max_contexts=args.max_contexts,
                max_path_length=args.max_path_length,
            )
        )
        split_summaries[split] = audit_contexts(
            contexts,
            k=args.k,
            hyperbolicity_samples=args.hyperbolicity_samples,
            seed=args.sample_seed + 1000 + offset,
        )
        all_contexts.extend(contexts)

    combined_summary = audit_contexts(
        all_contexts,
        k=args.k,
        hyperbolicity_samples=args.hyperbolicity_samples,
        seed=args.sample_seed + 2000,
    )
    config = {
        "data_root": str(args.data_root),
        "splits": requested_splits,
        "record_limit_per_split": args.record_limit,
        "sample_seed": args.sample_seed,
        "max_contexts": args.max_contexts,
        "max_path_length": args.max_path_length,
        "tie_k": args.k,
        "hyperbolicity_samples": args.hyperbolicity_samples,
    }
    payload = {
        "config": config,
        "split_summaries": split_summaries,
        "combined_summary": combined_summary,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    _write_markdown(args.output_md, config, split_summaries, combined_summary)
    args.distance_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.distance_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["distance", "pairs", "share"])
        writer.writeheader()
        writer.writerows(combined_summary["top_prefix_distance_levels"])  # type: ignore[arg-type]
    print(f"Wrote {args.output_json}")
    print(f"Wrote {args.output_md}")
    print(f"Wrote {args.distance_csv}")


if __name__ == "__main__":
    main()
