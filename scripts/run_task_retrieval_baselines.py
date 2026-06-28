#!/usr/bin/env python3
"""Run simple task-level retrieval baselines on Code2Hyp split artifacts.

The script uses exactly the train/query/gallery split stored in the existing
confirmatory JSON files. It evaluates deterministic non-neural baselines that
are useful for manuscript sanity checks: AST node bags, AST path label bigrams,
lexical token bags, and identifier/literal-stripped token shapes.
"""

from __future__ import annotations

import argparse
import io
import json
import math
import sys
import tokenize
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from geometry_profile_research.python_raw_ast import parse_python_ast_tree
from geometry_profile_research.raw_ast import terminal_to_terminal_paths


@dataclass(frozen=True)
class BaselineInput:
    dataset: str
    path: Path


BASELINES = ("random_expected", "ast_node_bag", "ast_path_bigram_bag", "token_bag", "token_shape_bag")


def run_baselines(inputs: Sequence[BaselineInput]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for item in inputs:
        payload = json.loads(item.path.read_text(encoding="utf-8"))
        split = payload["split"]
        query_ids = [str(value) for value in split["query_ids"]]
        gallery_ids = [str(value) for value in split["gallery_ids"]]
        seed = int(payload["config"]["seed"])
        feature_cache: dict[tuple[str, str], Counter[str]] = {}
        for baseline in BASELINES:
            if baseline == "random_expected":
                gallery_count = len(gallery_ids)
                expected_mrr = sum(1.0 / rank for rank in range(1, gallery_count + 1)) / gallery_count
                expected_recall_at_1 = 1.0 / gallery_count
                expected_recall_at_5 = min(5, gallery_count) / gallery_count
                expected_mean_rank = (gallery_count + 1.0) / 2.0
                for query_id in query_ids:
                    rows.append(
                        {
                            "dataset": item.dataset,
                            "source_file": str(item.path),
                            "seed": seed,
                            "baseline": baseline,
                            "query_id": query_id,
                            "query_task": _task_from_id(query_id),
                            "rank": expected_mean_rank,
                            "mrr": expected_mrr,
                            "recall_at_1": expected_recall_at_1,
                            "recall_at_5": expected_recall_at_5,
                            "mean_rank": expected_mean_rank,
                        }
                    )
                continue
            gallery_features = {
                gallery_id: _features_for_id(gallery_id, baseline, feature_cache)
                for gallery_id in gallery_ids
            }
            for query_id in query_ids:
                query_task = _task_from_id(query_id)
                query_features = _features_for_id(query_id, baseline, feature_cache)
                scored: list[tuple[float, str]] = []
                for gallery_id, candidate_features in gallery_features.items():
                    scored.append((_cosine(query_features, candidate_features), gallery_id))
                scored.sort(key=lambda item: (-item[0], item[1]))
                rank = _first_positive_rank(scored, query_task)
                rows.append(
                    {
                        "dataset": item.dataset,
                        "source_file": str(item.path),
                        "seed": seed,
                        "baseline": baseline,
                        "query_id": query_id,
                        "query_task": query_task,
                        "rank": rank,
                        "mrr": 1.0 / rank,
                        "recall_at_1": 1.0 if rank <= 1 else 0.0,
                        "recall_at_5": 1.0 if rank <= 5 else 0.0,
                        "mean_rank": float(rank),
                    }
                )
    return {"inputs": [item.__dict__ | {"path": str(item.path)} for item in inputs], "query_rows": rows, "cell_summaries": _summaries(rows)}


def _features_for_id(item_id: str, baseline: str, cache: dict[tuple[str, str], Counter[str]]) -> Counter[str]:
    key = (item_id, baseline)
    if key not in cache:
        path = _path_from_id(item_id)
        if baseline == "ast_node_bag":
            cache[key] = _ast_node_bag(path)
        elif baseline == "ast_path_bigram_bag":
            cache[key] = _ast_path_bigram_bag(path)
        elif baseline == "token_bag":
            cache[key] = _token_bag(path, strip_values=False)
        elif baseline == "token_shape_bag":
            cache[key] = _token_bag(path, strip_values=True)
        else:
            raise ValueError(f"unknown baseline: {baseline}")
    return cache[key]


def _ast_node_bag(path: Path) -> Counter[str]:
    tree = parse_python_ast_tree(path.read_text(encoding="utf-8"))
    return Counter(tree.labels.values())


def _ast_path_bigram_bag(path: Path, *, max_paths: int = 128) -> Counter[str]:
    tree = parse_python_ast_tree(path.read_text(encoding="utf-8"))
    counts: Counter[str] = Counter()
    for path_object in terminal_to_terminal_paths(tree, max_paths=max_paths):
        labels = [tree.labels.get(node, "") for node in path_object.nodes]
        for left, right in zip(labels, labels[1:]):
            counts[f"{left}>{right}"] += 1
    if not counts:
        counts.update(tree.labels.values())
    return counts


def _token_bag(path: Path, *, strip_values: bool) -> Counter[str]:
    source = path.read_text(encoding="utf-8")
    counts: Counter[str] = Counter()
    ignored = {
        tokenize.ENCODING,
        tokenize.ENDMARKER,
        tokenize.INDENT,
        tokenize.DEDENT,
        tokenize.NEWLINE,
        tokenize.NL,
        tokenize.COMMENT,
    }
    for token in tokenize.generate_tokens(io.StringIO(source).readline):
        if token.type in ignored:
            continue
        if strip_values and token.type == tokenize.NAME:
            value = "NAME"
        elif strip_values and token.type == tokenize.NUMBER:
            value = "NUMBER"
        elif strip_values and token.type == tokenize.STRING:
            value = "STRING"
        else:
            value = token.string
        counts[value] += 1
    return counts


def _cosine(left: Counter[str], right: Counter[str]) -> float:
    if not left or not right:
        return 0.0
    dot = sum(value * right.get(key, 0) for key, value in left.items())
    left_norm = math.sqrt(sum(value * value for value in left.values()))
    right_norm = math.sqrt(sum(value * value for value in right.values()))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return dot / (left_norm * right_norm)


def _first_positive_rank(scored: Sequence[tuple[float, str]], query_task: str) -> int:
    for index, (_, gallery_id) in enumerate(scored, start=1):
        if _task_from_id(gallery_id) == query_task:
            return index
    raise ValueError(f"no positive gallery item for task {query_task!r}")


def _path_from_id(item_id: str) -> Path:
    return PROJECT_ROOT / item_id.split(":", 1)[1].rsplit(":module:", 1)[0]


def _task_from_id(item_id: str) -> str:
    return Path(item_id.split(":", 1)[1].rsplit(":module:", 1)[0]).parent.name


def _summaries(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["dataset"], row["baseline"])].append(row)
    summaries = []
    for (dataset, baseline), block in sorted(grouped.items()):
        summaries.append(
            {
                "dataset": dataset,
                "baseline": baseline,
                "query_count": len(block),
                "seed_count": len({row["seed"] for row in block}),
                "task_count": len({row["query_task"] for row in block}),
                "mrr": _mean(row["mrr"] for row in block),
                "recall_at_1": _mean(row["recall_at_1"] for row in block),
                "recall_at_5": _mean(row["recall_at_5"] for row in block),
                "mean_rank": _mean(row["mean_rank"] for row in block),
            }
        )
    return summaries


def format_markdown(result: dict[str, Any]) -> str:
    lines = [
        "# Task-level retrieval baseline benchmark",
        "",
        "| Dataset | Baseline | Queries | Tasks | Seeds | MRR | Recall@1 | Recall@5 | Mean rank |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in result["cell_summaries"]:
        lines.append(
            f"| {row['dataset']} | {row['baseline']} | {row['query_count']} | {row['task_count']} | {row['seed_count']} | "
            f"{row['mrr']:.4f} | {row['recall_at_1']:.4f} | {row['recall_at_5']:.4f} | {row['mean_rank']:.4f} |"
        )
    lines.append("")
    return "\n".join(lines)


def _mean(values: Iterable[float]) -> float:
    values_list = list(values)
    if not values_list:
        raise ValueError("cannot average empty values")
    return sum(values_list) / len(values_list)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", nargs=2, action="append", metavar=("DATASET", "PATH"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--json-output", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = run_baselines([BaselineInput(dataset, Path(path)) for dataset, path in args.input])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(format_markdown(result), encoding="utf-8")
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
