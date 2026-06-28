#!/usr/bin/env python3
"""Evaluate Code2Hyp-Structural sensitivity to AST-label encoding."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch

from geometry_profile_research.code2hyp_tool import Code2Hyp, Code2HypConfig, EncodedProgram

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LABEL_MODES = ("scalar_hash", "categorical", "none")


@dataclass(frozen=True)
class SplitInput:
    dataset: str
    path: Path


def run_label_mode_sensitivity(
    inputs: Sequence[SplitInput],
    *,
    max_paths: int = 128,
    sinkhorn_iterations: int = 128,
    distance_mode: str = "sinkhorn",
    label_modes: Sequence[str] = LABEL_MODES,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    encoded_cache: dict[tuple[str, str], EncodedProgram] = {}
    for label_mode in label_modes:
        model = Code2Hyp(
            Code2HypConfig(
                max_paths=max_paths,
                label_mode=label_mode,
                sinkhorn_iterations=sinkhorn_iterations,
            )
        )
        for item in inputs:
            payload = json.loads(item.path.read_text(encoding="utf-8"))
            split = payload["split"]
            seed = int(payload["config"]["seed"])
            query_ids = [str(value) for value in split["query_ids"]]
            gallery_ids = [str(value) for value in split["gallery_ids"]]
            gallery_entries = {
                gallery_id: _encoded_for_id(gallery_id, model=model, label_mode=label_mode, cache=encoded_cache)
                for gallery_id in gallery_ids
            }
            for query_id in query_ids:
                query_task = _task_from_id(query_id)
                query_program = _encoded_for_id(query_id, model=model, label_mode=label_mode, cache=encoded_cache)
                scored = [
                    (_distance(query_program, gallery_program, model=model, distance_mode=distance_mode), gallery_id)
                    for gallery_id, gallery_program in gallery_entries.items()
                ]
                scored.sort(key=lambda value: (value[0], value[1]))
                rank = _first_positive_rank(scored, query_task)
                rows.append(
                    {
                        "dataset": item.dataset,
                        "source_file": str(item.path),
                        "seed": seed,
                        "variant": f"structural_{label_mode}",
                        "label_mode": label_mode,
                        "query_id": query_id,
                        "query_task": query_task,
                        "rank": rank,
                        "mrr": 1.0 / rank,
                        "recall_at_1": 1.0 if rank <= 1 else 0.0,
                        "recall_at_5": 1.0 if rank <= 5 else 0.0,
                        "mean_rank": float(rank),
                    }
                )
    return {
        "inputs": [{"dataset": item.dataset, "path": str(item.path)} for item in inputs],
        "max_paths": max_paths,
        "sinkhorn_iterations": sinkhorn_iterations,
        "distance_mode": distance_mode,
        "label_modes": list(label_modes),
        "query_rows": rows,
        "cell_summaries": _summaries(rows),
    }


def _encoded_for_id(
    item_id: str,
    *,
    model: Code2Hyp,
    label_mode: str,
    cache: dict[tuple[str, str], EncodedProgram],
) -> EncodedProgram:
    key = (label_mode, item_id)
    if key not in cache:
        cache[key] = model.encode_file(_path_from_id(item_id))
    return cache[key]


def _distance(left: EncodedProgram, right: EncodedProgram, *, model: Code2Hyp, distance_mode: str) -> float:
    if distance_mode == "sinkhorn":
        return model.distance(left, right)
    if distance_mode == "centroid_proxy":
        left_vector = _centroid_proxy(left)
        right_vector = _centroid_proxy(right)
        return float(torch.linalg.vector_norm(left_vector - right_vector).detach())
    raise ValueError(f"unknown distance_mode: {distance_mode!r}")


def _centroid_proxy(program: EncodedProgram) -> torch.Tensor:
    measure = program.measure
    weights = measure.mass.view(-1, 1, 1)
    point_centroid = torch.sum(weights * measure.points, dim=0).reshape(-1)
    if measure.side_features is None:
        return point_centroid
    side_centroid = torch.sum(measure.mass.view(-1, 1) * measure.side_features, dim=0)
    return torch.cat((point_centroid, side_centroid))


def _first_positive_rank(scored: Sequence[tuple[float, str]], query_task: str) -> int:
    for index, (_score, gallery_id) in enumerate(scored, start=1):
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
        grouped[(row["dataset"], row["variant"])].append(row)
    summaries: list[dict[str, Any]] = []
    for (dataset, variant), block in sorted(grouped.items()):
        summaries.append(
            {
                "dataset": dataset,
                "variant": variant,
                "label_mode": block[0]["label_mode"],
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


def _mean(values: Sequence[float] | Any) -> float:
    values_list = list(values)
    if not values_list:
        raise ValueError("cannot average empty values")
    return sum(float(value) for value in values_list) / len(values_list)


def format_markdown(result: dict[str, Any]) -> str:
    lines = [
        "# Code2Hyp-Structural label-encoding sensitivity",
        "",
        "This diagnostic evaluates whether the deterministic structural layer depends on the arbitrary scalar AST-label hash.",
        f"Configuration: max_paths={result['max_paths']}; distance_mode={result['distance_mode']}; sinkhorn_iterations={result['sinkhorn_iterations']}.",
        "",
        "| Dataset | Label mode | Queries | Tasks | MRR | R@1 | R@5 | Mean rank |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in result["cell_summaries"]:
        lines.append(
            f"| {row['dataset']} | {row['label_mode']} | {row['query_count']} | {row['task_count']} | "
            f"{row['mrr']:.4f} | {row['recall_at_1']:.4f} | {row['recall_at_5']:.4f} | {row['mean_rank']:.4f} |"
        )
    lines.extend(
        [
            "",
            "Interpretation rule: if categorical labels and no-label controls preserve the qualitative ordering, the LCA-path object claim is not merely an artifact of scalar label hashing. If the scalar mode is uniquely strong, the manuscript should treat the label hash as a limitation rather than as part of the main method.",
            "",
        ]
    )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", nargs=2, action="append", metavar=("DATASET", "PATH"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--json-output", type=Path, required=True)
    parser.add_argument("--max-paths", type=int, default=128)
    parser.add_argument("--sinkhorn-iterations", type=int, default=128)
    parser.add_argument("--distance-mode", choices=("sinkhorn", "centroid_proxy"), default="sinkhorn")
    parser.add_argument("--label-modes", default=",".join(LABEL_MODES))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    label_modes = tuple(value.strip() for value in args.label_modes.split(",") if value.strip())
    invalid = [value for value in label_modes if value not in LABEL_MODES]
    if invalid:
        raise ValueError(f"unknown label modes: {invalid!r}")
    result = run_label_mode_sensitivity(
        [SplitInput(dataset, Path(path)) for dataset, path in args.input],
        max_paths=args.max_paths,
        sinkhorn_iterations=args.sinkhorn_iterations,
        distance_mode=args.distance_mode,
        label_modes=label_modes,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(format_markdown(result), encoding="utf-8")
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
