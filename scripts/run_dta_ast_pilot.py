from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from geometry_profile_research.aggregate import summarize_geometry_profiles
from geometry_profile_research.analysis import GeometryProfile, geometry_profile_for_ast_source
from geometry_profile_research.ast_features import ast_markov_probabilities, ast_node_histogram
from geometry_profile_research.dta import DtaRecord, load_dta_records


def _mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def _summarize_task_records(
    records: list[DtaRecord],
    profiles_by_id: dict[str, GeometryProfile],
    histogram_sizes: dict[str, int],
    transition_sizes: dict[str, int],
) -> dict[str, object]:
    by_task: defaultdict[int, list[DtaRecord]] = defaultdict(list)
    for record in records:
        by_task[record.task_id].append(record)

    task_summary: dict[str, object] = {}
    for task_id, task_records in sorted(by_task.items()):
        task_profiles = [
            profiles_by_id[record.record_id]
            for record in task_records
            if record.record_id in profiles_by_id
        ]
        task_summary[f"task-{task_id:02d}"] = {
            "record_count": len(task_records),
            "profile_count": len(task_profiles),
            "mean_histogram_size": _mean(
                [
                    float(histogram_sizes[record.record_id])
                    for record in task_records
                    if record.record_id in histogram_sizes
                ]
            ),
            "mean_markov_transition_count": _mean(
                [
                    float(transition_sizes[record.record_id])
                    for record in task_records
                    if record.record_id in transition_sizes
                ]
            ),
            "geometry": summarize_geometry_profiles(task_profiles),
        }
    return task_summary


def run_pilot(
    dataset_dir: Path,
    *,
    limit_per_task: int,
) -> dict[str, object]:
    records = load_dta_records(dataset_dir, limit_per_task=limit_per_task)
    profiles_by_id: dict[str, GeometryProfile] = {}
    histogram_sizes: dict[str, int] = {}
    transition_sizes: dict[str, int] = {}
    errors: list[dict[str, str]] = []

    for record in records:
        try:
            histogram = ast_node_histogram(record.code)
            transitions = ast_markov_probabilities(record.code)
            profile = geometry_profile_for_ast_source(record.code)
        except SyntaxError as exc:
            errors.append(
                {
                    "record_id": record.record_id,
                    "source_file": record.source_file,
                    "error": str(exc),
                }
            )
            continue

        histogram_sizes[record.record_id] = len(histogram)
        transition_sizes[record.record_id] = len(transitions)
        profiles_by_id[record.record_id] = profile

    profiles = list(profiles_by_id.values())
    return {
        "dataset": {
            "path": str(dataset_dir),
            "limit_per_task": limit_per_task,
        },
        "records": {
            "loaded": len(records),
            "profiled": len(profiles),
            "syntax_errors": len(errors),
        },
        "overall_geometry": summarize_geometry_profiles(profiles),
        "tasks": _summarize_task_records(
            records,
            profiles_by_id,
            histogram_sizes,
            transition_sizes,
        ),
        "errors": errors[:20],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a DTA AST/Markov/geometry pilot experiment."
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("data/dta_zenodo_7799972/extracted"),
        help="Directory containing task-00.csv ... task-10.csv.",
    )
    parser.add_argument(
        "--limit-per-task",
        type=int,
        default=3,
        help="Maximum number of programs loaded from each task CSV.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/dta_ast_pilot.json"),
        help="Output JSON path.",
    )
    args = parser.parse_args()

    payload = run_pilot(args.dataset_dir, limit_per_task=args.limit_per_task)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"wrote {args.output}")
    print(json.dumps(payload["records"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
