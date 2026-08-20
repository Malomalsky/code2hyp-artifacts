from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import re
import tarfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROBLEM_ID = re.compile(r"p\d{5}")


def audit_java_stage_b_frame(
    *,
    metadata_archive: Path,
    source_archive: Path,
    stage_a_clusters_path: Path,
    official_duplicates_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Build a metadata-only upper bound for an independent Java Stage B frame."""

    opened_problem_ids = {
        str(problem_id)
        for row in _iter_jsonl(stage_a_clusters_path)
        for problem_id in row["problem_ids"]
    }
    duplicate_groups = _read_duplicate_groups(official_duplicates_path)
    accepted_by_problem, users_by_problem, metadata_counts = _read_java_metadata(metadata_archive)

    problem_ids = set(accepted_by_problem) | opened_problem_ids
    problem_ids.update(problem_id for group in duplicate_groups for problem_id in group)
    components = _connected_components(problem_ids, duplicate_groups)

    component_rows = []
    for component in components:
        java_problem_ids = sorted(component & accepted_by_problem.keys())
        if not java_problem_ids:
            continue
        accepted = sum(accepted_by_problem[problem_id] for problem_id in java_problem_ids)
        users = set().union(*(users_by_problem[problem_id] for problem_id in java_problem_ids))
        opened = sorted(component & opened_problem_ids)
        component_rows.append(
            {
                "component_id": _component_id(component),
                "problem_ids": sorted(component),
                "java_problem_ids": java_problem_ids,
                "accepted_java_submissions": accepted,
                "distinct_java_users": len(users),
                "opened_stage_a_problem_ids": opened,
                "independent_of_opened_stage_a": not opened,
                "metadata_eligible_for_evaluation_minimum_16": not opened and accepted >= 16 and len(users) >= 16,
                "metadata_eligible_for_train_minimum_64": not opened and accepted >= 64 and len(users) >= 16,
            }
        )
    component_rows.sort(key=lambda row: row["component_id"])

    independent = [row for row in component_rows if row["independent_of_opened_stage_a"]]
    evaluation = [row for row in independent if row["metadata_eligible_for_evaluation_minimum_16"]]
    train = [row for row in independent if row["metadata_eligible_for_train_minimum_64"]]
    evaluation_after_train = len(evaluation) - len(train)

    payload = {
        "schema_version": "code2hyp-codenet-java-stage-b-frame-v1",
        "status": "metadata_precheck_only_before_source_parse_and_D0_D4",
        "inputs": {
            "metadata_archive": _file_record(metadata_archive),
            "source_archive": _file_record(source_archive),
            "stage_a_clusters": _file_record(stage_a_clusters_path),
            "official_identical_problem_clusters": _file_record(official_duplicates_path),
        },
        "counts": {
            **metadata_counts,
            "opened_stage_a_problem_ids": len(opened_problem_ids),
            "java_official_components": len(component_rows),
            "java_components_overlapping_opened_stage_a": sum(
                not row["independent_of_opened_stage_a"] for row in component_rows
            ),
            "independent_java_components": len(independent),
            "independent_evaluation_components_minimum_16": len(evaluation),
            "independent_train_components_minimum_64": len(train),
            "evaluation_candidates_remaining_if_all_train_components_are_reserved": evaluation_after_train,
        },
        "role_specific_upper_bound": {
            "train_components": len(train),
            "validation_components": 81 if evaluation_after_train >= 407 else None,
            "test_components": 326 if evaluation_after_train >= 407 else None,
            "reserve_evaluation_components": evaluation_after_train - 407 if evaluation_after_train >= 407 else None,
            "note": "244/81/326 is a metadata-only upper bound; final quotas require Java parsing and D0-D4.",
        },
        "threshold_sensitivity_independent_components": {
            str(threshold): sum(
                row["accepted_java_submissions"] >= threshold and row["distinct_java_users"] >= 16
                for row in independent
            )
            for threshold in (16, 32, 64, 128, 136, 256)
        },
        "components": component_rows,
        "interpretation": {
            "may_define_an_independent_stage_b_sampling_frame": True,
            "is_final_eligibility": False,
            "required_next_gate": "extract accepted Java sources, parse raw AST, then apply D0-D4 before split",
            "java250_is_not_rehabilitated_as_independent": True,
        },
    }
    content = _canonical_json_bytes(payload)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and output_path.read_bytes() != content:
        raise ValueError(f"refusing to overwrite a different Java Stage B frame audit: {output_path}")
    output_path.write_bytes(content)
    return payload


def _read_java_metadata(metadata_archive: Path) -> tuple[dict[str, int], dict[str, set[str]], dict[str, int]]:
    accepted_by_problem: dict[str, int] = defaultdict(int)
    users_by_problem: dict[str, set[str]] = defaultdict(set)
    problem_files = 0
    submission_rows = 0
    with tarfile.open(metadata_archive, "r:gz") as archive:
        for member in archive:
            if not member.isfile() or not member.name.endswith(".csv") or member.name.endswith("/problem_list.csv"):
                continue
            problem_files += 1
            stream = archive.extractfile(member)
            if stream is None:
                raise ValueError(f"cannot read metadata member {member.name}")
            reader = csv.DictReader(io.TextIOWrapper(stream, encoding="utf-8", errors="strict", newline=""))
            for row in reader:
                submission_rows += 1
                if row.get("language") != "Java" or row.get("status") != "Accepted":
                    continue
                problem_id = str(row["problem_id"])
                if not PROBLEM_ID.fullmatch(problem_id):
                    raise ValueError(f"invalid problem ID in metadata: {problem_id!r}")
                accepted_by_problem[problem_id] += 1
                users_by_problem[problem_id].add(str(row["user_id"]))
    return dict(accepted_by_problem), dict(users_by_problem), {
        "metadata_problem_files": problem_files,
        "metadata_submission_rows": submission_rows,
        "java_problems_with_accepted_submissions": len(accepted_by_problem),
        "accepted_java_submissions": sum(accepted_by_problem.values()),
    }


def _read_duplicate_groups(path: Path) -> list[set[str]]:
    groups = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        group = {value.strip() for value in line.split(",") if value.strip()}
        if len(group) < 2 or any(not PROBLEM_ID.fullmatch(value) for value in group):
            raise ValueError(f"invalid duplicate group at {path}:{line_number}")
        groups.append(group)
    return groups


def _connected_components(problem_ids: set[str], groups: Iterable[set[str]]) -> list[set[str]]:
    neighbors = {problem_id: set() for problem_id in problem_ids}
    for group in groups:
        for problem_id in group:
            neighbors.setdefault(problem_id, set()).update(group - {problem_id})
    remaining = set(neighbors)
    components = []
    while remaining:
        stack = [remaining.pop()]
        component = set(stack)
        while stack:
            for neighbor in neighbors[stack.pop()]:
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    component.add(neighbor)
                    stack.append(neighbor)
        components.append(component)
    return components


def _component_id(problem_ids: set[str]) -> str:
    digest = hashlib.sha256("\n".join(sorted(problem_ids)).encode("ascii")).hexdigest()
    return f"official-{digest[:20]}"


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSONL at {path}:{line_number}") from error


def _file_record(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def _canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit the independent full-CodeNet Java Stage B sampling frame.")
    parser.add_argument("--metadata-archive", required=True, type=Path)
    parser.add_argument("--source-archive", required=True, type=Path)
    parser.add_argument("--stage-a-clusters", required=True, type=Path)
    parser.add_argument("--official-duplicates", required=True, type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "reports/codenet_java_stage_b_frame_v1.json",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = audit_java_stage_b_frame(
        metadata_archive=args.metadata_archive,
        source_archive=args.source_archive,
        stage_a_clusters_path=args.stage_a_clusters,
        official_duplicates_path=args.official_duplicates,
        output_path=args.output,
    )
    print(json.dumps(result["counts"], indent=2, sort_keys=True))
    print(json.dumps(result["role_specific_upper_bound"], indent=2, sort_keys=True))
    print(f"report={args.output}")


if __name__ == "__main__":
    main()
