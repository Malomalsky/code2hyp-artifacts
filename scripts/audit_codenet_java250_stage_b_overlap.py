from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from geometry_profile_research.codenet_eligibility import canonical_json_bytes  # noqa: E402


PROBLEM_ID = re.compile(r"p\d{5}")


def audit_java250_overlap(
    *,
    java_root: Path,
    python_clusters_path: Path,
    stage_a_assignments_path: Path,
    archive_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Determine whether Java250 is independent of the opened Python800 Stage A tasks."""

    java_problem_ids = {
        path.name
        for path in java_root.iterdir()
        if path.is_dir() and PROBLEM_ID.fullmatch(path.name)
    }
    if not java_problem_ids:
        raise ValueError(f"no Java250 problem directories found under {java_root}")

    cluster_by_problem: dict[str, str] = {}
    python_cluster_ids: set[str] = set()
    for row in _iter_jsonl(python_clusters_path):
        cluster_id = str(row["cluster_id"])
        python_cluster_ids.add(cluster_id)
        for problem_id in row["problem_ids"]:
            problem_id = str(problem_id)
            previous = cluster_by_problem.setdefault(problem_id, cluster_id)
            if previous != cluster_id:
                raise ValueError(f"problem {problem_id} occurs in multiple duplicate components")

    assigned_cluster_ids = {
        str(row["cluster_id"])
        for row in _iter_jsonl(stage_a_assignments_path)
    }
    if not assigned_cluster_ids <= python_cluster_ids:
        raise ValueError("Stage A assignments contain unknown duplicate components")

    python_problem_ids = set(cluster_by_problem)
    stage_a_problem_ids = {
        problem_id
        for problem_id, cluster_id in cluster_by_problem.items()
        if cluster_id in assigned_cluster_ids
    }
    overlap = java_problem_ids & stage_a_problem_ids
    novel = java_problem_ids - stage_a_problem_ids
    overlap_components = {cluster_by_problem[problem_id] for problem_id in overlap}
    java_files = tuple(java_root.glob("p?????/*.java"))

    payload = {
        "schema_version": "code2hyp-codenet-java250-stage-b-overlap-v1",
        "status": "not_eligible_as_independent_confirmatory_stage_b",
        "inputs": {
            "java250_archive": {
                "bytes": archive_path.stat().st_size,
                "sha256": _sha256_file(archive_path),
                "source": "https://huggingface.co/datasets/qiankunmu/Project_CodeNet_Python800_and_Java250",
                "source_commit": "9e4c7b4da87e8ed66a9311d643d3edee675929a2",
            },
            "python_clusters_sha256": _sha256_file(python_clusters_path),
            "stage_a_assignments_sha256": _sha256_file(stage_a_assignments_path),
        },
        "counts": {
            "java250_problem_count": len(java_problem_ids),
            "java250_program_count": len(java_files),
            "python800_post_d3_problem_count": len(python_problem_ids),
            "python800_post_d3_duplicate_component_count": len(python_cluster_ids),
            "stage_a_assigned_duplicate_component_count": len(assigned_cluster_ids),
            "java250_problem_overlap_with_opened_stage_a": len(overlap),
            "java250_overlapping_stage_a_duplicate_components": len(overlap_components),
            "java250_problem_ids_not_in_opened_stage_a": len(novel),
        },
        "java250_problem_ids_not_in_opened_stage_a": sorted(novel),
        "interpretation": {
            "java250_may_be_used_for_cross_language_replication": True,
            "java250_must_not_be_described_as_an_independent_confirmatory_corpus": True,
            "reason": "242_of_250_problem_labels_were_already_opened_in_python800_stage_a",
        },
    }
    content = canonical_json_bytes(payload)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and output_path.read_bytes() != content:
        raise ValueError(f"refusing to overwrite a different overlap audit: {output_path}")
    output_path.write_bytes(content)
    return payload


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSONL at {path}:{line_number}") from error


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit Java250 overlap with opened Python800 Stage A tasks.")
    parser.add_argument("--java-root", required=True, type=Path)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument(
        "--python-clusters",
        type=Path,
        default=PROJECT_ROOT / "data/codenet_python800_eligibility_d0_d3/post_d3_problem_clusters.jsonl",
    )
    parser.add_argument(
        "--stage-a-assignments",
        type=Path,
        default=PROJECT_ROOT / "data/codenet_python800_stage_a_split/cluster_assignments.jsonl",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "reports/codenet_java250_stage_b_overlap_v1.json",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = audit_java250_overlap(
        java_root=args.java_root,
        python_clusters_path=args.python_clusters,
        stage_a_assignments_path=args.stage_a_assignments,
        archive_path=args.archive,
        output_path=args.output,
    )
    print(json.dumps(result["counts"], indent=2, sort_keys=True))
    print(f"report={args.output}")


if __name__ == "__main__":
    main()
