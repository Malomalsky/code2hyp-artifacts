from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from geometry_profile_research.codenet_eligibility import canonical_json_bytes, stable_sha256
from scripts.build_codenet_python800_statement_d4 import normalized_problem_statement


def audit_statement_overlap(
    *,
    frame_report_path: Path,
    d0_d2_manifest_path: Path,
    stage_a_clusters_path: Path,
    descriptions_root: Path,
    output_path: Path,
) -> dict[str, Any]:
    frame = json.loads(frame_report_path.read_text(encoding="utf-8"))
    d0_d2 = json.loads(d0_d2_manifest_path.read_text(encoding="utf-8"))
    component_by_problem = {
        str(problem_id): str(row["component_id"])
        for row in frame["components"]
        if row["metadata_eligible_for_evaluation_minimum_16"]
        for problem_id in row["java_problem_ids"]
    }
    problems_by_component = {
        str(row["component_id"]): tuple(str(value) for value in row["problem_ids"])
        for row in frame["components"]
    }
    opened_problem_ids = {
        str(problem_id)
        for row in _iter_jsonl(stage_a_clusters_path)
        for problem_id in row["problem_ids"]
    }
    new_statements, new_missing = _statement_hashes(set(component_by_problem), descriptions_root)
    opened_statements, opened_missing = _statement_hashes(opened_problem_ids, descriptions_root)

    new_by_hash = _invert(new_statements)
    opened_by_hash = _invert(opened_statements)
    cross_frame = [
        {
            "normalized_text_sha256": digest,
            "java_problem_ids": new_by_hash[digest],
            "opened_stage_a_problem_ids": opened_by_hash[digest],
        }
        for digest in sorted(new_by_hash.keys() & opened_by_hash.keys())
    ]
    cross_component = []
    for digest, problem_ids in sorted(new_by_hash.items()):
        component_ids = sorted({component_by_problem[problem_id] for problem_id in problem_ids})
        if len(component_ids) > 1:
            cross_component.append(
                {
                    "normalized_text_sha256": digest,
                    "problem_ids": problem_ids,
                    "official_component_ids": component_ids,
                }
            )

    d2_edges = []
    for edge in d0_d2["preliminary_d4_edges"]:
        left = str(edge["left_problem_id"])
        right = str(edge["right_problem_id"])
        left_problems = problems_by_component[left]
        right_problems = problems_by_component[right]
        comparable = [
            (left_problem, right_problem)
            for left_problem in left_problems
            for right_problem in right_problems
            if left_problem in new_statements and right_problem in new_statements
        ]
        d2_edges.append(
            {
                **edge,
                "left_problem_ids": list(left_problems),
                "right_problem_ids": list(right_problems),
                "statement_comparison_available": bool(comparable),
                "any_identical_normalized_statement": any(
                    new_statements[left_problem] == new_statements[right_problem]
                    for left_problem, right_problem in comparable
                ),
            }
        )

    payload = {
        "schema_version": "code2hyp-codenet-java-stage-b-statement-overlap-v1",
        "status": "exact_statement_gate_complete_semantic_equivalence_for_missing_descriptions_pending",
        "inputs": {
            "frame_report_sha256": _sha256_file(frame_report_path),
            "d0_d2_manifest_sha256": _sha256_file(d0_d2_manifest_path),
            "stage_a_clusters_sha256": _sha256_file(stage_a_clusters_path),
        },
        "counts": {
            "java_candidate_problem_ids": len(component_by_problem),
            "java_descriptions_available": len(new_statements),
            "java_descriptions_missing": len(new_missing),
            "opened_stage_a_problem_ids": len(opened_problem_ids),
            "opened_stage_a_descriptions_available": len(opened_statements),
            "opened_stage_a_descriptions_missing": len(opened_missing),
            "exact_normalized_statement_collisions_java_vs_opened_stage_a": len(cross_frame),
            "exact_normalized_statement_groups_across_java_official_components": len(cross_component),
        },
        "java_descriptions_missing": new_missing,
        "opened_stage_a_descriptions_missing": opened_missing,
        "cross_frame_exact_statement_collisions": cross_frame,
        "cross_component_exact_statement_groups": cross_component,
        "preliminary_shared_d2_edges": d2_edges,
        "interpretation": {
            "no_exact_statement_leak_detected_among_available_descriptions": not cross_frame,
            "exact_text_identity_is_not_a_test_of_semantic_equivalence": True,
            "missing_java_descriptions_require_conservative_component_retention_or_manual_adjudication": True,
            "split_status": "not_generated",
        },
    }
    content = canonical_json_bytes(payload)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and output_path.read_bytes() != content:
        raise ValueError(f"refusing to overwrite a different statement audit: {output_path}")
    output_path.write_bytes(content)
    return payload


def _statement_hashes(problem_ids: set[str], root: Path) -> tuple[dict[str, str], list[str]]:
    hashes = {}
    missing = []
    for problem_id in sorted(problem_ids):
        path = root / f"{problem_id}.html"
        if not path.is_file():
            missing.append(problem_id)
            continue
        normalized = normalized_problem_statement(path.read_text(encoding="utf-8", errors="strict"))
        hashes[problem_id] = stable_sha256(normalized)
    return hashes, missing


def _invert(values: dict[str, str]) -> dict[str, list[str]]:
    result: dict[str, list[str]] = defaultdict(list)
    for key, value in values.items():
        result[value].append(key)
    return {digest: sorted(problem_ids) for digest, problem_ids in result.items()}


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit Java Stage B statements against opened Stage A tasks.")
    parser.add_argument("--frame-report", required=True, type=Path)
    parser.add_argument("--d0-d2-manifest", required=True, type=Path)
    parser.add_argument("--stage-a-clusters", required=True, type=Path)
    parser.add_argument("--descriptions-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = audit_statement_overlap(
        frame_report_path=args.frame_report,
        d0_d2_manifest_path=args.d0_d2_manifest,
        stage_a_clusters_path=args.stage_a_clusters,
        descriptions_root=args.descriptions_root,
        output_path=args.output,
    )
    print(json.dumps(result["counts"], indent=2, sort_keys=True))
    print(f"report={args.output}")


if __name__ == "__main__":
    main()
