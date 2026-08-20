from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from geometry_profile_research.codenet_eligibility import (
    DisjointSet,
    canonical_json_bytes,
    jsonl_bytes,
    portable_manifest_path,
    stable_sha256,
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_and_hash(path: Path, content: bytes) -> dict[str, Any]:
    path.write_bytes(content)
    return {"path": path.name, "bytes": len(content), "sha256": stable_sha256(content)}


def _file_record(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return {
        "path": portable_manifest_path(path, project_root=PROJECT_ROOT),
        "bytes": path.stat().st_size,
        "sha256": digest.hexdigest(),
    }


def _hamilton(total: int) -> tuple[int, int, int]:
    numerators = (3, 1, 4)
    floors = [total * value // 8 for value in numerators]
    remainders = [total * value % 8 for value in numerators]
    for index in sorted(range(3), key=lambda item: (-remainders[item], item))[: total - sum(floors)]:
        floors[index] += 1
    return floors[0], floors[1], floors[2]


def _maximum_role_upper_bound(train_count: int, evaluation_count: int) -> dict[str, int]:
    for total in range(evaluation_count, 0, -1):
        train, validation, test = _hamilton(total)
        if train <= train_count and validation + test <= evaluation_count - train:
            return {
                "train_clusters": train,
                "validation_clusters": validation,
                "test_clusters": test,
                "reserve_evaluation_clusters": evaluation_count - total,
            }
    return {"train_clusters": 0, "validation_clusters": 0, "test_clusters": 0,
            "reserve_evaluation_clusters": evaluation_count}


def finalize_java_stage_b_d4(
    *,
    frame_report_path: Path,
    d3_dir: Path,
    statement_audit_path: Path,
    output_dir: Path,
    train_programs_per_cluster: int = 64,
) -> dict[str, Any]:
    if train_programs_per_cluster < 16:
        raise ValueError("train programs per cluster cannot be below the evaluation minimum")
    frame = json.loads(frame_report_path.read_text(encoding="utf-8"))
    statement_audit = json.loads(statement_audit_path.read_text(encoding="utf-8"))
    d3_manifest_path = d3_dir / "d3_manifest.json"
    d3_clusters = _read_jsonl(d3_dir / "post_d3_problem_clusters.jsonl")
    d3_index = _read_jsonl(d3_dir / "d3_index.jsonl")

    component_index = {
        component: index
        for index, cluster in enumerate(d3_clusters)
        for component in cluster["problem_ids"]
    }
    raw_to_component = {
        str(problem_id): str(row["component_id"])
        for row in frame["components"]
        for problem_id in row["java_problem_ids"]
        if str(row["component_id"]) in component_index
    }
    java_problems_by_component = {
        str(row["component_id"]): sorted(str(value) for value in row["java_problem_ids"])
        for row in frame["components"]
        if str(row["component_id"]) in component_index
    }

    dsu = DisjointSet(len(d3_clusters))
    exact_statement_edges = []
    for group in statement_audit["cross_component_exact_statement_groups"]:
        components = [str(value) for value in group["official_component_ids"]]
        present = [component for component in components if component in component_index]
        for component in present[1:]:
            dsu.union(component_index[present[0]], component_index[component])
            exact_statement_edges.append(
                {
                    "left_component_id": present[0],
                    "right_component_id": component,
                    "normalized_text_sha256": group["normalized_text_sha256"],
                }
            )

    missing_components = {
        raw_to_component[problem]
        for problem in statement_audit["java_descriptions_missing"]
        if problem in raw_to_component
    }
    opened_collision_components = {
        raw_to_component[problem]
        for collision in statement_audit["cross_frame_exact_statement_collisions"]
        for problem in collision["java_problem_ids"]
        if problem in raw_to_component
    }

    cluster_indices_by_root: dict[int, list[int]] = defaultdict(list)
    for index in range(len(d3_clusters)):
        cluster_indices_by_root[dsu.find(index)].append(index)
    root_by_component = {
        component: dsu.find(index) for component, index in component_index.items()
    }
    retained_programs_by_root: dict[int, int] = defaultdict(int)
    users_by_root: dict[int, set[str]] = defaultdict(set)
    for record in d3_index:
        if not record["retained_after_d0_d3"]:
            continue
        root = root_by_component[str(record["problem_id"])]
        retained_programs_by_root[root] += 1
        users_by_root[root].add(str(record["user_id"]))

    final_clusters = []
    for root, indices in cluster_indices_by_root.items():
        components = sorted(
            component
            for index in indices
            for component in d3_clusters[index]["problem_ids"]
        )
        raw_problem_ids = sorted(
            problem
            for component in components
            for problem in java_problems_by_component[component]
        )
        missing = sorted(set(components) & missing_components)
        opened = sorted(set(components) & opened_collision_components)
        programs = retained_programs_by_root[root]
        users = len(users_by_root[root])
        eligible = not missing and not opened
        final_clusters.append(
            {
                "cluster_id": f"problem-{stable_sha256('|'.join(components))[:20]}",
                "official_component_ids": components,
                "java_problem_ids": raw_problem_ids,
                "problem_count": len(components),
                "retained_programs_after_d0_d4": programs,
                "distinct_users_after_d0_d4": users,
                "missing_statement_component_ids": missing,
                "opened_stage_a_statement_collision_component_ids": opened,
                "eligible_evaluation_minimum_16": eligible and programs >= 16 and users >= 16,
                "eligible_train": (
                    eligible
                    and programs >= train_programs_per_cluster
                    and users >= train_programs_per_cluster
                ),
            }
        )
    final_clusters.sort(key=lambda item: item["cluster_id"])

    evaluation_count = sum(cluster["eligible_evaluation_minimum_16"] for cluster in final_clusters)
    train_count = sum(cluster["eligible_train"] for cluster in final_clusters)
    summary = {
        "d3_problem_clusters": len(d3_clusters),
        "exact_statement_edges_added_after_d3": len(exact_statement_edges),
        "problem_clusters_after_d4_union": len(final_clusters),
        "problem_clusters_excluded_for_missing_statements": sum(
            bool(cluster["missing_statement_component_ids"]) for cluster in final_clusters
        ),
        "problem_clusters_excluded_for_opened_stage_a_statement_collision": sum(
            bool(cluster["opened_stage_a_statement_collision_component_ids"])
            for cluster in final_clusters
        ),
        "eligible_evaluation_clusters_minimum_16_users_16": evaluation_count,
        "eligible_train_clusters": train_count,
        "retained_programs_in_eligible_clusters": sum(
            cluster["retained_programs_after_d0_d4"]
            for cluster in final_clusters
            if cluster["eligible_evaluation_minimum_16"]
        ),
    }
    role_upper_bound = _maximum_role_upper_bound(train_count, evaluation_count)
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts = [
        _write_and_hash(output_dir / "exact_statement_edges.jsonl", jsonl_bytes(exact_statement_edges)),
        _write_and_hash(output_dir / "post_d4_problem_clusters.jsonl", jsonl_bytes(final_clusters)),
        _write_and_hash(output_dir / "d4_summary.json", canonical_json_bytes(summary)),
    ]
    manifest = {
        "schema_version": "codenet-java-stage-b-d4-v1",
        "experiment_role": "full_pre_split_D4_eligibility_without_retrieval_metrics",
        "inputs": {
            "frame_report": _file_record(frame_report_path),
            "d3_manifest": _file_record(d3_manifest_path),
            "statement_audit": _file_record(statement_audit_path),
        },
        "protocol": {
            "official_identical_problem_map": "applied_before_source_materialization",
            "additional_edge_rule": "identical normalized statement SHA-256 across official components",
            "missing_statement_rule": "exclude the complete D3/D4 cluster",
            "opened_stage_a_exact_statement_collision_rule": "exclude the complete D3/D4 cluster",
            "train_programs_per_cluster": train_programs_per_cluster,
            "train_user_distinct_requirement": train_programs_per_cluster,
            "role_ratio": "3:1:4 by Hamilton method; tie order train, validation, test",
            "split_status": "not_generated",
            "retrieval_metrics_opened": False,
        },
        "summary": summary,
        "role_specific_upper_bound": role_upper_bound,
        "gate_precheck": {
            "statement_complete_eligible_frame_available": evaluation_count > 0 and train_count > 0,
            "final_eligibility": "passed_before_power_analysis",
        },
        "artifacts": sorted(artifacts, key=lambda item: item["path"]),
    }
    manifest_bytes = canonical_json_bytes(manifest)
    (output_dir / "d4_manifest.json").write_bytes(manifest_bytes)
    (output_dir / "d4_manifest.sha256").write_text(
        f"{stable_sha256(manifest_bytes)}  d4_manifest.json\n", encoding="ascii"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Finalize conservative Java Stage B D4 eligibility.")
    parser.add_argument("--frame-report", required=True, type=Path)
    parser.add_argument("--d3-dir", required=True, type=Path)
    parser.add_argument("--statement-audit", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--train-programs-per-cluster", type=int, default=64)
    args = parser.parse_args()
    manifest = finalize_java_stage_b_d4(
        frame_report_path=args.frame_report,
        d3_dir=args.d3_dir,
        statement_audit_path=args.statement_audit,
        output_dir=args.output_dir,
        train_programs_per_cluster=args.train_programs_per_cluster,
    )
    print(json.dumps(manifest["summary"], ensure_ascii=False, indent=2, sort_keys=True))
    print(json.dumps(manifest["role_specific_upper_bound"], ensure_ascii=False, indent=2, sort_keys=True))
    print(f"manifest={args.output_dir / 'd4_manifest.json'}")


if __name__ == "__main__":
    main()
