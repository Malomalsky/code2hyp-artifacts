from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from geometry_profile_research.codenet_eligibility import (
    canonical_json_bytes,
    jsonl_bytes,
    portable_manifest_path,
    stable_sha256,
)


def build_metadata_rows(
    d3_rows: Sequence[Mapping[str, Any]],
    d4_clusters: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Join retained programs to their final duplicate-closed D4 clusters."""

    eligible = {
        str(cluster["cluster_id"]): cluster
        for cluster in d4_clusters
        if cluster.get("eligible_evaluation_minimum_16") is True
    }
    component_to_cluster = {
        str(component): cluster_id
        for cluster_id, cluster in eligible.items()
        for component in cluster["official_component_ids"]
    }
    rows = []
    counts: dict[str, int] = defaultdict(int)
    users: dict[str, set[str]] = defaultdict(set)
    seen_sources: set[str] = set()
    for record in d3_rows:
        if record.get("retained_after_d0_d3") is not True:
            continue
        cluster_id = component_to_cluster.get(str(record["problem_id"]))
        if cluster_id is None:
            continue
        source = str(record["source_relpath"])
        if source in seen_sources:
            raise ValueError(f"duplicate retained source in Java metadata: {source}")
        seen_sources.add(source)
        user_hash = stable_sha256(str(record["user_id"]))
        rows.append(
            {
                "problem_cluster_id": cluster_id,
                "problem_id": str(record["original_problem_id"]),
                "source_relpath": source,
                "status": "Accepted",
                "submission_id": str(record["submission_id"]),
                "user_id_sha256": user_hash,
            }
        )
        counts[cluster_id] += 1
        users[cluster_id].add(user_hash)
    rows.sort(key=lambda row: (row["problem_cluster_id"], row["source_relpath"]))
    for cluster_id, cluster in eligible.items():
        if counts[cluster_id] != int(cluster["retained_programs_after_d0_d4"]):
            raise ValueError(f"Java metadata program count differs from D4 for {cluster_id}")
        if len(users[cluster_id]) != int(cluster["distinct_users_after_d0_d4"]):
            raise ValueError(f"Java metadata user count differs from D4 for {cluster_id}")
    return rows, {
        "eligible_problem_clusters": len(eligible),
        "programs": len(rows),
        "distinct_users_global": len({row["user_id_sha256"] for row in rows}),
        "minimum_programs_per_cluster": min(counts.values()),
        "minimum_distinct_users_per_cluster": min(len(value) for value in users.values()),
    }


def build_metadata_artifacts(
    *,
    d3_index_path: Path,
    d4_clusters_path: Path,
    d4_manifest_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    d3_bytes = d3_index_path.read_bytes()
    d4_clusters_bytes = d4_clusters_path.read_bytes()
    d4_manifest_bytes = d4_manifest_path.read_bytes()
    d4_manifest = json.loads(d4_manifest_bytes)
    if d4_manifest.get("protocol", {}).get("split_status") != "not_generated":
        raise ValueError("Java D5 metadata must be built before the Stage B split")
    if d4_manifest.get("protocol", {}).get("retrieval_metrics_opened") is not False:
        raise ValueError("Java retrieval metrics were opened before D5 metadata")
    rows, summary = build_metadata_rows(
        [json.loads(line) for line in d3_bytes.splitlines() if line.strip()],
        [json.loads(line) for line in d4_clusters_bytes.splitlines() if line.strip()],
    )
    content = jsonl_bytes(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    index_path = output_dir / "d5_metadata_index.jsonl"
    if index_path.exists() and index_path.read_bytes() != content:
        raise ValueError(f"refusing to overwrite a different Java D5 index: {index_path}")
    index_path.write_bytes(content)
    manifest = {
        "schema_version": "codenet-java-stage-b-d5-metadata-v1",
        "status": "pre_registration_pre_split_without_retrieval_metrics",
        "inputs": {
            "d3_index": {
                "path": portable_manifest_path(d3_index_path, project_root=PROJECT_ROOT),
                "sha256": stable_sha256(d3_bytes),
            },
            "d4_clusters": {
                "path": portable_manifest_path(d4_clusters_path, project_root=PROJECT_ROOT),
                "sha256": stable_sha256(d4_clusters_bytes),
            },
            "d4_manifest": {
                "path": portable_manifest_path(d4_manifest_path, project_root=PROJECT_ROOT),
                "sha256": stable_sha256(d4_manifest_bytes),
            },
        },
        "artifacts": [
            {"path": index_path.name, "bytes": len(content), "sha256": stable_sha256(content)}
        ],
        "summary": summary,
        "user_identifier_policy": "unsalted SHA-256 for internal deterministic user-distinct sampling; omitted from public selected-program artifacts",
        "split_generated": False,
        "retrieval_metrics_opened": False,
    }
    manifest_bytes = canonical_json_bytes(manifest)
    manifest_path = output_dir / "d5_metadata_manifest.json"
    if manifest_path.exists() and manifest_path.read_bytes() != manifest_bytes:
        raise ValueError(f"refusing to overwrite a different Java D5 manifest: {manifest_path}")
    manifest_path.write_bytes(manifest_bytes)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the pre-split Java Stage B metadata index.")
    parser.add_argument("--d3-index", required=True, type=Path)
    parser.add_argument("--d4-clusters", required=True, type=Path)
    parser.add_argument("--d4-manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    result = build_metadata_artifacts(
        d3_index_path=args.d3_index,
        d4_clusters_path=args.d4_clusters,
        d4_manifest_path=args.d4_manifest,
        output_dir=args.output_dir,
    )
    print(json.dumps(result["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
