from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from geometry_profile_research.codenet_eligibility import (
    SCHEMA_VERSION,
    analyze_python_file,
    build_exact_duplicate_audit,
    canonical_json_bytes,
    environment_record,
    jsonl_bytes,
    portable_manifest_path,
    stable_sha256,
)


def _analyze_worker(payload: tuple[str, str]) -> dict[str, Any]:
    path_value, root_value = payload
    return analyze_python_file(Path(path_value), Path(root_value))


def build_eligibility_artifacts(
    *,
    input_root: Path,
    output_dir: Path,
    archive_sha256: str,
    workers: int,
    minimum_cluster_programs: int = 64,
    d4_min_shared_d2: int = 5,
    d4_min_fraction: float = 0.05,
) -> dict[str, Any]:
    paths = sorted(path for path in input_root.glob("*/*.py") if path.is_file())
    if not paths:
        raise ValueError(f"no <problem>/<submission>.py files found under {input_root}")
    payloads = [(str(path), str(input_root)) for path in paths]
    if workers == 1:
        records = [_analyze_worker(payload) for payload in payloads]
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            records = list(executor.map(_analyze_worker, payloads, chunksize=128))

    audit = build_exact_duplicate_audit(
        records,
        minimum_cluster_programs=minimum_cluster_programs,
        d4_min_shared_d2=d4_min_shared_d2,
        d4_min_fraction=d4_min_fraction,
    )
    canonical_index = audit.pop("canonical_index_by_record")
    for index, record in enumerate(records):
        canonical = int(canonical_index[index])
        record["retained_after_d0_d2"] = bool(record["parse_ok"]) and canonical == index
        record["canonical_source_relpath"] = records[canonical]["source_relpath"]

    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts: dict[str, bytes] = {
        "file_inventory.jsonl": jsonl_bytes(records),
        "exact_duplicate_components.jsonl": jsonl_bytes(audit["duplicate_components"]),
        "problem_summary.jsonl": jsonl_bytes(audit["problem_summaries"]),
        "d4_exact_problem_edges.jsonl": jsonl_bytes(audit["d4_edges"]),
        "preliminary_problem_clusters.jsonl": jsonl_bytes(audit["problem_clusters"]),
    }
    artifact_records: list[dict[str, Any]] = []
    for filename, content in artifacts.items():
        path = output_dir / filename
        path.write_bytes(content)
        artifact_records.append(
            {
                "path": filename,
                "bytes": len(content),
                "sha256": hashlib.sha256(content).hexdigest(),
            }
        )

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "experiment_role": "pre_split_data_eligibility_without_retrieval_metrics",
        "dataset": {
            "name": "Project CodeNet Python800",
            "source_archive_sha256": archive_sha256,
            "input_root": portable_manifest_path(input_root, project_root=PROJECT_ROOT),
            "observed_problem_count": len({record["problem_id"] for record in records}),
            "observed_source_file_count": len(records),
        },
        "protocol": {
            "minimum_cluster_programs": minimum_cluster_programs,
            "required_parse_rate": 0.95,
            "D0": "UTF-8 source; normalized line endings and trailing horizontal whitespace",
            "D1": "exact Python token stream without comments or formatting",
            "D2": "first-occurrence alpha-normalized Python AST with typed literal placeholders",
            "D3": "pending separate MinHash candidate generation and exact Jaccard verification",
            "D4": {
                "shared_d2_minimum": d4_min_shared_d2,
                "fraction_of_smaller_problem_minimum": d4_min_fraction,
                "official_duplicate_map": "pending: not included in the standalone Python800 archive",
            },
            "split_status": "not_generated",
            "retrieval_metrics_opened": False,
        },
        "environment": environment_record(),
        "summary": audit["summary"],
        "gate_precheck": {
            "parse_rate_at_least_0_95": audit["summary"]["parse_rate"] >= 0.95,
            "at_least_764_eligible_clusters_for_300_100_364": (
                audit["summary"]["eligible_problem_clusters_minimum_64"] >= 764
            ),
            "final_eligibility": "pending_D3_and_official_D4",
        },
        "artifacts": sorted(artifact_records, key=lambda item: item["path"]),
    }
    manifest_bytes = canonical_json_bytes(manifest)
    (output_dir / "eligibility_manifest.json").write_bytes(manifest_bytes)
    manifest_sha = stable_sha256(manifest_bytes)
    (output_dir / "eligibility_manifest.sha256").write_text(
        f"{manifest_sha}  eligibility_manifest.json\n",
        encoding="ascii",
    )
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build the pre-split D0-D2 and parse eligibility audit for CodeNet Python800."
    )
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--archive-sha256", required=True)
    parser.add_argument("--workers", type=int, default=max(1, min(6, os.cpu_count() or 1)))
    parser.add_argument("--minimum-cluster-programs", type=int, default=64)
    parser.add_argument("--d4-min-shared-d2", type=int, default=5)
    parser.add_argument("--d4-min-fraction", type=float, default=0.05)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.workers <= 0:
        raise ValueError("workers must be positive")
    manifest = build_eligibility_artifacts(
        input_root=args.input_root,
        output_dir=args.output_dir,
        archive_sha256=args.archive_sha256,
        workers=args.workers,
        minimum_cluster_programs=args.minimum_cluster_programs,
        d4_min_shared_d2=args.d4_min_shared_d2,
        d4_min_fraction=args.d4_min_fraction,
    )
    print(json.dumps(manifest["summary"], ensure_ascii=False, indent=2, sort_keys=True))
    print(f"manifest={args.output_dir / 'eligibility_manifest.json'}")


if __name__ == "__main__":
    main()
