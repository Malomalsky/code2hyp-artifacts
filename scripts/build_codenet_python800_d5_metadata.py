from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
import sys
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from geometry_profile_research.codenet_eligibility import (
    canonical_json_bytes,
    jsonl_bytes,
    stable_sha256,
)


D5_SCHEMA_VERSION = "codenet-python800-d5-metadata-v1"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_and_hash(path: Path, content: bytes) -> dict[str, Any]:
    path.write_bytes(content)
    return {
        "path": path.name,
        "bytes": len(content),
        "sha256": hashlib.sha256(content).hexdigest(),
    }


def _private_user_hash(user_id: str) -> str:
    return stable_sha256(f"Project-CodeNet-1.0.0\x00{user_id}")


def build_d5_metadata_artifacts(
    *,
    metadata_root: Path,
    d3_dir: Path,
    statement_d4_dir: Path,
    output_dir: Path,
    metadata_archive_sha256: str,
) -> dict[str, Any]:
    d3_manifest_path = d3_dir / "d3_manifest.json"
    d4_manifest_path = statement_d4_dir / "statement_d4_manifest.json"
    d3_manifest_sha = stable_sha256(d3_manifest_path.read_bytes())
    d4_manifest_sha = stable_sha256(d4_manifest_path.read_bytes())

    d3_rows = [row for row in _read_jsonl(d3_dir / "d3_index.jsonl") if row["retained_after_d0_d3"]]
    needed_by_problem: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in d3_rows:
        needed_by_problem[str(row["problem_id"])][str(row["submission_id"])] = row
    cluster_for_problem = {
        str(problem): str(cluster["cluster_id"])
        for cluster in _read_jsonl(statement_d4_dir / "post_statement_d4_problem_clusters.jsonl")
        for problem in cluster["problem_ids"]
    }

    metadata_rows_scanned = 0
    matched_rows: list[dict[str, Any]] = []
    missing: list[str] = []
    duplicate_matches: list[str] = []
    for problem in sorted(needed_by_problem):
        path = metadata_root / f"{problem}.csv"
        found: set[str] = set()
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                metadata_rows_scanned += 1
                submission = str(row["submission_id"])
                if submission not in needed_by_problem[problem]:
                    continue
                key = f"{problem}/{submission}"
                if submission in found:
                    duplicate_matches.append(key)
                    continue
                found.add(submission)
                source = needed_by_problem[problem][submission]
                matched_rows.append(
                    {
                        "problem_id": problem,
                        "problem_cluster_id": cluster_for_problem[problem],
                        "submission_id": submission,
                        "source_relpath": source["source_relpath"],
                        "user_id_sha256": _private_user_hash(str(row["user_id"])),
                        "date_unix": int(row["date"]),
                        "language": str(row["language"]),
                        "original_language": str(row["original_language"]),
                        "status": str(row["status"]),
                    }
                )
        missing.extend(
            f"{problem}/{submission}"
            for submission in sorted(set(needed_by_problem[problem]) - found)
        )
    if missing or duplicate_matches:
        raise ValueError(
            f"metadata join failed: missing={len(missing)} duplicate_matches={len(duplicate_matches)}"
        )
    matched_rows.sort(key=lambda item: item["source_relpath"])

    programs_by_user: Counter[str] = Counter()
    clusters_by_user: dict[str, set[str]] = defaultdict(set)
    problems_by_user: dict[str, set[str]] = defaultdict(set)
    users_by_cluster: dict[str, set[str]] = defaultdict(set)
    original_languages: Counter[str] = Counter()
    statuses: Counter[str] = Counter()
    dates: list[int] = []
    for row in matched_rows:
        user = str(row["user_id_sha256"])
        programs_by_user[user] += 1
        clusters_by_user[user].add(str(row["problem_cluster_id"]))
        problems_by_user[user].add(str(row["problem_id"]))
        users_by_cluster[str(row["problem_cluster_id"])].add(user)
        original_languages[str(row["original_language"])] += 1
        statuses[str(row["status"])] += 1
        dates.append(int(row["date_unix"]))
    user_rows = [
        {
            "user_id_sha256": user,
            "retained_programs": programs_by_user[user],
            "problem_count": len(problems_by_user[user]),
            "problem_cluster_count": len(clusters_by_user[user]),
            "cross_cluster": len(clusters_by_user[user]) > 1,
        }
        for user in sorted(programs_by_user)
    ]
    cross_cluster_users = {row["user_id_sha256"] for row in user_rows if row["cross_cluster"]}
    cluster_counts = [int(row["problem_cluster_count"]) for row in user_rows]
    distinct_users_per_cluster = sorted(len(users) for users in users_by_cluster.values())
    summary = {
        "retained_programs_after_d0_d3": len(d3_rows),
        "metadata_rows_scanned": metadata_rows_scanned,
        "metadata_rows_matched": len(matched_rows),
        "accepted_rows": statuses.get("Accepted", 0),
        "distinct_users": len(user_rows),
        "users_spanning_multiple_problem_clusters": len(cross_cluster_users),
        "fraction_users_spanning_multiple_problem_clusters": (
            len(cross_cluster_users) / len(user_rows) if user_rows else 0.0
        ),
        "programs_from_cross_cluster_users": sum(
            programs_by_user[user] for user in cross_cluster_users
        ),
        "maximum_problem_clusters_per_user": max(cluster_counts, default=0),
        "median_problem_clusters_per_user": statistics.median(cluster_counts) if cluster_counts else 0.0,
        "minimum_distinct_users_per_problem_cluster": min(distinct_users_per_cluster, default=0),
        "median_distinct_users_per_problem_cluster": (
            statistics.median(distinct_users_per_cluster) if distinct_users_per_cluster else 0.0
        ),
        "problem_clusters_with_at_least_80_distinct_users": sum(
            count >= 80 for count in distinct_users_per_cluster
        ),
        "earliest_submission_utc": datetime.fromtimestamp(min(dates), tz=UTC).isoformat() if dates else None,
        "latest_submission_utc": datetime.fromtimestamp(max(dates), tz=UTC).isoformat() if dates else None,
        "original_language_counts": dict(sorted(original_languages.items())),
        "status_counts": dict(sorted(statuses.items())),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts = [
        _write_and_hash(output_dir / "d5_metadata_index.jsonl", jsonl_bytes(matched_rows)),
        _write_and_hash(output_dir / "d5_user_cluster_summary.jsonl", jsonl_bytes(user_rows)),
        _write_and_hash(output_dir / "d5_metadata_summary.json", canonical_json_bytes(summary)),
    ]
    manifest = {
        "schema_version": D5_SCHEMA_VERSION,
        "experiment_role": "pre_split_author_metadata_for_registered_sensitivity_only",
        "input": {
            "metadata_root": str(metadata_root.resolve()),
            "metadata_archive_sha256": metadata_archive_sha256,
            "d3_manifest_sha256": d3_manifest_sha,
            "statement_d4_manifest_sha256": d4_manifest_sha,
        },
        "protocol": {
            "user_identifier": "SHA256(Project-CodeNet-1.0.0 NUL anonymized_user_id)",
            "primary_D0_D4_changed": False,
            "within_cluster_selection": "one retained program per user before choosing 64 train or 8+8 query/gallery",
            "global_author_overlap": "reported as an attrition diagnostic; it does not replace the primary estimand",
            "split_status": "not_generated",
            "retrieval_metrics_opened": False,
        },
        "summary": summary,
        "artifacts": sorted(artifacts, key=lambda item: item["path"]),
    }
    manifest_bytes = canonical_json_bytes(manifest)
    (output_dir / "d5_metadata_manifest.json").write_bytes(manifest_bytes)
    manifest_sha = stable_sha256(manifest_bytes)
    (output_dir / "d5_metadata_manifest.sha256").write_text(
        f"{manifest_sha}  d5_metadata_manifest.json\n",
        encoding="ascii",
    )
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Join CodeNet metadata for the pre-registered D5 sensitivity.")
    parser.add_argument("--metadata-root", type=Path, required=True)
    parser.add_argument("--d3-dir", type=Path, required=True)
    parser.add_argument("--statement-d4-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--metadata-archive-sha256", required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    manifest = build_d5_metadata_artifacts(
        metadata_root=args.metadata_root,
        d3_dir=args.d3_dir,
        statement_d4_dir=args.statement_d4_dir,
        output_dir=args.output_dir,
        metadata_archive_sha256=args.metadata_archive_sha256,
    )
    print(json.dumps(manifest["summary"], ensure_ascii=False, indent=2, sort_keys=True))
    print(f"manifest={args.output_dir / 'd5_metadata_manifest.json'}")


if __name__ == "__main__":
    main()
