from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import tarfile
from pathlib import Path
from typing import Any


def materialize_candidates(
    *,
    frame_report_path: Path,
    metadata_archive: Path,
    source_archive: Path,
    output_archive: Path,
    output_manifest: Path,
) -> dict[str, Any]:
    """Copy accepted Java sources from the metadata-eligible independent frame."""

    frame = json.loads(frame_report_path.read_text(encoding="utf-8"))
    if frame.get("schema_version") != "code2hyp-codenet-java-stage-b-frame-v1":
        raise ValueError("unsupported Java Stage B frame report")
    eligible_components = {
        str(row["component_id"]): tuple(str(value) for value in row["java_problem_ids"])
        for row in frame["components"]
        if row["metadata_eligible_for_evaluation_minimum_16"]
    }
    component_by_problem = {
        problem_id: component_id
        for component_id, problem_ids in eligible_components.items()
        for problem_id in problem_ids
    }
    expected = _accepted_java_members(metadata_archive, component_by_problem)
    if not expected:
        raise ValueError("the eligible frame contains no accepted Java submissions")

    output_archive.parent.mkdir(parents=True, exist_ok=True)
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    if output_archive.exists() or output_manifest.exists():
        raise ValueError("candidate archive and manifest outputs must not already exist")
    temporary_archive = output_archive.with_name(output_archive.name + ".tmp")
    temporary_manifest = output_manifest.with_name(output_manifest.name + ".tmp")

    inventory = []
    found = set()
    try:
        with tarfile.open(source_archive, "r|gz") as source, tarfile.open(temporary_archive, "w") as target:
            for member in source:
                record = expected.get(member.name)
                if record is None:
                    continue
                stream = source.extractfile(member)
                if stream is None:
                    raise ValueError(f"cannot read source member {member.name}")
                payload = stream.read()
                arcname = f"{record['problem_id']}/{record['submission_id']}.java"
                target_member = tarfile.TarInfo(arcname)
                target_member.size = len(payload)
                target_member.mode = 0o644
                target_member.mtime = 0
                target.addfile(target_member, io.BytesIO(payload))
                found.add(member.name)
                inventory.append(
                    {
                        **record,
                        "candidate_archive_member": arcname,
                        "source_bytes": len(payload),
                        "source_sha256": hashlib.sha256(payload).hexdigest(),
                    }
                )
        missing = sorted(set(expected) - found)
        if missing:
            raise ValueError(f"source archive is missing {len(missing)} accepted Java submissions; first={missing[0]}")
        inventory.sort(key=lambda row: (row["component_id"], row["problem_id"], row["submission_id"]))
        manifest = {
            "schema_version": "code2hyp-codenet-java-stage-b-candidates-v1",
            "status": "materialized_before_parse_and_D0_D4",
            "inputs": {
                "frame_report_sha256": _sha256_file(frame_report_path),
                "metadata_archive_sha256": frame["inputs"]["metadata_archive"]["sha256"],
                "source_archive_sha256": frame["inputs"]["source_archive"]["sha256"],
            },
            "counts": {
                "eligible_official_components": len(eligible_components),
                "problem_ids": len(component_by_problem),
                "accepted_java_sources": len(inventory),
                "distinct_users": len({row["user_id"] for row in inventory}),
            },
            "candidate_archive": {
                "path": str(output_archive),
                "bytes": temporary_archive.stat().st_size,
                "sha256": _sha256_file(temporary_archive),
            },
            "required_next_gate": "parse Java raw AST and apply D0-D4 before any split",
            "inventory": inventory,
        }
        temporary_manifest.write_bytes(_canonical_json_bytes(manifest))
        os.replace(temporary_archive, output_archive)
        os.replace(temporary_manifest, output_manifest)
        return manifest
    finally:
        temporary_archive.unlink(missing_ok=True)
        temporary_manifest.unlink(missing_ok=True)


def _accepted_java_members(metadata_archive: Path, component_by_problem: dict[str, str]) -> dict[str, dict[str, str]]:
    expected = {}
    with tarfile.open(metadata_archive, "r:gz") as archive:
        for member in archive:
            if not member.isfile() or not member.name.endswith(".csv") or member.name.endswith("/problem_list.csv"):
                continue
            stream = archive.extractfile(member)
            if stream is None:
                raise ValueError(f"cannot read metadata member {member.name}")
            reader = csv.DictReader(io.TextIOWrapper(stream, encoding="utf-8", errors="strict", newline=""))
            for row in reader:
                problem_id = str(row.get("problem_id", ""))
                if problem_id not in component_by_problem or row.get("language") != "Java" or row.get("status") != "Accepted":
                    continue
                extension = str(row["filename_ext"])
                submission_id = str(row["submission_id"])
                source_member = f"Project_CodeNet/data/{problem_id}/Java/{submission_id}.{extension}"
                if source_member in expected:
                    raise ValueError(f"duplicate metadata record for {source_member}")
                expected[source_member] = {
                    "component_id": component_by_problem[problem_id],
                    "problem_id": problem_id,
                    "submission_id": submission_id,
                    "user_id": str(row["user_id"]),
                    "source_archive_member": source_member,
                }
    return expected


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Materialize independent full-CodeNet Java Stage B candidates.")
    parser.add_argument("--frame-report", required=True, type=Path)
    parser.add_argument("--metadata-archive", required=True, type=Path)
    parser.add_argument("--source-archive", required=True, type=Path)
    parser.add_argument("--output-archive", required=True, type=Path)
    parser.add_argument("--output-manifest", required=True, type=Path)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = materialize_candidates(
        frame_report_path=args.frame_report,
        metadata_archive=args.metadata_archive,
        source_archive=args.source_archive,
        output_archive=args.output_archive,
        output_manifest=args.output_manifest,
    )
    print(json.dumps(result["counts"], indent=2, sort_keys=True))
    print(json.dumps(result["candidate_archive"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
