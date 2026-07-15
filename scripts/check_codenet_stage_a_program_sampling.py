from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from geometry_profile_research.codenet_eligibility import canonical_json_bytes, jsonl_bytes, stable_sha256
from geometry_profile_research.codenet_sampling import (
    SAMPLING_SCHEMA_VERSION,
    iter_jsonl,
    select_non_test_programs,
    validate_sampling_protocol,
)


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def audit_program_sampling(
    *,
    protocol_path: Path,
    design_path: Path,
    registration_path: Path,
    split_manifest_path: Path,
    assignments_path: Path,
    d5_manifest_path: Path,
    d5_index_path: Path,
    sampling_dir: Path,
) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []

    def add(identifier: str, passed: bool, detail: str, *, blocking: bool = True) -> None:
        checks.append({"id": identifier, "passed": passed, "blocking": blocking, "detail": detail})

    protocol_bytes = protocol_path.read_bytes()
    protocol = json.loads(protocol_bytes)
    design_bytes = design_path.read_bytes()
    design = json.loads(design_bytes)
    registration_bytes = registration_path.read_bytes()
    registration = json.loads(registration_bytes)
    split_manifest_bytes = split_manifest_path.read_bytes()
    split_manifest = json.loads(split_manifest_bytes)
    d5_manifest_bytes = d5_manifest_path.read_bytes()
    d5_index_sha = stable_sha256(d5_index_path.read_bytes())
    try:
        beacon_key = validate_sampling_protocol(
            protocol=protocol,
            design=design,
            design_bytes=design_bytes,
            registration=registration,
            registration_bytes=registration_bytes,
            split_manifest=split_manifest,
            split_manifest_bytes=split_manifest_bytes,
            d5_manifest_bytes=d5_manifest_bytes,
            d5_index_bytes_sha256=d5_index_sha,
        )
    except (KeyError, TypeError, ValueError) as error:
        add("frozen_sampling_provenance", False, str(error))
        return {
            "schema_version": "code2hyp-stage-a-program-sampling-audit-v1",
            "valid_for_validation_pipeline": False,
            "blocking_failures": ["frozen_sampling_provenance"],
            "checks": checks,
        }
    add("frozen_sampling_provenance", True, stable_sha256(protocol_bytes))

    assignments_bytes = assignments_path.read_bytes()
    assignments = _load_jsonl(assignments_path)
    add(
        "cluster_assignments_pinned",
        stable_sha256(assignments_bytes) == str(protocol["cluster_split"]["assignment_sha256"]),
        stable_sha256(assignments_bytes),
    )
    randomness = protocol["randomness"]
    sampling = design["sampling"]
    expected_train, expected_validation, expected_summary = select_non_test_programs(
        metadata_rows=iter_jsonl(d5_index_path),
        assignments=assignments,
        beacon_key=beacon_key,
        dataset_revision=str(design["dataset"]["revision"]),
        program_domain=str(randomness["program_domain"]),
        user_domain=str(randomness["user_domain"]),
        train_programs_per_cluster=int(sampling["train_programs_per_cluster"]),
        validation_queries_per_cluster=int(sampling["validation_queries_per_cluster"]),
        validation_gallery_per_cluster=int(sampling["validation_gallery_per_cluster"]),
    )
    expected_train_bytes = jsonl_bytes(expected_train)
    expected_validation_bytes = jsonl_bytes(expected_validation)
    actual_train_bytes = (sampling_dir / "train_programs.jsonl").read_bytes()
    actual_validation_bytes = (sampling_dir / "validation_programs.jsonl").read_bytes()
    add(
        "train_sampling_rederived",
        actual_train_bytes == expected_train_bytes,
        f"actual={stable_sha256(actual_train_bytes)}, expected={stable_sha256(expected_train_bytes)}",
    )
    add(
        "validation_sampling_rederived",
        actual_validation_bytes == expected_validation_bytes,
        f"actual={stable_sha256(actual_validation_bytes)}, expected={stable_sha256(expected_validation_bytes)}",
    )

    actual_train = _load_jsonl(sampling_dir / "train_programs.jsonl")
    actual_validation = _load_jsonl(sampling_dir / "validation_programs.jsonl")
    all_public_rows = actual_train + actual_validation
    test_clusters = {str(row["cluster_id"]) for row in assignments if row["split"] == "test"}
    leaked_test_clusters = sorted({str(row.get("cluster_id")) for row in all_public_rows} & test_clusters)
    add("test_program_ids_sealed", not leaked_test_clusters, ",".join(leaked_test_clusters) or "none")
    leaked_user_keys = sorted(
        {key for row in all_public_rows for key in row if "user" in key.casefold()}
    )
    add("no_user_hashes_published", not leaked_user_keys, ",".join(leaked_user_keys) or "none")
    sources = [str(row.get("source_relpath", "")) for row in all_public_rows]
    add("selected_sources_unique", len(sources) == len(set(sources)) and all(sources), str(len(set(sources))))

    validation_roles: dict[str, Counter[str]] = defaultdict(Counter)
    for row in actual_validation:
        validation_roles[str(row["cluster_id"])][str(row["role"])] += 1
    expected_validation_clusters = {
        str(row["cluster_id"]) for row in assignments if row["split"] == "validation"
    }
    valid_role_counts = set(validation_roles) == expected_validation_clusters and all(
        counts == {"query": int(sampling["validation_queries_per_cluster"]),
                   "gallery": int(sampling["validation_gallery_per_cluster"])}
        for counts in validation_roles.values()
    )
    add("validation_role_counts", valid_role_counts, f"clusters={len(validation_roles)}")

    source_to_user: dict[str, str] = {}
    selected_sources = set(sources)
    for row in iter_jsonl(d5_index_path):
        source = str(row["source_relpath"])
        if source in selected_sources:
            source_to_user[source] = str(row["user_id_sha256"])
    add("selected_sources_join_d5", set(source_to_user) == selected_sources, str(len(source_to_user)))
    users_by_cluster: dict[str, list[str]] = defaultdict(list)
    for row in all_public_rows:
        users_by_cluster[str(row["cluster_id"])].append(source_to_user.get(str(row["source_relpath"]), ""))
    add(
        "one_program_per_user_within_cluster",
        all(len(users) == len(set(users)) and all(users) for users in users_by_cluster.values()),
        f"clusters={len(users_by_cluster)}",
    )

    manifest_path = sampling_dir / "program_sampling_manifest.json"
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    add("sampling_manifest_schema", manifest.get("schema_version") == SAMPLING_SCHEMA_VERSION, str(manifest.get("schema_version")))
    manifest_protocol = manifest.get("protocol", {})
    add(
        "downstream_test_state_sealed",
        manifest_protocol.get("test_program_sampling_generated") is False
        and manifest_protocol.get("test_relevance_labels_opened") is False
        and manifest_protocol.get("test_retrieval_metrics_computed") is False
        and manifest_protocol.get("validation_retrieval_metrics_computed") is False,
        json.dumps(
            {
                "test_program_sampling_generated": manifest_protocol.get("test_program_sampling_generated"),
                "test_relevance_labels_opened": manifest_protocol.get("test_relevance_labels_opened"),
                "test_retrieval_metrics_computed": manifest_protocol.get("test_retrieval_metrics_computed"),
                "validation_retrieval_metrics_computed": manifest_protocol.get("validation_retrieval_metrics_computed"),
            },
            sort_keys=True,
        ),
    )
    expected_inputs = {
        "sampling_protocol": stable_sha256(protocol_bytes),
        "registered_design": stable_sha256(design_bytes),
        "registration": stable_sha256(registration_bytes),
        "split_manifest": stable_sha256(split_manifest_bytes),
        "cluster_assignments": stable_sha256(assignments_bytes),
        "d5_manifest": stable_sha256(d5_manifest_bytes),
        "d5_index": d5_index_sha,
    }
    actual_inputs = {
        "sampling_protocol": manifest.get("input", {}).get("sampling_protocol", {}).get("sha256"),
        "registered_design": manifest.get("input", {}).get("registered_design_sha256"),
        "registration": manifest.get("input", {}).get("registration_sha256"),
        "split_manifest": manifest.get("input", {}).get("split_manifest_sha256"),
        "cluster_assignments": manifest.get("input", {}).get("cluster_assignments_sha256"),
        "d5_manifest": manifest.get("input", {}).get("d5_manifest_sha256"),
        "d5_index": manifest.get("input", {}).get("d5_metadata_index_sha256"),
    }
    add("manifest_inputs_pinned", actual_inputs == expected_inputs, json.dumps(actual_inputs, sort_keys=True))
    artifact_hashes = {str(item["path"]): str(item["sha256"]) for item in manifest.get("artifacts", [])}
    expected_artifact_hashes = {
        "train_programs.jsonl": stable_sha256(actual_train_bytes),
        "validation_programs.jsonl": stable_sha256(actual_validation_bytes),
        "program_sampling_summary.json": stable_sha256(
            (sampling_dir / "program_sampling_summary.json").read_bytes()
        ),
    }
    add("artifact_hashes_pinned", artifact_hashes == expected_artifact_hashes, json.dumps(artifact_hashes, sort_keys=True))
    expected_sidecar = f"{stable_sha256(manifest_bytes)}  program_sampling_manifest.json\n"
    actual_sidecar = (sampling_dir / "program_sampling_manifest.sha256").read_text(encoding="ascii")
    add("manifest_sidecar", actual_sidecar == expected_sidecar, actual_sidecar.strip())
    add("summary_rederived", manifest.get("summary") == expected_summary, json.dumps(expected_summary, sort_keys=True))

    blocking_failures = [check["id"] for check in checks if check["blocking"] and not check["passed"]]
    return {
        "schema_version": "code2hyp-stage-a-program-sampling-audit-v1",
        "valid_for_validation_pipeline": not blocking_failures,
        "blocking_failures": blocking_failures,
        "summary": {
            "sampling_protocol_sha256": stable_sha256(protocol_bytes),
            "train_programs_sha256": stable_sha256(actual_train_bytes),
            "validation_programs_sha256": stable_sha256(actual_validation_bytes),
            **expected_summary,
        },
        "checks": checks,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Re-derive and audit CodeNet Stage A train/validation sampling.")
    parser.add_argument("--protocol", type=Path, default=PROJECT_ROOT / "configs/codenet_python800_stage_a_sampling_protocol_v1.json")
    parser.add_argument("--design", type=Path, default=PROJECT_ROOT / "configs/codenet_python800_stage_a_draft.json")
    parser.add_argument("--registration", type=Path, default=PROJECT_ROOT / "registrations/codenet_python800_stage_a_registration_v1.json")
    parser.add_argument("--split-manifest", type=Path, default=PROJECT_ROOT / "data/codenet_python800_stage_a_split/split_manifest.json")
    parser.add_argument("--assignments", type=Path, default=PROJECT_ROOT / "data/codenet_python800_stage_a_split/cluster_assignments.jsonl")
    parser.add_argument("--d5-manifest", type=Path, default=PROJECT_ROOT / "data/codenet_python800_d5_metadata/d5_metadata_manifest.json")
    parser.add_argument("--d5-index", type=Path, default=PROJECT_ROOT / "data/codenet_python800_d5_metadata/d5_metadata_index.jsonl")
    parser.add_argument("--sampling-dir", type=Path, default=PROJECT_ROOT / "data/codenet_python800_stage_a_program_sampling")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "reports/codenet_stage_a_program_sampling_audit.json")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = audit_program_sampling(
        protocol_path=args.protocol,
        design_path=args.design,
        registration_path=args.registration,
        split_manifest_path=args.split_manifest,
        assignments_path=args.assignments,
        d5_manifest_path=args.d5_manifest,
        d5_index_path=args.d5_index,
        sampling_dir=args.sampling_dir,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(canonical_json_bytes(report))
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    if not report["valid_for_validation_pipeline"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
