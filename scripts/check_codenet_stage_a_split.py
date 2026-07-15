from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from geometry_profile_research.codenet_eligibility import canonical_json_bytes, jsonl_bytes, stable_sha256
from geometry_profile_research.codenet_split import (
    SPLIT_SCHEMA_VERSION,
    assign_cluster_ids,
    eligible_cluster_ids,
    hamilton_quotas,
    validate_registration,
)


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def audit_split(
    *,
    design_path: Path,
    registration_path: Path,
    clusters_path: Path,
    statement_d4_manifest_path: Path,
    split_dir: Path,
) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []

    def add(identifier: str, passed: bool, detail: str, *, blocking: bool = True) -> None:
        checks.append({"id": identifier, "passed": passed, "blocking": blocking, "detail": detail})

    design_bytes = design_path.read_bytes()
    design = json.loads(design_bytes)
    registration_bytes = registration_path.read_bytes()
    registration = json.loads(registration_bytes)
    try:
        beacon_key = validate_registration(
            design=design,
            registration=registration,
            design_bytes=design_bytes,
        )
    except (KeyError, TypeError, ValueError) as error:
        add("registration_provenance", False, str(error))
        blocking_failures = [check["id"] for check in checks if check["blocking"] and not check["passed"]]
        return {
            "schema_version": "code2hyp-stage-a-split-audit-v1",
            "valid_for_program_sampling": False,
            "blocking_failures": blocking_failures,
            "checks": checks,
        }
    add("registration_provenance", True, str(registration["registration"]["doi"]))

    d4_manifest_bytes = statement_d4_manifest_path.read_bytes()
    d4_manifest = json.loads(d4_manifest_bytes)
    add(
        "d4_pre_split_state",
        d4_manifest["protocol"].get("split_status") == "not_generated"
        and d4_manifest["protocol"].get("retrieval_metrics_opened") is False,
        f"split={d4_manifest['protocol'].get('split_status')}, metrics={d4_manifest['protocol'].get('retrieval_metrics_opened')}",
    )

    expected_count = int(registration["design"]["eligible_problem_clusters"])
    cluster_ids = eligible_cluster_ids(_load_jsonl(clusters_path), expected_count=expected_count)
    weights = tuple(int(value) for value in design["split"]["weights_train_validation_test"])
    quotas = hamilton_quotas(expected_count, weights)
    beacon_assignments = assign_cluster_ids(
        cluster_ids=cluster_ids,
        beacon_key=beacon_key,
        dataset_revision=str(design["dataset"]["revision"]),
        quotas=quotas,
    )
    expected_assignment_bytes = jsonl_bytes(beacon_assignments)

    assignment_path = split_dir / "cluster_assignments.jsonl"
    actual_assignment_bytes = assignment_path.read_bytes()
    assignment_matches = actual_assignment_bytes == expected_assignment_bytes
    add(
        "assignment_rederived",
        assignment_matches,
        f"actual={stable_sha256(actual_assignment_bytes)}, expected={stable_sha256(expected_assignment_bytes)}",
    )
    actual_assignments = _load_jsonl(assignment_path)
    actual_counts = Counter(str(row.get("split")) for row in actual_assignments)
    expected_counts = {"train": quotas[0], "validation": quotas[1], "test": quotas[2]}
    add("split_quotas", dict(actual_counts) == expected_counts, json.dumps(dict(actual_counts), sort_keys=True))
    add(
        "unique_cluster_assignments",
        len({str(row.get("cluster_id")) for row in actual_assignments}) == expected_count,
        str(len({str(row.get('cluster_id')) for row in actual_assignments})),
    )
    add(
        "unique_hmac_ordering_keys",
        len({str(row.get("hmac_sha256")) for row in actual_assignments}) == expected_count,
        str(len({str(row.get('hmac_sha256')) for row in actual_assignments})),
    )
    forbidden_assignment_keys = {
        "problem_ids",
        "source_relpath",
        "query_id",
        "gallery_id",
        "relevance",
        "metric",
        "score",
    }
    leaked_keys = sorted(
        {
            key
            for row in actual_assignments
            for key in row
            if key.casefold() in forbidden_assignment_keys
        }
    )
    add("no_program_labels_or_metrics", not leaked_keys, ",".join(leaked_keys) or "none")

    manifest_path = split_dir / "split_manifest.json"
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    add("split_manifest_schema", manifest.get("schema_version") == SPLIT_SCHEMA_VERSION, str(manifest.get("schema_version")))
    protocol = manifest.get("protocol", {})
    add(
        "sealed_downstream_state",
        protocol.get("program_sampling_generated") is False
        and protocol.get("test_relevance_labels_opened") is False
        and protocol.get("retrieval_metrics_computed") is False,
        json.dumps(
            {
                "program_sampling_generated": protocol.get("program_sampling_generated"),
                "test_relevance_labels_opened": protocol.get("test_relevance_labels_opened"),
                "retrieval_metrics_computed": protocol.get("retrieval_metrics_computed"),
            },
            sort_keys=True,
        ),
    )
    expected_inputs = {
        "design": stable_sha256(design_bytes),
        "registration": stable_sha256(registration_bytes),
        "eligible_clusters": stable_sha256(clusters_path.read_bytes()),
        "statement_d4_manifest": stable_sha256(d4_manifest_bytes),
    }
    actual_inputs = {
        "design": manifest.get("input", {}).get("design", {}).get("sha256"),
        "registration": manifest.get("input", {}).get("registration", {}).get("sha256"),
        "eligible_clusters": manifest.get("input", {}).get("eligible_clusters", {}).get("sha256"),
        "statement_d4_manifest": manifest.get("input", {}).get("statement_d4_manifest_sha256"),
    }
    add("manifest_inputs_pinned", actual_inputs == expected_inputs, json.dumps(actual_inputs, sort_keys=True))
    artifact_hashes = {
        str(item["path"]): str(item["sha256"])
        for item in manifest.get("artifacts", [])
    }
    expected_artifact_hashes = {
        "cluster_assignments.jsonl": stable_sha256(actual_assignment_bytes),
        "split_summary.json": stable_sha256((split_dir / "split_summary.json").read_bytes()),
    }
    add(
        "artifact_hashes_pinned",
        artifact_hashes == expected_artifact_hashes,
        json.dumps(artifact_hashes, sort_keys=True),
    )
    expected_sidecar = f"{stable_sha256(manifest_bytes)}  split_manifest.json\n"
    actual_sidecar = (split_dir / "split_manifest.sha256").read_text(encoding="ascii")
    add("manifest_sidecar", actual_sidecar == expected_sidecar, actual_sidecar.strip())
    add(
        "summary_assignment_hash",
        manifest.get("summary", {}).get("assignment_sha256") == stable_sha256(actual_assignment_bytes),
        str(manifest.get("summary", {}).get("assignment_sha256")),
    )

    blocking_failures = [check["id"] for check in checks if check["blocking"] and not check["passed"]]
    return {
        "schema_version": "code2hyp-stage-a-split-audit-v1",
        "valid_for_program_sampling": not blocking_failures,
        "blocking_failures": blocking_failures,
        "summary": {
            "registration_doi": registration["registration"]["doi"],
            "beacon_uri": registration["nist_randomness_beacon"]["uri"],
            "cluster_count": len(actual_assignments),
            "quotas_train_validation_test": list(quotas),
            "assignment_sha256": stable_sha256(actual_assignment_bytes),
        },
        "checks": checks,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Re-derive and audit the registered CodeNet Stage A split.")
    parser.add_argument(
        "--design",
        type=Path,
        default=PROJECT_ROOT / "configs/codenet_python800_stage_a_draft.json",
    )
    parser.add_argument(
        "--registration",
        type=Path,
        default=PROJECT_ROOT / "registrations/codenet_python800_stage_a_registration_v1.json",
    )
    parser.add_argument(
        "--clusters",
        type=Path,
        default=(
            PROJECT_ROOT
            / "data/codenet_python800_eligibility_d4_statements/post_statement_d4_problem_clusters.jsonl"
        ),
    )
    parser.add_argument(
        "--statement-d4-manifest",
        type=Path,
        default=(
            PROJECT_ROOT / "data/codenet_python800_eligibility_d4_statements/statement_d4_manifest.json"
        ),
    )
    parser.add_argument(
        "--split-dir",
        type=Path,
        default=PROJECT_ROOT / "data/codenet_python800_stage_a_split",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "reports/codenet_stage_a_split_audit.json",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = audit_split(
        design_path=args.design,
        registration_path=args.registration,
        clusters_path=args.clusters,
        statement_d4_manifest_path=args.statement_d4_manifest,
        split_dir=args.split_dir,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(canonical_json_bytes(report))
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    if not report["valid_for_program_sampling"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
