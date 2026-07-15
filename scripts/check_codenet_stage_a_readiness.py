from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from geometry_profile_research.codenet_eligibility import canonical_json_bytes


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def evaluate_readiness(
    *,
    project_root: Path,
    design: dict[str, Any],
    d0_d2: dict[str, Any],
    d3: dict[str, Any],
    d4: dict[str, Any],
    d5: dict[str, Any],
    attrition: dict[str, Any],
    manifest_hashes: dict[str, str] | None = None,
) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []

    def add(identifier: str, passed: bool, detail: str, *, blocking: bool = True) -> None:
        checks.append({"id": identifier, "passed": passed, "blocking": blocking, "detail": detail})

    add("dataset_file_count", d0_d2["summary"]["source_files"] == 240000, str(d0_d2["summary"]["source_files"]))
    parse_rate = float(d0_d2["summary"]["parse_rate"])
    add("language_parse_rate", parse_rate >= float(design["eligibility"]["required_parse_rate"]), f"{parse_rate:.6f}")
    eligible = int(d4["summary"]["eligible_problem_clusters_minimum_64"])
    required = int(design["eligibility"]["minimum_clusters_for_practical_claim"])
    add("power_cluster_count", eligible >= required, f"eligible={eligible}, required={required}")
    add("d3_full_scan", d3["experiment_role"].startswith("full_pre_split"), d3["experiment_role"])
    add("statements_complete", d4["summary"]["descriptions_missing"] == 0, f"missing={d4['summary']['descriptions_missing']}")
    official_map_complete = d4["protocol"].get("official_identical_problem_map") == "applied_and_verified"
    add("official_identical_problem_map", official_map_complete, str(d4["protocol"].get("official_identical_problem_map")))
    expected_map_sha = (
        design.get("dataset", {}).get("official_identical_problem_map", {}).get("sha256")
    )
    actual_map_sha = d4.get("input", {}).get("official_identical_problem_map_sha256")
    if expected_map_sha is not None:
        add(
            "official_identical_problem_map_checksum",
            actual_map_sha == expected_map_sha,
            f"actual={actual_map_sha}, expected={expected_map_sha}",
        )
    expected_archive_sha = design.get("dataset", {}).get("official_full_archive", {}).get("sha256")
    actual_archive_sha = d4.get("input", {}).get("official_full_archive_sha256")
    if expected_archive_sha is not None:
        add(
            "official_full_archive_checksum",
            actual_archive_sha == expected_archive_sha,
            f"actual={actual_archive_sha}, expected={expected_archive_sha}",
        )
    minimum_users = int(d5["summary"]["minimum_distinct_users_per_problem_cluster"])
    add("user_distinct_sampling", minimum_users >= 80, f"minimum_distinct_users={minimum_users}")
    add(
        "author_removal_not_primary",
        "not estimand-preserving" in attrition["summary"]["decision"],
        attrition["summary"]["decision"],
    )
    audit = design.get("eligibility_audit")
    if audit is not None:
        add(
            "eligibility_cluster_count_pinned",
            int(audit["eligible_problem_clusters_minimum_64"]) == eligible,
            f"manifest={eligible}, design={audit['eligible_problem_clusters_minimum_64']}",
        )
        add(
            "eligibility_user_count_pinned",
            int(audit["minimum_distinct_users_per_problem_cluster"]) == minimum_users,
            f"manifest={minimum_users}, design={audit['minimum_distinct_users_per_problem_cluster']}",
        )
        if manifest_hashes is not None:
            for key, audit_key in (
                ("d0_d2", "d0_d2_manifest_sha256"),
                ("d3", "d3_manifest_sha256"),
                ("d4", "statement_d4_manifest_sha256"),
                ("d5", "d5_metadata_manifest_sha256"),
                ("attrition", "d5_attrition_manifest_sha256"),
            ):
                actual = manifest_hashes[key]
                expected = str(audit[audit_key])
                add(
                    f"{key}_manifest_pinned",
                    actual == expected,
                    f"actual={actual}, expected={expected}",
                )
    add(
        "registration_doi_pending",
        design.get("registration_doi") is None,
        "expected before Stage A publication",
        blocking=False,
    )
    add(
        "design_is_draft",
        design.get("status") == "draft_not_registered",
        str(design.get("status")),
    )
    add("split_not_generated", design["split"]["generated"] is False, "must remain false before Stage A", blocking=False)
    add("test_not_opened", design["test_policy"]["test_labels_opened"] is False, "must remain false", blocking=False)

    git_dir = project_root / ".git"
    add("independent_git_repository", git_dir.exists(), str(git_dir))
    commit = None
    clean = False
    if git_dir.exists():
        commit_result = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=project_root, text=True, capture_output=True, check=False
        )
        status_result = subprocess.run(
            ["git", "status", "--porcelain"], cwd=project_root, text=True, capture_output=True, check=False
        )
        commit = commit_result.stdout.strip() or None
        clean = status_result.returncode == 0 and not status_result.stdout.strip()
    add("immutable_commit", bool(commit), str(commit))
    add("clean_worktree", clean, str(clean))
    lockfiles = [project_root / "uv.lock", project_root / "poetry.lock", project_root / "requirements.lock"]
    existing_lockfiles = [str(path.name) for path in lockfiles if path.is_file()]
    add("dependency_lockfile", bool(existing_lockfiles), ",".join(existing_lockfiles) or "missing")

    blocking_failures = [check["id"] for check in checks if check["blocking"] and not check["passed"]]
    return {
        "schema_version": "code2hyp-stage-a-readiness-v1",
        "ready_for_stage_a_registration": not blocking_failures,
        "blocking_failures": blocking_failures,
        "checks": checks,
    }


def build_readiness_report(*, project_root: Path, design_path: Path, output_path: Path) -> dict[str, Any]:
    d0_d2_path = project_root / "data/codenet_python800_eligibility_d0_d2/eligibility_manifest.json"
    d3_path = project_root / "data/codenet_python800_eligibility_d0_d3/d3_manifest.json"
    d4_path = project_root / "data/codenet_python800_eligibility_d4_statements/statement_d4_manifest.json"
    d5_path = project_root / "data/codenet_python800_d5_metadata/d5_metadata_manifest.json"
    attrition_path = project_root / "data/codenet_python800_d5_attrition/d5_attrition_manifest.json"
    report = evaluate_readiness(
        project_root=project_root,
        design=_load(design_path),
        d0_d2=_load(d0_d2_path),
        d3=_load(d3_path),
        d4=_load(d4_path),
        d5=_load(d5_path),
        attrition=_load(attrition_path),
        manifest_hashes={
            "d0_d2": _sha256(d0_d2_path),
            "d3": _sha256(d3_path),
            "d4": _sha256(d4_path),
            "d5": _sha256(d5_path),
            "attrition": _sha256(attrition_path),
        },
    )
    report["inputs"] = {
        "design": {"path": str(design_path), "sha256": _sha256(design_path)},
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(canonical_json_bytes(report))
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Check whether Code2Hyp CodeNet Stage A can be registered.")
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--design", type=Path, default=PROJECT_ROOT / "configs/codenet_python800_stage_a_draft.json")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "reports/codenet_stage_a_readiness.json")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = build_readiness_report(project_root=args.project_root, design_path=args.design, output_path=args.output)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    if not report["ready_for_stage_a_registration"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
