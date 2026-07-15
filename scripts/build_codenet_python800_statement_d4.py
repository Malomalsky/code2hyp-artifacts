from __future__ import annotations

import argparse
import hashlib
import html.parser
import json
import sys
import unicodedata
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


D4_STATEMENT_SCHEMA_VERSION = "codenet-python800-statement-d4-v2"


class _VisibleTextParser(html.parser.HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.hidden_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        del attrs
        if tag.casefold() in {"script", "style"}:
            self.hidden_depth += 1

    def handle_endtag(self, tag: str) -> None:
        if tag.casefold() in {"script", "style"} and self.hidden_depth:
            self.hidden_depth -= 1

    def handle_data(self, data: str) -> None:
        if self.hidden_depth == 0:
            self.parts.append(data)


def normalized_problem_statement(html_source: str) -> str:
    parser = _VisibleTextParser()
    parser.feed(html_source)
    parser.close()
    visible = " ".join(parser.parts)
    normalized = unicodedata.normalize("NFKC", visible).casefold()
    return " ".join(normalized.split())


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


def read_official_identical_problem_clusters(path: Path) -> list[list[str]]:
    clusters: list[list[str]] = []
    cluster_by_problem: dict[str, int] = {}
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        problems = sorted({part.strip() for part in line.split(",") if part.strip()})
        if len(problems) < 2:
            raise ValueError(f"Official cluster at line {line_number} has fewer than two problem IDs")
        for problem in problems:
            if len(problem) != 6 or not problem.startswith("p") or not problem[1:].isdigit():
                raise ValueError(f"Invalid official problem ID at line {line_number}: {problem!r}")
            previous = cluster_by_problem.get(problem)
            if previous is not None:
                raise ValueError(
                    f"Official problem ID {problem} occurs in clusters at lines {previous} and {line_number}"
                )
            cluster_by_problem[problem] = line_number
        clusters.append(problems)
    if not clusters:
        raise ValueError("Official identical-problem map is empty")
    return clusters


def build_statement_d4_artifacts(
    *,
    descriptions_root: Path,
    d3_dir: Path,
    output_dir: Path,
    minimum_cluster_programs: int = 64,
    official_identical_problem_map: Path | None = None,
    official_source_readme: Path | None = None,
    official_archive_sha256: str | None = None,
) -> dict[str, Any]:
    d3_manifest_path = d3_dir / "d3_manifest.json"
    d3_manifest_sha = stable_sha256(d3_manifest_path.read_bytes())
    d3_clusters = _read_jsonl(d3_dir / "post_d3_problem_clusters.jsonl")
    problem_ids = sorted({str(problem) for cluster in d3_clusters for problem in cluster["problem_ids"]})
    problem_id_set = set(problem_ids)
    problem_index = {problem: index for index, problem in enumerate(problem_ids)}
    problem_dsu = DisjointSet(len(problem_ids))
    retained_by_problem: dict[str, int] = {}
    for cluster in d3_clusters:
        problems = [str(problem) for problem in cluster["problem_ids"]]
        for problem in problems[1:]:
            problem_dsu.union(problem_index[problems[0]], problem_index[problem])
        # Counts are available only at cluster level after cross-problem D3.
        if len(problems) == 1:
            retained_by_problem[problems[0]] = int(cluster["retained_programs_after_d0_d3"])

    statement_rows: list[dict[str, Any]] = []
    problems_by_statement_hash: dict[str, list[str]] = defaultdict(list)
    for problem in problem_ids:
        path = descriptions_root / f"{problem}.html"
        if not path.is_file():
            statement_rows.append(
                {
                    "problem_id": problem,
                    "description_path": None,
                    "description_available": False,
                    "raw_html_sha256": None,
                    "normalized_text_sha256": None,
                    "normalized_text_characters": 0,
                }
            )
            continue
        raw = path.read_bytes()
        source = raw.decode("utf-8", errors="strict")
        normalized = normalized_problem_statement(source)
        normalized_hash = stable_sha256(normalized)
        statement_rows.append(
            {
                "problem_id": problem,
                "description_path": path.name,
                "description_available": True,
                "raw_html_sha256": stable_sha256(raw),
                "normalized_text_sha256": normalized_hash,
                "normalized_text_characters": len(normalized),
            }
        )
        problems_by_statement_hash[normalized_hash].append(problem)

    statement_groups: list[dict[str, Any]] = []
    statement_edges: list[dict[str, Any]] = []
    for digest, problems in sorted(problems_by_statement_hash.items()):
        if len(problems) <= 1:
            continue
        ordered = sorted(problems)
        statement_groups.append(
            {
                "normalized_text_sha256": digest,
                "problem_ids": ordered,
                "problem_count": len(ordered),
            }
        )
        for left_index, left in enumerate(ordered):
            for right in ordered[left_index + 1 :]:
                problem_dsu.union(problem_index[left], problem_index[right])
                statement_edges.append(
                    {
                        "left_problem_id": left,
                        "right_problem_id": right,
                        "normalized_text_sha256": digest,
                        "rule": "identical_normalized_problem_statement",
                    }
                )

    official_clusters: list[list[str]] = []
    official_intersections: list[dict[str, Any]] = []
    official_edges: list[dict[str, Any]] = []
    official_map_sha256: str | None = None
    official_source_readme_sha256: str | None = None
    if official_identical_problem_map is not None:
        if official_archive_sha256 is None:
            raise ValueError("official_archive_sha256 is required with an official identical-problem map")
        normalized_archive_sha = official_archive_sha256.strip().casefold()
        if len(normalized_archive_sha) != 64 or any(
            character not in "0123456789abcdef" for character in normalized_archive_sha
        ):
            raise ValueError("official_archive_sha256 must contain exactly 64 hexadecimal characters")
        official_archive_sha256 = normalized_archive_sha
        official_map_bytes = official_identical_problem_map.read_bytes()
        official_map_sha256 = stable_sha256(official_map_bytes)
        official_clusters = read_official_identical_problem_clusters(official_identical_problem_map)
        for official_cluster_index, official_problems in enumerate(official_clusters, start=1):
            in_frame = sorted(set(official_problems) & problem_id_set)
            if not in_frame:
                continue
            outside_frame = sorted(set(official_problems) - problem_id_set)
            official_intersections.append(
                {
                    "official_cluster_index": official_cluster_index,
                    "official_problem_ids": official_problems,
                    "problem_ids_in_sampling_frame": in_frame,
                    "problem_ids_outside_sampling_frame": outside_frame,
                    "sampling_frame_member_count": len(in_frame),
                }
            )
            for left_index, left in enumerate(in_frame):
                for right in in_frame[left_index + 1 :]:
                    problem_dsu.union(problem_index[left], problem_index[right])
                    official_edges.append(
                        {
                            "left_problem_id": left,
                            "right_problem_id": right,
                            "official_cluster_index": official_cluster_index,
                            "rule": "official_identical_problem_cluster",
                        }
                    )
        if official_source_readme is not None:
            official_source_readme_sha256 = stable_sha256(official_source_readme.read_bytes())

    retained_by_d3_cluster: dict[frozenset[str], int] = {
        frozenset(str(problem) for problem in cluster["problem_ids"]): int(cluster["retained_programs_after_d0_d3"])
        for cluster in d3_clusters
    }
    problems_by_root: dict[int, list[str]] = defaultdict(list)
    for problem in problem_ids:
        problems_by_root[problem_dsu.find(problem_index[problem])].append(problem)
    final_clusters: list[dict[str, Any]] = []
    for problems in problems_by_root.values():
        ordered = sorted(problems)
        problem_set = set(ordered)
        retained = sum(
            count
            for cluster_problems, count in retained_by_d3_cluster.items()
            if cluster_problems.issubset(problem_set)
        )
        final_clusters.append(
            {
                "cluster_id": f"problem-{stable_sha256('|'.join(ordered))[:20]}",
                "problem_ids": ordered,
                "problem_count": len(ordered),
                "retained_programs_after_statement_d4": retained,
                "eligible_minimum_64": retained >= minimum_cluster_programs,
            }
        )
    final_clusters.sort(key=lambda item: item["cluster_id"])

    summary = {
        "problem_count": len(problem_ids),
        "descriptions_available": sum(int(row["description_available"]) for row in statement_rows),
        "descriptions_missing": sum(int(not row["description_available"]) for row in statement_rows),
        "identical_statement_groups": len(statement_groups),
        "identical_statement_edges": len(statement_edges),
        "official_identical_problem_clusters_total": len(official_clusters),
        "official_identical_problem_clusters_touching_sampling_frame": len(official_intersections),
        "official_identical_problem_clusters_with_multiple_sampling_frame_members": sum(
            int(row["sampling_frame_member_count"] > 1) for row in official_intersections
        ),
        "official_identical_problem_edges_within_sampling_frame": len(official_edges),
        "problem_cluster_count_after_statement_d4": len(final_clusters),
        "eligible_problem_clusters_minimum_64": sum(
            int(cluster["eligible_minimum_64"]) for cluster in final_clusters
        ),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts = [
        _write_and_hash(output_dir / "problem_statement_inventory.jsonl", jsonl_bytes(statement_rows)),
        _write_and_hash(output_dir / "identical_statement_groups.jsonl", jsonl_bytes(statement_groups)),
        _write_and_hash(output_dir / "identical_statement_edges.jsonl", jsonl_bytes(statement_edges)),
        _write_and_hash(
            output_dir / "official_identical_problem_intersections.jsonl",
            jsonl_bytes(official_intersections),
        ),
        _write_and_hash(output_dir / "official_identical_problem_edges.jsonl", jsonl_bytes(official_edges)),
        _write_and_hash(output_dir / "post_statement_d4_problem_clusters.jsonl", jsonl_bytes(final_clusters)),
        _write_and_hash(output_dir / "statement_d4_summary.json", canonical_json_bytes(summary)),
    ]
    manifest = {
        "schema_version": D4_STATEMENT_SCHEMA_VERSION,
        "experiment_role": "pre_split_statement_D4_without_retrieval_metrics",
        "input": {
            "descriptions_root": portable_manifest_path(descriptions_root, project_root=PROJECT_ROOT),
            "d3_manifest": portable_manifest_path(d3_manifest_path, project_root=PROJECT_ROOT),
            "d3_manifest_sha256": d3_manifest_sha,
            "official_identical_problem_map": (
                portable_manifest_path(official_identical_problem_map, project_root=PROJECT_ROOT)
                if official_identical_problem_map is not None
                else None
            ),
            "official_identical_problem_map_sha256": official_map_sha256,
            "official_source_readme": (
                portable_manifest_path(official_source_readme, project_root=PROJECT_ROOT)
                if official_source_readme is not None
                else None
            ),
            "official_source_readme_sha256": official_source_readme_sha256,
            "official_full_archive_sha256": official_archive_sha256,
        },
        "protocol": {
            "normalization": "visible HTML text; Unicode NFKC; casefold; collapsed whitespace",
            "edge_rule": "identical SHA-256 of normalized statement text",
            "manual_adjudication": False,
            "split_status": "not_generated",
            "retrieval_metrics_opened": False,
            "official_identical_problem_map": (
                "applied_and_verified"
                if official_identical_problem_map is not None
                else "pending_full_CodeNet_derived_metadata"
            ),
        },
        "summary": summary,
        "gate_precheck": {
            "at_least_764_eligible_clusters_for_300_100_364": (
                summary["eligible_problem_clusters_minimum_64"] >= 764
            ),
            "final_eligibility": (
                "passed"
                if official_identical_problem_map is not None
                and summary["eligible_problem_clusters_minimum_64"] >= 764
                else "failed"
                if official_identical_problem_map is not None
                else "pending_official_identical_problem_map"
            ),
        },
        "artifacts": sorted(artifacts, key=lambda item: item["path"]),
    }
    manifest_bytes = canonical_json_bytes(manifest)
    (output_dir / "statement_d4_manifest.json").write_bytes(manifest_bytes)
    manifest_sha = stable_sha256(manifest_bytes)
    (output_dir / "statement_d4_manifest.sha256").write_text(
        f"{manifest_sha}  statement_d4_manifest.json\n",
        encoding="ascii",
    )
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit identical CodeNet problem statements before split generation.")
    parser.add_argument("--descriptions-root", type=Path, required=True)
    parser.add_argument("--d3-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--minimum-cluster-programs", type=int, default=64)
    parser.add_argument("--official-identical-problem-map", type=Path)
    parser.add_argument("--official-source-readme", type=Path)
    parser.add_argument("--official-archive-sha256")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    manifest = build_statement_d4_artifacts(
        descriptions_root=args.descriptions_root,
        d3_dir=args.d3_dir,
        output_dir=args.output_dir,
        minimum_cluster_programs=args.minimum_cluster_programs,
        official_identical_problem_map=args.official_identical_problem_map,
        official_source_readme=args.official_source_readme,
        official_archive_sha256=args.official_archive_sha256,
    )
    print(json.dumps(manifest["summary"], ensure_ascii=False, indent=2, sort_keys=True))
    print(f"manifest={args.output_dir / 'statement_d4_manifest.json'}")


if __name__ == "__main__":
    main()
