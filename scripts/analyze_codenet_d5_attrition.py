from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from geometry_profile_research.codenet_eligibility import canonical_json_bytes, jsonl_bytes, stable_sha256


ATTRITION_SCHEMA_VERSION = "codenet-d5-attrition-planning-v1"


def hamilton_quotas(total: int, weights: tuple[int, ...]) -> tuple[int, ...]:
    if total <= 0 or not weights or any(weight <= 0 for weight in weights):
        raise ValueError("total and all weights must be positive")
    weight_sum = sum(weights)
    exact = [total * weight / weight_sum for weight in weights]
    floors = [math.floor(value) for value in exact]
    remainder = total - sum(floors)
    order = sorted(range(len(weights)), key=lambda index: (-(exact[index] - floors[index]), index))
    for index in order[:remainder]:
        floors[index] += 1
    return tuple(floors)


def _quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round(probability * (len(ordered) - 1))))
    return ordered[index]


def analyze_attrition(
    *,
    d5_index_rows: list[dict[str, Any]],
    cluster_ids: list[str],
    simulations: int,
    seed_offset: int,
    minimum_train_programs: int = 64,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if simulations <= 0:
        raise ValueError("simulations must be positive")
    quotas = hamilton_quotas(len(cluster_ids), (3, 1, 4))
    users_by_cluster: dict[str, list[str]] = defaultdict(list)
    for row in d5_index_rows:
        users_by_cluster[str(row["problem_cluster_id"])].append(str(row["user_id_sha256"]))
    simulation_rows: list[dict[str, Any]] = []
    for simulation in range(simulations):
        ordered = list(cluster_ids)
        random.Random(seed_offset + simulation).shuffle(ordered)
        train = ordered[: quotas[0]]
        test = ordered[quotas[0] + quotas[1] :]
        test_users = {user for cluster in test for user in users_by_cluster[cluster]}
        retained_counts = [
            sum(user not in test_users for user in users_by_cluster[cluster])
            for cluster in train
        ]
        original_train_programs = sum(len(users_by_cluster[cluster]) for cluster in train)
        retained_train_programs = sum(retained_counts)
        simulation_rows.append(
            {
                "simulation": simulation,
                "seed": seed_offset + simulation,
                "train_clusters": quotas[0],
                "validation_clusters": quotas[1],
                "test_clusters": quotas[2],
                "original_train_programs": original_train_programs,
                "retained_train_programs": retained_train_programs,
                "retained_fraction": retained_train_programs / original_train_programs,
                "train_clusters_with_at_least_64_programs": sum(
                    count >= minimum_train_programs for count in retained_counts
                ),
                "minimum_retained_programs_in_train_cluster": min(retained_counts),
            }
        )
    metric_names = (
        "retained_fraction",
        "train_clusters_with_at_least_64_programs",
        "minimum_retained_programs_in_train_cluster",
    )
    distributions: dict[str, dict[str, float]] = {}
    for name in metric_names:
        values = [float(row[name]) for row in simulation_rows]
        distributions[name] = {
            "mean": statistics.mean(values),
            "median": statistics.median(values),
            "p2_5": _quantile(values, 0.025),
            "p97_5": _quantile(values, 0.975),
            "minimum": min(values),
            "maximum": max(values),
        }
    summary = {
        "simulation_count": simulations,
        "seed_offset": seed_offset,
        "cluster_count": len(cluster_ids),
        "hamilton_quotas_train_validation_test": list(quotas),
        "minimum_train_programs": minimum_train_programs,
        "distributions": distributions,
        "decision": (
            "global author removal is not estimand-preserving; use user-distinct within-cluster sampling "
            "and report overlap as a diagnostic"
        ),
    }
    return summary, simulation_rows


def build_attrition_artifacts(
    *,
    d5_dir: Path,
    statement_d4_dir: Path,
    output_dir: Path,
    simulations: int = 1000,
    seed_offset: int = 100000,
) -> dict[str, Any]:
    d5_manifest_path = d5_dir / "d5_metadata_manifest.json"
    d4_manifest_path = statement_d4_dir / "statement_d4_manifest.json"
    d5_rows = [
        json.loads(line)
        for line in (d5_dir / "d5_metadata_index.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]
    cluster_ids = sorted(
        json.loads(line)["cluster_id"]
        for line in (statement_d4_dir / "post_statement_d4_problem_clusters.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    )
    summary, rows = analyze_attrition(
        d5_index_rows=d5_rows,
        cluster_ids=cluster_ids,
        simulations=simulations,
        seed_offset=seed_offset,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_payloads = {
        "d5_attrition_simulations.jsonl": jsonl_bytes(rows),
        "d5_attrition_summary.json": canonical_json_bytes(summary),
    }
    artifacts = []
    for filename, content in artifact_payloads.items():
        path = output_dir / filename
        path.write_bytes(content)
        artifacts.append(
            {"path": filename, "bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()}
        )
    manifest = {
        "schema_version": ATTRITION_SCHEMA_VERSION,
        "experiment_role": "pre_split_attrition_planning_not_a_dataset_split",
        "input": {
            "d5_manifest_sha256": stable_sha256(d5_manifest_path.read_bytes()),
            "statement_d4_manifest_sha256": stable_sha256(d4_manifest_path.read_bytes()),
        },
        "protocol": {
            "assignment": "independent planning permutations of problem cluster IDs",
            "quota_rule": "Hamilton apportionment for weights 3:1:4",
            "actual_beacon_used": False,
            "split_manifest_created": False,
            "retrieval_metrics_opened": False,
        },
        "summary": summary,
        "artifacts": sorted(artifacts, key=lambda item: item["path"]),
    }
    manifest_bytes = canonical_json_bytes(manifest)
    (output_dir / "d5_attrition_manifest.json").write_bytes(manifest_bytes)
    manifest_sha = stable_sha256(manifest_bytes)
    (output_dir / "d5_attrition_manifest.sha256").write_text(
        f"{manifest_sha}  d5_attrition_manifest.json\n",
        encoding="ascii",
    )
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plan the attrition of a global CodeNet author-removal control.")
    parser.add_argument("--d5-dir", type=Path, required=True)
    parser.add_argument("--statement-d4-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--simulations", type=int, default=1000)
    parser.add_argument("--seed-offset", type=int, default=100000)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    manifest = build_attrition_artifacts(
        d5_dir=args.d5_dir,
        statement_d4_dir=args.statement_d4_dir,
        output_dir=args.output_dir,
        simulations=args.simulations,
        seed_offset=args.seed_offset,
    )
    print(json.dumps(manifest["summary"], ensure_ascii=False, indent=2, sort_keys=True))
    print(f"manifest={args.output_dir / 'd5_attrition_manifest.json'}")


if __name__ == "__main__":
    main()
