from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from geometry_profile_research.codenet_eligibility import canonical_json_bytes, stable_sha256
from geometry_profile_research.codenet_split import hamilton_quotas


def analyze_sampling_design(
    cluster_rows: Sequence[Mapping[str, Any]],
    *,
    candidate_train_programs: Sequence[int],
    minimum_train_eligibility_headroom: float = 1.25,
) -> dict[str, Any]:
    """Choose the largest full-frame train sample with eligibility headroom."""

    evaluation = [row for row in cluster_rows if row.get("eligible_evaluation_minimum_16") is True]
    candidates = tuple(sorted({int(value) for value in candidate_train_programs}))
    if not evaluation or not candidates or candidates[0] < 16:
        raise ValueError("sampling design requires evaluation clusters and K >= 16 candidates")
    if minimum_train_eligibility_headroom <= 1.0:
        raise ValueError("train eligibility headroom must exceed one")
    full_quotas = hamilton_quotas(len(evaluation), (3, 1, 4))
    rows = []
    for programs_per_cluster in candidates:
        eligible_train = sum(
            int(row["retained_programs_after_d0_d4"]) >= programs_per_cluster
            and int(row["distinct_users_after_d0_d4"]) >= programs_per_cluster
            for row in evaluation
        )
        maximum_total, maximum_quotas = _maximum_role_design(
            evaluation_clusters=len(evaluation),
            train_eligible_clusters=eligible_train,
        )
        full_frame_feasible = eligible_train >= full_quotas[0]
        headroom = eligible_train / full_quotas[0]
        rows.append(
            {
                "train_programs_per_cluster": programs_per_cluster,
                "eligible_train_clusters": eligible_train,
                "full_frame_feasible": full_frame_feasible,
                "full_frame_quotas_train_validation_test": list(full_quotas),
                "train_eligibility_headroom_ratio": headroom,
                "train_programs_if_full_frame": (
                    full_quotas[0] * programs_per_cluster if full_frame_feasible else None
                ),
                "maximum_feasible_total_clusters": maximum_total,
                "maximum_feasible_quotas_train_validation_test": list(maximum_quotas),
                "reserve_clusters_at_maximum": len(evaluation) - maximum_total,
            }
        )
    eligible_candidates = [
        row
        for row in rows
        if row["full_frame_feasible"]
        and row["train_eligibility_headroom_ratio"] >= minimum_train_eligibility_headroom
    ]
    if not eligible_candidates:
        raise ValueError("no candidate preserves the full frame and train-eligibility headroom")
    selected = max(eligible_candidates, key=lambda row: row["train_programs_per_cluster"])
    return {
        "schema_version": "codenet-java-stage-b-sampling-design-v1",
        "status": "pre_registration_pre_split_design_selection_without_retrieval_metrics",
        "evaluation_clusters": len(evaluation),
        "role_weights_train_validation_test": [3, 1, 4],
        "selection_rule": (
            "largest K retaining the complete evaluation frame and at least the frozen "
            "train-eligibility headroom"
        ),
        "minimum_train_eligibility_headroom": minimum_train_eligibility_headroom,
        "candidates": rows,
        "selected_train_programs_per_cluster": selected["train_programs_per_cluster"],
        "selected_quotas_train_validation_test": list(full_quotas),
        "retrieval_metrics_opened": False,
    }


def _maximum_role_design(
    *,
    evaluation_clusters: int,
    train_eligible_clusters: int,
) -> tuple[int, tuple[int, ...]]:
    for total in range(evaluation_clusters, 0, -1):
        quotas = hamilton_quotas(total, (3, 1, 4))
        if quotas[0] <= train_eligible_clusters:
            return total, quotas
    raise ValueError("no positive Stage B split is feasible")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare pre-split Java Stage B sampling budgets.")
    parser.add_argument("--clusters", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--candidates", default="16,24,32,40,48,56,64")
    parser.add_argument("--minimum-headroom", type=float, default=1.25)
    args = parser.parse_args()
    cluster_bytes = args.clusters.read_bytes()
    rows = [json.loads(line) for line in cluster_bytes.splitlines() if line.strip()]
    result = analyze_sampling_design(
        rows,
        candidate_train_programs=tuple(int(value) for value in args.candidates.split(",")),
        minimum_train_eligibility_headroom=args.minimum_headroom,
    )
    result["input"] = {
        "path": str(args.clusters),
        "sha256": stable_sha256(cluster_bytes),
    }
    content = canonical_json_bytes(result)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists() and args.output.read_bytes() != content:
        raise ValueError(f"refusing to overwrite a different sampling-design report: {args.output}")
    args.output.write_bytes(content)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
