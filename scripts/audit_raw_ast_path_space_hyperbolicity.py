from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from statistics import mean, median
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from geometry_profile_research.path_space_hyperbolicity import (
    matrix_max_four_point_delta,
    upper_triangle_records_to_distance_matrix,
)


DISTANCE_KEYS = (
    "oriented_endpoint_distance",
    "unoriented_endpoint_distance",
    "edge_symmetric_difference",
    "edge_jaccard_distance",
)


def _quantile(values: list[float], probability: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * probability)))
    return ordered[index]


def _summarize(values: list[float]) -> dict[str, float]:
    if not values:
        return {"count": 0.0, "mean": 0.0, "median": 0.0, "p90": 0.0, "max": 0.0}
    return {
        "count": float(len(values)),
        "mean": float(mean(values)),
        "median": float(median(values)),
        "p90": float(_quantile(values, 0.9)),
        "max": float(max(values)),
    }


def audit_path_space_hyperbolicity(input_path: Path) -> dict[str, Any]:
    deltas: dict[str, list[float]] = {key: [] for key in DISTANCE_KEYS}
    status_counts: dict[str, int] = {}
    eligible_scopes = 0
    total_scopes = 0
    for line in input_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        status = str(payload.get("status", "unknown"))
        status_counts[status] = status_counts.get(status, 0) + 1
        total_scopes += 1
        if status != "ok":
            continue
        path_count = int(payload.get("path_count", 0))
        records = payload.get("records", [])
        if path_count < 4 or not records:
            continue
        eligible_scopes += 1
        for key in DISTANCE_KEYS:
            matrix = upper_triangle_records_to_distance_matrix(records, path_count=path_count, distance_key=key)
            deltas[key].append(matrix_max_four_point_delta(matrix))
    return {
        "input": str(input_path),
        "total_scopes": total_scopes,
        "status_counts": status_counts,
        "eligible_scopes": eligible_scopes,
        "delta_summary": {key: _summarize(values) for key, values in deltas.items()},
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit four-point hyperbolicity of raw-AST path-space metrics.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("outputs/raw_ast_path_space_hyperbolicity_audit.json"))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = audit_path_space_hyperbolicity(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
