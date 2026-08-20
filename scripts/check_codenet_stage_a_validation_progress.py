from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def seed_state(output_dir: Path, seed: int) -> dict[str, Any]:
    result_path = output_dir / f"seed_{seed}_validation.json"
    partial_path = output_dir / f"seed_{seed}_validation.partial.json"
    checkpoint_path = output_dir / f"seed_{seed}_encoder.pt"
    seal_path = output_dir / f"seed_{seed}_validation_seal.json"
    if result_path.exists():
        payload = load_json(result_path)
        cells = payload.get("cells", {})
        return {
            "seed": seed,
            "state": "complete" if payload.get("status") == "complete" else str(payload.get("status")),
            "cell_count": len(cells),
            "cells": sorted(cells),
            "seal_exists": seal_path.exists(),
            "test_program_ids_materialized": bool(payload.get("test_program_ids_materialized")),
            "test_relevance_labels_opened": bool(payload.get("test_relevance_labels_opened")),
            "test_retrieval_metrics_computed": bool(payload.get("test_retrieval_metrics_computed")),
        }
    if partial_path.exists():
        payload = load_json(partial_path)
        cells = payload.get("cells", {})
        return {
            "seed": seed,
            "state": "partial",
            "cell_count": len(cells),
            "cells": sorted(cells),
            "seal_exists": False,
            "checkpoint_exists": checkpoint_path.exists(),
            "test_program_ids_materialized": bool(payload.get("test_program_ids_materialized")),
            "test_relevance_labels_opened": bool(payload.get("test_relevance_labels_opened")),
            "test_retrieval_metrics_computed": bool(payload.get("test_retrieval_metrics_computed")),
        }
    if checkpoint_path.exists():
        return {
            "seed": seed,
            "state": "checkpoint",
            "cell_count": 0,
            "cells": [],
            "seal_exists": False,
            "checkpoint_exists": True,
        }
    return {
        "seed": seed,
        "state": "missing",
        "cell_count": 0,
        "cells": [],
        "seal_exists": False,
        "checkpoint_exists": False,
    }


def build_progress_report(
    *,
    output_dir: Path,
    protocol_path: Path,
) -> dict[str, Any]:
    protocol = load_json(protocol_path)
    seeds = [int(seed) for seed in protocol["encoder_training"]["model_seeds"]]
    seed_rows = [seed_state(output_dir, seed) for seed in seeds]
    selection_path = output_dir / "validation_selection_record.json"
    selection = load_json(selection_path) if selection_path.exists() else None
    forbidden_test_flags = [
        (row["seed"], key)
        for row in seed_rows
        for key in (
            "test_program_ids_materialized",
            "test_relevance_labels_opened",
            "test_retrieval_metrics_computed",
        )
        if bool(row.get(key))
    ]
    complete_count = sum(row["state"] == "complete" for row in seed_rows)
    sealed_count = sum(bool(row.get("seal_exists")) for row in seed_rows)
    return {
        "schema_version": "code2hyp-stage-a-validation-progress-v1",
        "output_dir": str(output_dir),
        "registered_seed_count": len(seeds),
        "complete_seed_count": complete_count,
        "sealed_seed_count": sealed_count,
        "states": seed_rows,
        "selection_record_exists": selection is not None,
        "selected_active_curvature": None if selection is None else selection.get("selected_active_curvature"),
        "selected_cell_id": None if selection is None else selection.get("selected_cell_id"),
        "validation_only_boundary_ok": not forbidden_test_flags,
        "forbidden_test_flags": forbidden_test_flags,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize CodeNet Stage A validation progress without reading test data.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs/codenet_python800_stage_a_validation_v1",
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        default=PROJECT_ROOT / "configs/codenet_python800_stage_a_model_analysis_protocol_v1.json",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = build_progress_report(output_dir=args.output_dir, protocol_path=args.protocol)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    if not report["validation_only_boundary_ok"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
