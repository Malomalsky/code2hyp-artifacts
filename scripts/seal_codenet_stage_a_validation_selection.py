from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from geometry_profile_research.codenet_eligibility import canonical_json_bytes, stable_sha256
from geometry_profile_research.codenet_stage_a_runner import build_validation_selection_record
from scripts.seal_codenet_stage_a_validation_seed import RUNNER_COMMIT, RUNNER_TAG


def seal_validation_selection(
    *,
    selection_path: Path,
    protocol_path: Path,
    calibration_manifest_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Verify a validation selection against all registered sealed seed results."""

    selection_bytes = selection_path.read_bytes()
    protocol_bytes = protocol_path.read_bytes()
    calibration_bytes = calibration_manifest_path.read_bytes()
    selection = json.loads(selection_bytes)
    protocol = json.loads(protocol_bytes)
    registered_seeds = tuple(int(seed) for seed in protocol["encoder_training"]["model_seeds"])
    result_dir = selection_path.parent
    seed_payloads = []
    seed_inputs = []
    for seed in registered_seeds:
        result_path = result_dir / f"seed_{seed}_validation.json"
        seal_path = result_dir / f"seed_{seed}_validation_seal.json"
        result_bytes = result_path.read_bytes()
        seal_bytes = seal_path.read_bytes()
        result = json.loads(result_bytes)
        seed_seal = json.loads(seal_bytes)
        result_sha256 = stable_sha256(result_bytes)
        if int(result.get("seed", -1)) != seed or result.get("status") != "complete":
            raise ValueError(f"registered seed {seed} result is missing or incomplete")
        if int(seed_seal.get("seed", -1)) != seed:
            raise ValueError(f"registered seed {seed} seal identifies another seed")
        if seed_seal.get("implementation", {}).get("commit") != RUNNER_COMMIT:
            raise ValueError(f"registered seed {seed} used an unexpected runner commit")
        if seed_seal.get("inputs", {}).get("result", {}).get("sha256") != result_sha256:
            raise ValueError(f"registered seed {seed} result differs from its seal")
        if seed_seal.get("checks", {}).get("validation_only") is not True:
            raise ValueError(f"registered seed {seed} is not sealed as validation-only")
        if any(
            bool(result.get(flag)) or bool(seed_seal.get(flag))
            for flag in (
                "test_program_ids_materialized",
                "test_relevance_labels_opened",
                "test_retrieval_metrics_computed",
            )
        ):
            raise ValueError(f"registered seed {seed} indicates forbidden test access")
        seed_payloads.append(result)
        seed_inputs.append(
            {
                "seed": seed,
                "result_path": result_path.name,
                "result_sha256": result_sha256,
                "seal_path": seal_path.name,
                "seal_sha256": stable_sha256(seal_bytes),
            }
        )

    expected = build_validation_selection_record(
        seed_payloads,
        protocol_bytes=protocol_bytes,
        calibration_manifest_bytes=calibration_bytes,
    )
    if canonical_json_bytes(selection) != canonical_json_bytes(expected):
        raise ValueError("validation selection record is not the frozen-rule recomputation")

    manifest = {
        "schema_version": "code2hyp-stage-a-validation-selection-seal-v1",
        "experiment_role": "verified_validation_only_curvature_selection",
        "implementation": {
            "repository": "https://github.com/Malomalsky/code2hyp-artifacts",
            "runner_commit": RUNNER_COMMIT,
            "runner_tag": RUNNER_TAG,
        },
        "inputs": {
            "selection": {
                "path": selection_path.name,
                "bytes": len(selection_bytes),
                "sha256": stable_sha256(selection_bytes),
            },
            "protocol": {
                "path": protocol_path.name,
                "sha256": stable_sha256(protocol_bytes),
            },
            "calibration_manifest": {
                "path": calibration_manifest_path.name,
                "sha256": stable_sha256(calibration_bytes),
            },
            "seeds": seed_inputs,
        },
        "selected_active_curvature": float(selection["selected_active_curvature"]),
        "selected_cell_id": str(selection["selected_cell_id"]),
        "checks": {
            "registered_seed_set_complete": True,
            "all_seed_results_match_their_seals": True,
            "selection_recomputed_from_frozen_rule": True,
            "validation_only": True,
        },
        "test_program_ids_materialized": False,
        "test_relevance_labels_opened": False,
        "test_retrieval_metrics_computed": False,
    }
    content = canonical_json_bytes(manifest)
    if output_path.exists() and output_path.read_bytes() != content:
        raise ValueError(f"refusing to overwrite a different selection seal: {output_path}")
    output_path.write_bytes(content)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Verify and seal the Stage A validation selection.")
    parser.add_argument("selection", type=Path)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=PROJECT_ROOT / "configs/codenet_python800_stage_a_model_analysis_protocol_v1.json",
    )
    parser.add_argument(
        "--calibration-manifest",
        type=Path,
        default=PROJECT_ROOT / "data/codenet_python800_stage_a_calibration_pairs/calibration_pair_manifest.json",
    )
    parser.add_argument("--output", type=Path, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output = args.output or args.selection.with_name(args.selection.stem + "_seal.json")
    manifest = seal_validation_selection(
        selection_path=args.selection,
        protocol_path=args.protocol,
        calibration_manifest_path=args.calibration_manifest,
        output_path=output,
    )
    print(json.dumps(manifest["checks"], indent=2, sort_keys=True))
    print(f"seal={output}")


if __name__ == "__main__":
    main()
