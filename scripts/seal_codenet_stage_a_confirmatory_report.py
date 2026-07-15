from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from geometry_profile_research.codenet_eligibility import (  # noqa: E402
    canonical_json_bytes,
    stable_sha256,
)
from geometry_profile_research.codenet_stage_a_inference import (  # noqa: E402
    analyze_confirmatory_test,
)
from geometry_profile_research.codenet_stage_a_test_runner import (  # noqa: E402
    aggregate_all_test_cells,
)
from scripts.seal_codenet_stage_a_test_seed import (  # noqa: E402
    TEST_RUNNER_COMMIT,
    TEST_RUNNER_TAG,
)


def seal_confirmatory_report(
    *,
    report_path: Path,
    test_execution_protocol_path: Path,
    model_protocol_path: Path,
    inference_protocol_path: Path,
    registration_path: Path,
    validation_selection_path: Path,
    validation_selection_seal_path: Path,
    test_materialization_manifest_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Recompute the complete confirmatory report from sealed test seeds."""

    report_bytes = report_path.read_bytes()
    report = json.loads(report_bytes)
    execution_bytes = test_execution_protocol_path.read_bytes()
    model_bytes = model_protocol_path.read_bytes()
    model_protocol = json.loads(model_bytes)
    inference_bytes = inference_protocol_path.read_bytes()
    inference_protocol = json.loads(inference_bytes)
    registration_bytes = registration_path.read_bytes()
    registration = json.loads(registration_bytes)
    selection_bytes = validation_selection_path.read_bytes()
    selection = json.loads(selection_bytes)
    selection_seal_bytes = validation_selection_seal_path.read_bytes()
    materialization_bytes = test_materialization_manifest_path.read_bytes()
    materialization = json.loads(materialization_bytes)
    if report.get("schema_version") != "code2hyp-stage-a-confirmatory-test-v1":
        raise ValueError("unexpected confirmatory-report schema")
    if report.get("status") != "complete" or report.get("opening_count") != 1:
        raise ValueError("confirmatory report is not a complete single-opening result")
    implementation = report.get("implementation", {})
    if implementation.get("commit") != TEST_RUNNER_COMMIT or implementation.get("tag") != TEST_RUNNER_TAG:
        raise ValueError("confirmatory report used an unexpected test runner")
    if materialization.get("implementation") != implementation:
        raise ValueError("confirmatory report and test materialization used different implementations")

    registered_seeds = tuple(int(seed) for seed in model_protocol["encoder_training"]["model_seeds"])
    seed_payloads = []
    seed_inputs = []
    for seed in registered_seeds:
        result_path = report_path.parent / f"seed_{seed}_test.json"
        seal_path = report_path.parent / f"seed_{seed}_test_seal.json"
        result_bytes = result_path.read_bytes()
        seal_bytes = seal_path.read_bytes()
        result = json.loads(result_bytes)
        seal = json.loads(seal_bytes)
        if result.get("status") != "complete" or int(result.get("seed", -1)) != seed:
            raise ValueError(f"test seed {seed} is missing or incomplete")
        if seal.get("schema_version") != "code2hyp-stage-a-test-seed-seal-v1":
            raise ValueError(f"test seed {seed} has an unexpected seal")
        if seal.get("inputs", {}).get("result", {}).get("sha256") != stable_sha256(result_bytes):
            raise ValueError(f"test seed {seed} differs from its seal")
        if any(value is not True for value in seal.get("checks", {}).values()):
            raise ValueError(f"test seed {seed} seal contains a failed check")
        seed_payloads.append(result)
        seed_inputs.append(
            {
                "seed": seed,
                "result_path": result_path.name,
                "result_sha256": stable_sha256(result_bytes),
                "seal_path": seal_path.name,
                "seal_sha256": stable_sha256(seal_bytes),
            }
        )

    bootstrap = inference_protocol["bootstrap"]
    decisions = inference_protocol["decision_rules"]
    recomputed_inference = analyze_confirmatory_test(
        seed_payloads,
        selected_active_cell_id=str(selection["selected_cell_id"]),
        expected_seeds=registered_seeds,
        beacon_output_hex=str(registration["nist_randomness_beacon"]["output_value_hex"]),
        bootstrap_domain=str(bootstrap["rng_seed_derivation"]["domain"]),
        bootstrap_resamples=int(bootstrap["resamples"]),
        practical_delta=float(decisions["minimum_practically_significant_delta_MAP_at_8"]),
        lower_quantile=float(bootstrap["two_sided_interval"]["lower_quantile"]),
        upper_quantile=float(bootstrap["two_sided_interval"]["upper_quantile"]),
    )
    expected_report = {
        "schema_version": "code2hyp-stage-a-confirmatory-test-v1",
        "status": "complete",
        "implementation": implementation,
        "inputs": {
            "test_execution_protocol_sha256": stable_sha256(execution_bytes),
            "model_analysis_protocol_sha256": stable_sha256(model_bytes),
            "test_inference_protocol_sha256": stable_sha256(inference_bytes),
            "registration_sha256": stable_sha256(registration_bytes),
            "validation_selection_sha256": stable_sha256(selection_bytes),
            "validation_selection_seal_sha256": stable_sha256(selection_seal_bytes),
            "test_materialization_manifest_sha256": stable_sha256(materialization_bytes),
            "test_seed_results": [
                {
                    "seed": row["seed"],
                    "path": row["result_path"],
                    "sha256": row["result_sha256"],
                }
                for row in seed_inputs
            ],
        },
        "opening_count": 1,
        "selected_active_curvature": float(selection["selected_active_curvature"]),
        "selected_cell_id": str(selection["selected_cell_id"]),
        "all_planned_cells": aggregate_all_test_cells(
            seed_payloads,
            expected_seeds=registered_seeds,
        ),
        "confirmatory_inference": recomputed_inference,
        "test_program_ids_materialized": True,
        "test_relevance_labels_opened": True,
        "test_retrieval_metrics_computed": True,
    }
    if canonical_json_bytes(report) != canonical_json_bytes(expected_report):
        raise ValueError("confirmatory report is not the frozen-rule recomputation")

    manifest = {
        "schema_version": "code2hyp-stage-a-confirmatory-test-seal-v1",
        "experiment_role": "independently_recomputed_confirmatory_test_report",
        "implementation": {
            "repository": "https://github.com/Malomalsky/code2hyp-artifacts",
            "test_runner_commit": TEST_RUNNER_COMMIT,
            "test_runner_tag": TEST_RUNNER_TAG,
        },
        "inputs": {
            "report": {
                "path": report_path.name,
                "bytes": len(report_bytes),
                "sha256": stable_sha256(report_bytes),
            },
            "test_seed_results_and_seals": seed_inputs,
        },
        "selected_active_curvature": float(selection["selected_active_curvature"]),
        "selected_cell_id": str(selection["selected_cell_id"]),
        "decisions": recomputed_inference["decisions"],
        "checks": {
            "all_registered_test_seeds_present": True,
            "all_test_seed_results_match_their_seals": True,
            "all_planned_cells_reaggregated": True,
            "cluster_bootstrap_recomputed": True,
            "single_test_opening": True,
        },
        "test_program_ids_materialized": True,
        "test_relevance_labels_opened": True,
        "test_retrieval_metrics_computed": True,
    }
    content = canonical_json_bytes(manifest)
    if output_path.exists() and output_path.read_bytes() != content:
        raise ValueError(f"refusing to overwrite a different confirmatory-report seal: {output_path}")
    output_path.write_bytes(content)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Verify and seal the complete Stage A confirmatory test.")
    parser.add_argument("report", type=Path)
    parser.add_argument(
        "--test-execution-protocol",
        type=Path,
        default=PROJECT_ROOT / "configs/codenet_python800_stage_a_test_execution_protocol_v1.json",
    )
    parser.add_argument(
        "--model-protocol",
        type=Path,
        default=PROJECT_ROOT / "configs/codenet_python800_stage_a_model_analysis_protocol_v1.json",
    )
    parser.add_argument(
        "--inference-protocol",
        type=Path,
        default=PROJECT_ROOT / "configs/codenet_python800_stage_a_test_inference_protocol_v1.json",
    )
    parser.add_argument(
        "--registration",
        type=Path,
        default=PROJECT_ROOT / "registrations/codenet_python800_stage_a_registration_v1.json",
    )
    parser.add_argument("--validation-selection", type=Path, required=True)
    parser.add_argument("--validation-selection-seal", type=Path, required=True)
    parser.add_argument("--test-materialization-manifest", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output = args.output or args.report.with_name(args.report.stem + "_seal.json")
    materialization = args.test_materialization_manifest or (
        args.report.parent / "test_materialization_manifest.json"
    )
    manifest = seal_confirmatory_report(
        report_path=args.report,
        test_execution_protocol_path=args.test_execution_protocol,
        model_protocol_path=args.model_protocol,
        inference_protocol_path=args.inference_protocol,
        registration_path=args.registration,
        validation_selection_path=args.validation_selection,
        validation_selection_seal_path=args.validation_selection_seal,
        test_materialization_manifest_path=materialization,
        output_path=output,
    )
    print(json.dumps(manifest["checks"], indent=2, sort_keys=True))
    print(json.dumps(manifest["decisions"], indent=2, sort_keys=True))
    print(f"seal={output}")


if __name__ == "__main__":
    main()
