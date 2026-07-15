from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from geometry_profile_research.codenet_eligibility import (  # noqa: E402
    canonical_json_bytes,
    stable_sha256,
)
from geometry_profile_research.codenet_stage_a import load_stage_a_test_split  # noqa: E402
from geometry_profile_research.codenet_stage_a_inference import (  # noqa: E402
    analyze_confirmatory_test,
)
from geometry_profile_research.codenet_stage_a_test import (  # noqa: E402
    materialize_and_audit_test_programs,
)
from geometry_profile_research.codenet_stage_a_test_runner import (  # noqa: E402
    aggregate_all_test_cells,
    run_stage_a_test_seed,
)
from scripts.materialize_codenet_stage_a_test import (  # noqa: E402
    verified_implementation_state,
)
from scripts.seal_codenet_stage_a_validation_selection import (  # noqa: E402
    seal_validation_selection,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Perform the single CodeNet Stage A test opening, all planned cells and frozen inference."
    )
    parser.add_argument(
        "--test-execution-protocol",
        type=Path,
        default=PROJECT_ROOT / "configs/codenet_python800_stage_a_test_execution_protocol_v1.json",
    )
    parser.add_argument(
        "--test-runtime-addendum",
        type=Path,
        default=PROJECT_ROOT / "configs/codenet_python800_stage_a_test_runtime_addendum_v1.json",
    )
    parser.add_argument(
        "--test-resumability-addendum",
        type=Path,
        default=PROJECT_ROOT
        / "configs/codenet_python800_stage_a_test_resumability_addendum_v1.json",
    )
    parser.add_argument(
        "--relevance-addendum",
        type=Path,
        default=PROJECT_ROOT / "configs/codenet_python800_stage_a_relevance_identity_addendum_v1.json",
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
    parser.add_argument(
        "--calibration-manifest",
        type=Path,
        default=PROJECT_ROOT / "data/codenet_python800_stage_a_calibration_pairs/calibration_pair_manifest.json",
    )
    parser.add_argument(
        "--validation-output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs/codenet_python800_stage_a_validation_v1",
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=PROJECT_ROOT / "data/external_raw/codenet_python800_extracted/Project_CodeNet_Python800",
    )
    parser.add_argument(
        "--d5-index",
        type=Path,
        default=PROJECT_ROOT / "data/codenet_python800_d5_metadata/d5_metadata_index.jsonl",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs/codenet_python800_stage_a_test_v1",
    )
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="*",
        default=None,
        help="Optional registered seed subset. Confirmatory inference is written only when all seeds exist.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    implementation = verified_implementation_state(PROJECT_ROOT)
    runtime_addendum_bytes = args.test_runtime_addendum.read_bytes()
    runtime_addendum = json.loads(runtime_addendum_bytes)
    if runtime_addendum.get("schema_version") != "code2hyp-stage-a-test-runtime-addendum-v1":
        raise ValueError("unexpected Stage A test-runtime addendum")
    if runtime_addendum.get("status") != (
        "frozen_during_validation_before_validation_selection_or_test_unseal"
    ):
        raise ValueError("Stage A test-runtime addendum was not frozen before unseal")
    resumability_addendum_bytes = args.test_resumability_addendum.read_bytes()
    resumability_addendum = json.loads(resumability_addendum_bytes)
    if resumability_addendum.get("schema_version") != (
        "code2hyp-stage-a-test-resumability-addendum-v1"
    ):
        raise ValueError("unexpected Stage A test-resumability addendum")
    if resumability_addendum.get("status") != (
        "frozen_during_validation_before_validation_selection_or_test_unseal"
    ):
        raise ValueError("Stage A test-resumability addendum was not frozen before unseal")
    relevance_addendum_bytes = args.relevance_addendum.read_bytes()
    relevance_addendum = json.loads(relevance_addendum_bytes)
    if relevance_addendum.get("schema_version") != "code2hyp-stage-a-relevance-identity-addendum-v1":
        raise ValueError("unexpected Stage A relevance-identity addendum")
    if relevance_addendum.get("correction", {}).get("relevance_key_after") != "cluster_id":
        raise ValueError("Stage A relevance must use duplicate-closed cluster IDs")
    selection_path = args.validation_output_dir / "validation_selection_record.json"
    selection_seal_path = args.validation_output_dir / "validation_selection_record_seal.json"
    seal_validation_selection(
        selection_path=selection_path,
        protocol_path=args.model_protocol,
        calibration_manifest_path=args.calibration_manifest,
        output_path=selection_seal_path,
    )
    materialize_and_audit_test_programs(
        project_root=PROJECT_ROOT,
        protocol_path=args.test_execution_protocol,
        selection_path=selection_path,
        selection_seal_path=selection_seal_path,
        source_root=args.source_root,
        output_dir=args.output_dir,
        implementation=implementation,
        d5_metadata_index_path=args.d5_index,
        workers=args.workers,
    )
    test_split = load_stage_a_test_split(
        source_root=args.source_root,
        test_path=args.output_dir / "test_programs.jsonl",
        ast_index_path=args.output_dir / "test_source_ast_index.jsonl",
    )

    model_protocol_bytes = args.model_protocol.read_bytes()
    model_protocol = json.loads(model_protocol_bytes)
    execution_protocol_bytes = args.test_execution_protocol.read_bytes()
    inference_protocol_bytes = args.inference_protocol.read_bytes()
    inference_protocol = json.loads(inference_protocol_bytes)
    registration_bytes = args.registration.read_bytes()
    registration = json.loads(registration_bytes)
    registered_seeds = tuple(int(seed) for seed in model_protocol["encoder_training"]["model_seeds"])
    seeds = registered_seeds if args.seeds is None else tuple(args.seeds)
    if not seeds or any(seed not in registered_seeds for seed in seeds):
        raise ValueError("all test seeds must belong to the registered model-seed set")
    selection_seal = json.loads(selection_seal_path.read_text(encoding="utf-8"))
    sealed_seed_inputs = {
        int(row["seed"]): row for row in selection_seal["inputs"]["seeds"]
    }
    start_time = time.perf_counter()

    def progress(event: dict) -> None:
        elapsed = time.perf_counter() - start_time
        print(json.dumps({"elapsed_seconds": round(elapsed, 1), **event}, sort_keys=True), flush=True)

    seed_payloads_by_seed = {}
    materialization_path = args.output_dir / "test_materialization_manifest.json"
    for seed in seeds:
        progress({"phase": "test_seed_start", "seed": seed})
        seed_input = sealed_seed_inputs[seed]
        payload = run_stage_a_test_seed(
            test_split=test_split,
            seed=seed,
            validation_result_path=args.validation_output_dir / str(seed_input["result_path"]),
            validation_result_expected_sha256=str(seed_input["result_sha256"]),
            validation_seed_seal_path=args.validation_output_dir / str(seed_input["seal_path"]),
            test_materialization_manifest_path=materialization_path,
            output_dir=args.output_dir,
            test_execution_protocol_sha256=stable_sha256(execution_protocol_bytes),
            test_runtime_addendum_sha256=stable_sha256(runtime_addendum_bytes),
            test_resumability_addendum_sha256=stable_sha256(resumability_addendum_bytes),
            relevance_identity_addendum_sha256=stable_sha256(relevance_addendum_bytes),
            implementation=implementation,
            progress_callback=progress,
        )
        seed_payloads_by_seed[seed] = payload
        progress({"phase": "test_seed_complete", "seed": seed, "cell_count": len(payload["cells"])})

    for seed in registered_seeds:
        if seed in seed_payloads_by_seed:
            continue
        path = args.output_dir / f"seed_{seed}_test.json"
        if path.exists():
            seed_payloads_by_seed[seed] = json.loads(path.read_text(encoding="utf-8"))
    if set(seed_payloads_by_seed) != set(registered_seeds):
        progress(
            {
                "phase": "confirmatory_inference_pending",
                "complete_seed_count": len(seed_payloads_by_seed),
                "required_seed_count": len(registered_seeds),
            }
        )
        return

    ordered_payloads = tuple(seed_payloads_by_seed[seed] for seed in registered_seeds)
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    bootstrap = inference_protocol["bootstrap"]
    decisions = inference_protocol["decision_rules"]
    inference = analyze_confirmatory_test(
        ordered_payloads,
        selected_active_cell_id=str(selection["selected_cell_id"]),
        expected_seeds=registered_seeds,
        beacon_output_hex=str(registration["nist_randomness_beacon"]["output_value_hex"]),
        bootstrap_domain=str(bootstrap["rng_seed_derivation"]["domain"]),
        bootstrap_resamples=int(bootstrap["resamples"]),
        practical_delta=float(decisions["minimum_practically_significant_delta_MAP_at_8"]),
        lower_quantile=float(bootstrap["two_sided_interval"]["lower_quantile"]),
        upper_quantile=float(bootstrap["two_sided_interval"]["upper_quantile"]),
    )
    test_result_inputs = []
    for seed in registered_seeds:
        path = args.output_dir / f"seed_{seed}_test.json"
        test_result_inputs.append(
            {"seed": seed, "path": path.name, "sha256": stable_sha256(path.read_bytes())}
        )
    final = {
        "schema_version": "code2hyp-stage-a-confirmatory-test-v1",
        "status": "complete",
        "implementation": implementation,
        "inputs": {
            "test_execution_protocol_sha256": stable_sha256(execution_protocol_bytes),
            "model_analysis_protocol_sha256": stable_sha256(model_protocol_bytes),
            "test_inference_protocol_sha256": stable_sha256(inference_protocol_bytes),
            "relevance_identity_addendum_sha256": stable_sha256(relevance_addendum_bytes),
            "registration_sha256": stable_sha256(registration_bytes),
            "validation_selection_sha256": stable_sha256(selection_path.read_bytes()),
            "validation_selection_seal_sha256": stable_sha256(selection_seal_path.read_bytes()),
            "test_materialization_manifest_sha256": stable_sha256(materialization_path.read_bytes()),
            "test_seed_results": test_result_inputs,
        },
        "opening_count": 1,
        "selected_active_curvature": float(selection["selected_active_curvature"]),
        "selected_cell_id": str(selection["selected_cell_id"]),
        "all_planned_cells": aggregate_all_test_cells(
            ordered_payloads,
            expected_seeds=registered_seeds,
        ),
        "confirmatory_inference": inference,
        "test_program_ids_materialized": True,
        "test_relevance_labels_opened": True,
        "test_retrieval_metrics_computed": True,
    }
    report_path = args.output_dir / "confirmatory_test_report.json"
    _write_once_or_verify(report_path, canonical_json_bytes(final))
    progress(
        {
            "phase": "confirmatory_inference_complete",
            "H1_confirmatory_success": inference["decisions"]["H1_confirmatory_success"],
            "H3_confirmatory_success": inference["decisions"]["H3_confirmatory_success"],
            "report": str(report_path),
        }
    )


def _write_once_or_verify(path: Path, content: bytes) -> None:
    if path.exists():
        if path.read_bytes() != content:
            raise ValueError(f"refusing to overwrite a different confirmatory test report: {path}")
        return
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(content)
    os.replace(temporary, path)


if __name__ == "__main__":
    main()
