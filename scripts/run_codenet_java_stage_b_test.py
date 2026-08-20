from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNNER_TAG = "codenet-java-stage-b-test-runner-v1"
RESULT_SCHEMA = "code2hyp-codenet-java-stage-b-test-seed-v1"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from geometry_profile_research.codenet_eligibility import canonical_json_bytes, stable_sha256
from geometry_profile_research.codenet_stage_a import load_codenet_test_split
from geometry_profile_research.codenet_stage_a_inference import analyze_stage_b_confirmatory_test
from geometry_profile_research.codenet_stage_a_runner import configure_torch_runtime, torch_device_metadata
from geometry_profile_research.codenet_stage_a_test_runner import aggregate_all_test_cells, run_stage_a_test_seed
from geometry_profile_research.codenet_stage_b import (
    STAGE_B_TEST_MATERIALIZATION_SCHEMA,
    materialize_stage_b_test_programs,
)
from scripts.run_codenet_stage_a_validation import _verified_implementation_state
from scripts.seal_codenet_java_stage_b_test import seal_stage_b_confirmatory_report
from scripts.seal_codenet_java_stage_b_validation import seal_stage_b_validation


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Perform the single registered CodeNet Java Stage B test opening and seven-cell inference."
    )
    parser.add_argument("--design", type=Path, default=PROJECT_ROOT / "configs/codenet_java_stage_b_draft_v1.json")
    parser.add_argument("--registration", type=Path, default=PROJECT_ROOT / "registrations/codenet_java_stage_b_registration_v1.json")
    parser.add_argument("--split-manifest", type=Path, default=PROJECT_ROOT / "data/codenet_java_stage_b_split_v1/split_manifest.json")
    parser.add_argument("--assignments", type=Path, default=PROJECT_ROOT / "data/codenet_java_stage_b_split_v1/cluster_assignments.jsonl")
    parser.add_argument("--d5-manifest", type=Path, default=PROJECT_ROOT / "data/codenet_java_stage_b_metadata_v2/d5_metadata_manifest.json")
    parser.add_argument("--d5-index", type=Path, default=PROJECT_ROOT / "data/codenet_java_stage_b_metadata_v2/d5_metadata_index.jsonl")
    parser.add_argument("--candidate-manifest", type=Path, default=PROJECT_ROOT / "data/codenet_java_stage_b_candidates_v1/manifest.json")
    parser.add_argument("--candidate-archive", type=Path, default=PROJECT_ROOT / "data/codenet_java_stage_b_candidates_v1/accepted_java_sources.tar")
    parser.add_argument("--d0-d2-manifest", type=Path, default=PROJECT_ROOT / "data/codenet_java_stage_b_eligibility_d0_d2_v1/manifest.json")
    parser.add_argument("--d0-d2-inventory", type=Path, default=PROJECT_ROOT / "data/codenet_java_stage_b_eligibility_d0_d2_v1/file_inventory.jsonl")
    parser.add_argument("--source-root", type=Path, default=PROJECT_ROOT / "data/codenet_java_stage_b_test_sources_v1")
    parser.add_argument("--validation-output-dir", type=Path, default=PROJECT_ROOT / "outputs/codenet_java_stage_b_validation_v1")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs/codenet_java_stage_b_test_v1")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--seeds", type=int, nargs="*", default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    design_bytes = args.design.read_bytes()
    design = json.loads(design_bytes)
    design_sha = stable_sha256(design_bytes)
    freeze = design["freeze"]
    implementation = {
        **_verified_implementation_state(PROJECT_ROOT, runner_tag=RUNNER_TAG),
        "container_digest": freeze["container_digest"],
        "design_sha256": design_sha,
    }
    if implementation["commit"] != freeze["implementation_commit"]:
        raise ValueError("Stage B test runner commit differs from the frozen design")
    if implementation["tag"] != freeze["test_runner_tag"]:
        raise ValueError("Stage B test runner tag differs from the frozen design")
    if not str(implementation["container_digest"] or "").startswith("sha256:"):
        raise ValueError("Stage B test container digest is not frozen")
    encoder = design["encoder_training"]
    registered_seeds = tuple(int(seed) for seed in encoder["model_seeds"])
    device = configure_torch_runtime(
        compute_device=str(encoder["compute_device"]),
        torch_num_threads=int(encoder["torch_num_threads_per_process"]),
        seed=registered_seeds[0],
    )
    print(json.dumps({"phase": "runtime_preflight", **torch_device_metadata(device)}, sort_keys=True), flush=True)

    validation_seal = seal_stage_b_validation(
        design_path=args.design,
        calibration_manifest_path=PROJECT_ROOT / "data/codenet_java_stage_b_calibration_pairs_v1/calibration_pair_manifest.json",
        sampling_manifest_path=PROJECT_ROOT / "data/codenet_java_stage_b_program_sampling_v1/program_sampling_manifest.json",
        selected_ast_manifest_path=PROJECT_ROOT / "data/codenet_java_stage_b_selected_source_ast_v1/selected_source_ast_manifest.json",
        validation_programs_path=PROJECT_ROOT / "data/codenet_java_stage_b_program_sampling_v1/validation_programs.jsonl",
        validation_output_dir=args.validation_output_dir,
    )
    selection_path = args.validation_output_dir / "validation_selection_record.json"
    selection_seal_path = args.validation_output_dir / "validation_selection_record_seal.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    materialize_stage_b_test_programs(
        design_path=args.design,
        registration_path=args.registration,
        split_manifest_path=args.split_manifest,
        assignments_path=args.assignments,
        d5_manifest_path=args.d5_manifest,
        d5_index_path=args.d5_index,
        selection_path=selection_path,
        selection_seal_path=selection_seal_path,
        candidate_manifest_path=args.candidate_manifest,
        candidate_archive_path=args.candidate_archive,
        d0_d2_manifest_path=args.d0_d2_manifest,
        d0_d2_inventory_path=args.d0_d2_inventory,
        source_root=args.source_root,
        output_dir=args.output_dir,
        implementation=implementation,
        workers=args.workers,
    )

    quotas = design["eligibility"]["primary_role_upper_bound"]
    sampling = design["sampling"]
    test_split = load_codenet_test_split(
        source_root=args.source_root,
        test_path=args.output_dir / "test_programs.jsonl",
        ast_index_path=args.output_dir / "test_source_ast_index.jsonl",
        language="java",
        test_clusters=int(quotas["test_clusters"]),
        queries_per_cluster=int(sampling["test_queries_per_cluster"]),
        gallery_per_cluster=int(sampling["test_gallery_per_cluster"]),
    )
    seeds = registered_seeds if args.seeds is None else tuple(args.seeds)
    if not seeds or len(seeds) != len(set(seeds)) or any(seed not in registered_seeds for seed in seeds):
        raise ValueError("requested Stage B test seeds must be a unique subset of the frozen sequence")
    seed_inputs = {
        (str(row["model"]), int(row["seed"])): row
        for row in validation_seal["inputs"]["seeds"]
    }
    active_cell = str(selection["selected_prefix_HEE_cell_id"])
    hhh_cell = str(selection["selected_prefix_HHH_cell_id"])
    cells_by_model = {
        "prefix": ("EEE_true_LCA", "EEE_zero_anchor", "HEE_near_zero_true_LCA", active_cell, hhh_cell),
        "label_only": ("EEE_true_LCA", active_cell),
    }
    materialization_path = args.output_dir / "test_materialization_manifest.json"
    start = time.perf_counter()

    def progress(event: dict[str, Any]) -> None:
        print(json.dumps({"elapsed_seconds": round(time.perf_counter() - start, 1), **event}, sort_keys=True), flush=True)

    for seed in seeds:
        for model in ("prefix", "label_only"):
            seed_input = seed_inputs[(model, seed)]
            progress({"phase": "test_seed_start", "model": model, "seed": seed})
            run_stage_a_test_seed(
                test_split=test_split,
                seed=seed,
                validation_result_path=args.validation_output_dir / str(seed_input["result_path"]),
                validation_result_expected_sha256=str(seed_input["result_sha256"]),
                validation_seed_seal_path=args.validation_output_dir / str(seed_input["seal_path"]),
                test_materialization_manifest_path=materialization_path,
                output_dir=args.output_dir / model,
                test_execution_protocol_sha256=design_sha,
                test_runtime_addendum_sha256=design_sha,
                test_resumability_addendum_sha256=design_sha,
                relevance_identity_addendum_sha256=design_sha,
                implementation=implementation,
                expected_cell_ids=cells_by_model[model],
                expected_validation_commit=str(freeze["implementation_commit"]),
                expected_validation_tag=str(freeze["validation_runner_tag"]),
                expected_materialization_schema=STAGE_B_TEST_MATERIALIZATION_SCHEMA,
                result_schema=RESULT_SCHEMA,
                progress_callback=progress,
            )
            progress({"phase": "test_seed_complete", "model": model, "seed": seed})

    payloads = {
        model: _complete_payloads(args.output_dir / model, registered_seeds)
        for model in ("prefix", "label_only")
    }
    if any(len(values) != len(registered_seeds) for values in payloads.values()):
        progress(
            {
                "phase": "confirmatory_inference_pending",
                "prefix_seed_count": len(payloads["prefix"]),
                "label_only_seed_count": len(payloads["label_only"]),
                "required_seed_count": len(registered_seeds),
            }
        )
        return

    registration = json.loads(args.registration.read_text(encoding="utf-8"))
    inference_config = design["inference"]
    inference = analyze_stage_b_confirmatory_test(
        payloads["prefix"],
        payloads["label_only"],
        selected_active_cell_id=active_cell,
        selected_hhh_cell_id=hhh_cell,
        expected_seeds=registered_seeds,
        beacon_output_hex=str(registration["nist_randomness_beacon"]["output_value_hex"]),
        bootstrap_domain=str(inference_config["bootstrap_domain"]),
        bootstrap_resamples=int(inference_config["bootstrap_resamples"]),
        practical_delta=float(inference_config["minimum_practically_significant_delta_MAP_at_8"]),
    )
    aggregates = {
        model: aggregate_all_test_cells(values, expected_seeds=registered_seeds)
        for model, values in payloads.items()
    }
    public_cells = {
        str(row["cell_id"]): aggregates[str(row["model"])][str(row["validation_cell_id"])]
        for row in selection["test_cell_plan"]
    }
    result_inputs = [
        {
            "model": model,
            "seed": seed,
            "path": f"{model}/seed_{seed}_test.json",
            "sha256": stable_sha256((args.output_dir / model / f"seed_{seed}_test.json").read_bytes()),
        }
        for model in ("prefix", "label_only")
        for seed in registered_seeds
    ]
    report = {
        "schema_version": "code2hyp-codenet-java-stage-b-confirmatory-test-v1",
        "status": "complete",
        "implementation": implementation,
        "inputs": {
            "design_sha256": design_sha,
            "registration_sha256": stable_sha256(args.registration.read_bytes()),
            "validation_selection_sha256": stable_sha256(selection_path.read_bytes()),
            "validation_selection_seal_sha256": stable_sha256(selection_seal_path.read_bytes()),
            "test_materialization_manifest_sha256": stable_sha256(materialization_path.read_bytes()),
            "test_seed_results": result_inputs,
        },
        "opening_count": 1,
        "selected_active_curvature": float(selection["selected_active_curvature"]),
        "selected_prefix_HEE_cell_id": active_cell,
        "selected_prefix_HHH_cell_id": hhh_cell,
        "all_seven_planned_cells": public_cells,
        "confirmatory_inference": inference,
        "test_program_ids_materialized": True,
        "test_relevance_labels_opened": True,
        "test_retrieval_metrics_computed": True,
    }
    report_path = args.output_dir / "confirmatory_test_report.json"
    content = canonical_json_bytes(report)
    if report_path.exists() and report_path.read_bytes() != content:
        raise FileExistsError("refusing to overwrite a different Stage B confirmatory report")
    report_path.write_bytes(content)
    seal_stage_b_confirmatory_report(
        design_path=args.design,
        registration_path=args.registration,
        selection_path=selection_path,
        selection_seal_path=selection_seal_path,
        materialization_manifest_path=materialization_path,
        test_programs_path=args.output_dir / "test_programs.jsonl",
        validation_output_dir=args.validation_output_dir,
        test_output_dir=args.output_dir,
        report_path=report_path,
        output_path=args.output_dir / "confirmatory_test_report_seal.json",
    )
    progress({"phase": "confirmatory_inference_complete", "report": str(report_path)})


def _complete_payloads(directory: Path, seeds: tuple[int, ...]) -> tuple[dict[str, Any], ...]:
    payloads = []
    for seed in seeds:
        path = directory / f"seed_{seed}_test.json"
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("status") == "complete":
            payloads.append(payload)
    return tuple(payloads)


if __name__ == "__main__":
    main()
