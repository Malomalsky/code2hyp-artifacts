from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEST_RUNNER_COMMIT = "38b9334ce3777c48fc1aa45d0118a8c54f11bbe7"
TEST_RUNNER_TAG = "codenet-stage-a-test-runner-v3"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from geometry_profile_research.codenet_eligibility import (  # noqa: E402
    canonical_json_bytes,
    stable_sha256,
)
from geometry_profile_research.codenet_stage_a_evaluation import (  # noqa: E402
    summarize_problem_macro_retrieval,
)


EXPECTED_CELLS = (
    "EEE_true_LCA",
    "EEE_zero_anchor",
    "HEE_near_zero_true_LCA",
    "HEE_c0p1_true_LCA",
    "HEE_c0p3_true_LCA",
    "HEE_c1_true_LCA",
    "HEE_c3_true_LCA",
)


def seal_test_seed_result(
    *,
    result_path: Path,
    test_execution_protocol_path: Path,
    test_runtime_addendum_path: Path,
    test_resumability_addendum_path: Path,
    test_materialization_manifest_path: Path,
    test_programs_path: Path,
    validation_selection_seal_path: Path,
    output_path: Path,
    expected_query_count: int = 3_088,
    expected_gallery_count: int = 3_088,
    expected_problem_count: int = 386,
    relevant_count: int = 8,
) -> dict[str, Any]:
    """Recompute one complete test seed from its stored distance matrices."""

    result_bytes = result_path.read_bytes()
    result = json.loads(result_bytes)
    protocol_bytes = test_execution_protocol_path.read_bytes()
    runtime_addendum_bytes = test_runtime_addendum_path.read_bytes()
    resumability_addendum_bytes = test_resumability_addendum_path.read_bytes()
    materialization_bytes = test_materialization_manifest_path.read_bytes()
    materialization = json.loads(materialization_bytes)
    selection_seal_bytes = validation_selection_seal_path.read_bytes()
    selection_seal = json.loads(selection_seal_bytes)
    if result.get("schema_version") != "code2hyp-stage-a-test-seed-v1":
        raise ValueError("unexpected Stage A test-seed schema")
    if result.get("status") != "complete":
        raise ValueError("test seed result is not complete")
    identity = result.get("identity", {})
    implementation = identity.get("implementation", {})
    if implementation.get("commit") != TEST_RUNNER_COMMIT or implementation.get("tag") != TEST_RUNNER_TAG:
        raise ValueError("test seed used an unexpected implementation")
    if identity.get("test_execution_protocol_sha256") != stable_sha256(protocol_bytes):
        raise ValueError("test seed differs from the frozen execution protocol")
    if identity.get("test_runtime_addendum_sha256") != stable_sha256(runtime_addendum_bytes):
        raise ValueError("test seed differs from the frozen runtime addendum")
    if identity.get("test_resumability_addendum_sha256") != stable_sha256(
        resumability_addendum_bytes
    ):
        raise ValueError("test seed differs from the frozen resumability addendum")
    runtime = identity.get("test_runtime", {})
    if runtime.get("torch_num_threads") != 1 or runtime.get("deterministic_algorithms") is not True:
        raise ValueError("test seed did not use the frozen deterministic runtime")
    if identity.get("test_materialization_manifest_sha256") != stable_sha256(materialization_bytes):
        raise ValueError("test seed differs from its materialized test split")
    if materialization.get("implementation") != implementation:
        raise ValueError("test seed and materialization used different implementations")
    if selection_seal.get("schema_version") != "code2hyp-stage-a-validation-selection-seal-v1":
        raise ValueError("unexpected validation-selection seal schema")
    seed = int(result["seed"])
    try:
        validation_input = next(
            row for row in selection_seal["inputs"]["seeds"] if int(row["seed"]) == seed
        )
    except StopIteration as error:
        raise ValueError("test seed is absent from the validation-selection seal") from error
    if identity.get("validation_result_sha256") != str(validation_input["result_sha256"]):
        raise ValueError("test seed references an unexpected validation result")
    if identity.get("validation_seed_seal_sha256") != str(validation_input["seal_sha256"]):
        raise ValueError("test seed references an unexpected validation seed seal")
    required_flags = (
        "test_program_ids_materialized",
        "test_relevance_labels_opened",
        "test_retrieval_metrics_computed",
    )
    if any(result.get(flag) is not True for flag in required_flags):
        raise ValueError("test seed does not record complete test access and metrics")
    if set(result.get("cells", {})) != set(EXPECTED_CELLS):
        raise ValueError("test seed does not contain the exact seven-cell design")

    rows = _iter_jsonl(test_programs_path)
    if any(str(row.get("split")) != "test" for row in rows):
        raise ValueError("test program artifact contains a non-test row")
    if any(any("user" in str(key).casefold() for key in row) for row in rows):
        raise ValueError("test program artifact publishes user identifiers")
    query_rows = [row for row in rows if str(row.get("role")) == "query"]
    gallery_rows = [row for row in rows if str(row.get("role")) == "gallery"]
    if len(query_rows) != expected_query_count or len(gallery_rows) != expected_gallery_count:
        raise ValueError("test program cardinalities differ from the registered design")
    if len({str(row["cluster_id"]) for row in rows}) != expected_problem_count:
        raise ValueError("test problem-cluster count differs from the registered design")
    query_ids = tuple(str(row["source_relpath"]) for row in query_rows)
    query_clusters = tuple(str(row["cluster_id"]) for row in query_rows)
    gallery_ids = tuple(str(row["source_relpath"]) for row in gallery_rows)
    gallery_clusters = tuple(str(row["cluster_id"]) for row in gallery_rows)
    if len(set(query_ids)) != len(query_ids) or len(set(gallery_ids)) != len(gallery_ids):
        raise ValueError("test query or gallery identifiers are not unique")

    cell_checks = []
    for cell_id in EXPECTED_CELLS:
        cell = result["cells"][cell_id]
        distance_meta = cell["distance_matrix"]
        distance_path = result_path.parent / str(distance_meta["path"])
        distance_bytes = distance_path.read_bytes()
        if stable_sha256(distance_bytes) != str(distance_meta["sha256"]):
            raise ValueError(f"test distance matrix hash mismatch: {cell_id}")
        distances = torch.load(distance_path, map_location="cpu", weights_only=True)
        if not isinstance(distances, torch.Tensor):
            raise ValueError(f"test distance artifact is not a tensor: {cell_id}")
        if distances.dtype != torch.float64:
            raise ValueError(f"test distance matrix is not float64: {cell_id}")
        if tuple(distances.shape) != (expected_query_count, expected_gallery_count):
            raise ValueError(f"test distance matrix shape mismatch: {cell_id}")
        if list(distances.shape) != list(distance_meta["shape"]) or distance_meta.get("dtype") != "float64":
            raise ValueError(f"test distance metadata mismatch: {cell_id}")
        if not torch.isfinite(distances).all():
            raise ValueError(f"test distance matrix contains a non-finite value: {cell_id}")
        observed = {
            "minimum": float(distances.min()),
            "maximum": float(distances.max()),
            "negative_count": int((distances < 0.0).sum()),
        }
        if any(not math.isclose(observed[name], float(distance_meta[name]), rel_tol=0.0, abs_tol=0.0) for name in ("minimum", "maximum")):
            raise ValueError(f"test distance extrema metadata mismatch: {cell_id}")
        if observed["negative_count"] != int(distance_meta["negative_count"]):
            raise ValueError(f"test negative-distance count mismatch: {cell_id}")
        recomputed = summarize_problem_macro_retrieval(
            distances,
            query_ids=query_ids,
            query_cluster_ids=query_clusters,
            gallery_ids=gallery_ids,
            gallery_cluster_ids=gallery_clusters,
            r=relevant_count,
        )
        if canonical_json_bytes(asdict(recomputed)) != canonical_json_bytes(cell["metrics"]):
            raise ValueError(f"test retrieval metrics do not match stored distances: {cell_id}")
        cell_checks.append(
            {
                "cell_id": cell_id,
                "distance_path": distance_path.name,
                "distance_sha256": stable_sha256(distance_bytes),
                "metrics_recomputed": True,
            }
        )

    manifest = {
        "schema_version": "code2hyp-stage-a-test-seed-seal-v1",
        "experiment_role": "independently_recomputed_confirmatory_test_seed",
        "seed": seed,
        "implementation": {
            "repository": "https://github.com/Malomalsky/code2hyp-artifacts",
            "test_runner_commit": TEST_RUNNER_COMMIT,
            "test_runner_tag": TEST_RUNNER_TAG,
        },
        "inputs": {
            "result": {
                "path": result_path.name,
                "bytes": len(result_bytes),
                "sha256": stable_sha256(result_bytes),
            },
            "test_execution_protocol_sha256": stable_sha256(protocol_bytes),
            "test_runtime_addendum_sha256": stable_sha256(runtime_addendum_bytes),
            "test_resumability_addendum_sha256": stable_sha256(
                resumability_addendum_bytes
            ),
            "test_materialization_manifest_sha256": stable_sha256(materialization_bytes),
            "test_programs_sha256": stable_sha256(test_programs_path.read_bytes()),
            "validation_selection_seal_sha256": stable_sha256(selection_seal_bytes),
        },
        "cardinalities": {
            "queries": len(query_rows),
            "gallery": len(gallery_rows),
            "problem_clusters": expected_problem_count,
            "relevant_gallery_items_per_query": relevant_count,
        },
        "cells": cell_checks,
        "checks": {
            "all_seven_cells_present": True,
            "all_distance_hashes_match": True,
            "all_distance_matrices_are_finite_float64": True,
            "all_metrics_recomputed_from_distances": True,
            "registered_test_cardinalities_match": True,
        },
        "test_program_ids_materialized": True,
        "test_relevance_labels_opened": True,
        "test_retrieval_metrics_computed": True,
    }
    content = canonical_json_bytes(manifest)
    if output_path.exists() and output_path.read_bytes() != content:
        raise ValueError(f"refusing to overwrite a different test-seed seal: {output_path}")
    output_path.write_bytes(content)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Verify and seal one complete Stage A test seed.")
    parser.add_argument("result", type=Path)
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
    parser.add_argument("--test-materialization-manifest", type=Path, default=None)
    parser.add_argument("--test-programs", type=Path, default=None)
    parser.add_argument("--validation-selection-seal", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result_dir = args.result.parent
    materialization = args.test_materialization_manifest or result_dir / "test_materialization_manifest.json"
    programs = args.test_programs or result_dir / "test_programs.jsonl"
    selection_seal = args.validation_selection_seal or (
        PROJECT_ROOT / "outputs/codenet_python800_stage_a_validation_v1/validation_selection_record_seal.json"
    )
    output = args.output or args.result.with_name(args.result.stem + "_seal.json")
    manifest = seal_test_seed_result(
        result_path=args.result,
        test_execution_protocol_path=args.test_execution_protocol,
        test_runtime_addendum_path=args.test_runtime_addendum,
        test_resumability_addendum_path=args.test_resumability_addendum,
        test_materialization_manifest_path=materialization,
        test_programs_path=programs,
        validation_selection_seal_path=selection_seal,
        output_path=output,
    )
    print(json.dumps(manifest["checks"], indent=2, sort_keys=True))
    print(f"seal={output}")


def _iter_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSONL at {path}:{line_number}") from error
    return rows


if __name__ == "__main__":
    main()
