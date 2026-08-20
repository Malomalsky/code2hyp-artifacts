from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from geometry_profile_research.codenet_eligibility import canonical_json_bytes  # noqa: E402
from geometry_profile_research.codenet_stage_a_inference import (  # noqa: E402
    _bootstrap_contrast,
    _paired_differences,
    derive_cluster_bootstrap_seed,
    seed_averaged_problem_scores,
)


H1_CELLS = ("EEE_true_LCA", "EEE_zero_anchor")
SECONDARY_METRICS = ("mrr", "recall_at_1", "recall_at_5", "recall_at_10", "mean_first_relevant_rank")


def summarize_interim_h1(
    *,
    output_dir: Path,
    model_protocol_path: Path,
    inference_protocol_path: Path,
    registration_path: Path,
) -> dict[str, Any]:
    """Recompute H1 from completed cells without treating the test as complete."""

    model_protocol = json.loads(model_protocol_path.read_text(encoding="utf-8"))
    inference_protocol = json.loads(inference_protocol_path.read_text(encoding="utf-8"))
    registration = json.loads(registration_path.read_text(encoding="utf-8"))
    expected_seeds = tuple(int(seed) for seed in model_protocol["encoder_training"]["model_seeds"])

    payloads = []
    matrix_inputs = []
    for seed in expected_seeds:
        complete_path = output_dir / f"seed_{seed}_test.json"
        partial_path = output_dir / f"seed_{seed}_test.partial.json"
        result_path = complete_path if complete_path.exists() else partial_path
        if not result_path.exists():
            raise ValueError(f"missing test payload for registered seed {seed}")
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        if int(payload.get("seed", -1)) != seed or payload.get("status") not in {"partial", "complete"}:
            raise ValueError(f"invalid test payload for registered seed {seed}")
        for cell_id in H1_CELLS:
            try:
                matrix = payload["cells"][cell_id]["distance_matrix"]
                task_scores = payload["cells"][cell_id]["metrics"]["task_scores"]
            except KeyError as error:
                raise ValueError(f"seed {seed} is missing completed H1 cell {cell_id}") from error
            if matrix.get("shape") != [3088, 3088] or len(task_scores) != 386:
                raise ValueError(f"seed {seed} has an incomplete H1 cell {cell_id}")
            matrix_path = output_dir / str(matrix["path"])
            digest = _sha256_file(matrix_path)
            if digest != matrix.get("sha256"):
                raise ValueError(f"matrix hash mismatch for seed {seed}, cell {cell_id}")
            matrix_inputs.append(
                {
                    "seed": seed,
                    "cell_id": cell_id,
                    "path": matrix_path.name,
                    "bytes": matrix_path.stat().st_size,
                    "sha256": digest,
                }
            )
        reusable = dict(payload)
        reusable["status"] = "complete"
        payloads.append(reusable)

    true_scores = seed_averaged_problem_scores(
        payloads,
        cell_id="EEE_true_LCA",
        expected_seeds=expected_seeds,
    )
    zero_scores = seed_averaged_problem_scores(
        payloads,
        cell_id="EEE_zero_anchor",
        expected_seeds=expected_seeds,
    )
    differences = _paired_differences(true_scores, zero_scores)
    bootstrap = inference_protocol["bootstrap"]
    bootstrap_seed = derive_cluster_bootstrap_seed(
        registration["nist_randomness_beacon"]["output_value_hex"],
        bootstrap["rng_seed_derivation"]["domain"],
    )
    problems = tuple(differences)
    generator = torch.Generator(device="cpu").manual_seed(bootstrap_seed)
    indices = torch.randint(
        len(problems),
        (int(bootstrap["resamples"]), len(problems)),
        generator=generator,
        dtype=torch.int64,
        device="cpu",
    )
    practical_delta = float(
        inference_protocol["decision_rules"]["minimum_practically_significant_delta_MAP_at_8"]
    )
    contrast = _bootstrap_contrast(
        differences,
        problems=problems,
        indices=indices,
        practical_delta=practical_delta,
        lower_quantile=float(bootstrap["two_sided_interval"]["lower_quantile"]),
        upper_quantile=float(bootstrap["two_sided_interval"]["upper_quantile"]),
    )
    primary_means = {
        cell_id: sum(
            float(payload["cells"][cell_id]["metrics"]["problem_macro_map_at_r"])
            for payload in payloads
        )
        / len(payloads)
        for cell_id in H1_CELLS
    }
    secondary_means = {
        cell_id: {
            metric: sum(float(payload["cells"][cell_id]["metrics"][metric]) for payload in payloads)
            / len(payloads)
            for metric in SECONDARY_METRICS
        }
        for cell_id in H1_CELLS
    }
    return {
        "schema_version": "code2hyp-stage-a-interim-h1-v1",
        "status": "provisional_h1_only_full_confirmatory_report_not_yet_sealable",
        "claim_scope": "registered_H1_only",
        "registered_seed_count": len(expected_seeds),
        "problem_cluster_count": len(problems),
        "primary_cell_means": primary_means,
        "secondary_cell_means_descriptive_only": secondary_means,
        "H1_EEE_true_LCA_minus_EEE_zero_anchor": contrast,
        "decisions": {
            "H1_statistical_support": bool(contrast["statistical_support"]),
            "H1_practical_support": bool(contrast["practical_support"]),
            "H1_confirmatory_criterion_satisfied_numerically": bool(
                contrast["statistical_support"] and contrast["practical_support"]
            ),
            "formal_confirmatory_report_complete": False,
        },
        "bootstrap": {
            "resamples": int(bootstrap["resamples"]),
            "rng_seed": bootstrap_seed,
            "shared_resample_index_sha256": hashlib.sha256(indices.numpy().tobytes(order="C")).hexdigest(),
        },
        "integrity": {
            "matrix_count": len(matrix_inputs),
            "matrix_bytes": sum(item["bytes"] for item in matrix_inputs),
            "all_matrix_hashes_verified": True,
            "matrices": matrix_inputs,
        },
        "interpretation": {
            "remaining_cells_can_no_longer_change_H1": True,
            "H3_not_evaluated": True,
            "all_seven_cells_must_complete_before_the_registered_report_can_be_sealed": True,
        },
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Recompute the completed Stage A H1 cells without unsealing H3.")
    parser.add_argument("output_dir", type=Path)
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
    parser.add_argument("--output", type=Path, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = summarize_interim_h1(
        output_dir=args.output_dir,
        model_protocol_path=args.model_protocol,
        inference_protocol_path=args.inference_protocol,
        registration_path=args.registration,
    )
    output_path = args.output or args.output_dir / "interim_h1_report.json"
    content = canonical_json_bytes(result)
    if output_path.exists() and output_path.read_bytes() != content:
        raise ValueError(f"refusing to overwrite a different interim H1 report: {output_path}")
    output_path.write_bytes(content)
    print(json.dumps(result["decisions"], indent=2, sort_keys=True))
    print(json.dumps(result["H1_EEE_true_LCA_minus_EEE_zero_anchor"], indent=2, sort_keys=True))
    print(f"report={output_path}")


if __name__ == "__main__":
    main()
