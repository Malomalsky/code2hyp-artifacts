from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from geometry_profile_research.codenet_eligibility import canonical_json_bytes
from geometry_profile_research.codenet_stage_a_inference import (
    _paired_differences,
    seed_averaged_problem_scores,
)


def simulate_location_shift_power(
    residuals: Iterable[float],
    *,
    cluster_count: int,
    simulations: int,
    rng_seed: int,
    effects: tuple[float, ...],
    variance_scales: tuple[float, ...],
) -> list[dict[str, Any]]:
    values = np.asarray(tuple(residuals), dtype=np.float64)
    values -= values.mean()
    if values.size < 2 or cluster_count < 2 or simulations < 100:
        raise ValueError("power simulation requires at least two residuals, two clusters, and 100 simulations")
    rng = np.random.default_rng(rng_seed)
    sample_means = np.empty(simulations, dtype=np.float64)
    for start in range(0, simulations, 4096):
        stop = min(start + 4096, simulations)
        indices = rng.integers(0, values.size, size=(stop - start, cluster_count))
        sample_means[start:stop] = values[indices].mean(axis=1)

    rows = []
    for scale in variance_scales:
        null_means = sample_means * scale
        critical = float(np.quantile(null_means, 0.975))
        mde_80 = critical - float(np.quantile(null_means, 0.20))
        mde_90 = critical - float(np.quantile(null_means, 0.10))
        for effect in effects:
            power = float(np.mean(null_means + effect > critical))
            rows.append(
                {
                    "variance_scale": scale,
                    "true_location_shift": effect,
                    "marginal_power": power,
                    "two_contrast_joint_power_lower_bound": max(0.0, 2.0 * power - 1.0),
                    "null_97p5_critical_mean": critical,
                    "minimum_detectable_effect_80_percent": mde_80,
                    "minimum_detectable_effect_90_percent": mde_90,
                }
            )
    return rows


def analyze_stage_b_power(
    *,
    stage_a_output_dir: Path,
    model_protocol_path: Path,
    d4_manifest_path: Path,
    output_path: Path,
    simulations: int = 200_000,
    rng_seed: int = 20260820,
) -> dict[str, Any]:
    model_protocol = json.loads(model_protocol_path.read_text(encoding="utf-8"))
    d4 = json.loads(d4_manifest_path.read_text(encoding="utf-8"))
    expected_seeds = tuple(int(seed) for seed in model_protocol["encoder_training"]["model_seeds"])
    payloads = []
    source_records = []
    for seed in expected_seeds:
        complete = stage_a_output_dir / f"seed_{seed}_test.json"
        partial = stage_a_output_dir / f"seed_{seed}_test.partial.json"
        path = complete if complete.exists() else partial
        payload = json.loads(path.read_text(encoding="utf-8"))
        for cell in ("EEE_true_LCA", "EEE_zero_anchor"):
            if cell not in payload.get("cells", {}):
                raise ValueError(f"seed {seed} is missing completed reference cell {cell}")
        reusable = dict(payload)
        reusable["status"] = "complete"
        payloads.append(reusable)
        source_records.append({"path": path.name, "sha256": _sha256(path)})

    true_scores = seed_averaged_problem_scores(
        payloads, cell_id="EEE_true_LCA", expected_seeds=expected_seeds
    )
    zero_scores = seed_averaged_problem_scores(
        payloads, cell_id="EEE_zero_anchor", expected_seeds=expected_seeds
    )
    differences = np.asarray(
        tuple(_paired_differences(true_scores, zero_scores).values()), dtype=np.float64
    )
    test_clusters = int(d4["role_specific_upper_bound"]["test_clusters"])
    rows = simulate_location_shift_power(
        differences,
        cluster_count=test_clusters,
        simulations=simulations,
        rng_seed=rng_seed,
        effects=(0.0025, 0.005, 0.0075, 0.01),
        variance_scales=(1.0, 1.5, 2.0, 3.0),
    )
    payload = {
        "schema_version": "codenet-java-stage-b-power-precheck-v1",
        "status": "pre_split_location_shift_planning_not_retrieval_evidence",
        "inputs": {
            "d4_manifest": {"path": str(d4_manifest_path), "sha256": _sha256(d4_manifest_path)},
            "stage_a_reference_payloads": source_records,
        },
        "protocol": {
            "reference_distribution": "seed-averaged Stage A per-cluster EEE true-LCA minus zero-anchor differences",
            "reference_centering": "subtract empirical mean before simulation",
            "sampling": "independent empirical resampling of centered cluster residuals",
            "decision_surrogate": "one-sided 0.025 location-shift test on the cluster mean",
            "joint_bound": "Bonferroni lower bound max(0, 2*marginal_power-1) for H_B2's two contrasts",
            "simulations": simulations,
            "rng_seed": rng_seed,
            "retrieval_metrics_from_java_opened": False,
        },
        "reference_diagnostics": {
            "stage_a_cluster_count": int(differences.size),
            "observed_mean": float(differences.mean()),
            "observed_sample_standard_deviation": float(differences.std(ddof=1)),
        },
        "planned_test_clusters": test_clusters,
        "results": rows,
        "interpretation": {
            "planning_assumption": "Stage B centered cluster residuals have the Stage A shape up to the stated variance multiplier",
            "not_a_guarantee": True,
            "power_gate_scenario": "variance_scale=2 and true_location_shift=0.01",
            "power_gate_threshold": 0.80,
        },
    }
    gate = next(
        row for row in rows if row["variance_scale"] == 2.0 and row["true_location_shift"] == 0.01
    )
    payload["interpretation"]["power_gate_marginal_passed"] = gate["marginal_power"] >= 0.80
    payload["interpretation"]["power_gate_joint_lower_bound_passed"] = (
        gate["two_contrast_joint_power_lower_bound"] >= 0.80
    )
    content = canonical_json_bytes(payload)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and output_path.read_bytes() != content:
        raise ValueError(f"refusing to overwrite a different power report: {output_path}")
    output_path.write_bytes(content)
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the pre-split Java Stage B power precheck.")
    parser.add_argument("--stage-a-output-dir", required=True, type=Path)
    parser.add_argument(
        "--model-protocol",
        type=Path,
        default=PROJECT_ROOT / "configs/codenet_python800_stage_a_model_analysis_protocol_v1.json",
    )
    parser.add_argument("--d4-manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--simulations", type=int, default=200_000)
    parser.add_argument("--rng-seed", type=int, default=20260820)
    args = parser.parse_args()
    result = analyze_stage_b_power(
        stage_a_output_dir=args.stage_a_output_dir,
        model_protocol_path=args.model_protocol,
        d4_manifest_path=args.d4_manifest,
        output_path=args.output,
        simulations=args.simulations,
        rng_seed=args.rng_seed,
    )
    print(json.dumps(result["reference_diagnostics"], indent=2, sort_keys=True))
    print(json.dumps(result["interpretation"], indent=2, sort_keys=True))
    print(f"report={args.output}")


if __name__ == "__main__":
    main()
