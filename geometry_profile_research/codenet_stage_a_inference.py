from __future__ import annotations

import hashlib
import math
from typing import Any, Mapping, Sequence

import torch


def derive_cluster_bootstrap_seed(beacon_output_hex: str, domain: str) -> int:
    """Derive the frozen CPU generator seed from the registered beacon output."""

    beacon = bytes.fromhex(beacon_output_hex)
    if len(beacon) != 64:
        raise ValueError("registered beacon output must contain exactly 64 bytes")
    digest = hashlib.sha256(beacon + domain.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big", signed=False) % (2**63 - 1)


def seed_averaged_problem_scores(
    seed_payloads: Sequence[Mapping[str, Any]],
    *,
    cell_id: str,
    expected_seeds: Sequence[int],
) -> dict[str, float]:
    """Average one planned cell over model seeds within each problem cluster."""

    observed_seeds = [int(payload["seed"]) for payload in seed_payloads]
    if len(observed_seeds) != len(set(observed_seeds)):
        raise ValueError("test inference received duplicate model seeds")
    if set(observed_seeds) != {int(seed) for seed in expected_seeds}:
        raise ValueError("test inference does not contain the registered model seed set")
    per_problem: dict[str, list[float]] = {}
    reference_problems: set[str] | None = None
    for payload in seed_payloads:
        if payload.get("status") != "complete":
            raise ValueError("all test seed payloads must be complete")
        try:
            scores = payload["cells"][cell_id]["metrics"]["task_scores"]
        except KeyError as error:
            raise ValueError(f"test seed is missing planned cell {cell_id!r}") from error
        problems = {str(problem) for problem in scores}
        if reference_problems is None:
            reference_problems = problems
        elif problems != reference_problems:
            raise ValueError(f"problem set differs across seeds for cell {cell_id!r}")
        for problem, score in scores.items():
            value = float(score)
            if not math.isfinite(value):
                raise ValueError(f"non-finite test score in cell {cell_id!r}")
            per_problem.setdefault(str(problem), []).append(value)
    return {
        problem: sum(values) / len(values)
        for problem, values in sorted(per_problem.items())
    }


def analyze_confirmatory_test(
    seed_payloads: Sequence[Mapping[str, Any]],
    *,
    selected_active_cell_id: str,
    expected_seeds: Sequence[int],
    beacon_output_hex: str,
    bootstrap_domain: str,
    bootstrap_resamples: int = 20_000,
    practical_delta: float = 0.01,
    lower_quantile: float = 0.025,
    upper_quantile: float = 0.975,
) -> dict[str, Any]:
    """Apply the frozen problem-cluster bootstrap to H1 and H3."""

    if bootstrap_resamples <= 0:
        raise ValueError("bootstrap_resamples must be positive")
    if practical_delta < 0.0:
        raise ValueError("practical_delta must be non-negative")
    if not 0.0 < lower_quantile < upper_quantile < 1.0:
        raise ValueError("bootstrap quantiles must satisfy 0 < lower < upper < 1")

    cell_ids = {
        "EEE_true_LCA": "EEE_true_LCA",
        "EEE_zero_anchor": "EEE_zero_anchor",
        "HEE_near_zero_true_LCA": "HEE_near_zero_true_LCA",
        "selected_active_HEE": selected_active_cell_id,
    }
    cell_scores = {
        name: seed_averaged_problem_scores(
            seed_payloads,
            cell_id=cell_id,
            expected_seeds=expected_seeds,
        )
        for name, cell_id in cell_ids.items()
    }
    problem_sets = {tuple(scores) for scores in cell_scores.values()}
    if len(problem_sets) != 1:
        raise ValueError("planned test cells do not contain the same problem set")
    problems = next(iter(problem_sets))
    contrasts = {
        "H1_EEE_true_LCA_minus_EEE_zero_anchor": _paired_differences(
            cell_scores["EEE_true_LCA"],
            cell_scores["EEE_zero_anchor"],
        ),
        "H3a_selected_active_HEE_minus_EEE_true_LCA": _paired_differences(
            cell_scores["selected_active_HEE"],
            cell_scores["EEE_true_LCA"],
        ),
        "H3b_selected_active_HEE_minus_HEE_near_zero_true_LCA": _paired_differences(
            cell_scores["selected_active_HEE"],
            cell_scores["HEE_near_zero_true_LCA"],
        ),
    }
    seed = derive_cluster_bootstrap_seed(beacon_output_hex, bootstrap_domain)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    indices = torch.randint(
        len(problems),
        (bootstrap_resamples, len(problems)),
        generator=generator,
        dtype=torch.int64,
        device="cpu",
    )
    index_sha256 = hashlib.sha256(indices.numpy().tobytes(order="C")).hexdigest()
    contrast_results = {
        name: _bootstrap_contrast(
            differences,
            problems=problems,
            indices=indices,
            practical_delta=practical_delta,
            lower_quantile=lower_quantile,
            upper_quantile=upper_quantile,
        )
        for name, differences in contrasts.items()
    }
    h1 = contrast_results["H1_EEE_true_LCA_minus_EEE_zero_anchor"]
    h3a = contrast_results["H3a_selected_active_HEE_minus_EEE_true_LCA"]
    h3b = contrast_results["H3b_selected_active_HEE_minus_HEE_near_zero_true_LCA"]
    return {
        "problem_count": len(problems),
        "model_seed_count": len(expected_seeds),
        "selected_active_cell_id": selected_active_cell_id,
        "bootstrap": {
            "resamples": bootstrap_resamples,
            "rng_seed": seed,
            "shared_resample_index_sha256": index_sha256,
            "lower_quantile": lower_quantile,
            "upper_quantile": upper_quantile,
        },
        "contrasts": contrast_results,
        "decisions": {
            "H1_statistical_support": h1["statistical_support"],
            "H1_practical_support": h1["practical_support"],
            "H1_confirmatory_success": h1["statistical_support"] and h1["practical_support"],
            "H3_statistical_support": h3a["statistical_support"] and h3b["statistical_support"],
            "H3_practical_support": h3a["practical_support"] and h3b["practical_support"],
            "H3_confirmatory_success": (
                h3a["statistical_support"]
                and h3b["statistical_support"]
                and h3a["practical_support"]
                and h3b["practical_support"]
            ),
        },
        "aggregation_order": [
            "AP_at_8_per_query",
            "mean_over_queries_within_problem",
            "mean_over_registered_model_seeds_within_problem",
            "paired_bootstrap_over_problem_clusters",
        ],
    }


def _paired_differences(
    treatment: Mapping[str, float],
    control: Mapping[str, float],
) -> dict[str, float]:
    if set(treatment) != set(control):
        raise ValueError("paired contrast cells do not contain the same problem set")
    return {
        problem: float(treatment[problem]) - float(control[problem])
        for problem in sorted(treatment)
    }


def _bootstrap_contrast(
    differences: Mapping[str, float],
    *,
    problems: Sequence[str],
    indices: torch.Tensor,
    practical_delta: float,
    lower_quantile: float,
    upper_quantile: float,
) -> dict[str, Any]:
    values = torch.tensor([differences[problem] for problem in problems], dtype=torch.float64)
    bootstrap_means = values[indices].mean(dim=1)
    bounds = torch.quantile(
        bootstrap_means,
        torch.tensor([lower_quantile, upper_quantile], dtype=torch.float64),
        interpolation="linear",
    )
    estimate = float(values.mean())
    lower = float(bounds[0])
    upper = float(bounds[1])
    return {
        "point_estimate_delta_problem_macro_MAP_at_8": estimate,
        "percentile_interval": {
            "lower": lower,
            "upper": upper,
        },
        "one_sided_lower_bound": lower,
        "positive_problem_count": int((values > 0.0).sum()),
        "zero_problem_count": int((values == 0.0).sum()),
        "negative_problem_count": int((values < 0.0).sum()),
        "statistical_support": lower > 0.0,
        "practical_support": estimate >= practical_delta,
    }
