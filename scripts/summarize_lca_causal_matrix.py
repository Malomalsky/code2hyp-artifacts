from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any, Sequence


PLANNED_CONTRASTS = (
    ("H1_true_lca_vs_zero_anchor", "EEE__true_lca__measure", "EEE__zero_anchor__measure"),
    ("true_lca_vs_endpoint_only", "EEE__true_lca__measure", "EEE__endpoint_only__measure"),
    ("true_lca_vs_program_shuffled_lca", "EEE__true_lca__measure", "EEE__program_shuffled_lca__measure"),
    ("true_lca_vs_full_path_pool", "EEE__true_lca__measure", "EEE__full_path_no_explicit_lca__measure"),
    ("product_vs_equal_capacity_concat_identity", "EEE__true_lca__measure", "EEE_concat__true_lca__measure"),
    ("HEE_vs_EEE", "HEE__true_lca__measure", "EEE__true_lca__measure"),
    ("HEE_vs_near_zero_HEE", "HEE__true_lca__measure", "HEE_near_zero__true_lca__measure"),
    ("HEE_vs_HHH", "HEE__true_lca__measure", "HHH__true_lca__measure"),
)


def summarize_lca_causal_matrix(
    inputs: Sequence[Path],
    *,
    bootstrap_resamples: int = 20_000,
    bootstrap_seed: int = 20260711,
) -> dict[str, Any]:
    if not inputs:
        raise ValueError("at least one input is required")
    if bootstrap_resamples <= 0:
        raise ValueError("bootstrap_resamples must be positive")
    payloads = [_load_payload(path) for path in inputs]
    study_stage = str(payloads[0]["config"].get("study_stage", "pilot"))
    seeds = [int(payload["config"]["seed"]) for payload in payloads]
    if len(set(seeds)) != len(seeds):
        raise ValueError("input files must have distinct seeds")
    _validate_compatible(payloads)

    rows_by_seed = {
        int(payload["config"]["seed"]): {row["cell_id"]: row for row in payload["runs"]}
        for payload in payloads
    }
    cell_ids = sorted(set.intersection(*(set(rows) for rows in rows_by_seed.values())))
    cell_summaries = {
        cell_id: _cell_summary(rows_by_seed, cell_id=cell_id)
        for cell_id in cell_ids
    }
    contrasts = []
    for name, treatment_id, control_id in PLANNED_CONTRASTS:
        if treatment_id not in cell_summaries or control_id not in cell_summaries:
            continue
        contrasts.append(
            _contrast_summary(
                name=name,
                treatment_id=treatment_id,
                control_id=control_id,
                treatment=cell_summaries[treatment_id],
                control=cell_summaries[control_id],
                bootstrap_resamples=bootstrap_resamples,
                bootstrap_seed=bootstrap_seed,
            )
        )
    by_name = {contrast["name"]: contrast for contrast in contrasts}
    h1 = by_name.get("H1_true_lca_vs_zero_anchor")
    hee_e = by_name.get("HEE_vs_EEE")
    hee_nz = by_name.get("HEE_vs_near_zero_HEE")
    gate_c_pass = bool(
        hee_e
        and hee_nz
        and hee_e["mean_task_difference"] > 0.0
        and hee_nz["mean_task_difference"] > 0.0
        and hee_e["bootstrap_ci95"][0] > 0.0
        and hee_nz["bootstrap_ci95"][0] > 0.0
    )
    return {
        "experiment": "code2hyp_lca_causal_matrix_multiseed_summary",
        "status": "pilot" if study_stage == "pilot" else "confirmatory",
        "study_stage": study_stage,
        "inputs": [str(path) for path in inputs],
        "seeds": seeds,
        "seed_count": len(seeds),
        "bootstrap_resamples": bootstrap_resamples,
        "bootstrap_seed": bootstrap_seed,
        "ap_at_r": payloads[0]["config"]["ap_at_r"],
        "task_count": len(next(iter(cell_summaries.values()))["task_scores"]),
        "cells": cell_summaries,
        "contrasts": contrasts,
        "decision": {
            "gate_a_h1_direction_positive": bool(h1 and h1["mean_task_difference"] > 0.0),
            "gate_a_h1_ci_excludes_zero": bool(h1 and h1["bootstrap_ci95"][0] > 0.0),
            "gate_c_requires_both_controls": True,
            "gate_c_pass": gate_c_pass,
            "confirmatory_claim_allowed": study_stage == "confirmatory" and len(payloads) >= 10 and gate_c_pass,
        },
    }


def render_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Code2Hyp LCA causal matrix summary",
        "",
        f"- Status: `{summary['status']}`",
        f"- Study stage: `{summary['study_stage']}`",
        f"- Seeds: {summary['seed_count']}",
        f"- Tasks: {summary['task_count']}",
        f"- Primary metric: MAP@{summary['ap_at_r']}",
        f"- Task-cluster bootstrap resamples: {summary['bootstrap_resamples']}",
        "",
        "## Cells",
        "",
        "| Cell | MAP@R | Seed range |",
        "|---|---:|---:|",
    ]
    for cell_id, cell in sorted(summary["cells"].items()):
        lines.append(
            f"| `{cell_id}` | {cell['mean_task_score']:.4f} | "
            f"{cell['seed_macro_min']:.4f}..{cell['seed_macro_max']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Planned contrasts",
            "",
            "| Contrast | Delta MAP@R | 95% task-bootstrap CI | +/=/- tasks | Exact sign p |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for contrast in summary["contrasts"]:
        low, high = contrast["bootstrap_ci95"]
        signs = contrast["task_signs"]
        lines.append(
            f"| `{contrast['name']}` | {contrast['mean_task_difference']:+.4f} | "
            f"[{low:+.4f}, {high:+.4f}] | {signs['positive']}/{signs['tie']}/{signs['negative']} | "
            f"{contrast['exact_sign_test_two_sided_p']:.4g} |"
        )
    decision = summary["decision"]
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"- Gate A H1 positive direction: `{decision['gate_a_h1_direction_positive']}`.",
            f"- Gate A H1 CI excludes zero: `{decision['gate_a_h1_ci_excludes_zero']}`.",
            f"- Gate C pass: `{decision['gate_c_pass']}`. It requires HEE to beat both EEE and near-zero HEE with CIs above zero.",
            f"- Confirmatory claim allowed: `{decision['confirmatory_claim_allowed']}`.",
            "",
        ]
    )
    return "\n".join(lines)


def _load_payload(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("experiment") != "code2hyp_lca_causal_and_role_geometry_matrix":
        raise ValueError(f"{path} is not an LCA causal matrix payload")
    if payload.get("status") != "complete":
        raise ValueError(f"{path} is not complete")
    return payload


def _validate_compatible(payloads: Sequence[dict[str, Any]]) -> None:
    reference = payloads[0]
    keys = (
        "benchmark_level",
        "language",
        "dim",
        "epochs",
        "max_paths",
        "path_selection_policy",
        "item_scope",
        "ap_at_r",
    )
    for payload in payloads[1:]:
        for key in keys:
            if payload["config"].get(key) != reference["config"].get(key):
                raise ValueError(f"incompatible config field {key!r}")
        if payload["config"].get("study_stage", "pilot") != reference["config"].get("study_stage", "pilot"):
            raise ValueError("incompatible config field 'study_stage'")
        if {task["label"] for task in payload["tasks"]} != {task["label"] for task in reference["tasks"]}:
            raise ValueError("input task sets differ")


def _cell_summary(rows_by_seed: dict[int, dict[str, dict[str, Any]]], *, cell_id: str) -> dict[str, Any]:
    task_values: dict[str, list[float]] = {}
    seed_macro: dict[str, float] = {}
    for seed, rows in sorted(rows_by_seed.items()):
        row = rows[cell_id]
        seed_macro[str(seed)] = float(row["map_at_r"])
        for task, score in row["task_scores"].items():
            task_values.setdefault(task, []).append(float(score))
    task_scores = {task: sum(values) / len(values) for task, values in sorted(task_values.items())}
    seed_scores = list(seed_macro.values())
    return {
        "cell_id": cell_id,
        "mean_task_score": sum(task_scores.values()) / len(task_scores),
        "task_scores": task_scores,
        "seed_macro_scores": seed_macro,
        "seed_macro_min": min(seed_scores),
        "seed_macro_max": max(seed_scores),
    }


def _contrast_summary(
    *,
    name: str,
    treatment_id: str,
    control_id: str,
    treatment: dict[str, Any],
    control: dict[str, Any],
    bootstrap_resamples: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    tasks = sorted(set(treatment["task_scores"]) & set(control["task_scores"]))
    differences = [float(treatment["task_scores"][task]) - float(control["task_scores"][task]) for task in tasks]
    mean_difference = sum(differences) / len(differences)
    rng = random.Random(_derived_seed(bootstrap_seed, name))
    bootstrap = [
        sum(differences[rng.randrange(len(differences))] for _ in differences) / len(differences)
        for _ in range(bootstrap_resamples)
    ]
    positive = sum(value > 0.0 for value in differences)
    negative = sum(value < 0.0 for value in differences)
    ties = len(differences) - positive - negative
    return {
        "name": name,
        "treatment_cell": treatment_id,
        "control_cell": control_id,
        "mean_task_difference": mean_difference,
        "median_task_difference": _quantile(sorted(differences), 0.5),
        "bootstrap_ci95": [_quantile(bootstrap, 0.025), _quantile(bootstrap, 0.975)],
        "bootstrap_probability_positive": sum(value > 0.0 for value in bootstrap) / len(bootstrap),
        "task_signs": {"positive": positive, "tie": ties, "negative": negative},
        "exact_sign_test_two_sided_p": _exact_sign_test_two_sided(positive, negative),
        "task_differences": dict(zip(tasks, differences)),
    }


def _derived_seed(seed: int, label: str) -> int:
    value = int(seed)
    for character in label:
        value = (value * 131 + ord(character)) % (2**32)
    return value


def _exact_sign_test_two_sided(positive: int, negative: int) -> float:
    n = positive + negative
    if n == 0:
        return 1.0
    smaller = min(positive, negative)
    lower_tail = sum(math.comb(n, k) for k in range(smaller + 1)) / (2**n)
    return min(1.0, 2.0 * lower_tail)


def _quantile(values: Sequence[float], probability: float) -> float:
    if not values:
        raise ValueError("quantile requires non-empty values")
    ordered = sorted(float(value) for value in values)
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize one or more Code2Hyp LCA causal matrix seeds.")
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--bootstrap-resamples", type=int, default=20_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260711)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = summarize_lca_causal_matrix(
        args.input,
        bootstrap_resamples=args.bootstrap_resamples,
        bootstrap_seed=args.bootstrap_seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(render_markdown(summary), encoding="utf-8")
    print(f"status={summary['status']} seeds={summary['seed_count']} output={args.output} report={args.report}")


if __name__ == "__main__":
    main()
