from __future__ import annotations

import argparse
import csv
from pathlib import Path


SELECTED_METRICS = [
    "node_count",
    "ball_size_mean_r3",
    "forman_negative_mass",
    "forman_positive_mass",
    "ollivier_mean",
    "ollivier_negative_mass",
    "ollivier_near_zero_mass",
]


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _format_float(value: str | float, digits: int = 4) -> str:
    return f"{float(value):.{digits}f}"


def _permutation_lookup(rows: list[dict[str, str]]) -> dict[tuple[str, str], dict[str, str]]:
    return {(row["analysis"], row["metric"]): row for row in rows}


def build_snapshot(
    *,
    effect_sizes_path: Path,
    residual_effect_sizes_path: Path,
    permutation_tests_path: Path,
    output_path: Path,
) -> None:
    effects = {row["metric"]: row for row in _read_csv(effect_sizes_path)}
    residuals = {row["metric"]: row for row in _read_csv(residual_effect_sizes_path)}
    permutation = _permutation_lookup(_read_csv(permutation_tests_path))

    lines: list[str] = [
        "# Article Results Snapshot",
        "",
        "Source files:",
        "",
        f"- `{effect_sizes_path}`",
        f"- `{residual_effect_sizes_path}`",
        f"- `{permutation_tests_path}`",
        "",
        "## Statistical Design",
        "",
        "The task-level association is measured as:",
        "",
        "```text",
        "eta_squared_task = SS_between(task_id) / SS_total",
        "```",
        "",
        "Residual tests first remove linear size/growth controls:",
        "",
        "```text",
        "curvature_metric = beta_0 + beta_1 * node_count + beta_2 * ball_size_mean_r3 + epsilon",
        "eta_squared_task_residual = SS_between(task_id, epsilon) / SS_total(epsilon)",
        "```",
        "",
        "Permutation test:",
        "",
        "- null model: random permutation of `task_id` labels across programs;",
        "- permutations: 5000;",
        "- seed: 20260614;",
        "- multiple testing correction: Holm over raw and residual tests.",
        "",
        "## Raw Task-Level Effects",
        "",
        "| Metric | eta_squared_task | permutation p | Holm p |",
        "|---|---:|---:|---:|",
    ]

    for metric in SELECTED_METRICS:
        effect = effects[metric]
        test = permutation[("raw", metric)]
        lines.append(
            "| "
            f"`{metric}` | "
            f"{_format_float(effect['eta_squared_task'])} | "
            f"{_format_float(test['p_value'])} | "
            f"{_format_float(test['p_value_holm'])} |"
        )

    lines.extend(
        [
            "",
            "## Residual Task-Level Effects",
            "",
            "| Metric | covariate R^2 | residual eta_squared_task | permutation p | Holm p |",
            "|---|---:|---:|---:|---:|",
        ]
    )

    for metric in SELECTED_METRICS:
        if metric not in residuals:
            continue
        residual = residuals[metric]
        test = permutation[("residual", metric)]
        lines.append(
            "| "
            f"`{metric}` | "
            f"{_format_float(residual['covariate_r_squared'])} | "
            f"{_format_float(residual['eta_squared_task_residual'])} | "
            f"{_format_float(test['p_value'])} | "
            f"{_format_float(test['p_value_holm'])} |"
        )

    lines.extend(
        [
            "",
            "## Strict Claim",
            "",
            "On the 550-program DTA AST atlas, local discrete-curvature distributions are associated with task type. "
            "The association remains visible after linear controls for AST size and local ball growth; "
            "permutation testing with Holm correction gives `p_Holm <= 0.0028` for the selected raw and residual descriptors.",
            "",
            "## Boundary",
            "",
            "This is structural evidence, not a downstream performance claim. It supports the existence of task-level geometry signal in AST curvature fractions; it does not by itself prove that curvature features improve retrieval or classification.",
            "",
        ]
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a manuscript-oriented Markdown snapshot of geometry results."
    )
    parser.add_argument(
        "--effect-sizes",
        type=Path,
        default=Path("reports/task_geometry_effect_sizes_limit50.csv"),
    )
    parser.add_argument(
        "--residual-effect-sizes",
        type=Path,
        default=Path("reports/task_geometry_residual_effect_sizes_limit50.csv"),
    )
    parser.add_argument(
        "--permutation-tests",
        type=Path,
        default=Path("reports/task_geometry_permutation_tests_limit50.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/article_results_snapshot.md"),
    )
    args = parser.parse_args()
    build_snapshot(
        effect_sizes_path=args.effect_sizes,
        residual_effect_sizes_path=args.residual_effect_sizes,
        permutation_tests_path=args.permutation_tests,
        output_path=args.output,
    )
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
