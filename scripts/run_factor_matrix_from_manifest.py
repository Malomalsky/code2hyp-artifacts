from __future__ import annotations

import argparse
import json
import shlex
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_dta_factor_matrix import run_dta_factor_matrix
from scripts.run_dta_level_b_c_retrieval import TaskSource


def task_sources_from_manifest(manifest_path: Path) -> tuple[TaskSource, ...]:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    tasks = payload.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        raise ValueError(f"{manifest_path} does not contain a non-empty 'tasks' list")
    result: list[TaskSource] = []
    for task in tasks:
        if isinstance(task, str):
            output_dir = Path(str(payload.get("output_dir", "")).strip())
            if not output_dir:
                raise ValueError(f"{manifest_path} has string tasks but no output_dir")
            label = f"task-{task}"
            result.append(TaskSource(label=label, source=output_dir / label))
            continue
        if not isinstance(task, dict):
            raise ValueError("manifest task entries must be objects")
        label = str(task.get("label", "")).strip()
        source = Path(str(task.get("path", "")).strip())
        if not label or not source:
            raise ValueError(f"invalid task entry in {manifest_path}: {task!r}")
        result.append(TaskSource(label=label, source=source))
    return tuple(result)


def language_from_manifest(manifest_path: Path) -> str:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    value = str(payload.get("language", "")).strip()
    if not value and payload.get("output_dir") and all(isinstance(task, str) for task in payload.get("tasks", [])):
        return "python"
    if value not in {"python", "java"}:
        raise ValueError(f"{manifest_path} has unsupported language={value!r}")
    return value


def item_scope_from_manifest(manifest_path: Path) -> str:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    value = str(payload.get("item_scope", "callable")).strip() or "callable"
    if value not in {"callable", "module", "callable_or_module"}:
        raise ValueError(f"{manifest_path} has unsupported item_scope={value!r}")
    return value


def dry_run_command(args: argparse.Namespace, tasks: tuple[TaskSource, ...], language: str, item_scope: str) -> str:
    command = [
        "python",
        "scripts/run_dta_factor_matrix.py",
        "--output",
        str(args.output),
        "--benchmark-level",
        args.benchmark_level,
        "--language",
        language,
        "--geometries",
        args.geometries,
        "--path-object-modes",
        args.path_object_modes,
        "--method-aggregations",
        args.method_aggregations,
        "--dim",
        str(args.dim),
        "--epochs",
        str(args.epochs),
        "--seed",
        str(args.seed),
        "--item-scope",
        item_scope,
        "--max-ball-fraction",
        str(args.max_ball_fraction),
        "--encoder-policy",
        args.encoder_policy,
    ]
    cost_modes = getattr(args, "cost_modes", None)
    if cost_modes:
        command.extend(["--cost-modes", str(cost_modes)])
    side_weights = getattr(args, "side_weights", None)
    if side_weights:
        command.extend(["--side-weights", str(side_weights)])
    else:
        command.extend(["--side-weight", str(args.side_weight)])
    point_weights = getattr(args, "point_weights", None)
    if point_weights:
        command.extend(["--point-weights", str(point_weights)])
    for task in tasks:
        command.extend(["--task", task.label, str(task.source)])
    return " ".join(shlex.quote(part) for part in command)


def _parse_csv(value: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in value.split(",") if part.strip())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Code2Hyp factor matrix from a materialized external corpus manifest.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "outputs/external_code2hyp_factor_matrix.json")
    parser.add_argument("--benchmark-level", choices=("B_independent_solution", "C_structural_hard_negative"), default="B_independent_solution")
    parser.add_argument("--language", choices=("python", "java"), default=None, help="Override manifest language.")
    parser.add_argument("--item-scope", choices=("callable", "module", "callable_or_module"), default=None, help="Override manifest item scope.")
    parser.add_argument("--geometries", default="E,H_1e-4,H_1")
    parser.add_argument("--path-object-modes", default="single_point,lca_product")
    parser.add_argument("--method-aggregations", default="centroid,measure")
    parser.add_argument("--dim", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-2)
    parser.add_argument("--max-files-per-task", type=int, default=128)
    parser.add_argument("--max-methods-per-task", type=int, default=48)
    parser.add_argument("--train-per-task", type=int, default=16)
    parser.add_argument("--query-per-task", type=int, default=8)
    parser.add_argument("--gallery-per-task", type=int, default=1)
    parser.add_argument("--max-paths", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260625)
    parser.add_argument("--sinkhorn-iterations", type=int, default=6)
    parser.add_argument("--sinkhorn-projection-iterations", type=int, default=512)
    parser.add_argument("--kappa", type=float, default=0.05)
    parser.add_argument("--side-weight", type=float, default=1.0)
    parser.add_argument("--side-weights", default=None, help="Comma-separated side-feature weights for a sweep.")
    parser.add_argument("--point-weights", default=None, help="Comma-separated point-channel weights for train_weighted_combined.")
    parser.add_argument(
        "--cost-modes",
        default=None,
        help=(
            "Comma-separated cost modes: point_only,side_only,unnormalized_combined,"
            "train_normalized_combined,train_weighted_combined,validation_selected_combined."
        ),
    )
    parser.add_argument("--max-ball-fraction", type=float, default=0.35)
    parser.add_argument("--hard-negatives-per-query", type=int, default=6)
    parser.add_argument("--encoder-policy", choices=("shared_euclidean", "geometry_aware"), default="shared_euclidean")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    tasks = task_sources_from_manifest(args.manifest)
    language = args.language or language_from_manifest(args.manifest)
    item_scope = args.item_scope or item_scope_from_manifest(args.manifest)
    if args.dry_run:
        print(dry_run_command(args, tasks, language, item_scope))
        return
    payload: dict[str, Any] = run_dta_factor_matrix(
        tasks=tasks,
        output_path=args.output,
        benchmark_level=args.benchmark_level,
        geometries=_parse_csv(args.geometries),  # type: ignore[arg-type]
        path_object_modes=_parse_csv(args.path_object_modes),  # type: ignore[arg-type]
        method_aggregations=_parse_csv(args.method_aggregations),  # type: ignore[arg-type]
        language=language,  # type: ignore[arg-type]
        dim=args.dim,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        max_files_per_task=args.max_files_per_task,
        max_methods_per_task=args.max_methods_per_task,
        train_per_task=args.train_per_task,
        query_per_task=args.query_per_task,
        gallery_per_task=args.gallery_per_task,
        max_paths=args.max_paths,
        seed=args.seed,
        item_scope=item_scope,  # type: ignore[arg-type]
        sinkhorn_iterations=args.sinkhorn_iterations,
        sinkhorn_projection_iterations=args.sinkhorn_projection_iterations,
        kappa=args.kappa,
        side_weight=args.side_weight,
        side_weights=_parse_float_csv(args.side_weights),
        point_weights=_parse_float_csv(args.point_weights),
        cost_modes=_parse_cost_modes(args.cost_modes),
        max_ball_fraction=args.max_ball_fraction,
        hard_negatives_per_query=args.hard_negatives_per_query,
        encoder_policy=args.encoder_policy,
    )
    print(f"status={payload['status']} completed={payload['completed_runs']}/{payload['expected_runs']} output={args.output}")


def _parse_float_csv(value: str | None) -> tuple[float, ...] | None:
    if value is None:
        return None
    return tuple(float(part.strip()) for part in value.split(",") if part.strip())


def _parse_cost_modes(value: str | None) -> tuple[str, ...] | None:
    if value is None:
        return None
    return tuple(part.strip() for part in value.split(",") if part.strip())


if __name__ == "__main__":
    main()
