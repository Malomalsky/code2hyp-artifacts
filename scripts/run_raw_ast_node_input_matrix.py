from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from geometry_profile_research.raw_ast_code2hyp import MethodAggregation, NodeInputMode, PathCostOrientation, PathObjectMode, TerminalPolicy
from geometry_profile_research.raw_ast_retrieval import PositiveMode
from scripts.run_raw_ast_code2hyp_retrieval import Geometry, Language, run_retrieval_experiment


DEFAULT_PROJECT = PROJECT_ROOT / "data/code2seq_java_small_raw/extracted/java-small/validation/libgdx"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs/raw_ast_node_input_matrix.json"
DEFAULT_RUNS_DIR = PROJECT_ROOT / "outputs/raw_ast_node_input_matrix_runs"
DEFAULT_NODE_INPUT_MODES: tuple[NodeInputMode, ...] = ("label_only", "label_depth", "label_depth_prefix")
DEFAULT_GEOMETRIES: tuple[Geometry, ...] = ("euclidean", "poincare")


@dataclass(frozen=True)
class MatrixProject:
    label: str
    source: Path


def run_node_input_matrix(
    *,
    projects: Sequence[MatrixProject],
    output_path: Path,
    runs_dir: Path,
    node_input_modes: Sequence[NodeInputMode] = DEFAULT_NODE_INPUT_MODES,
    path_object_modes: Sequence[PathObjectMode] = ("lca_product",),
    method_aggregations: Sequence[MethodAggregation] = ("measure",),
    path_cost_orientation: PathCostOrientation = "directed",
    path_cost_orientations: Sequence[PathCostOrientation] | None = None,
    geometries: Sequence[Geometry] = DEFAULT_GEOMETRIES,
    curvatures: Sequence[float] = (1.0,),
    dims: Sequence[int] = (4,),
    seeds: Sequence[int] = (20260623,),
    language: Language = "java",
    epochs: int = 3,
    learning_rate: float = 1e-2,
    max_files: int | None = 200,
    max_methods: int | None = 32,
    max_paths: int = 16,
    min_structural_gap: float = 0.05,
    sinkhorn_iterations: int = 30,
    sinkhorn_epsilon: float = 0.05,
    terminal_policy: TerminalPolicy = "class",
    positive_mode: PositiveMode = "alpha_structural_noop",
    resume: bool = True,
) -> dict[str, Any]:
    """Run the reviewer-directed raw-AST node-input matrix.

    The matrix isolates whether hyperbolic structure helps when lexical path
    information is restricted: label-only, label+depth, and full prefix input
    modes are evaluated under matched Euclidean and Poincare geometries.
    """

    if not projects:
        raise ValueError("at least one project is required")
    if not node_input_modes:
        raise ValueError("at least one node_input_mode is required")
    if not path_object_modes:
        raise ValueError("at least one path_object_mode is required")
    if not method_aggregations:
        raise ValueError("at least one method_aggregation is required")
    orientation_values = tuple(path_cost_orientations) if path_cost_orientations is not None else (path_cost_orientation,)
    if not orientation_values:
        raise ValueError("at least one path_cost_orientation is required")
    invalid_orientations = [orientation for orientation in orientation_values if orientation not in {"directed", "unoriented"}]
    if invalid_orientations:
        raise ValueError(f"unknown path_cost_orientation values: {invalid_orientations!r}")
    if not geometries:
        raise ValueError("at least one geometry is required")
    if not curvatures:
        raise ValueError("at least one curvature is required")
    if not dims:
        raise ValueError("at least one dim is required")
    if not seeds:
        raise ValueError("at least one seed is required")

    runs_dir.mkdir(parents=True, exist_ok=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    geometry_curvature_pairs = _geometry_curvature_pairs(geometries, curvatures)
    expected_runs = (
        len(projects)
        * len(node_input_modes)
        * len(path_object_modes)
        * len(method_aggregations)
        * len(orientation_values)
        * len(geometry_curvature_pairs)
        * len(dims)
        * len(seeds)
    )
    for project in projects:
        for node_input_mode in node_input_modes:
            for path_object_mode in path_object_modes:
                for method_aggregation in method_aggregations:
                    for path_cost_orientation_value in orientation_values:
                        for geometry, curvature in geometry_curvature_pairs:
                            for dim in dims:
                                for seed in seeds:
                                    run_path = runs_dir / _run_filename(
                                        project_label=project.label,
                                        node_input_mode=node_input_mode,
                                        path_object_mode=path_object_mode,
                                        method_aggregation=method_aggregation,
                                        path_cost_orientation=path_cost_orientation_value,
                                        geometry=geometry,
                                        curvature=curvature,
                                        include_orientation=path_cost_orientation_value != "directed",
                                        include_curvature=geometry == "poincare" and (len(curvatures) > 1 or float(curvature) != 1.0),
                                        dim=dim,
                                        seed=seed,
                                    )
                                    resumed = False
                                    if resume and run_path.exists():
                                        payload = _load_json(run_path)
                                        resumed = True
                                    else:
                                        payload = run_retrieval_experiment(
                                            sources=(project.source,),
                                            output_path=run_path,
                                            language=language,
                                            geometry=geometry,
                                            dim=dim,
                                            epochs=epochs,
                                            learning_rate=learning_rate,
                                            max_files=max_files,
                                            max_methods=max_methods,
                                            max_paths=max_paths,
                                            seed=seed,
                                            min_structural_gap=min_structural_gap,
                                            sinkhorn_iterations=sinkhorn_iterations,
                                            sinkhorn_epsilon=sinkhorn_epsilon,
                                            terminal_policy=terminal_policy,
                                            node_input_mode=node_input_mode,
                                            path_object_mode=path_object_mode,
                                            method_aggregation=method_aggregation,
                                            path_cost_orientation=path_cost_orientation_value,
                                            curvature=curvature,
                                            positive_mode=positive_mode,
                                        )
                                    row = _summary_row(
                                        project=project,
                                        node_input_mode=node_input_mode,
                                        path_object_mode=path_object_mode,
                                        method_aggregation=method_aggregation,
                                        path_cost_orientation=path_cost_orientation_value,
                                        geometry=geometry,
                                        curvature=curvature,
                                        dim=dim,
                                        seed=seed,
                                        run_path=run_path,
                                        payload=payload,
                                        resumed=resumed,
                                    )
                                    rows.append(row)
                                    _write_summary(
                                        output_path,
                                        _summary_payload(
                                            projects=projects,
                                            output_path=output_path,
                                            runs_dir=runs_dir,
                                            node_input_modes=node_input_modes,
                                            path_object_modes=path_object_modes,
                                            method_aggregations=method_aggregations,
                                            path_cost_orientations=orientation_values,
                                            geometries=geometries,
                                            curvatures=curvatures,
                                            dims=dims,
                                            seeds=seeds,
                                            language=language,
                                            epochs=epochs,
                                            learning_rate=learning_rate,
                                            max_files=max_files,
                                            max_methods=max_methods,
                                            max_paths=max_paths,
                                            min_structural_gap=min_structural_gap,
                                            sinkhorn_iterations=sinkhorn_iterations,
                                            sinkhorn_epsilon=sinkhorn_epsilon,
                                            terminal_policy=terminal_policy,
                                            positive_mode=positive_mode,
                                            expected_runs=expected_runs,
                                            rows=rows,
                                        ),
                                    )

    summary = _summary_payload(
        projects=projects,
        output_path=output_path,
        runs_dir=runs_dir,
        node_input_modes=node_input_modes,
        path_object_modes=path_object_modes,
        method_aggregations=method_aggregations,
        path_cost_orientations=orientation_values,
        geometries=geometries,
        curvatures=curvatures,
        dims=dims,
        seeds=seeds,
        language=language,
        epochs=epochs,
        learning_rate=learning_rate,
        max_files=max_files,
        max_methods=max_methods,
        max_paths=max_paths,
        min_structural_gap=min_structural_gap,
        sinkhorn_iterations=sinkhorn_iterations,
        sinkhorn_epsilon=sinkhorn_epsilon,
        terminal_policy=terminal_policy,
        positive_mode=positive_mode,
        expected_runs=expected_runs,
        rows=rows,
    )
    _write_summary(output_path, summary)
    return summary


def _summary_row(
    *,
    project: MatrixProject,
    node_input_mode: NodeInputMode,
    path_object_mode: PathObjectMode,
    method_aggregation: MethodAggregation,
    path_cost_orientation: PathCostOrientation,
    geometry: Geometry,
    curvature: float,
    dim: int,
    seed: int,
    run_path: Path,
    payload: dict[str, Any],
    resumed: bool,
) -> dict[str, Any]:
    metrics = payload.get("metrics", {})
    row: dict[str, Any] = {
        "project": project.label,
        "source": str(project.source),
        "node_input_mode": node_input_mode,
        "path_object_mode": path_object_mode,
        "method_aggregation": method_aggregation,
        "path_cost_orientation": path_cost_orientation,
        "geometry": geometry,
        "curvature": float(curvature),
        "dim": int(dim),
        "seed": int(seed),
        "output_path": str(run_path),
        "resumed": resumed,
        "item_count": int(payload.get("item_count", 0)),
        "vocab_size": int(payload.get("vocab_size", 0)),
        "query_record_count": len(payload.get("query_records", [])),
        "geometry_diagnostics": payload.get("geometry_diagnostics", {}),
    }
    for metric_name in (
        "recall_at_1",
        "recall_at_3",
        "recall_at_5",
        "ndcg_at_1",
        "ndcg_at_3",
        "ndcg_at_5",
        "mrr",
        "mean_rank",
        "positive_distance_mean",
        "nearest_negative_distance_mean",
        "margin_mean",
        "margin_min",
    ):
        if metric_name in metrics:
            row[metric_name] = float(metrics[metric_name])
    return row


def _summary_payload(
    *,
    projects: Sequence[MatrixProject],
    output_path: Path,
    runs_dir: Path,
    node_input_modes: Sequence[NodeInputMode],
    path_object_modes: Sequence[PathObjectMode],
    method_aggregations: Sequence[MethodAggregation],
    path_cost_orientations: Sequence[PathCostOrientation],
    geometries: Sequence[Geometry],
    curvatures: Sequence[float],
    dims: Sequence[int],
    seeds: Sequence[int],
    language: Language,
    epochs: int,
    learning_rate: float,
    max_files: int | None,
    max_methods: int | None,
    max_paths: int,
    min_structural_gap: float,
    sinkhorn_iterations: int,
    sinkhorn_epsilon: float,
    terminal_policy: TerminalPolicy,
    positive_mode: PositiveMode,
    expected_runs: int,
    rows: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "experiment": "raw_ast_node_input_matrix",
        "status": "complete" if len(rows) == expected_runs else "partial",
        "expected_runs": expected_runs,
        "completed_runs": len(rows),
        "config": {
            "projects": [{"label": project.label, "source": str(project.source)} for project in projects],
            "output_path": str(output_path),
            "runs_dir": str(runs_dir),
            "node_input_modes": list(node_input_modes),
            "path_object_modes": list(path_object_modes),
            "method_aggregations": list(method_aggregations),
            "path_cost_orientation": path_cost_orientations[0] if len(path_cost_orientations) == 1 else None,
            "path_cost_orientations": list(path_cost_orientations),
            "geometries": list(geometries),
            "curvatures": [float(curvature) for curvature in curvatures],
            "dims": [int(dim) for dim in dims],
            "seeds": [int(seed) for seed in seeds],
            "language": language,
            "epochs": epochs,
            "learning_rate": learning_rate,
            "max_files": max_files,
            "max_methods": max_methods,
            "max_paths": max_paths,
            "min_structural_gap": min_structural_gap,
            "sinkhorn_iterations": sinkhorn_iterations,
            "sinkhorn_epsilon": sinkhorn_epsilon,
            "terminal_policy": terminal_policy,
            "positive_mode": positive_mode,
        },
        "runs": sorted(
            rows,
            key=lambda row: (
                str(row["project"]),
                str(row["node_input_mode"]),
                str(row["path_object_mode"]),
                str(row["method_aggregation"]),
                str(row.get("path_cost_orientation", "directed")),
                str(row["geometry"]),
                float(row.get("curvature", 0.0)),
                int(row["dim"]),
                int(row["seed"]),
            ),
        ),
    }


def _write_summary(path: Path, payload: dict[str, Any]) -> None:
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary_path.replace(path)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _run_filename(
    *,
    project_label: str,
    node_input_mode: NodeInputMode,
    path_object_mode: PathObjectMode,
    method_aggregation: MethodAggregation,
    geometry: Geometry,
    curvature: float,
    path_cost_orientation: PathCostOrientation,
    include_orientation: bool,
    include_curvature: bool,
    dim: int,
    seed: int,
) -> str:
    safe_label = re.sub(r"[^A-Za-z0-9_.-]+", "_", project_label.strip()).strip("._") or "project"
    curvature_suffix = f"_c{_format_curvature(curvature)}" if include_curvature else ""
    orientation_suffix = f"{path_cost_orientation}_" if include_orientation else ""
    return (
        f"{safe_label}_{node_input_mode}_{path_object_mode}_{method_aggregation}_"
        f"{orientation_suffix}{geometry}{curvature_suffix}_d{dim}_seed{seed}.json"
    )


def _geometry_curvature_pairs(
    geometries: Sequence[Geometry],
    curvatures: Sequence[float],
) -> tuple[tuple[Geometry, float], ...]:
    pairs: list[tuple[Geometry, float]] = []
    for geometry in geometries:
        if geometry == "euclidean":
            pairs.append((geometry, 1.0))
        else:
            pairs.extend((geometry, float(curvature)) for curvature in curvatures)
    return tuple(pairs)


def _format_curvature(curvature: float) -> str:
    return f"{curvature:.0e}" if curvature < 0.001 or curvature >= 1000 else f"{curvature:g}"


def _parse_projects(values: Sequence[Sequence[str]] | None) -> tuple[MatrixProject, ...]:
    if not values:
        return (MatrixProject("java-small-libgdx-validation", DEFAULT_PROJECT),)
    return tuple(MatrixProject(label, Path(path)) for label, path in values)


def _parse_csv_ints(value: str) -> tuple[int, ...]:
    return tuple(int(part.strip()) for part in value.split(",") if part.strip())


def _parse_csv_floats(value: str) -> tuple[float, ...]:
    return tuple(float(part.strip()) for part in value.split(",") if part.strip())


def _parse_csv_literals(value: str, *, allowed: set[str], option_name: str) -> tuple[str, ...]:
    parsed = tuple(part.strip() for part in value.split(",") if part.strip())
    invalid = [part for part in parsed if part not in allowed]
    if invalid:
        raise argparse.ArgumentTypeError(f"{option_name} has unsupported values: {', '.join(invalid)}")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a raw-AST Code2Hyp node-input matrix benchmark.")
    parser.add_argument(
        "--project",
        action="append",
        nargs=2,
        metavar=("LABEL", "PATH"),
        help="Project label and source file/directory. Can be supplied multiple times.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS_DIR)
    parser.add_argument(
        "--node-input-modes",
        default="label_only,label_depth,label_depth_prefix",
        help="Comma-separated subset of label_only,label_depth,label_depth_prefix.",
    )
    parser.add_argument(
        "--path-object-modes",
        default="lca_product",
        help="Comma-separated subset of single_point,lca_product.",
    )
    parser.add_argument(
        "--method-aggregations",
        default="measure",
        help="Comma-separated subset of centroid,measure.",
    )
    parser.add_argument("--path-cost-orientation", choices=("directed", "unoriented"), default="directed")
    parser.add_argument(
        "--path-cost-orientations",
        default=None,
        help="Comma-separated subset of directed,unoriented. Overrides --path-cost-orientation.",
    )
    parser.add_argument("--geometries", default="euclidean,poincare", help="Comma-separated subset of euclidean,poincare.")
    parser.add_argument("--curvatures", default=None, help="Comma-separated Poincare curvature controls.")
    parser.add_argument("--dims", default="4", help="Comma-separated structural dimensions.")
    parser.add_argument("--seeds", default="20260623", help="Comma-separated random seeds.")
    parser.add_argument("--language", choices=("auto", "java", "python"), default="java")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=1e-2)
    parser.add_argument("--max-files", type=int, default=200)
    parser.add_argument("--max-methods", type=int, default=32)
    parser.add_argument("--max-paths", type=int, default=16)
    parser.add_argument("--min-structural-gap", type=float, default=0.05)
    parser.add_argument("--sinkhorn-iterations", type=int, default=30)
    parser.add_argument("--sinkhorn-epsilon", type=float, default=0.05)
    parser.add_argument("--terminal-policy", choices=("type", "class", "value"), default="class")
    parser.add_argument("--curvature", type=float, default=1.0)
    parser.add_argument(
        "--positive-mode",
        choices=("alpha_rename", "structural_noop", "alpha_structural_noop"),
        default="alpha_structural_noop",
    )
    parser.add_argument("--no-resume", action="store_true", help="Recompute runs even if per-run JSON already exists.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    node_input_modes = _parse_csv_literals(
        args.node_input_modes,
        allowed={"label_only", "label_depth", "label_depth_prefix"},
        option_name="--node-input-modes",
    )
    geometries = _parse_csv_literals(args.geometries, allowed={"euclidean", "poincare"}, option_name="--geometries")
    path_object_modes = _parse_csv_literals(
        args.path_object_modes,
        allowed={"single_point", "lca_product"},
        option_name="--path-object-modes",
    )
    method_aggregations = _parse_csv_literals(
        args.method_aggregations,
        allowed={"centroid", "measure"},
        option_name="--method-aggregations",
    )
    path_cost_orientations = (
        _parse_csv_literals(
            args.path_cost_orientations,
            allowed={"directed", "unoriented"},
            option_name="--path-cost-orientations",
        )
        if args.path_cost_orientations
        else (args.path_cost_orientation,)
    )
    summary = run_node_input_matrix(
        projects=_parse_projects(args.project),
        output_path=args.output,
        runs_dir=args.runs_dir,
        node_input_modes=node_input_modes,  # type: ignore[arg-type]
        path_object_modes=path_object_modes,  # type: ignore[arg-type]
        method_aggregations=method_aggregations,  # type: ignore[arg-type]
        path_cost_orientations=path_cost_orientations,  # type: ignore[arg-type]
        geometries=geometries,  # type: ignore[arg-type]
        curvatures=_parse_csv_floats(args.curvatures) if args.curvatures else (float(args.curvature),),
        dims=_parse_csv_ints(args.dims),
        seeds=_parse_csv_ints(args.seeds),
        language=args.language,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        max_files=args.max_files,
        max_methods=args.max_methods,
        max_paths=args.max_paths,
        min_structural_gap=args.min_structural_gap,
        sinkhorn_iterations=args.sinkhorn_iterations,
        sinkhorn_epsilon=args.sinkhorn_epsilon,
        terminal_policy=args.terminal_policy,  # type: ignore[arg-type]
        positive_mode=args.positive_mode,  # type: ignore[arg-type]
        resume=not args.no_resume,
    )
    print(
        f"status={summary['status']} completed={summary['completed_runs']}/{summary['expected_runs']} "
        f"output={args.output}"
    )


if __name__ == "__main__":
    main()
