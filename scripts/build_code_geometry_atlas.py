from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path
from statistics import fmean, pstdev
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from geometry_profile_research.code_geometry_profile import code_geometry_profile
from geometry_profile_research.dta import load_dta_records
from geometry_profile_research.program_views import ProgramView, extract_python_program_views


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _iter_available_program_views(code: str) -> list[ProgramView]:
    views = extract_python_program_views(code)
    available = [views.ast]
    if views.cfg is not None:
        available.append(views.cfg)
    if views.dfg is not None:
        available.append(views.dfg)
    if views.cpg is not None:
        available.append(views.cpg)
    return available


def build_code_geometry_atlas_rows(
    dataset_dir: Path,
    *,
    limit_per_task: int | None,
    include_ollivier: bool,
    ollivier_idleness: float,
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    rows: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    for record in load_dta_records(dataset_dir, limit_per_task=limit_per_task):
        try:
            for view in _iter_available_program_views(record.code):
                profile = code_geometry_profile(
                    view.graph,
                    include_ollivier=include_ollivier,
                    ollivier_idleness=ollivier_idleness,
                )
                rows.append(
                    {
                        "record_id": record.record_id,
                        "task_id": record.task_id,
                        "row_index": record.row_index,
                        "source_file": record.source_file,
                        "view": view.kind,
                        **profile,
                    }
                )
        except SyntaxError as exc:
            errors.append(
                {
                    "record_id": record.record_id,
                    "source_file": record.source_file,
                    "error": str(exc),
                }
            )
    return rows, errors


def summarize_atlas_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["task_id"]), str(row["view"]))].append(row)

    summary_rows: list[dict[str, Any]] = []
    for (task_id, view), group_rows in sorted(grouped.items(), key=lambda item: (int(item[0][0]), item[0][1])):
        numeric_keys = sorted(
            key
            for key in {key for row in group_rows for key in row}
            if key not in {"record_id", "source_file", "view"}
            and all(isinstance(row.get(key), (int, float)) for row in group_rows)
        )
        summary: dict[str, Any] = {
            "task_id": int(task_id),
            "view": view,
            "n": len(group_rows),
        }
        for key in numeric_keys:
            values = [float(row[key]) for row in group_rows]
            summary[f"{key}_mean"] = fmean(values)
            summary[f"{key}_std"] = pstdev(values) if len(values) > 1 else 0.0
        summary_rows.append(summary)
    return summary_rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the first Code Geometry Atlas table for DTA program views."
    )
    parser.add_argument("--dataset-dir", type=Path, default=Path("data/dta_zenodo_7799972/extracted"))
    parser.add_argument("--limit-per-task", type=int, default=20)
    parser.add_argument(
        "--include-ollivier",
        action="store_true",
        help="Include exact local Ollivier-Ricci summaries. Slower; use for confirmatory atlas runs.",
    )
    parser.add_argument(
        "--ollivier-idleness",
        type=float,
        default=0.0,
        help="Lazy random-walk idleness alpha for Ollivier-Ricci curvature.",
    )
    parser.add_argument("--output", type=Path, default=Path("reports/code_geometry_atlas.csv"))
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=Path("reports/code_geometry_view_summary.csv"),
    )
    parser.add_argument("--errors-output", type=Path, default=Path("reports/code_geometry_atlas_errors.csv"))
    args = parser.parse_args()

    rows, errors = build_code_geometry_atlas_rows(
        args.dataset_dir,
        limit_per_task=args.limit_per_task,
        include_ollivier=args.include_ollivier,
        ollivier_idleness=args.ollivier_idleness,
    )
    summary_rows = summarize_atlas_rows(rows)
    _write_csv(args.output, rows)
    _write_csv(args.summary_output, summary_rows)
    _write_csv(args.errors_output, errors)
    print(f"wrote {args.output} ({len(rows)} rows)")
    print(f"wrote {args.summary_output} ({len(summary_rows)} rows)")
    print(f"wrote {args.errors_output} ({len(errors)} rows)")
    print("implemented_views=ast")
    print("unsupported_views=cfg,dfg,cpg")
    print(f"include_ollivier={args.include_ollivier}")
    print(f"ollivier_idleness={args.ollivier_idleness}")


if __name__ == "__main__":
    main()
