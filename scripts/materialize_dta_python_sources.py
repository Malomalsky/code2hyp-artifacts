from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from geometry_profile_research.python_raw_ast import parse_python_ast_tree
from geometry_profile_research.raw_ast import terminal_to_terminal_paths
from geometry_profile_research.raw_ast_retrieval import callable_subtrees


def materialize_dta_python_sources(
    *,
    input_dir: Path,
    output_dir: Path,
    tasks: Sequence[str] | None = None,
    max_records_per_task: int = 128,
    min_paths: int = 2,
    max_paths: int = 16,
) -> dict[str, Any]:
    """Materialize DTA CSV code snippets as deterministic Python source files.

    The raw DTA release stores solutions in CSV cells. The raw-AST retrieval
    runner consumes source files, so this adapter creates a stable file-based
    view without changing the model or the retrieval pipeline.
    """

    if max_records_per_task <= 0:
        raise ValueError("max_records_per_task must be positive")
    if min_paths <= 0:
        raise ValueError("min_paths must be positive")
    selected_tasks = set(_normalize_task(task) for task in tasks) if tasks else None
    task_files = sorted(input_dir.glob("task-*.csv"))
    if selected_tasks is not None:
        task_files = [path for path in task_files if _task_id_from_path(path) in selected_tasks]
    if not task_files:
        raise ValueError(f"no task CSV files found in {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_rows: list[dict[str, Any]] = []
    manifest_records: list[dict[str, Any]] = []
    for csv_path in task_files:
        task_id = _task_id_from_path(csv_path)
        task_output = output_dir / f"task-{task_id}"
        task_output.mkdir(parents=True, exist_ok=True)
        seen_hashes: set[str] = set()
        written = 0
        parse_errors = 0
        skipped_duplicates = 0
        skipped_no_callable = 0
        skipped_few_paths = 0
        total_rows = 0
        with csv_path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if "code" not in (reader.fieldnames or ()):
                raise ValueError(f"{csv_path} does not contain a 'code' column")
            for row_index, row in enumerate(reader):
                total_rows += 1
                code = _normalize_code(row.get("code", ""))
                code_hash = hashlib.sha256(code.encode("utf-8")).hexdigest()[:16]
                if code_hash in seen_hashes:
                    skipped_duplicates += 1
                    continue
                seen_hashes.add(code_hash)
                try:
                    tree = parse_python_ast_tree(code)
                except SyntaxError:
                    parse_errors += 1
                    continue
                callables = callable_subtrees(tree)
                if not callables:
                    skipped_no_callable += 1
                    continue
                if not any(len(terminal_to_terminal_paths(callable_tree, max_paths=max_paths)) >= min_paths for callable_tree in callables):
                    skipped_few_paths += 1
                    continue
                output_path = task_output / f"task_{task_id}_{written:04d}_{code_hash}.py"
                output_path.write_text(code + "\n", encoding="utf-8")
                manifest_records.append(
                    {
                        "task": task_id,
                        "source_csv": str(csv_path),
                        "row_index": row_index,
                        "code_sha256_16": code_hash,
                        "path": str(output_path),
                    }
                )
                written += 1
                if written >= max_records_per_task:
                    break
        summary_rows.append(
            {
                "task": task_id,
                "source_csv": str(csv_path),
                "total_rows_scanned": total_rows,
                "written": written,
                "parse_errors": parse_errors,
                "skipped_duplicates": skipped_duplicates,
                "skipped_no_callable": skipped_no_callable,
                "skipped_few_paths": skipped_few_paths,
            }
        )

    payload = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "max_records_per_task": max_records_per_task,
        "min_paths": min_paths,
        "max_paths": max_paths,
        "tasks": [row["task"] for row in summary_rows],
        "summary": summary_rows,
        "files": manifest_records,
    }
    (output_dir / "manifest.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return payload


def _normalize_task(task: str) -> str:
    value = task.strip()
    if value.startswith("task-"):
        value = value.removeprefix("task-")
    return f"{int(value):02d}"


def _task_id_from_path(path: Path) -> str:
    return _normalize_task(path.stem.removeprefix("task-"))


def _normalize_code(code: str) -> str:
    return code.replace("\r\n", "\n").replace("\r", "\n").strip()


def _parse_tasks(value: str | None) -> tuple[str, ...] | None:
    if value is None or not value.strip():
        return None
    return tuple(item.strip() for item in value.split(",") if item.strip())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Materialize DTA Python CSV snippets as .py source files.")
    parser.add_argument("--input-dir", type=Path, default=Path("data/dta_zenodo_7799972/extracted"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/dta_zenodo_7799972/materialized_python"))
    parser.add_argument("--tasks", type=str, default=None, help="Comma-separated task ids, for example: 00,01,08")
    parser.add_argument("--max-records-per-task", type=int, default=128)
    parser.add_argument("--min-paths", type=int, default=2)
    parser.add_argument("--max-paths", type=int, default=16)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    payload = materialize_dta_python_sources(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        tasks=_parse_tasks(args.tasks),
        max_records_per_task=args.max_records_per_task,
        min_paths=args.min_paths,
        max_paths=args.max_paths,
    )
    total = sum(int(row["written"]) for row in payload["summary"])
    print(f"wrote {total} Python files to {args.output_dir}")
    for row in payload["summary"]:
        print(f"task-{row['task']}: written={row['written']} scanned={row['total_rows_scanned']}")


if __name__ == "__main__":
    main()
