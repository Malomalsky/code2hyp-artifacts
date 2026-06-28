from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from geometry_profile_research.java_raw_ast import parse_java_ast_tree
from geometry_profile_research.python_raw_ast import parse_python_ast_tree
from geometry_profile_research.raw_ast import terminal_to_terminal_paths
from geometry_profile_research.raw_ast_retrieval import ItemScope, retrieval_item_trees


Language = Literal["python", "java"]
Layout = Literal["generic", "codenet", "poj104", "table"]


@dataclass(frozen=True)
class CandidateFile:
    task_label: str
    path: Path
    record_id: str | None = None
    source: str | None = None


def materialize_task_labeled_corpus(
    *,
    input_root: Path,
    output_dir: Path,
    layout: Layout = "generic",
    language: Language = "python",
    max_tasks: int | None = None,
    max_files_per_task: int = 64,
    min_files_per_task: int = 24,
    min_paths: int = 2,
    max_paths: int = 16,
    seed: int = 20260625,
    validate_parse: bool = True,
    table_code_column: str = "code",
    table_label_column: str = "label",
    table_id_column: str | None = None,
    item_scope: ItemScope = "callable",
) -> dict[str, Any]:
    """Create a deterministic task-directory view of an external code corpus.

    Supported layouts:
    - generic: input_root/<task>/**/*.py|java
    - codenet: Project_CodeNet/data/<problem>/<language-subdir>/**/*.py|java
    - poj104: POJ-104/<problem>/**/*.java, or nested train/test folders.
    - table: CSV/JSONL file(s) with code and task-label columns.

    The output is intentionally compatible with scripts/run_dta_factor_matrix.py:
    every selected task is represented as output_dir/task-XXX_<label>/source files.
    """

    if max_files_per_task <= 0:
        raise ValueError("max_files_per_task must be positive")
    if min_files_per_task <= 0:
        raise ValueError("min_files_per_task must be positive")
    if min_files_per_task > max_files_per_task:
        raise ValueError("min_files_per_task cannot exceed max_files_per_task")
    if min_paths <= 0:
        raise ValueError("min_paths must be positive")
    if not input_root.exists():
        raise FileNotFoundError(input_root)

    rng = random.Random(seed)
    grouped = _discover_candidates(
        input_root=input_root,
        layout=layout,
        language=language,
        table_code_column=table_code_column,
        table_label_column=table_label_column,
        table_id_column=table_id_column,
    )
    if not grouped:
        raise ValueError(f"no {language} files found for layout={layout!r} under {input_root}")

    output_dir.mkdir(parents=True, exist_ok=True)
    task_records: list[dict[str, Any]] = []
    file_records: list[dict[str, Any]] = []
    skipped_tasks: list[dict[str, Any]] = []
    selected_task_count = 0
    for task_index, task_label in enumerate(sorted(grouped)):
        candidates = sorted(grouped[task_label], key=lambda candidate: (str(candidate.path), candidate.record_id or ""))
        rng.shuffle(candidates)
        selected: list[dict[str, Any]] = []
        counters = {
            "seen": 0,
            "parse_errors": 0,
            "skipped_few_paths": 0,
            "skipped_duplicate_content": 0,
        }
        seen_hashes: set[str] = set()
        for candidate in candidates:
            counters["seen"] += 1
            source = _candidate_source(candidate)
            if not source:
                counters["parse_errors"] += 1
                continue
            digest = hashlib.sha256(source.encode("utf-8")).hexdigest()
            if digest in seen_hashes:
                counters["skipped_duplicate_content"] += 1
                continue
            seen_hashes.add(digest)
            if validate_parse and not _has_supported_item(
                source,
                language=language,
                min_paths=min_paths,
                max_paths=max_paths,
                item_scope=item_scope,
            ):
                counters["skipped_few_paths"] += 1
                continue
            selected.append({"source_path": candidate, "sha256": digest, "source": source})
            if len(selected) >= max_files_per_task:
                break
        if len(selected) < min_files_per_task:
            skipped_tasks.append(
                {
                    "task_label": task_label,
                    "candidate_count": len(candidates),
                    "selected_count": len(selected),
                    **counters,
                }
            )
            continue
        safe_label = _safe_label(task_label)
        task_output = output_dir / f"task-{selected_task_count:03d}_{safe_label}"
        task_output.mkdir(parents=True, exist_ok=True)
        extension = _extension_for_language(language)
        task_files: list[str] = []
        for file_index, selected_file in enumerate(selected):
            short_hash = str(selected_file["sha256"])[:16]
            target = task_output / f"{safe_label}_{file_index:04d}_{short_hash}{extension}"
            target.write_text(str(selected_file["source"]) + "\n", encoding="utf-8")
            task_files.append(str(target))
            file_records.append(
                {
                    "task": f"task-{selected_task_count:03d}_{safe_label}",
                    "source_task_label": task_label,
                    "source_path": _candidate_source_id(selected_file["source_path"]),
                    "path": str(target),
                    "sha256": selected_file["sha256"],
                }
            )
        task_records.append(
            {
                "label": f"task-{selected_task_count:03d}_{safe_label}",
                "source_task_label": task_label,
                "path": str(task_output),
                "selected_files": len(task_files),
                "candidate_count": len(candidates),
                **counters,
            }
        )
        selected_task_count += 1
        if max_tasks is not None and selected_task_count >= max_tasks:
            break

    if not task_records:
        raise ValueError("no tasks satisfied min_files_per_task after filtering")
    payload = {
        "experiment_role": "external_task_labeled_corpus_materialization",
        "input_root": str(input_root),
        "output_dir": str(output_dir),
        "layout": layout,
        "language": language,
        "seed": seed,
        "validate_parse": validate_parse,
        "table_code_column": table_code_column if layout == "table" else None,
        "table_label_column": table_label_column if layout == "table" else None,
        "table_id_column": table_id_column if layout == "table" else None,
        "item_scope": item_scope,
        "min_files_per_task": min_files_per_task,
        "max_files_per_task": max_files_per_task,
        "min_paths": min_paths,
        "max_paths": max_paths,
        "task_count": len(task_records),
        "file_count": len(file_records),
        "tasks": task_records,
        "files": file_records,
        "skipped_tasks": skipped_tasks,
        "factor_matrix_task_args": [item for task in task_records for item in ("--task", task["label"], task["path"])],
        "recommended_factor_matrix_command": _factor_matrix_command(task_records, language=language, item_scope=item_scope),
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def _discover_candidates(
    *,
    input_root: Path,
    layout: Layout,
    language: Language,
    table_code_column: str,
    table_label_column: str,
    table_id_column: str | None,
) -> dict[str, list[CandidateFile]]:
    if layout == "generic":
        return _discover_generic(input_root=input_root, language=language)
    if layout == "codenet":
        return _discover_codenet(input_root=input_root, language=language)
    if layout == "poj104":
        return _discover_poj104(input_root=input_root, language=language)
    if layout == "table":
        return _discover_table(
            input_root=input_root,
            code_column=table_code_column,
            label_column=table_label_column,
            id_column=table_id_column,
        )
    raise ValueError(f"unknown layout: {layout!r}")


def _discover_generic(*, input_root: Path, language: Language) -> dict[str, list[CandidateFile]]:
    grouped: dict[str, list[CandidateFile]] = {}
    extension = _extension_for_language(language)
    for task_dir in sorted(path for path in input_root.iterdir() if path.is_dir()):
        files = [CandidateFile(task_label=task_dir.name, path=path) for path in task_dir.rglob(f"*{extension}") if path.is_file()]
        if files:
            grouped[task_dir.name] = files
    return grouped


def _discover_codenet(*, input_root: Path, language: Language) -> dict[str, list[CandidateFile]]:
    data_root = input_root / "data" if (input_root / "data").is_dir() else input_root
    extension = _extension_for_language(language)
    language_dirs = _codenet_language_dirs(language)
    grouped: dict[str, list[CandidateFile]] = {}
    for problem_dir in sorted(path for path in data_root.iterdir() if path.is_dir()):
        files: list[CandidateFile] = []
        for language_dir in language_dirs:
            source_dir = problem_dir / language_dir
            if source_dir.is_dir():
                files.extend(CandidateFile(task_label=problem_dir.name, path=path) for path in source_dir.rglob(f"*{extension}") if path.is_file())
        if files:
            grouped[problem_dir.name] = files
    return grouped


def _discover_poj104(*, input_root: Path, language: Language) -> dict[str, list[CandidateFile]]:
    extension = _extension_for_language(language)
    grouped: dict[str, list[CandidateFile]] = {}
    for path in sorted(input_root.rglob(f"*{extension}")):
        if not path.is_file():
            continue
        task_label = _poj_task_label(input_root, path)
        if task_label:
            grouped.setdefault(task_label, []).append(CandidateFile(task_label=task_label, path=path))
    return grouped


def _discover_table(
    *,
    input_root: Path,
    code_column: str,
    label_column: str,
    id_column: str | None,
) -> dict[str, list[CandidateFile]]:
    grouped: dict[str, list[CandidateFile]] = {}
    table_paths = _table_paths(input_root)
    if not table_paths:
        raise ValueError(f"no .csv or .jsonl files found under {input_root}")
    for table_path in table_paths:
        for row_index, row in enumerate(_iter_table_rows(table_path)):
            code = str(row.get(code_column, "") or "").strip()
            label = str(row.get(label_column, "") or "").strip()
            if not code or not label:
                continue
            record_id = str(row.get(id_column, "")).strip() if id_column else str(row_index)
            grouped.setdefault(label, []).append(
                CandidateFile(
                    task_label=label,
                    path=table_path,
                    record_id=record_id or str(row_index),
                    source=code,
                )
            )
    return grouped


def _table_paths(input_root: Path) -> list[Path]:
    if input_root.is_file():
        return [input_root] if input_root.suffix.lower() in {".csv", ".jsonl", ".json"} else []
    return sorted(
        path
        for suffix in ("*.csv", "*.jsonl", "*.json")
        for path in input_root.rglob(suffix)
        if path.is_file()
    )


def _iter_table_rows(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        with path.open(newline="", encoding="utf-8") as handle:
            return list(csv.DictReader(handle))
    if suffix == ".json":
        parsed = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(parsed, list) or not all(isinstance(item, dict) for item in parsed):
            raise ValueError(f"{path} must contain a JSON array of objects")
        return parsed
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            value = line.strip()
            if not value:
                continue
            parsed = json.loads(value)
            if not isinstance(parsed, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            rows.append(parsed)
    return rows


def _poj_task_label(input_root: Path, file_path: Path) -> str:
    relative = file_path.relative_to(input_root)
    parts = relative.parts
    for part in parts[:-1]:
        if re.fullmatch(r"\d{1,4}|problem-\d+|class-\d+|p\d+", part, flags=re.IGNORECASE):
            return part
    return parts[0] if len(parts) > 1 else ""


def _has_supported_item(
    source: str,
    *,
    language: Language,
    min_paths: int,
    max_paths: int,
    item_scope: ItemScope,
) -> bool:
    try:
        tree = parse_python_ast_tree(source) if language == "python" else parse_java_ast_tree(source)
    except Exception:
        return False
    return any(
        len(terminal_to_terminal_paths(subtree, max_paths=max_paths)) >= min_paths
        for subtree in retrieval_item_trees(tree, item_scope=item_scope)
    )


def _candidate_source(candidate: CandidateFile) -> str:
    if candidate.source is not None:
        return candidate.source.replace("\r\n", "\n").replace("\r", "\n").strip()
    return candidate.path.read_text(encoding="utf-8", errors="ignore").replace("\r\n", "\n").replace("\r", "\n").strip()


def _candidate_source_id(candidate: CandidateFile) -> str:
    if candidate.record_id is None:
        return str(candidate.path)
    return f"{candidate.path}#{candidate.record_id}"


def _extension_for_language(language: Language) -> str:
    return ".py" if language == "python" else ".java"


def _codenet_language_dirs(language: Language) -> tuple[str, ...]:
    if language == "python":
        return ("Python", "Python3", "PyPy", "PyPy3")
    return ("Java",)


def _safe_label(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip()).strip("._") or "task"


def _factor_matrix_command(tasks: Sequence[dict[str, Any]], *, language: Language, item_scope: ItemScope) -> str:
    task_args = " ".join(f"--task {task['label']} {shlex_quote(task['path'])}" for task in tasks)
    return (
        "python scripts/run_dta_factor_matrix.py "
        f"{task_args} --language {language} --benchmark-level B_independent_solution "
        f"--item-scope {item_scope} "
        "--output outputs/external_code2hyp_factor_matrix.json"
    )


def shlex_quote(value: str) -> str:
    if re.fullmatch(r"[A-Za-z0-9_./:@%+=,-]+", value):
        return value
    return "'" + value.replace("'", "'\"'\"'") + "'"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Materialize a task-labeled code corpus for Code2Hyp factor-matrix experiments.")
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--layout", choices=("generic", "codenet", "poj104", "table"), default="generic")
    parser.add_argument("--language", choices=("python", "java"), default="python")
    parser.add_argument("--table-code-column", default="code")
    parser.add_argument("--table-label-column", default="label")
    parser.add_argument("--table-id-column", default=None)
    parser.add_argument(
        "--item-scope",
        choices=("callable", "module", "callable_or_module"),
        default="callable",
        help="Retrieval unit used for parse/path validation.",
    )
    parser.add_argument("--max-tasks", type=int, default=None)
    parser.add_argument("--max-files-per-task", type=int, default=64)
    parser.add_argument("--min-files-per-task", type=int, default=24)
    parser.add_argument("--min-paths", type=int, default=2)
    parser.add_argument("--max-paths", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260625)
    parser.add_argument("--no-validate-parse", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    payload = materialize_task_labeled_corpus(
        input_root=args.input_root,
        output_dir=args.output_dir,
        layout=args.layout,
        language=args.language,
        max_tasks=args.max_tasks,
        max_files_per_task=args.max_files_per_task,
        min_files_per_task=args.min_files_per_task,
        min_paths=args.min_paths,
        max_paths=args.max_paths,
        seed=args.seed,
        validate_parse=not args.no_validate_parse,
        table_code_column=args.table_code_column,
        table_label_column=args.table_label_column,
        table_id_column=args.table_id_column,
        item_scope=args.item_scope,
    )
    print(f"materialized tasks={payload['task_count']} files={payload['file_count']} manifest={args.output_dir / 'manifest.json'}")
    print(payload["recommended_factor_matrix_command"])


if __name__ == "__main__":
    main()
