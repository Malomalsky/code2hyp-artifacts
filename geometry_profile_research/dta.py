from __future__ import annotations

import csv
import random
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


_TASK_FILE_PATTERN = re.compile(r"task-(\d+)\.csv$")


@dataclass(frozen=True)
class DtaRecord:
    """Single accepted program from the Digital Teaching Assistant dataset."""

    record_id: str
    task_id: int
    row_index: int
    source_file: str
    code: str


def _task_id_from_path(path: Path) -> int | None:
    match = _TASK_FILE_PATTERN.match(path.name)
    if not match:
        return None
    return int(match.group(1))


def iter_task_files(dataset_dir: Path) -> list[Path]:
    """Return DTA `task-XX.csv` files in task-id order."""
    task_files = [
        path
        for path in Path(dataset_dir).glob("task-*.csv")
        if _task_id_from_path(path) is not None
    ]
    return sorted(task_files, key=lambda path: _task_id_from_path(path) or -1)


def load_dta_records(
    dataset_dir: Path,
    *,
    limit_per_task: int | None = None,
) -> list[DtaRecord]:
    """Load accepted source-code records from DTA task CSV files.

    The published dataset stores one normalized Python program per row in the
    `code` column. The task identifier is encoded in file names `task-00.csv`
    ... `task-10.csv`, so the loader keeps it as the supervised label for
    classification and retrieval experiments.
    """
    records: list[DtaRecord] = []
    for path in iter_task_files(Path(dataset_dir)):
        task_id = _task_id_from_path(path)
        if task_id is None:
            continue

        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if "code" not in (reader.fieldnames or []):
                raise ValueError(f"{path} does not contain required 'code' column")

            kept_for_task = 0
            for row_index, row in enumerate(reader):
                code = (row.get("code") or "").strip()
                if not code:
                    continue
                if limit_per_task is not None and kept_for_task >= limit_per_task:
                    break

                records.append(
                    DtaRecord(
                        record_id=f"task-{task_id:02d}:{row_index}",
                        task_id=task_id,
                        row_index=row_index,
                        source_file=path.name,
                        code=code,
                    )
                )
                kept_for_task += 1

    return records


def stratified_sample_records(
    records: list[DtaRecord],
    *,
    per_task: int,
    seed: int,
) -> list[DtaRecord]:
    """Return a deterministic balanced sample from each DTA task group."""
    if per_task <= 0:
        raise ValueError("per_task must be positive")

    rng = random.Random(seed)
    by_task: defaultdict[int, list[DtaRecord]] = defaultdict(list)
    for record in records:
        by_task[record.task_id].append(record)

    sampled: list[DtaRecord] = []
    for task_id in sorted(by_task):
        task_records = sorted(by_task[task_id], key=lambda record: record.row_index)
        if len(task_records) <= per_task:
            selected = task_records
        else:
            selected = rng.sample(task_records, per_task)
            selected.sort(key=lambda record: record.row_index)
        sampled.extend(selected)
    return sampled


def stratified_validation_test_split(
    records: list[DtaRecord],
    *,
    validation_per_task: int,
    test_per_task: int,
    seed: int,
) -> dict[str, list[DtaRecord]]:
    """Build disjoint deterministic validation/test splits per DTA task."""
    if validation_per_task <= 0:
        raise ValueError("validation_per_task must be positive")
    if test_per_task <= 0:
        raise ValueError("test_per_task must be positive")

    rng = random.Random(seed)
    by_task: defaultdict[int, list[DtaRecord]] = defaultdict(list)
    for record in records:
        by_task[record.task_id].append(record)

    validation: list[DtaRecord] = []
    test: list[DtaRecord] = []
    required = validation_per_task + test_per_task
    for task_id in sorted(by_task):
        task_records = sorted(by_task[task_id], key=lambda record: record.row_index)
        if len(task_records) < required:
            raise ValueError(
                f"task {task_id} has {len(task_records)} records, "
                f"but {required} are required"
            )
        selected = rng.sample(task_records, required)
        validation_selected = sorted(
            selected[:validation_per_task],
            key=lambda record: record.row_index,
        )
        test_selected = sorted(
            selected[validation_per_task:],
            key=lambda record: record.row_index,
        )
        validation.extend(validation_selected)
        test.extend(test_selected)

    return {"validation": validation, "test": test}
