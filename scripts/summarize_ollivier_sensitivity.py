from __future__ import annotations

import argparse
import csv
from pathlib import Path
from statistics import fmean, pstdev
from typing import Any


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


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


def summarize_ollivier_sensitivity(inputs: list[tuple[float, Path]]) -> list[dict[str, Any]]:
    metrics = [
        "ollivier_mean",
        "ollivier_negative_mass",
        "ollivier_near_zero_mass",
        "ollivier_positive_mass",
    ]
    rows: list[dict[str, Any]] = []
    for alpha, path in inputs:
        records = _read_rows(path)
        row: dict[str, Any] = {
            "ollivier_idleness": alpha,
            "source_file": str(path),
            "n_records": len(records),
        }
        for metric in metrics:
            values = [float(record[metric]) for record in records]
            row[f"{metric}_mean"] = fmean(values)
            row[f"{metric}_std"] = pstdev(values) if len(values) > 1 else 0.0
            row[f"{metric}_min"] = min(values)
            row[f"{metric}_max"] = max(values)
        rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize Ollivier-Ricci idleness sensitivity atlas runs."
    )
    parser.add_argument(
        "--input",
        nargs=2,
        action="append",
        metavar=("ALPHA", "CSV"),
        required=True,
        help="Pair of idleness alpha and atlas CSV path. Repeat for each alpha.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/ollivier_idleness_sensitivity.csv"),
    )
    args = parser.parse_args()

    inputs = [(float(alpha), Path(path)) for alpha, path in args.input]
    rows = summarize_ollivier_sensitivity(inputs)
    _write_csv(args.output, rows)
    print(f"wrote {args.output} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
