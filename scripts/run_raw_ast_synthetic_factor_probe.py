from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from geometry_profile_research.raw_ast_synthetic_geometry import synthetic_path_distortion_rows


def run_synthetic_factor_probe(
    *,
    output_json: Path,
    output_csv: Path | None = None,
    dims: tuple[int, ...] = (2, 4, 8),
    poincare_curvatures: tuple[float, ...] = (1e-4, 1.0),
    steps: int = 300,
    learning_rate: float = 0.05,
    max_paths: int = 48,
    seed: int = 20260624,
) -> dict[str, Any]:
    rows = synthetic_path_distortion_rows(
        dims=dims,
        poincare_curvatures=poincare_curvatures,
        steps=steps,
        learning_rate=learning_rate,
        max_paths=max_paths,
        seed=seed,
    )
    payload: dict[str, Any] = {
        "experiment": "raw_ast_synthetic_factor_probe",
        "description": (
            "Mechanistic metric-distortion probe over synthetic tree families: "
            "chain-like comb, star, balanced tree, repeated labels, and product-like tree."
        ),
        "config": {
            "dims": list(dims),
            "poincare_curvatures": list(poincare_curvatures),
            "steps": steps,
            "learning_rate": learning_rate,
            "max_paths": max_paths,
            "seed": seed,
        },
        "rows": rows,
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if output_csv is not None:
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        _write_csv(output_csv, rows)
    return payload


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _parse_csv_ints(value: str) -> tuple[int, ...]:
    return tuple(int(part.strip()) for part in value.split(",") if part.strip())


def _parse_csv_floats(value: str) -> tuple[float, ...]:
    return tuple(float(part.strip()) for part in value.split(",") if part.strip())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run synthetic raw-AST Code2Hyp factor probes.")
    parser.add_argument("--output-json", type=Path, default=Path("outputs/raw_ast_synthetic_factor_probe.json"))
    parser.add_argument("--output-csv", type=Path, default=Path("outputs/raw_ast_synthetic_factor_probe.csv"))
    parser.add_argument("--dims", default="2,4,8")
    parser.add_argument("--poincare-curvatures", default="1e-4,1.0")
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--max-paths", type=int, default=48)
    parser.add_argument("--seed", type=int, default=20260624)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    payload = run_synthetic_factor_probe(
        output_json=args.output_json,
        output_csv=args.output_csv,
        dims=_parse_csv_ints(args.dims),
        poincare_curvatures=_parse_csv_floats(args.poincare_curvatures),
        steps=args.steps,
        learning_rate=args.learning_rate,
        max_paths=args.max_paths,
        seed=args.seed,
    )
    print(f"wrote {args.output_json} rows={len(payload['rows'])}")


if __name__ == "__main__":
    main()
