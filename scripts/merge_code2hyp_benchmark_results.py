from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any


DATASET_COMPARABILITY_KEYS = (
    "train_path",
    "validation_path",
    "train_records",
    "validation_records_loaded",
    "validation_records_after_known_target_filter",
    "target_subtoken_vocab_size",
    "token_vocab_size",
    "ast_node_vocab_size",
    "path_encoder",
    "representation_transform",
    "lexical_ablation",
)

TRAINING_COMPARABILITY_KEYS = (
    "epochs",
    "batch_size",
    "learning_rate",
    "use_positive_weighting",
    "max_positive_weight",
    "curvature",
    "metric",
)

EVALUATION_COMPARABILITY_KEYS = ("split",)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path}: invalid JSON: {exc}") from exc


def _check_section(
    base: dict[str, Any],
    other: dict[str, Any],
    *,
    section: str,
    keys: tuple[str, ...],
    other_path: Path,
) -> None:
    base_section = base.get(section, {})
    other_section = other.get(section, {})
    for key in keys:
        base_value = base_section.get(key)
        other_value = other_section.get(key)
        if base_value != other_value:
            raise ValueError(
                f"{other_path}: incompatible {section}.{key}: "
                f"{other_value!r} != base {base_value!r}"
            )


def _check_comparable(base: dict[str, Any], other: dict[str, Any], other_path: Path) -> None:
    _check_section(base, other, section="dataset", keys=DATASET_COMPARABILITY_KEYS, other_path=other_path)
    _check_section(base, other, section="training", keys=TRAINING_COMPARABILITY_KEYS, other_path=other_path)
    _check_section(base, other, section="evaluation", keys=EVALUATION_COMPARABILITY_KEYS, other_path=other_path)


def _sorted_unique(values: list[Any]) -> list[Any]:
    return sorted(set(values))


def merge_results(paths: list[Path], *, allow_duplicate_runs: bool = False) -> dict[str, Any]:
    if not paths:
        raise ValueError("At least one input path is required")

    loaded = [(path, _read_json(path)) for path in paths]
    base_path, base = loaded[0]
    merged = deepcopy(base)
    merged_runs: list[dict[str, Any]] = []
    seen_runs: set[tuple[str, int]] = set()

    for path, result in loaded:
        _check_comparable(base, result, path)
        for run in result.get("runs", []):
            run_key = (str(run.get("variant")), int(run.get("model_seed")))
            if run_key in seen_runs and not allow_duplicate_runs:
                raise ValueError(
                    f"{path}: duplicate run {run_key}; use --allow-duplicate-runs only "
                    "when intentionally preserving repeated measurements"
                )
            seen_runs.add(run_key)
            merged_runs.append(run)

    merged["runs"] = merged_runs
    merged["experiment"] = "merged_code2hyp_benchmark"
    merged.setdefault("merge", {})
    merged["merge"] = {
        "input_files": [str(path) for path, _ in loaded],
        "base_file": str(base_path),
        "n_runs": len(merged_runs),
        "allow_duplicate_runs": allow_duplicate_runs,
    }

    training = merged.setdefault("training", {})
    training["model_seeds"] = _sorted_unique([int(run["model_seed"]) for run in merged_runs])
    training["variant_filter"] = _sorted_unique([str(run["variant"]) for run in merged_runs])

    return merged


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge comparable Code2Hyp benchmark JSON files.")
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--allow-duplicate-runs",
        action="store_true",
        help="Preserve duplicate (variant, model_seed) runs instead of failing.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        merged = merge_results(args.inputs, allow_duplicate_runs=args.allow_duplicate_runs)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(merged, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {args.output} runs={len(merged.get('runs', []))}")


if __name__ == "__main__":
    main()
