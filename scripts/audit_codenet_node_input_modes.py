from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from geometry_profile_research.codenet_eligibility import (  # noqa: E402
    canonical_json_bytes,
    normalize_python_source,
    stable_sha256,
)
from geometry_profile_research.codenet_stage_a_identifiability import (  # noqa: E402
    program_identifiability_diagnostics,
    summarize_identifiability_diagnostics,
)
from geometry_profile_research.python_raw_ast import parse_python_ast_tree  # noqa: E402


NODE_INPUT_MODES = ("label_only", "label_depth", "label_depth_prefix")


def audit_node_input_modes(
    *,
    train_path: Path,
    source_root: Path,
    output_path: Path,
    progress_every: int = 500,
) -> dict[str, Any]:
    """Compare encoder-input identifiability on the fixed CodeNet train split."""

    train_bytes = train_path.read_bytes()
    source_root = source_root.resolve()
    rows: dict[str, list[dict[str, float | int]]] = {mode: [] for mode in NODE_INPUT_MODES}
    source_digest = hashlib.sha256()
    for index, sample in enumerate(_iter_jsonl(train_path), start=1):
        if sample.get("split") != "train" or sample.get("role") != "train":
            raise ValueError("node-input audit encountered a non-training program")
        source_relpath = str(sample["source_relpath"])
        relative = Path(source_relpath)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"unsafe source path: {source_relpath!r}")
        source_path = (source_root / relative).resolve()
        try:
            source_path.relative_to(source_root)
        except ValueError as error:
            raise ValueError(f"source path escapes source root: {source_relpath!r}") from error
        raw = source_path.read_bytes()
        source_digest.update(source_relpath.encode("utf-8"))
        source_digest.update(b"\0")
        source_digest.update(hashlib.sha256(raw).digest())
        canonical = normalize_python_source(raw)
        if not canonical.decode_ok:
            raise ValueError(f"training source decode failed: {source_relpath}")
        tree = parse_python_ast_tree(canonical.text)
        for mode in NODE_INPUT_MODES:
            rows[mode].append(
                program_identifiability_diagnostics(
                    tree,
                    terminal_policy="class",
                    node_input_mode=mode,
                    path_selection_policy="lca_depth_affine_sampled",
                    max_paths=64,
                )
            )
        if progress_every > 0 and index % progress_every == 0:
            print(json.dumps({"phase": "node_input_audit", "programs_observed": index}), flush=True)

    summaries = {
        mode: summarize_identifiability_diagnostics(mode_rows)
        for mode, mode_rows in rows.items()
    }
    program_count = summaries["label_only"]["program_count"]
    if any(summary["program_count"] != program_count for summary in summaries.values()):
        raise AssertionError("node-input modes were not evaluated on identical programs")
    prefix_counts = summaries["label_depth_prefix"]["micro_counts"]
    if any(prefix_counts[field] != 0 for field in (
        "colliding_node_count",
        "colliding_true_lca_node_count",
        "colliding_path_object_count",
    )):
        raise AssertionError("label_depth_prefix violates the declared injectivity check")

    payload = {
        "schema_version": "code2hyp-codenet-node-input-identifiability-v1",
        "status": "descriptive_train_only_implementation_audit",
        "inputs": {
            "train_programs_sha256": stable_sha256(train_bytes),
            "ordered_raw_training_source_digest": source_digest.hexdigest(),
        },
        "protocol": {
            "node_input_modes": list(NODE_INPUT_MODES),
            "terminal_policy": "class",
            "path_selection_policy": "lca_depth_affine_sampled",
            "maximum_path_count": 64,
            "endpoint_orientation": "unoriented",
        },
        "summaries": summaries,
        "interpretation": {
            "label_depth_prefix_injectivity_check_passed": True,
            "rates_measure_input_equivalence_not_embedding_quality": True,
            "does_not_test_retrieval_or_geometry": True,
            "does_not_modify_stage_a": True,
        },
    }
    content = canonical_json_bytes(payload)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and output_path.read_bytes() != content:
        raise ValueError(f"refusing to overwrite a different audit result: {output_path}")
    output_path.write_bytes(content)
    return payload


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSONL at {path}:{line_number}") from error


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare CodeNet AST node-input identifiability modes.")
    parser.add_argument("--train-programs", required=True, type=Path)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "reports/codenet_python800_node_input_identifiability_v1.json",
    )
    parser.add_argument("--progress-every", type=int, default=500)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = audit_node_input_modes(
        train_path=args.train_programs,
        source_root=args.source_root,
        output_path=args.output,
        progress_every=args.progress_every,
    )
    print(json.dumps({
        mode: summary["micro_rates"]
        for mode, summary in result["summaries"].items()
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
