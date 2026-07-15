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

from geometry_profile_research.codenet_eligibility import (
    canonical_json_bytes,
    normalize_python_source,
    stable_sha256,
)
from geometry_profile_research.codenet_stage_a_identifiability import (
    program_identifiability_diagnostics,
    summarize_identifiability_diagnostics,
)
from geometry_profile_research.python_raw_ast import parse_python_ast_tree


def run_identifiability_audit(
    *,
    protocol_path: Path,
    model_protocol_path: Path,
    train_path: Path,
    source_root: Path,
    output_path: Path,
    progress_every: int = 500,
) -> dict[str, Any]:
    """Run the frozen train-only representation audit."""

    audit_protocol_bytes = protocol_path.read_bytes()
    model_protocol_bytes = model_protocol_path.read_bytes()
    train_bytes = train_path.read_bytes()
    audit_protocol = json.loads(audit_protocol_bytes)
    if audit_protocol.get("status") != "frozen_before_the_first_validation_retrieval_metric":
        raise ValueError("identifiability audit protocol is not frozen")
    expected_model = audit_protocol["inputs"]["model_analysis_protocol"]["sha256"]
    expected_train = audit_protocol["inputs"]["train_programs"]["sha256"]
    if stable_sha256(model_protocol_bytes) != expected_model:
        raise ValueError("model-analysis protocol differs from the audit protocol")
    if stable_sha256(train_bytes) != expected_train:
        raise ValueError("training manifest differs from the audit protocol")

    representation = audit_protocol["representation"]
    source_root = source_root.resolve()
    rows = []
    source_digest = hashlib.sha256()
    for index, sample in enumerate(_iter_jsonl(train_path), start=1):
        if sample.get("split") != "train" or sample.get("role") != "train":
            raise ValueError("identifiability audit encountered a non-training program")
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
        rows.append(
            program_identifiability_diagnostics(
                tree,
                terminal_policy=str(representation["terminal_policy"]),
                node_input_mode=str(representation["node_input_mode"]),
                path_selection_policy=str(representation["path_selection_policy"]),
                max_paths=int(representation["maximum_path_count"]),
            )
        )
        if progress_every > 0 and index % progress_every == 0:
            print(json.dumps({"phase": "train_only_identifiability", "programs_observed": index}), flush=True)

    summary = summarize_identifiability_diagnostics(
        rows,
        quantiles=tuple(float(value) for value in audit_protocol["aggregation"]["program_distribution_quantiles"]),
    )
    if summary["program_count"] != 18_560:
        raise ValueError(f"expected 18,560 training programs, observed {summary['program_count']}")
    payload = {
        "schema_version": "code2hyp-stage-a-representation-identifiability-result-v1",
        "experiment_role": "descriptive_train_only_scope_audit",
        "inputs": {
            "audit_protocol_sha256": stable_sha256(audit_protocol_bytes),
            "model_analysis_protocol_sha256": stable_sha256(model_protocol_bytes),
            "train_programs_sha256": stable_sha256(train_bytes),
            "ordered_raw_training_source_digest": source_digest.hexdigest(),
        },
        "summary": summary,
        "interpretation": {
            "rates_measure_encoder_input_equivalence_not_learned_embedding_quality": True,
            "does_not_modify_validation_selection": True,
            "inference_status": "descriptive_only",
        },
        "validation_programs_read": False,
        "validation_relevance_labels_read": False,
        "test_program_ids_materialized": False,
        "test_relevance_labels_opened": False,
        "test_retrieval_metrics_computed": False,
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
    parser = argparse.ArgumentParser(description="Audit Stage A label-only representation identifiability.")
    parser.add_argument(
        "--protocol",
        type=Path,
        default=PROJECT_ROOT / "configs/codenet_python800_stage_a_representation_identifiability_audit_v1.json",
    )
    parser.add_argument(
        "--model-protocol",
        type=Path,
        default=PROJECT_ROOT / "configs/codenet_python800_stage_a_model_analysis_protocol_v1.json",
    )
    parser.add_argument(
        "--train-programs",
        type=Path,
        default=PROJECT_ROOT / "data/codenet_python800_stage_a_program_sampling/train_programs.jsonl",
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=PROJECT_ROOT / "data/external_raw/codenet_python800_extracted/Project_CodeNet_Python800",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "reports/codenet_python800_stage_a_representation_identifiability_v1.json",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = run_identifiability_audit(
        protocol_path=args.protocol,
        model_protocol_path=args.model_protocol,
        train_path=args.train_programs,
        source_root=args.source_root,
        output_path=args.output,
    )
    print(json.dumps(result["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
