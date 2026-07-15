from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from geometry_profile_research.codenet_ast_audit import audit_selected_sources


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit every frozen CodeNet Stage A train/validation source and AST path sample."
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        default=PROJECT_ROOT / "configs/codenet_python800_stage_a_ast_path_protocol_v1.json",
    )
    parser.add_argument(
        "--train",
        type=Path,
        default=PROJECT_ROOT / "data/codenet_python800_stage_a_program_sampling/train_programs.jsonl",
    )
    parser.add_argument(
        "--validation",
        type=Path,
        default=PROJECT_ROOT / "data/codenet_python800_stage_a_program_sampling/validation_programs.jsonl",
    )
    parser.add_argument(
        "--sampling-manifest",
        type=Path,
        default=PROJECT_ROOT / "data/codenet_python800_stage_a_program_sampling/program_sampling_manifest.json",
    )
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data/codenet_python800_stage_a_selected_source_ast",
    )
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    manifest = audit_selected_sources(
        project_root=PROJECT_ROOT,
        protocol_path=args.protocol,
        train_path=args.train,
        validation_path=args.validation,
        sampling_manifest_path=args.sampling_manifest,
        source_root=args.source_root,
        output_dir=args.output_dir,
        workers=args.workers,
    )
    print(json.dumps(manifest["summary"], ensure_ascii=False, indent=2, sort_keys=True))
    print(f"manifest={args.output_dir / 'selected_source_ast_manifest.json'}")
    if not manifest["valid_for_stage_a_modeling"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
