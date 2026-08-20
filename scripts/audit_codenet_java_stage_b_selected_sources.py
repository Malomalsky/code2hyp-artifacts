from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from geometry_profile_research.codenet_ast_audit import audit_stage_b_selected_sources


def main() -> None:
    parser = argparse.ArgumentParser(description="Materialize and re-audit selected Java Stage B sources.")
    parser.add_argument("--design", type=Path, default=PROJECT_ROOT / "configs/codenet_java_stage_b_draft_v1.json")
    parser.add_argument("--sampling-manifest", type=Path, default=PROJECT_ROOT / "data/codenet_java_stage_b_program_sampling_v1/program_sampling_manifest.json")
    parser.add_argument("--train", type=Path, default=PROJECT_ROOT / "data/codenet_java_stage_b_program_sampling_v1/train_programs.jsonl")
    parser.add_argument("--validation", type=Path, default=PROJECT_ROOT / "data/codenet_java_stage_b_program_sampling_v1/validation_programs.jsonl")
    parser.add_argument("--candidate-manifest", type=Path, default=PROJECT_ROOT / "data/codenet_java_stage_b_candidates_v1/manifest.json")
    parser.add_argument("--candidate-archive", type=Path, default=PROJECT_ROOT / "data/codenet_java_stage_b_candidates_v1/accepted_java_sources.tar")
    parser.add_argument("--d0-d2-manifest", type=Path, default=PROJECT_ROOT / "data/codenet_java_stage_b_eligibility_d0_d2_v1/manifest.json")
    parser.add_argument("--d0-d2-inventory", type=Path, default=PROJECT_ROOT / "data/codenet_java_stage_b_eligibility_d0_d2_v1/file_inventory.jsonl")
    parser.add_argument("--source-root", type=Path, default=PROJECT_ROOT / "data/codenet_java_stage_b_sources_v1")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "data/codenet_java_stage_b_selected_source_ast_v1")
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    args = parser.parse_args()
    manifest = audit_stage_b_selected_sources(
        design_path=args.design,
        sampling_manifest_path=args.sampling_manifest,
        train_path=args.train,
        validation_path=args.validation,
        candidate_manifest_path=args.candidate_manifest,
        candidate_archive_path=args.candidate_archive,
        d0_d2_manifest_path=args.d0_d2_manifest,
        d0_d2_inventory_path=args.d0_d2_inventory,
        source_root=args.source_root,
        output_dir=args.output_dir,
        workers=args.workers,
    )
    print(json.dumps(manifest["summary"], indent=2, sort_keys=True))
    if not manifest["valid_for_stage_b_modeling"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
