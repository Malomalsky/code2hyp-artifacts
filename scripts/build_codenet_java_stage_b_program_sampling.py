from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from geometry_profile_research.codenet_stage_b import build_stage_b_program_sampling_artifacts


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample registered Java Stage B train/validation programs.")
    parser.add_argument("--design", type=Path, default=PROJECT_ROOT / "configs/codenet_java_stage_b_draft_v1.json")
    parser.add_argument("--registration", type=Path, default=PROJECT_ROOT / "registrations/codenet_java_stage_b_registration_v1.json")
    parser.add_argument("--split-manifest", type=Path, default=PROJECT_ROOT / "data/codenet_java_stage_b_split_v1/split_manifest.json")
    parser.add_argument("--assignments", type=Path, default=PROJECT_ROOT / "data/codenet_java_stage_b_split_v1/cluster_assignments.jsonl")
    parser.add_argument("--d5-manifest", type=Path, default=PROJECT_ROOT / "data/codenet_java_stage_b_metadata_v2/d5_metadata_manifest.json")
    parser.add_argument("--d5-index", type=Path, default=PROJECT_ROOT / "data/codenet_java_stage_b_metadata_v2/d5_metadata_index.jsonl")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "data/codenet_java_stage_b_program_sampling_v1")
    args = parser.parse_args()
    manifest = build_stage_b_program_sampling_artifacts(
        project_root=PROJECT_ROOT,
        design_path=args.design,
        registration_path=args.registration,
        split_manifest_path=args.split_manifest,
        assignments_path=args.assignments,
        d5_manifest_path=args.d5_manifest,
        d5_index_path=args.d5_index,
        output_dir=args.output_dir,
    )
    print(json.dumps(manifest["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
