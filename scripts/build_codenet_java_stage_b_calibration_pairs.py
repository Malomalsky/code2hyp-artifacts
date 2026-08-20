from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from geometry_profile_research.codenet_stage_b import build_stage_b_calibration_pair_artifacts


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze train-only Java Stage B calibration pairs.")
    parser.add_argument("--design", type=Path, default=PROJECT_ROOT / "configs/codenet_java_stage_b_draft_v1.json")
    parser.add_argument("--registration", type=Path, default=PROJECT_ROOT / "registrations/codenet_java_stage_b_registration_v1.json")
    parser.add_argument("--sampling-manifest", type=Path, default=PROJECT_ROOT / "data/codenet_java_stage_b_program_sampling_v1/program_sampling_manifest.json")
    parser.add_argument("--train-programs", type=Path, default=PROJECT_ROOT / "data/codenet_java_stage_b_program_sampling_v1/train_programs.jsonl")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "data/codenet_java_stage_b_calibration_pairs_v1")
    args = parser.parse_args()
    manifest = build_stage_b_calibration_pair_artifacts(
        project_root=PROJECT_ROOT,
        design_path=args.design,
        registration_path=args.registration,
        sampling_manifest_path=args.sampling_manifest,
        train_path=args.train_programs,
        output_dir=args.output_dir,
    )
    print(json.dumps(manifest["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
