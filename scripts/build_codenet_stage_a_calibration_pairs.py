from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from geometry_profile_research.codenet_stage_a import build_calibration_pair_artifacts


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Materialize the frozen train-only CodeNet Stage A calibration pairs."
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        default=PROJECT_ROOT / "configs/codenet_python800_stage_a_model_analysis_protocol_v1.json",
    )
    parser.add_argument(
        "--registration",
        type=Path,
        default=PROJECT_ROOT / "registrations/codenet_python800_stage_a_registration_v1.json",
    )
    parser.add_argument(
        "--train-programs",
        type=Path,
        default=PROJECT_ROOT / "data/codenet_python800_stage_a_program_sampling/train_programs.jsonl",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data/codenet_python800_stage_a_calibration_pairs",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    manifest = build_calibration_pair_artifacts(
        project_root=PROJECT_ROOT,
        protocol_path=args.protocol,
        registration_path=args.registration,
        train_path=args.train_programs,
        output_dir=args.output_dir,
    )
    print(json.dumps(manifest["summary"], ensure_ascii=False, indent=2, sort_keys=True))
    print(f"manifest={args.output_dir / 'calibration_pair_manifest.json'}")


if __name__ == "__main__":
    main()
