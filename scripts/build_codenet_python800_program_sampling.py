from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from geometry_profile_research.codenet_sampling import build_program_sampling_artifacts


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Materialize registered CodeNet train/validation programs while keeping test IDs sealed."
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        default=PROJECT_ROOT / "configs/codenet_python800_stage_a_sampling_protocol_v1.json",
    )
    parser.add_argument(
        "--design",
        type=Path,
        default=PROJECT_ROOT / "configs/codenet_python800_stage_a_draft.json",
    )
    parser.add_argument(
        "--registration",
        type=Path,
        default=PROJECT_ROOT / "registrations/codenet_python800_stage_a_registration_v1.json",
    )
    parser.add_argument(
        "--split-manifest",
        type=Path,
        default=PROJECT_ROOT / "data/codenet_python800_stage_a_split/split_manifest.json",
    )
    parser.add_argument(
        "--assignments",
        type=Path,
        default=PROJECT_ROOT / "data/codenet_python800_stage_a_split/cluster_assignments.jsonl",
    )
    parser.add_argument(
        "--d5-manifest",
        type=Path,
        default=PROJECT_ROOT / "data/codenet_python800_d5_metadata/d5_metadata_manifest.json",
    )
    parser.add_argument(
        "--d5-index",
        type=Path,
        default=PROJECT_ROOT / "data/codenet_python800_d5_metadata/d5_metadata_index.jsonl",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data/codenet_python800_stage_a_program_sampling",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    manifest = build_program_sampling_artifacts(
        project_root=PROJECT_ROOT,
        protocol_path=args.protocol,
        design_path=args.design,
        registration_path=args.registration,
        split_manifest_path=args.split_manifest,
        assignments_path=args.assignments,
        d5_manifest_path=args.d5_manifest,
        d5_index_path=args.d5_index,
        output_dir=args.output_dir,
    )
    print(json.dumps(manifest["summary"], ensure_ascii=False, indent=2, sort_keys=True))
    print(f"manifest={args.output_dir / 'program_sampling_manifest.json'}")


if __name__ == "__main__":
    main()
