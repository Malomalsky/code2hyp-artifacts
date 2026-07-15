from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from geometry_profile_research.codenet_split import build_split_artifacts


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build the registered NIST-Beacon Project CodeNet Python800 problem-cluster split."
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
        "--clusters",
        type=Path,
        default=(
            PROJECT_ROOT
            / "data/codenet_python800_eligibility_d4_statements/post_statement_d4_problem_clusters.jsonl"
        ),
    )
    parser.add_argument(
        "--statement-d4-manifest",
        type=Path,
        default=(
            PROJECT_ROOT / "data/codenet_python800_eligibility_d4_statements/statement_d4_manifest.json"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data/codenet_python800_stage_a_split",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    manifest = build_split_artifacts(
        project_root=PROJECT_ROOT,
        design_path=args.design,
        registration_path=args.registration,
        clusters_path=args.clusters,
        statement_d4_manifest_path=args.statement_d4_manifest,
        output_dir=args.output_dir,
    )
    print(json.dumps(manifest["summary"], ensure_ascii=False, indent=2, sort_keys=True))
    print(f"manifest={args.output_dir / 'split_manifest.json'}")


if __name__ == "__main__":
    main()
