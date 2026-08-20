from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from geometry_profile_research.codenet_stage_b import build_stage_b_split_artifacts


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the registered CodeNet Java Stage B cluster split.")
    parser.add_argument("--design", type=Path, default=PROJECT_ROOT / "configs/codenet_java_stage_b_draft_v1.json")
    parser.add_argument("--registration", type=Path, default=PROJECT_ROOT / "registrations/codenet_java_stage_b_registration_v1.json")
    parser.add_argument("--clusters", type=Path, default=PROJECT_ROOT / "data/codenet_java_stage_b_eligibility_d4_train32_v2/post_d4_problem_clusters.jsonl")
    parser.add_argument("--d4-manifest", type=Path, default=PROJECT_ROOT / "data/codenet_java_stage_b_eligibility_d4_train32_v2/d4_manifest.json")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "data/codenet_java_stage_b_split_v1")
    args = parser.parse_args()
    manifest = build_stage_b_split_artifacts(
        project_root=PROJECT_ROOT,
        design_path=args.design,
        registration_path=args.registration,
        clusters_path=args.clusters,
        d4_manifest_path=args.d4_manifest,
        output_dir=args.output_dir,
    )
    print(json.dumps(manifest["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
