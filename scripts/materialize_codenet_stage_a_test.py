from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEST_RUNNER_TAG = "codenet-stage-a-test-runner-v4"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from geometry_profile_research.codenet_stage_a_test import (  # noqa: E402
    materialize_and_audit_test_programs,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Perform the single registered CodeNet Stage A test opening and AST audit."
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        default=PROJECT_ROOT / "configs/codenet_python800_stage_a_test_execution_protocol_v1.json",
    )
    parser.add_argument(
        "--validation-selection",
        type=Path,
        default=PROJECT_ROOT / "outputs/codenet_python800_stage_a_validation_v1/validation_selection_record.json",
    )
    parser.add_argument(
        "--validation-selection-seal",
        type=Path,
        default=PROJECT_ROOT
        / "outputs/codenet_python800_stage_a_validation_v1/validation_selection_record_seal.json",
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=PROJECT_ROOT / "data/external_raw/codenet_python800_extracted/Project_CodeNet_Python800",
    )
    parser.add_argument(
        "--d5-index",
        type=Path,
        default=PROJECT_ROOT / "data/codenet_python800_d5_metadata/d5_metadata_index.jsonl",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs/codenet_python800_stage_a_test_v1",
    )
    parser.add_argument("--workers", type=int, default=1)
    return parser


def verified_implementation_state(project_root: Path) -> dict[str, Any]:
    """Require the immutable tagged test implementation before unsealing."""

    def git(*arguments: str) -> str:
        completed = subprocess.run(
            ("git", "-C", str(project_root), *arguments),
            check=True,
            capture_output=True,
            text=True,
        )
        return completed.stdout.strip()

    commit = git("rev-parse", "HEAD")
    if git("status", "--porcelain", "--untracked-files=no"):
        raise ValueError("official Stage A test opening requires a clean tracked worktree")
    tags = tuple(line for line in git("tag", "--points-at", "HEAD").splitlines() if line)
    if TEST_RUNNER_TAG not in tags:
        raise ValueError(f"official Stage A test opening requires tag {TEST_RUNNER_TAG!r}")
    return {
        "repository": "https://github.com/Malomalsky/code2hyp-artifacts",
        "commit": commit,
        "tag": TEST_RUNNER_TAG,
        "tracked_worktree_clean": True,
    }


def main() -> None:
    args = build_parser().parse_args()
    manifest = materialize_and_audit_test_programs(
        project_root=PROJECT_ROOT,
        protocol_path=args.protocol,
        selection_path=args.validation_selection,
        selection_seal_path=args.validation_selection_seal,
        source_root=args.source_root,
        output_dir=args.output_dir,
        implementation=verified_implementation_state(PROJECT_ROOT),
        d5_metadata_index_path=args.d5_index,
        workers=args.workers,
    )
    print(json.dumps(manifest["sampling_summary"], indent=2, sort_keys=True))
    print(f"manifest={args.output_dir / 'test_materialization_manifest.json'}")


if __name__ == "__main__":
    main()
