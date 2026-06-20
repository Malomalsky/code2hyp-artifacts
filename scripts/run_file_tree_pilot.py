from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from geometry_profile_research.analysis import geometry_profile_for_paths
from geometry_profile_research.io import extract_paths_from_text

SAMPLE_PATH_TEXT = """
src/utils/io.py
src/utils/path.py
src/services/auth/token.py
tests/test_io.py
docs/readme.md
"""


def _read_paths_file(path: Path) -> list[str]:
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]


def _resolve_input_paths(args: argparse.Namespace) -> list[str]:
    if args.paths_file:
        return _read_paths_file(args.paths_file)
    if args.text_file:
        return extract_paths_from_text(args.text_file.read_text(encoding="utf-8"))
    return extract_paths_from_text(SAMPLE_PATH_TEXT)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute a file-tree geometry profile for source-code paths."
    )
    parser.add_argument(
        "--paths-file",
        type=Path,
        help="Text file with one repository path per line.",
    )
    parser.add_argument(
        "--text-file",
        type=Path,
        help="Free-form text file; source-tree paths will be extracted automatically.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/file_tree_pilot.json"),
        help="Output JSON path.",
    )
    parser.add_argument("--beta", type=float, default=0.45)
    parser.add_argument("--gamma", type=float, default=0.0)
    parser.add_argument("--curvature", type=float, default=1.0)
    args = parser.parse_args()

    paths = _resolve_input_paths(args)
    profile = geometry_profile_for_paths(
        paths,
        beta=args.beta,
        gamma=args.gamma,
        curvature=args.curvature,
    )
    payload = {
        "input": {
            "path_count": len(paths),
            "paths": paths,
            "beta": args.beta,
            "gamma": args.gamma,
            "curvature": args.curvature,
        },
        "profile": profile.to_dict(),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"wrote {args.output}")
    print(json.dumps(profile.to_dict(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
