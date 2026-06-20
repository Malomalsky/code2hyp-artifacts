from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from geometry_profile_research.code2hyp_smoke import write_code2hyp_smoke_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Code2Hyp model smoke check.")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--output", default="outputs/code2hyp_smoke_report.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = write_code2hyp_smoke_report(args.output, seed=args.seed)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
