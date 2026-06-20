from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from .code2hyp_registry import (
    Code2HypVariantMetadata,
    available_profiles,
    format_variant_catalog,
    variant_catalog,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="code2hyp",
        description="Code2Hyp research and reproducibility command line interface.",
    )
    subparsers = parser.add_subparsers(dest="command")

    variants = subparsers.add_parser(
        "variants",
        help="Print supported Code2Hyp variants and curated run profiles.",
    )
    variants.add_argument(
        "--profile",
        choices=available_profiles(),
        default=None,
        help="Show only variants that belong to the selected profile.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "variants":
        catalog = variant_catalog()
        if args.profile is not None:
            catalog = _filter_catalog_by_profile(catalog, args.profile)
        print(format_variant_catalog(catalog))
        return 0
    parser.print_help()
    return 0


def _filter_catalog_by_profile(
    catalog: dict[str, Code2HypVariantMetadata],
    profile: str,
) -> dict[str, Code2HypVariantMetadata]:
    return {
        name: metadata
        for name, metadata in catalog.items()
        if profile in metadata.profiles
    }


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
