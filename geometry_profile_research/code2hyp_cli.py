from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence

from .code2hyp_registry import (
    Code2HypVariantMetadata,
    available_profiles,
    format_variant_catalog,
    variant_catalog,
)
from .code2hyp_tool import Code2Hyp, Code2HypIndex


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

    index = subparsers.add_parser(
        "index",
        help="Build a structural Code2Hyp index for a directory of source files.",
    )
    index.add_argument("root", help="Directory with source files to index.")
    index.add_argument("--output", required=True, help="Path to the JSON index file.")
    index.add_argument("--language", choices=("python",), default="python")
    index.add_argument("--model", default="code2hyp-v1")
    index.add_argument("--pattern", default="*.py")
    index.add_argument("--no-recursive", action="store_true")

    search = subparsers.add_parser(
        "search",
        help="Search a Code2Hyp index for structurally similar code.",
    )
    search.add_argument("query", help="Query source file.")
    search.add_argument("--index", required=True, help="Path to a JSON index file.")
    search.add_argument("--top-k", type=int, default=20)

    compare = subparsers.add_parser(
        "compare",
        help="Compare two source files under the Code2Hyp structural distance.",
    )
    compare.add_argument("left", help="First source file.")
    compare.add_argument("right", help="Second source file.")
    compare.add_argument("--model", default="code2hyp-v1")

    explain = subparsers.add_parser(
        "explain",
        help="Explain structural similarity by path-to-path transport alignment.",
    )
    explain.add_argument("left", help="Query source file.")
    explain.add_argument("right", help="Candidate source file.")
    explain.add_argument("--model", default="code2hyp-v1")
    explain.add_argument("--top-k", type=int, default=10)

    audit = subparsers.add_parser(
        "audit-geometry",
        help="Audit point/side cost shares for a directory or a saved Code2Hyp index.",
    )
    audit.set_defaults(command="audit_geometry")
    audit.add_argument("root", nargs="?", default=None, help="Directory with source files to audit.")
    audit.add_argument("--index", default=None, help="Path to a saved JSON index.")
    audit.add_argument("--language", choices=("python",), default="python")
    audit.add_argument("--model", default="code2hyp-v1")
    audit.add_argument("--pattern", default="*.py")
    audit.add_argument("--no-recursive", action="store_true")
    audit.add_argument("--max-pairs", type=int, default=256)
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
    if args.command == "index":
        model = Code2Hyp.load(args.model)
        index = model.index_directory(
            args.root,
            pattern=args.pattern,
            recursive=not args.no_recursive,
        )
        index.save(args.output)
        _print_json(
            {
                "model_name": index.model_name,
                "output": args.output,
                "entries": len(index.entries),
            }
        )
        return 0
    if args.command == "search":
        index = Code2HypIndex.load(args.index)
        results = index.search(args.query, top_k=args.top_k)
        _print_json(
            {
                "model_name": index.model_name,
                "query": args.query,
                "results": [result.as_dict() for result in results],
            }
        )
        return 0
    if args.command == "compare":
        model = Code2Hyp.load(args.model)
        _print_json(model.compare_files(args.left, args.right))
        return 0
    if args.command == "explain":
        model = Code2Hyp.load(args.model)
        _print_json(model.explain_files(args.left, args.right, top_k=args.top_k))
        return 0
    if args.command == "audit_geometry":
        model = Code2Hyp.load(args.model)
        if args.index is not None:
            index = Code2HypIndex.load(args.index)
            _print_json(model.audit_index(index, max_pairs=args.max_pairs))
            return 0
        if args.root is None:
            parser.error("audit-geometry requires ROOT or --index")
        _print_json(
            model.audit_directory(
                args.root,
                pattern=args.pattern,
                recursive=not args.no_recursive,
                max_pairs=args.max_pairs,
            )
        )
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


def _print_json(payload: object) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
