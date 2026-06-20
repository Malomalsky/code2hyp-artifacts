from __future__ import annotations

import re
from pathlib import PurePosixPath

from .graphs import normalize_path

_PATH_PATTERN = re.compile(
    r"(?:(?:a|b)/)?[A-Za-z0-9_.@+-]+(?:/[A-Za-z0-9_.@+-]+)+"
)


def _clean_candidate_path(candidate: str) -> str:
    candidate = candidate.strip().strip("\"'`,;:")
    if candidate.startswith(("a/", "b/")):
        candidate = candidate[2:]
    return normalize_path(candidate)


def _looks_like_source_path(path: str) -> bool:
    if not path or path == "dev/null":
        return False
    if "://" in path:
        return False
    if path.startswith(".git/"):
        return False

    name = PurePosixPath(path).name
    if name in {"Makefile", "Dockerfile", "CMakeLists.txt"}:
        return True
    return "." in name


def extract_paths_from_text(text: str) -> list[str]:
    """Extract stable source-tree paths from free-form technical text.

    The function is intentionally conservative: it keeps only path-like tokens
    that look like source-tree files, removes common `a/` and `b/` prefixes,
    drops `/dev/null`, deduplicates values, and returns a sorted list for
    reproducible experiments.
    """
    paths: set[str] = set()
    for match in _PATH_PATTERN.finditer(text):
        path = _clean_candidate_path(match.group(0))
        if _looks_like_source_path(path):
            paths.add(path)
    return sorted(paths)
