from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Mapping

from .graphs import normalize_path


@dataclass
class RepositoryPaths:
    repo_name: str
    paths: list[str] = field(default_factory=list)
    languages: set[str] = field(default_factory=set)


def _first_present(row: Mapping[str, object], candidates: tuple[str, ...]) -> str:
    for candidate in candidates:
        value = row.get(candidate)
        if value:
            return str(value)
    return ""


def group_records_by_repo(
    records: Iterable[Mapping[str, object]],
    *,
    repo_fields: tuple[str, ...] = ("repo_name", "repo", "repository_name"),
    path_fields: tuple[str, ...] = ("path", "filepath", "file", "filename"),
    language_fields: tuple[str, ...] = ("language", "lang", "repo_language"),
) -> dict[str, RepositoryPaths]:
    """Group external dataset rows into repository-level path collections."""
    grouped: dict[str, RepositoryPaths] = {}
    seen_paths: dict[str, set[str]] = {}

    for row in records:
        repo_name = _first_present(row, repo_fields).strip()
        path = normalize_path(_first_present(row, path_fields))
        if not repo_name or not path:
            continue

        if repo_name not in grouped:
            grouped[repo_name] = RepositoryPaths(repo_name=repo_name)
            seen_paths[repo_name] = set()

        if path not in seen_paths[repo_name]:
            grouped[repo_name].paths.append(path)
            seen_paths[repo_name].add(path)

        language = _first_present(row, language_fields).strip()
        if language:
            grouped[repo_name].languages.add(language)

    for repo in grouped.values():
        repo.paths.sort()
    return grouped
