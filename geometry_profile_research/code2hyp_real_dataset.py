from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DatasetDownloadSpec:
    name: str
    url: str
    archive_name: str
    expected_bytes: int
    source_repository: str
    citation_note: str


@dataclass(frozen=True)
class ArchiveStatus:
    path: Path
    exists: bool
    bytes: int | None
    size_matches: bool


@dataclass(frozen=True)
class Code2SeqPreprocessedInventory:
    root: Path
    split_paths: dict[str, Path]
    split_line_counts: dict[str, int]

    @property
    def has_all_required_splits(self) -> bool:
        return {"train", "val", "test"}.issubset(self.split_paths)

    @classmethod
    def from_directory(cls, root: str | Path) -> "Code2SeqPreprocessedInventory":
        root_path = Path(root)
        split_paths: dict[str, Path] = {}
        patterns = {
            "train": ("*.train.c2s", "*.train.c2v", "*train*.c2s", "*train*.c2v"),
            "val": ("*.val.c2s", "*.val.c2v", "*.valid.c2s", "*.valid.c2v", "*val*.c2s", "*val*.c2v"),
            "test": ("*.test.c2s", "*.test.c2v", "*test*.c2s", "*test*.c2v"),
        }
        for split, split_patterns in patterns.items():
            for pattern in split_patterns:
                matches = sorted(root_path.rglob(pattern))
                if matches:
                    split_paths[split] = matches[0]
                    break
        split_line_counts = {
            split: _count_lines(path)
            for split, path in split_paths.items()
        }
        return cls(
            root=root_path,
            split_paths=split_paths,
            split_line_counts=split_line_counts,
        )


def java_small_preprocessed_spec() -> DatasetDownloadSpec:
    return DatasetDownloadSpec(
        name="code2seq-java-small-preprocessed",
        url="https://s3.amazonaws.com/code2seq/datasets/java-small-preprocessed.tar.gz",
        archive_name="java-small-preprocessed.tar.gz",
        expected_bytes=479_663_374,
        source_repository="https://github.com/tech-srl/code2seq",
        citation_note=(
            "Use the official code2seq Java-small preprocessed corpus as a real "
            "code2vec/code2seq-style benchmark. Do not report synthetic data as "
            "research evidence."
        ),
    )


def inspect_archive_status(path: str | Path, expected_bytes: int) -> ArchiveStatus:
    archive_path = Path(path)
    if not archive_path.exists():
        return ArchiveStatus(
            path=archive_path,
            exists=False,
            bytes=None,
            size_matches=False,
        )
    actual_bytes = archive_path.stat().st_size
    return ArchiveStatus(
        path=archive_path,
        exists=True,
        bytes=actual_bytes,
        size_matches=actual_bytes == expected_bytes,
    )


def write_dataset_manifest(path: str | Path, spec: DatasetDownloadSpec) -> None:
    target = Path(path)
    target.write_text(
        "\n".join(
            [
                "# Code2Hyp Dataset Manifest",
                "",
                "## Real dataset source",
                "",
                f"- Name: {spec.name}",
                f"- Archive: `{spec.archive_name}`",
                f"- URL: {spec.url}",
                f"- Expected bytes: {spec.expected_bytes}",
                f"- Source repository: {spec.source_repository}",
                "",
                "## Claim boundary",
                "",
                "Synthetic datasets are not valid research evidence for the article.",
                "They may only be used for unit, smoke, and integration tests.",
                "",
            ]
        ),
        encoding="utf-8",
    )


def _count_lines(path: Path) -> int:
    count = 0
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for count, _ in enumerate(handle, start=1):
            pass
    return count
