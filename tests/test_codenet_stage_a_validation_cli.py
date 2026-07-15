from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from scripts.run_codenet_stage_a_validation import RUNNER_TAG, _verified_implementation_state


def _git(repository: Path, *arguments: str) -> None:
    subprocess.run(
        ("git", "-C", str(repository), *arguments),
        check=True,
        capture_output=True,
        text=True,
    )


def test_official_validation_requires_clean_tagged_worktree(tmp_path: Path) -> None:
    _git(tmp_path, "init")
    tracked = tmp_path / "tracked.txt"
    tracked.write_text("frozen\n", encoding="utf-8")
    _git(tmp_path, "add", "tracked.txt")
    _git(
        tmp_path,
        "-c",
        "user.name=Code2Hyp Test",
        "-c",
        "user.email=test@example.invalid",
        "commit",
        "-m",
        "frozen runner",
    )
    _git(tmp_path, "tag", RUNNER_TAG)

    state = _verified_implementation_state(tmp_path)

    assert state["tag"] == RUNNER_TAG
    assert state["tracked_worktree_clean"] is True
    assert len(state["commit"]) == 40

    tracked.write_text("modified\n", encoding="utf-8")
    with pytest.raises(ValueError, match="clean tracked worktree"):
        _verified_implementation_state(tmp_path)
