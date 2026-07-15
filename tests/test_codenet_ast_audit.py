from __future__ import annotations

import json
from pathlib import Path

from geometry_profile_research.codenet_ast_audit import audit_source_program


def _sample_row(source_relpath: str) -> dict[str, str]:
    return {
        "cluster_id": "problem-test",
        "problem_id": "p00000",
        "role": "train",
        "source_relpath": source_relpath,
        "split": "train",
        "submission_id": "s000000001",
    }


def test_source_audit_parses_module_and_hashes_unique_paths(tmp_path: Path) -> None:
    source = tmp_path / "p00000" / "s000000001.py"
    source.parent.mkdir()
    source.write_text("def f(x):\n    return x + 1\n", encoding="utf-8")

    result = audit_source_program(
        tmp_path,
        _sample_row("p00000/s000000001.py"),
        max_paths=64,
        selection_policy="lca_depth_affine_sampled",
    )

    assert result["audit_ok"] is True
    assert result["node_count"] > result["leaf_count"] >= 2
    assert result["selected_path_count"] == min(64, result["available_terminal_pair_count"])
    assert len(result["selected_endpoint_pairs_sha256"]) == 64
    json.dumps(result)


def test_source_audit_keeps_small_program_without_duplicate_paths(tmp_path: Path) -> None:
    source = tmp_path / "p00000" / "s000000001.py"
    source.parent.mkdir()
    source.write_text("x = 1\n", encoding="utf-8")

    result = audit_source_program(
        tmp_path,
        _sample_row("p00000/s000000001.py"),
        max_paths=64,
        selection_policy="lca_depth_affine_sampled",
    )

    assert result["audit_ok"] is True
    assert result["available_terminal_pair_count"] < 64
    assert result["selected_path_count"] == result["available_terminal_pair_count"]


def test_source_audit_fails_closed_on_syntax_error(tmp_path: Path) -> None:
    source = tmp_path / "p00000" / "s000000001.py"
    source.parent.mkdir()
    source.write_text("def broken(:\n", encoding="utf-8")

    result = audit_source_program(
        tmp_path,
        _sample_row("p00000/s000000001.py"),
        max_paths=64,
        selection_policy="lca_depth_affine_sampled",
    )

    assert result["audit_ok"] is False
    assert result["failure"] == "ast_parse:SyntaxError"
