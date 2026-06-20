from __future__ import annotations

import tomllib
from pathlib import Path

from geometry_profile_research.code2hyp_cli import build_parser, main


def test_code2hyp_cli_parses_variants_command() -> None:
    args = build_parser().parse_args(["variants"])

    assert args.command == "variants"


def test_code2hyp_cli_prints_variant_catalog(capsys) -> None:
    exit_code = main(["variants", "--profile", "balanced"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "B4_hyperbolic_code2vec" in captured.out
    assert "B31_hyperbolic_path_dual_attention_mp_soft_rank" in captured.out
    assert "B1_euclidean" not in captured.out


def test_pyproject_exposes_code2hyp_console_script() -> None:
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))

    assert pyproject["project"]["scripts"]["code2hyp"] == "geometry_profile_research.code2hyp_cli:main"
