from __future__ import annotations

import json
import tomllib
from pathlib import Path

from geometry_profile_research.code2hyp_cli import build_parser, main


def test_code2hyp_cli_parses_variants_command() -> None:
    args = build_parser().parse_args(["variants"])

    assert args.command == "variants"


def test_code2hyp_cli_parses_public_tool_commands() -> None:
    parser = build_parser()

    assert parser.parse_args(["index", ".", "--output", "index.json"]).command == "index"
    assert parser.parse_args(["search", "query.py", "--index", "index.json"]).command == "search"
    assert parser.parse_args(["compare", "left.py", "right.py"]).command == "compare"
    assert parser.parse_args(["explain", "left.py", "right.py"]).command == "explain"
    assert parser.parse_args(["audit-geometry", "."]).command == "audit_geometry"


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


def test_code2hyp_cli_indexes_searches_compares_and_explains(tmp_path: Path, capsys) -> None:
    query = tmp_path / "query.py"
    query.write_text(
        """
def total(values):
    answer = 0
    for value in values:
        answer += value
    return answer
""".strip()
        + "\n",
        encoding="utf-8",
    )
    candidate = tmp_path / "candidate.py"
    candidate.write_text(
        """
def count(items):
    result = 0
    for item in items:
        result += item
    return result
""".strip()
        + "\n",
        encoding="utf-8",
    )
    index_path = tmp_path / "code2hyp-index.json"

    assert main(["index", str(tmp_path), "--output", str(index_path)]) == 0
    index_payload = json.loads(index_path.read_text(encoding="utf-8"))
    assert index_payload["model_name"] == "code2hyp-v1"
    assert len(index_payload["entries"]) == 2
    capsys.readouterr()

    assert main(["search", str(query), "--index", str(index_path), "--top-k", "1"]) == 0
    search_payload = json.loads(capsys.readouterr().out)
    assert search_payload["results"][0]["path"] == str(query)

    assert main(["compare", str(query), str(candidate)]) == 0
    compare_payload = json.loads(capsys.readouterr().out)
    assert compare_payload["distance"] >= 0.0

    assert main(["explain", str(query), str(candidate), "--top-k", "2"]) == 0
    explain_payload = json.loads(capsys.readouterr().out)
    assert explain_payload["alignments"]

    assert main(["audit-geometry", str(tmp_path)]) == 0
    audit_payload = json.loads(capsys.readouterr().out)
    assert audit_payload["entries"] == 2
    assert 0.0 <= audit_payload["side_cost_share"] <= 1.0
