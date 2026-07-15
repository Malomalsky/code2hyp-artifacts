from __future__ import annotations

import json
from pathlib import Path

from geometry_profile_research.codenet_eligibility import (
    alpha_normalized_ast,
    analyze_python_file,
    build_exact_duplicate_audit,
    lexical_token_stream,
    normalize_python_source,
    portable_manifest_path,
    stable_sha256,
)
from scripts.build_codenet_python800_eligibility import build_eligibility_artifacts


def test_d0_normalizes_encoding_line_endings_and_trailing_whitespace() -> None:
    left = normalize_python_source(b"# coding: utf-8\r\nx = 1  \r\n")
    right = normalize_python_source(b"# coding: utf-8\nx = 1\n\n")
    assert left.decode_ok and right.decode_ok
    assert left.text == right.text == "# coding: utf-8\nx = 1\n"


def test_d1_ignores_comments_and_formatting_but_preserves_lexemes() -> None:
    left = "x = 1  # note\nprint(x)\n"
    right = "x=1\n\nprint( x )\n"
    changed = "x=2\nprint(x)\n"
    assert lexical_token_stream(left) == lexical_token_stream(right)
    assert lexical_token_stream(left) != lexical_token_stream(changed)


def test_d2_is_invariant_to_alpha_renaming_and_literal_values() -> None:
    left = "def total(values):\n    answer = 10\n    for value in values:\n        answer += value\n    return answer\n"
    right = "def sum_items(items):\n    result = 99\n    for item in items:\n        result += item\n    return result\n"
    structural_change = "def sum_items(items):\n    result = 99\n    return items[result]\n"
    assert alpha_normalized_ast(left) == alpha_normalized_ast(right)
    assert alpha_normalized_ast(left) != alpha_normalized_ast(structural_change)


def test_manifest_paths_are_relative_inside_project_root(tmp_path: Path) -> None:
    nested = tmp_path / "data" / "manifest.json"
    assert portable_manifest_path(nested, project_root=tmp_path) == "data/manifest.json"


def test_exact_audit_deduplicates_globally_and_builds_d4_edge() -> None:
    records = []
    for problem in ("p1", "p2"):
        for index in range(5):
            source = f"def f{index}(x):\n    return x + {index}\n"
            records.append(
                {
                    "problem_id": problem,
                    "source_relpath": f"{problem}/s{index}.py",
                    "decode_ok": True,
                    "tokenize_ok": True,
                    "parse_ok": True,
                    "d0_sha256": stable_sha256(source),
                    "d1_sha256": stable_sha256(f"tokens-{index}"),
                    "d2_sha256": stable_sha256(f"ast-{index}"),
                }
            )
    audit = build_exact_duplicate_audit(
        records,
        minimum_cluster_programs=1,
        d4_min_shared_d2=5,
        d4_min_fraction=0.05,
    )
    assert audit["summary"]["exact_duplicates_removed"] == 5
    assert audit["summary"]["multi_problem_cluster_count"] == 1
    assert audit["d4_edges"][0]["shared_d2_programs"] == 5


def test_builder_writes_hashed_artifacts_without_split_or_metrics(tmp_path: Path) -> None:
    root = tmp_path / "Python800"
    for problem in ("p1", "p2"):
        task = root / problem
        task.mkdir(parents=True)
        for index in range(3):
            (task / f"s{index}.py").write_text(
                f"def solve_{problem}_{index}(x):\n    return x + {index}\n",
                encoding="utf-8",
            )
    output = tmp_path / "audit"
    manifest = build_eligibility_artifacts(
        input_root=root,
        output_dir=output,
        archive_sha256="a" * 64,
        workers=1,
        minimum_cluster_programs=2,
    )
    assert manifest["protocol"]["split_status"] == "not_generated"
    assert manifest["protocol"]["retrieval_metrics_opened"] is False
    assert manifest["summary"]["source_files"] == 6
    for artifact in manifest["artifacts"]:
        path = output / artifact["path"]
        assert stable_sha256(path.read_bytes()) == artifact["sha256"]
    checksum, filename = (output / "eligibility_manifest.sha256").read_text().split()
    assert filename == "eligibility_manifest.json"
    assert checksum == stable_sha256((output / filename).read_bytes())
    loaded = json.loads((output / "eligibility_manifest.json").read_text())
    assert loaded["gate_precheck"]["final_eligibility"] == "pending_D3_and_official_D4"
