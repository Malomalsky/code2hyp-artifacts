from __future__ import annotations

import json
from pathlib import Path

from code2hyp import Code2Hyp as PublicCode2Hyp
from geometry_profile_research.code2hyp_tool import Code2Hyp, Code2HypConfig, Code2HypIndex


def test_public_code2hyp_api_exports_model_class() -> None:
    model = PublicCode2Hyp.load("code2hyp-v1")

    assert isinstance(model, Code2Hyp)


def _write_program(path: Path, source: str) -> Path:
    path.write_text(source.strip() + "\n", encoding="utf-8")
    return path


def test_code2hyp_indexes_directory_and_searches_structural_matches(tmp_path: Path) -> None:
    query = _write_program(
        tmp_path / "query.py",
        """
        def sum_values(values):
            total = 0
            for value in values:
                total += value
            return total
        """,
    )
    similar = _write_program(
        tmp_path / "similar.py",
        """
        def add_items(items):
            result = 0
            for item in items:
                result += item
            return result
        """,
    )
    different = _write_program(
        tmp_path / "different.py",
        """
        def choose(flag, left, right):
            if flag:
                return left
            return right
        """,
    )

    model = Code2Hyp.load("code2hyp-v1")
    index = model.index_directory(tmp_path)
    results = index.search(query, top_k=3)

    assert {Path(result.path).name for result in results} == {"query.py", "similar.py", "different.py"}
    assert results[0].path == str(query)
    assert results[0].distance <= results[1].distance
    assert any(Path(result.path).name == similar.name for result in results[:2])


def test_code2hyp_compare_and_explain_files_return_transport_alignment(tmp_path: Path) -> None:
    left = _write_program(
        tmp_path / "left.py",
        """
        def normalize(xs):
            total = sum(xs)
            return [x / total for x in xs]
        """,
    )
    right = _write_program(
        tmp_path / "right.py",
        """
        def scale(values):
            denom = sum(values)
            return [value / denom for value in values]
        """,
    )

    model = Code2Hyp.load("code2hyp-v1")
    comparison = model.compare_files(left, right)
    explanation = model.explain_files(left, right, top_k=4)

    assert comparison["distance"] >= 0.0
    assert comparison["left_path_count"] > 0
    assert comparison["right_path_count"] > 0
    assert explanation["distance"] == comparison["distance"]
    assert 1 <= len(explanation["alignments"]) <= 4
    first = explanation["alignments"][0]
    assert first["transport_mass"] > 0.0
    assert first["local_cost"] >= 0.0
    assert first["query_lca_label"]
    assert first["candidate_lca_label"]
    assert "query_source_span" in first
    assert "candidate_source_span" in first


def test_code2hyp_index_round_trips_json(tmp_path: Path) -> None:
    _write_program(
        tmp_path / "program.py",
        """
        def identity(value):
            return value
        """,
    )

    model = Code2Hyp.load("code2hyp-v1")
    index = model.index_directory(tmp_path)
    payload = index.to_json()
    restored = Code2HypIndex.from_json(json.loads(json.dumps(payload)))

    assert restored.model_name == "code2hyp-v1"
    assert len(restored.entries) == 1
    assert restored.entries[0].path.endswith("program.py")
    assert restored.search(tmp_path / "program.py", top_k=1)[0].distance == 0.0


def test_code2hyp_supports_categorical_label_side_channel(tmp_path: Path) -> None:
    program = _write_program(
        tmp_path / "program.py",
        """
        def identity(value):
            return value
        """,
    )

    scalar = Code2Hyp(Code2HypConfig(label_mode="scalar_hash")).encode_file(program)
    categorical = Code2Hyp(Code2HypConfig(label_mode="categorical")).encode_file(program)
    no_label = Code2Hyp(Code2HypConfig(label_mode="none")).encode_file(program)

    assert scalar.measure.side_features is not None
    assert categorical.measure.side_features is not None
    assert no_label.measure.side_features is not None
    assert scalar.measure.side_features.shape[1] == 7
    assert no_label.measure.side_features.shape[1] == 4
    assert categorical.measure.side_features.shape[1] > scalar.measure.side_features.shape[1]


def test_code2hyp_index_preserves_label_mode(tmp_path: Path) -> None:
    _write_program(
        tmp_path / "program.py",
        """
        def identity(value):
            return value
        """,
    )

    model = Code2Hyp(Code2HypConfig(label_mode="categorical"))
    index = model.index_directory(tmp_path)
    restored = Code2HypIndex.from_json(index.to_json())

    assert restored.model.config.label_mode == "categorical"
    assert restored.search(tmp_path / "program.py", top_k=1)[0].distance == 0.0


def test_code2hyp_audits_geometry_cost_shares(tmp_path: Path) -> None:
    _write_program(
        tmp_path / "loop.py",
        """
        def total(xs):
            answer = 0
            for x in xs:
                answer += x
            return answer
        """,
    )
    _write_program(
        tmp_path / "branch.py",
        """
        def choose(flag, left, right):
            if flag:
                return left
            return right
        """,
    )

    model = Code2Hyp.load("code2hyp-v1")
    audit = model.audit_directory(tmp_path)

    assert audit["model"] == "code2hyp-v1"
    assert audit["entries"] == 2
    assert audit["pair_count"] == 1
    assert 0.0 <= audit["point_cost_share"] <= 1.0
    assert 0.0 <= audit["side_cost_share"] <= 1.0
    assert audit["median_positive_full_cost"] > 0.0
