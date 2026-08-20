from __future__ import annotations

from scripts.audit_codenet_java_stage_b_d0_d2 import analyze_java_source


def test_java_d2_normalizes_identifiers_and_literals_but_preserves_operators() -> None:
    plus_a = analyze_java_source(
        b"class A { int f(int x) { return x + 1; } }",
        source_relpath="a.java",
    )
    plus_b = analyze_java_source(
        b"class B { int g(int y) { return y + 2; } }",
        source_relpath="b.java",
    )
    multiply = analyze_java_source(
        b"class C { int h(int z) { return z * 3; } }",
        source_relpath="c.java",
    )

    assert plus_a["parse_ok"] and plus_b["parse_ok"] and multiply["parse_ok"]
    assert plus_a["d0_sha256"] != plus_b["d0_sha256"]
    assert plus_a["d1_sha256"] != plus_b["d1_sha256"]
    assert plus_a["d2_sha256"] == plus_b["d2_sha256"]
    assert plus_a["d2_sha256"] != multiply["d2_sha256"]
