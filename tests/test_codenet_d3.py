from __future__ import annotations

import json
from pathlib import Path

from geometry_profile_research.codenet_eligibility import (
    exact_jaccard,
    lexical_token_stream,
    minhash_signature,
    token_ngrams,
)
from scripts.build_codenet_python800_d3 import build_d3_artifacts


def test_token_5gram_jaccard_is_exact_and_minhash_is_deterministic() -> None:
    left = token_ngrams(tuple(f"NAME:x{i}" for i in range(30)))
    right = token_ngrams(tuple(f"NAME:x{i}" for i in range(29)) + ("NAME:changed",))
    assert 0.8 < exact_jaccard(left, right) < 1.0
    first = minhash_signature(left, num_perm=32, seed=7)
    second = minhash_signature(left, num_perm=32, seed=7)
    assert first.tolist() == second.tolist()


def test_short_stream_has_no_5gram() -> None:
    assert not token_ngrams(("NAME:x", "OP:="), width=5)


def test_d3_builder_verifies_candidates_and_keeps_metrics_sealed(tmp_path: Path) -> None:
    source_root = tmp_path / "Python800"
    base_lines = [f"value_{index} = data + {index}" for index in range(80)]
    sources = {
        "p1/s1.py": "\n".join(["data = 1", *base_lines, "print(value_79)"]) + "\n",
        "p2/s2.py": "\n".join(["data = 1", *base_lines[:-1], "value_79 = data - 79", "print(value_79)"]) + "\n",
        "p3/s3.py": "def unrelated(items):\n    return sorted(set(items))\n",
    }
    inventory_rows = []
    for relative, source in sources.items():
        path = source_root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source, encoding="utf-8")
        problem, filename = relative.split("/")
        inventory_rows.append(
            {
                "problem_id": problem,
                "submission_id": Path(filename).stem,
                "source_relpath": relative,
                "retained_after_d0_d2": True,
            }
        )
    d0_d2 = tmp_path / "d0_d2"
    d0_d2.mkdir()
    (d0_d2 / "eligibility_manifest.json").write_text("{}\n", encoding="utf-8")
    (d0_d2 / "file_inventory.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in inventory_rows),
        encoding="utf-8",
    )
    clusters = [
        {"cluster_id": problem, "problem_ids": [problem]}
        for problem in ("p1", "p2", "p3")
    ]
    (d0_d2 / "preliminary_problem_clusters.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in clusters),
        encoding="utf-8",
    )

    output = tmp_path / "d3"
    manifest = build_d3_artifacts(
        input_root=source_root,
        d0_d2_dir=d0_d2,
        output_dir=output,
        workers=1,
        num_perm=32,
        bands=8,
        rows_per_band=4,
        minhash_seed=7,
        jaccard_threshold=0.90,
        minimum_cluster_programs=1,
    )
    assert manifest["protocol"]["retrieval_metrics_opened"] is False
    assert manifest["summary"]["verified_d3_edges"] == 1
    assert manifest["summary"]["d3_duplicates_removed"] == 1
    assert manifest["summary"]["problem_cluster_count_after_exact_d4_and_cross_problem_d3"] == 2
    edge = json.loads((output / "d3_near_duplicate_edges.jsonl").read_text())
    assert edge["set_jaccard"] >= 0.90
