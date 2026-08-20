from __future__ import annotations

import json
from pathlib import Path

from geometry_profile_research.codenet_ast_audit import audit_source_program
from geometry_profile_research.codenet_stage_a import load_codenet_split


def test_generic_codenet_loader_verifies_a_small_java_split(tmp_path: Path) -> None:
    source_root = tmp_path / "sources"
    rows = []
    ast_rows = []
    specs = (
        ("train", "train-a", "train", 0),
        ("train", "train-a", "train", 1),
        ("validation", "validation-a", "query", 2),
        ("validation", "validation-a", "gallery", 3),
    )
    for split, cluster, role, index in specs:
        relpath = f"p{index:05d}/s{index:09d}.java"
        path = source_root / relpath
        path.parent.mkdir(parents=True)
        path.write_text(
            f"class C{index} {{ int f(int x) {{ return x + {index}; }} }}\n",
            encoding="utf-8",
        )
        row = {
            "cluster_id": cluster,
            "problem_id": cluster,
            "role": role,
            "source_relpath": relpath,
            "split": split,
            "submission_id": f"s{index:09d}",
        }
        rows.append(row)
        ast_rows.append(
            audit_source_program(
                source_root,
                row,
                max_paths=64,
                selection_policy="lca_depth_affine_sampled",
                language="java",
            )
        )
    train_path = tmp_path / "train.jsonl"
    validation_path = tmp_path / "validation.jsonl"
    ast_path = tmp_path / "ast.jsonl"
    train_path.write_text("".join(json.dumps(row) + "\n" for row in rows[:2]), encoding="utf-8")
    validation_path.write_text("".join(json.dumps(row) + "\n" for row in rows[2:]), encoding="utf-8")
    ast_path.write_text("".join(json.dumps(row) + "\n" for row in ast_rows), encoding="utf-8")

    split = load_codenet_split(
        source_root=source_root,
        train_path=train_path,
        validation_path=validation_path,
        ast_index_path=ast_path,
        language="java",
        train_clusters=1,
        validation_clusters=1,
        train_programs_per_cluster=2,
        validation_queries_per_cluster=1,
        validation_gallery_per_cluster=1,
    )

    assert len(split.train) == 2
    assert len(split.query) == len(split.gallery) == 1
    assert split.query[0].cluster_id == split.gallery[0].cluster_id == "validation-a"
