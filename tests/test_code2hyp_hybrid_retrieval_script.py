from __future__ import annotations

import json
from pathlib import Path

from scripts.run_code2hyp_hybrid_retrieval import HybridInput, run_hybrid, _multiview_variants


def test_hybrid_retrieval_selects_train_weights_and_ranks_same_task_first(tmp_path: Path) -> None:
    split_path = _write_tiny_task_corpus(tmp_path)

    result = run_hybrid([HybridInput("synthetic", split_path)], max_paths=32)

    summaries = {(row["dataset"], row["variant"]): row for row in result["cell_summaries"]}
    multiview = summaries[("synthetic", "code2hyp_multiview_selected")]

    assert multiview["query_count"] == 2
    assert multiview["task_count"] == 2
    assert multiview["seed_count"] == 1
    weights = multiview["selected_weights_by_seed"]["20260625"]
    assert abs(sum(weights.values()) - 1.0) < 1e-9
    assert all(value >= 0.0 for value in weights.values())
    assert set(weights) == {
        "path_signature_plus_tokens",
        "token_bag",
        "ast_node_bag",
        "token_count_bag",
        "ast_node_count_bag",
    }

    rows = [row for row in result["query_rows"] if row["variant"] == "code2hyp_multiview_selected"]
    assert {row["rank"] for row in rows} == {1}
    assert {row["query_task"] for row in rows} == {"task_sum", "task_max"}


def test_hybrid_retrieval_records_hash_sorted_path_selection_policy(tmp_path: Path) -> None:
    split_path = _write_tiny_task_corpus(tmp_path)

    result = run_hybrid([HybridInput("synthetic", split_path)], max_paths=8, path_selection_policy="hash_sorted")

    assert result["path_selection_policy"] == "hash_sorted"
    assert result["max_paths"] == 8
    assert any(row["variant"] == "code2hyp_multiview_selected" for row in result["query_rows"])


def test_hybrid_retrieval_supports_clean_lca_view(tmp_path: Path) -> None:
    split_path = _write_tiny_task_corpus(tmp_path)

    result = run_hybrid(
        [HybridInput("synthetic", split_path)],
        max_paths=8,
        lca_view="code2hyp_path_signature_kernel",
    )

    assert result["lca_view"] == "code2hyp_path_signature_kernel"
    summaries = {(row["dataset"], row["variant"]): row for row in result["cell_summaries"]}
    weights = summaries[("synthetic", "code2hyp_multiview_selected")]["selected_weights_by_seed"]["20260625"]
    assert "code2hyp_path_signature_kernel" in weights
    assert "path_signature_plus_tokens" not in weights


def test_hybrid_retrieval_lca_selection_margin_can_fall_back_to_no_lca(tmp_path: Path) -> None:
    split_path = _write_tiny_task_corpus(tmp_path)

    result = run_hybrid(
        [HybridInput("synthetic", split_path)],
        max_paths=8,
        lca_view="code2hyp_path_signature_kernel",
        lca_selection_margin=1.0,
    )

    summaries = {(row["dataset"], row["variant"]): row for row in result["cell_summaries"]}
    weights = summaries[("synthetic", "code2hyp_multiview_selected")]["selected_weights_by_seed"]["20260625"]
    assert weights["code2hyp_path_signature_kernel"] == 0.0


def test_hybrid_retrieval_records_expanded_weight_grid_mode(tmp_path: Path) -> None:
    split_path = _write_tiny_task_corpus(tmp_path)

    result = run_hybrid(
        [HybridInput("synthetic", split_path)],
        max_paths=8,
        lca_view="code2hyp_path_signature_kernel",
        weight_grid_mode="expanded",
    )

    assert result["weight_grid_mode"] == "expanded"
    compact = run_hybrid(
        [HybridInput("synthetic", split_path)],
        max_paths=8,
        lca_view="code2hyp_path_signature_kernel",
        weight_grid_mode="compact",
    )
    assert len(result["weight_grid"]["code2hyp_multiview_selected"]) > len(
        compact["weight_grid"]["code2hyp_multiview_selected"]
    )


def test_multiview_grids_are_nested_over_token_ast_baseline() -> None:
    variants = _multiview_variants("code2hyp_path_signature_kernel", weight_grid_mode="expanded")
    full_grid = variants["code2hyp_multiview_selected"][1]
    no_lca_grid = variants["code2hyp_multiview_no_lca_selected"][1]
    token_ast_grid = variants["token_ast_selected"][1]

    expected_no_lca = {
        tuple(sorted((view, float(value)) for view, value in weights.items()))
        for weights in token_ast_grid
    }
    actual_no_lca = {
        tuple(
            sorted(
                (view, float(value))
                for view, value in weights.items()
                if view in {"token_bag", "ast_node_bag"}
            )
        )
        for weights in no_lca_grid
        if float(weights.get("token_count_bag", 0.0)) == 0.0
        and float(weights.get("ast_node_count_bag", 0.0)) == 0.0
    }
    actual_full_zero_lca = {
        tuple(
            sorted(
                (view, float(value))
                for view, value in weights.items()
                if view in {"token_bag", "ast_node_bag"}
            )
        )
        for weights in full_grid
        if float(weights.get("code2hyp_path_signature_kernel", 0.0)) == 0.0
        and float(weights.get("token_count_bag", 0.0)) == 0.0
        and float(weights.get("ast_node_count_bag", 0.0)) == 0.0
    }

    assert expected_no_lca <= actual_no_lca
    assert expected_no_lca <= actual_full_zero_lca


def _write_tiny_task_corpus(root: Path) -> Path:
    task_sum = root / "task_sum"
    task_max = root / "task_max"
    task_sum.mkdir()
    task_max.mkdir()

    train_ids = [
        _program(task_sum / "sum_train_1.py", "def solve():\n    n = int(input())\n    print(sum(range(n)))\n"),
        _program(task_sum / "sum_train_2.py", "def calc():\n    n = int(input())\n    total = sum(range(n))\n    print(total)\n"),
        _program(task_max / "max_train_1.py", "def solve():\n    a, b = map(int, input().split())\n    print(max(a, b))\n"),
        _program(task_max / "max_train_2.py", "def calc():\n    values = list(map(int, input().split()))\n    print(max(values))\n"),
    ]
    query_ids = [
        _program(task_sum / "sum_query.py", "def answer():\n    n = int(input())\n    print(sum(range(n)))\n"),
        _program(task_max / "max_query.py", "def answer():\n    a, b = map(int, input().split())\n    print(max(a, b))\n"),
    ]
    gallery_ids = [
        _program(task_sum / "sum_gallery.py", "def main():\n    n = int(input())\n    result = sum(range(n))\n    print(result)\n"),
        _program(task_max / "max_gallery.py", "def main():\n    nums = list(map(int, input().split()))\n    print(max(nums))\n"),
    ]

    payload = {
        "config": {"seed": 20260625},
        "split": {
            "train_ids": train_ids,
            "query_ids": query_ids,
            "gallery_ids": gallery_ids,
        },
    }
    split_path = root / "split.json"
    split_path.write_text(json.dumps(payload), encoding="utf-8")
    return split_path


def _program(path: Path, source: str) -> str:
    path.write_text(source, encoding="utf-8")
    return f"synthetic:{path}:module:0"
