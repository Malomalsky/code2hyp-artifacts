from __future__ import annotations

import json
from pathlib import Path

from scripts.run_dta_level_b_c_retrieval import TaskSource, run_dta_level_b_c_retrieval


def test_level_b_retrieval_uses_disjoint_train_query_gallery(tmp_path: Path) -> None:
    task_a = _write_task(tmp_path, "task-a", "a")
    task_b = _write_task(tmp_path, "task-b", "b")
    output = tmp_path / "level_b.json"

    payload = run_dta_level_b_c_retrieval(
        tasks=(TaskSource("task-a", task_a), TaskSource("task-b", task_b)),
        output_path=output,
        benchmark_level="B_independent_solution",
        curvatures=(0.0,),
        epochs=1,
        train_per_task=2,
        query_per_task=1,
        gallery_per_task=2,
        max_methods_per_task=5,
        max_paths=8,
        sinkhorn_iterations=6,
        sinkhorn_projection_iterations=512,
    )

    loaded = json.loads(output.read_text(encoding="utf-8"))
    train = set(loaded["split"]["train_ids"])
    query = set(loaded["split"]["query_ids"])
    gallery = set(loaded["split"]["gallery_ids"])
    assert payload["status"] == "complete"
    assert not (train & query)
    assert not (train & gallery)
    assert not (query & gallery)
    assert loaded["runs"][0]["benchmark_level"] == "B_independent_solution"
    assert loaded["runs"][0]["query_count"] == 2
    assert loaded["runs"][0]["positive_count_mean"] == 2


def test_level_c_retrieval_uses_structural_hard_negative_subset(tmp_path: Path) -> None:
    task_a = _write_task(tmp_path, "task-a", "a")
    task_b = _write_task(tmp_path, "task-b", "b")
    output = tmp_path / "level_c.json"

    payload = run_dta_level_b_c_retrieval(
        tasks=(TaskSource("task-a", task_a), TaskSource("task-b", task_b)),
        output_path=output,
        benchmark_level="C_structural_hard_negative",
        curvatures=(0.0,),
        epochs=1,
        train_per_task=2,
        query_per_task=1,
        gallery_per_task=3,
        max_methods_per_task=6,
        max_paths=8,
        sinkhorn_iterations=6,
        sinkhorn_projection_iterations=512,
        hard_negatives_per_query=1,
    )

    run = payload["runs"][0]
    assert run["benchmark_level"] == "C_structural_hard_negative"
    assert run["positive_count_mean"] == 3
    assert run["candidate_count_mean"] == 4


def _write_task(root: Path, task_name: str, prefix: str) -> Path:
    task_dir = root / task_name
    task_dir.mkdir()
    bodies = (
        "total = 0\n    for x in xs:\n        total += x\n    return total\n",
        "acc = 1\n    for x in xs:\n        acc *= (x + 1)\n    return acc\n",
        "out = []\n    for x in xs:\n        out.append(x + 1)\n    return out\n",
        "count = 0\n    for x in xs:\n        if x > 0:\n            count += 1\n    return count\n",
        "best = None\n    for x in xs:\n        if best is None or x > best:\n            best = x\n    return best\n",
        "return [x for x in xs if x % 2 == 0]\n",
    )
    for index, body in enumerate(bodies):
        (task_dir / f"{prefix}_{index}.py").write_text(f"def f_{prefix}_{index}(xs):\n    {body}", encoding="utf-8")
    return task_dir
