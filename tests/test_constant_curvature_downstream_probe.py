from __future__ import annotations

import json
from pathlib import Path

from scripts.run_constant_curvature_downstream_probe import ProbeProject, run_downstream_probe


def test_downstream_probe_runs_on_tiny_python_project(tmp_path: Path) -> None:
    source_dir = tmp_path / "src"
    source_dir.mkdir()
    for index, body in enumerate(
        (
            "total = 0\n    for x in xs:\n        total += x\n    return total\n",
            "acc = 1\n    for x in xs:\n        acc *= x\n    return acc\n",
            "return [x + 1 for x in xs]\n",
        )
    ):
        (source_dir / f"m{index}.py").write_text(f"def f(xs):\n    {body}", encoding="utf-8")
    output = tmp_path / "probe.json"

    payload = run_downstream_probe(
        projects=(ProbeProject("tiny", source_dir),),
        output_path=output,
        curvatures=(0.0, 1e-6),
        epochs=1,
        max_files=8,
        max_methods=3,
        max_paths=8,
        sinkhorn_iterations=8,
        sinkhorn_projection_iterations=512,
    )

    loaded = json.loads(output.read_text(encoding="utf-8"))
    assert payload["status"] == "complete"
    assert loaded["completed_runs"] == 2
    assert {run["curvature"] for run in loaded["runs"]} == {0.0, 1e-6}
    assert all("mrr" in run for run in loaded["runs"])
