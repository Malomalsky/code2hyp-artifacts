from __future__ import annotations

from pathlib import Path

from scripts.run_codenet_stage_a_gate0 import PROJECT_ROOT, run_gate0


def test_stage_a_numerical_gate0_passes_without_validation_data() -> None:
    payload = run_gate0(
        protocol_path=PROJECT_ROOT / "configs/codenet_python800_stage_a_model_analysis_protocol_v1.json"
    )

    assert payload["status"] == "passed"
    assert all(check["passed"] for check in payload["checks"].values())
    assert payload["validation_retrieval_metrics_computed"] is False
    assert payload["test_program_ids_materialized"] is False
