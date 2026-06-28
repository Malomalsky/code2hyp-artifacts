from __future__ import annotations

from pathlib import Path

from scripts.run_constant_curvature_limit_audit import run_audit, write_report


def test_constant_curvature_limit_audit_reports_convergence(tmp_path: Path) -> None:
    payload = run_audit(
        curvatures=(0.0, 1e-8, 1.0),
        method_count=5,
        train_methods=3,
        paths_per_method=4,
        dim=3,
        sinkhorn_iterations=24,
    )
    report_path = tmp_path / "audit.md"

    write_report(payload, report_path)

    assert payload["audit"] == "constant_curvature_euclidean_limit"
    assert len(payload["rows"]) == 3
    near_zero = next(row for row in payload["rows"] if row["curvature"] == 1e-8)
    active = next(row for row in payload["rows"] if row["curvature"] == 1.0)
    assert near_zero["distance_matrix_relative_frobenius_error"] < 1e-5
    assert near_zero["sinkhorn_matrix_relative_frobenius_error"] < 1e-4
    assert near_zero["transport_plan_l1_error"] < 1e-4
    assert near_zero["gradient_relative_error"] < 1e-3
    assert near_zero["gradient_cosine"] > 0.999
    assert near_zero["ranking_top1_agreement"] == 1.0
    assert near_zero["ranking_top3_jaccard"] == 1.0
    assert active["sinkhorn_matrix_relative_frobenius_error"] > near_zero["sinkhorn_matrix_relative_frobenius_error"]
    assert payload["quality_gate"]["near_zero_distance_converges"] is True
    assert payload["quality_gate"]["near_zero_sinkhorn_converges"] is True
    assert payload["quality_gate"]["near_zero_transport_converges"] is True
    assert payload["quality_gate"]["near_zero_gradients_converge"] is True
    assert payload["quality_gate"]["near_zero_rankings_stable"] is True
    assert "Constant-curvature Euclidean-limit audit" in report_path.read_text(encoding="utf-8")
