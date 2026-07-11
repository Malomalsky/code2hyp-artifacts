from __future__ import annotations

from scripts.analyze_codenet_d5_attrition import analyze_attrition, hamilton_quotas


def test_hamilton_quotas_match_the_773_cluster_design() -> None:
    assert hamilton_quotas(773, (3, 1, 4)) == (290, 97, 386)
    assert hamilton_quotas(800, (3, 1, 4)) == (300, 100, 400)


def test_attrition_analysis_is_planning_only_and_detects_shared_users() -> None:
    clusters = [f"c{index}" for index in range(8)]
    rows = []
    for cluster in clusters:
        rows.extend(
            {"problem_cluster_id": cluster, "user_id_sha256": user}
            for user in ("global-user", f"local-{cluster}-1", f"local-{cluster}-2")
        )
    summary, simulations = analyze_attrition(
        d5_index_rows=rows,
        cluster_ids=clusters,
        simulations=10,
        seed_offset=5,
        minimum_train_programs=2,
    )
    assert summary["hamilton_quotas_train_validation_test"] == [3, 1, 4]
    assert len(simulations) == 10
    assert all(row["retained_fraction"] < 1.0 for row in simulations)
    assert "not estimand-preserving" in summary["decision"]
