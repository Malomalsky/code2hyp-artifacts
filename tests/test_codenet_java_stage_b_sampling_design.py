from scripts.analyze_codenet_java_stage_b_sampling_design import analyze_sampling_design


def test_sampling_design_selects_largest_full_frame_candidate_with_headroom() -> None:
    rows = []
    for index in range(80):
        capacity = 40 if index < 30 else 32 if index < 40 else 16
        rows.append(
            {
                "eligible_evaluation_minimum_16": True,
                "retained_programs_after_d0_d4": capacity,
                "distinct_users_after_d0_d4": capacity,
            }
        )

    result = analyze_sampling_design(
        rows,
        candidate_train_programs=(16, 32, 40),
        minimum_train_eligibility_headroom=1.25,
    )

    assert result["selected_train_programs_per_cluster"] == 32
    assert result["selected_quotas_train_validation_test"] == [30, 10, 40]
    by_k = {row["train_programs_per_cluster"]: row for row in result["candidates"]}
    assert by_k[40]["full_frame_feasible"] is True
    assert by_k[40]["train_eligibility_headroom_ratio"] == 1.0
