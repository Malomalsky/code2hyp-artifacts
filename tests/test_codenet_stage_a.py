from __future__ import annotations

from geometry_profile_research.codenet_stage_a import select_calibration_pairs


def _train_rows(cluster_count: int = 4, programs_per_cluster: int = 5) -> list[dict[str, str]]:
    return [
        {
            "cluster_id": f"cluster-{cluster:02d}",
            "role": "train",
            "source_relpath": f"p{cluster:02d}/s{program:03d}.py",
            "split": "train",
        }
        for cluster in range(cluster_count)
        for program in range(programs_per_cluster)
    ]


def test_calibration_pairs_are_deterministic_unique_and_train_only() -> None:
    rows = _train_rows()
    key = bytes(range(64))

    first = select_calibration_pairs(
        rows,
        beacon_key=key,
        dataset_revision="1.0.0",
        domain="code2hyp/test/calibration/v1",
        same_cluster_count=12,
        cross_cluster_count=6,
    )
    second = select_calibration_pairs(
        tuple(reversed(rows)),
        beacon_key=key,
        dataset_revision="1.0.0",
        domain="code2hyp/test/calibration/v1",
        same_cluster_count=12,
        cross_cluster_count=6,
    )

    assert first == second
    assert len(first) == 18
    pairs = {(row["left_source_relpath"], row["right_source_relpath"]) for row in first}
    assert len(pairs) == 18
    assert [row["pair_index"] for row in first] == list(range(18))
    assert sum(row["pair_type"] == "same_cluster" for row in first) == 12
    assert sum(row["pair_type"] == "cross_cluster" for row in first) == 6
    assert all(len(row["selection_digest"]) == 64 for row in first)


def test_calibration_pairs_reject_validation_rows() -> None:
    rows = _train_rows()
    rows[0]["split"] = "validation"

    try:
        select_calibration_pairs(
            rows,
            beacon_key=bytes(range(64)),
            dataset_revision="1.0.0",
            domain="code2hyp/test/calibration/v1",
            same_cluster_count=1,
            cross_cluster_count=1,
        )
    except ValueError as error:
        assert "only frozen training" in str(error)
    else:
        raise AssertionError("validation rows must be rejected")
