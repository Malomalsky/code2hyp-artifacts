from __future__ import annotations

import torch

from geometry_profile_research.codenet_stage_a_evaluation import (
    calibrate_euclidean_role_weights,
    calibration_cost_scale,
    full_gallery_sinkhorn_divergence,
    precompute_self_objectives,
    summarize_problem_macro_retrieval,
)
from geometry_profile_research.constant_curvature import ProductMeasure, RoleProductGeometry


def _measure(values: list[list[list[float]]], mass: list[float] | None = None) -> ProductMeasure:
    points = torch.tensor(values, dtype=torch.float64)
    count = points.shape[0]
    return ProductMeasure(
        points=points,
        mass=torch.full((count,), 1.0 / count, dtype=torch.float64)
        if mass is None
        else torch.tensor(mass, dtype=torch.float64),
    )


def _calibration_fixture() -> tuple[dict[str, ProductMeasure], tuple[dict[str, str], ...]]:
    measures = {
        "a.py": _measure([[[0.0], [0.0], [1.0]], [[1.0], [1.0], [2.0]]]),
        "b.py": _measure([[[2.0], [2.0], [3.0]], [[3.0], [3.0], [4.0]]]),
        "c.py": _measure([[[4.0], [4.0], [5.0]], [[5.0], [5.0], [6.0]]]),
    }
    pairs = (
        {
            "pair_type": "same_cluster",
            "left_source_relpath": "a.py",
            "right_source_relpath": "b.py",
        },
        {
            "pair_type": "cross_cluster",
            "left_source_relpath": "a.py",
            "right_source_relpath": "c.py",
        },
    )
    return measures, pairs


def test_calibration_scale_gives_equal_weight_to_program_pairs() -> None:
    measures, pairs = _calibration_fixture()
    geometry = RoleProductGeometry(
        factor_curvatures=(0.0, 0.0, 0.0),
        factor_weights=(1.0, 0.0, 0.0),
        side_weight=0.0,
        unoriented=False,
    )

    summary = calibration_cost_scale(measures, pairs, geometry, batch_size=2)

    first_pair_median = torch.median(
        geometry.path_cost_matrix(measures["a.py"], measures["b.py"]).reshape(-1)
    )
    second_pair_median = torch.median(
        geometry.path_cost_matrix(measures["a.py"], measures["c.py"]).reshape(-1)
    )
    expected = float(torch.quantile(torch.stack((first_pair_median, second_pair_median)), 0.5))
    assert summary.cost_scale == expected
    assert summary.same_cluster_pair_count == 1
    assert summary.cross_cluster_pair_count == 1


def test_role_weight_calibration_enforces_equal_endpoint_weights() -> None:
    measures, pairs = _calibration_fixture()

    result = calibrate_euclidean_role_weights(measures, pairs, batch_size=2)

    assert result.euclidean_weights[1] == result.euclidean_weights[2]
    assert result.canonical_weights == tuple(value / 4.0 for value in result.euclidean_weights)
    assert result.pair_count == 2


def test_full_gallery_batch_matches_scalar_divergences() -> None:
    queries = (
        _measure([[[0.00], [0.10], [0.20]], [[0.05], [0.20], [0.30]]]),
        _measure([[[0.02], [0.12], [0.24]]]),
    )
    gallery = (
        _measure([[[0.01], [0.11], [0.19]], [[0.04], [0.18], [0.33]]]),
        _measure([[[0.08], [0.31], [0.12]]]),
        _measure(
            [[[0.03], [0.16], [0.27]], [[0.06], [0.21], [0.37]], [[0.01], [0.10], [0.29]]],
            [0.2, 0.3, 0.5],
        ),
    )
    geometry = RoleProductGeometry(
        factor_curvatures=(0.8, 0.0, 0.0),
        factor_weights=(0.25, 0.75, 0.75),
        side_weight=0.0,
        unoriented=True,
    )

    actual = full_gallery_sinkhorn_divergence(
        queries,
        gallery,
        geometry,
        epsilon=0.04,
        query_batch_size=2,
        gallery_batch_size=2,
        sinkhorn_iterations=300,
        projection_iterations=2048,
        marginal_tolerance=1e-9,
    )
    expected = torch.empty_like(actual)
    for query_index, query in enumerate(queries):
        for gallery_index, candidate in enumerate(gallery):
            expected[query_index, gallery_index] = geometry.sinkhorn_divergence(
                query,
                candidate,
                epsilon=0.04,
                iterations=300,
                projection_iterations=2048,
            )

    torch.testing.assert_close(actual, expected, atol=1e-9, rtol=1e-9)


def test_full_gallery_reuses_precomputed_self_objectives() -> None:
    measures = (
        _measure([[[0.0], [0.1], [0.2]]]),
        _measure([[[0.1], [0.2], [0.3]], [[0.2], [0.4], [0.6]]]),
    )
    geometry = RoleProductGeometry(
        factor_curvatures=(0.0, 0.0, 0.0),
        factor_weights=(1.0, 1.0, 1.0),
        side_weight=0.0,
        unoriented=True,
    )
    self_values = precompute_self_objectives(
        measures,
        geometry,
        epsilon=0.1,
        batch_size=2,
        sinkhorn_iterations=200,
        projection_iterations=1024,
    )

    scores = full_gallery_sinkhorn_divergence(
        measures,
        measures,
        geometry,
        epsilon=0.1,
        query_batch_size=2,
        gallery_batch_size=2,
        sinkhorn_iterations=200,
        projection_iterations=1024,
        query_self_objectives=self_values,
        gallery_self_objectives=self_values,
    )

    torch.testing.assert_close(torch.diagonal(scores), torch.zeros(2, dtype=torch.float64), atol=1e-10, rtol=0.0)


def test_problem_macro_map_weights_problems_not_queries() -> None:
    distances = torch.tensor(
        [
            [0.1, 0.2, 0.8, 0.9],
            [0.4, 0.5, 0.1, 0.2],
            [0.1, 0.2, 0.8, 0.9],
        ],
        dtype=torch.float64,
    )

    summary = summarize_problem_macro_retrieval(
        distances,
        query_ids=("qa1", "qa2", "qb1"),
        query_problem_ids=("A", "A", "B"),
        gallery_ids=("ga1", "ga2", "gb1", "gb2"),
        gallery_problem_ids=("A", "A", "B", "B"),
        r=2,
    )

    assert summary.task_scores["A"] == 0.5
    assert summary.task_scores["B"] == 0.0
    assert summary.problem_macro_map_at_r == 0.25
    assert summary.query_macro_map_at_r == 1.0 / 3.0


def test_retrieval_ties_are_broken_by_gallery_id() -> None:
    distances = torch.zeros((1, 4), dtype=torch.float64)

    summary = summarize_problem_macro_retrieval(
        distances,
        query_ids=("q",),
        query_problem_ids=("A",),
        gallery_ids=("z-b", "a-a", "y-b", "b-a"),
        gallery_problem_ids=("B", "A", "B", "A"),
        r=2,
    )

    assert summary.query_scores["q"] == 1.0
