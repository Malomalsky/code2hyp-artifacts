from __future__ import annotations

from itertools import combinations, product
from typing import Mapping, Sequence


DistanceMatrix = Sequence[Sequence[float]]


def four_point_delta(
    *,
    ab: float,
    ac: float,
    ad: float,
    bc: float,
    bd: float,
    cd: float,
) -> float:
    """Four-point hyperbolicity delta for one metric quadruple."""

    sums = sorted((ab + cd, ac + bd, ad + bc))
    return 0.5 * (sums[2] - sums[1])


def matrix_four_point_delta(distances: DistanceMatrix, a: int, b: int, c: int, d: int) -> float:
    return four_point_delta(
        ab=float(distances[a][b]),
        ac=float(distances[a][c]),
        ad=float(distances[a][d]),
        bc=float(distances[b][c]),
        bd=float(distances[b][d]),
        cd=float(distances[c][d]),
    )


def matrix_max_four_point_delta(distances: DistanceMatrix) -> float:
    """Maximum four-point delta over all quadruples in a finite metric space."""

    size = len(distances)
    if size < 4:
        return 0.0
    return max(matrix_four_point_delta(distances, *quadruple) for quadruple in combinations(range(size), 4))


def line_tree_distance_matrix(node_count: int) -> tuple[tuple[float, ...], ...]:
    if node_count < 0:
        raise ValueError("node_count must be non-negative")
    return tuple(tuple(float(abs(left - right)) for right in range(node_count)) for left in range(node_count))


def product_grid_l1_distance_matrix(side_length: int) -> tuple[tuple[float, ...], ...]:
    """L1 metric on the four corners of ``[0, side_length] x [0, side_length]``."""

    if side_length < 0:
        raise ValueError("side_length must be non-negative")
    points = tuple(product((0, side_length), repeat=2))
    return tuple(
        tuple(float(abs(left[0] - right[0]) + abs(left[1] - right[1])) for right in points)
        for left in points
    )


def upper_triangle_records_to_distance_matrix(
    records: Sequence[Mapping[str, object]],
    *,
    path_count: int,
    distance_key: str,
) -> tuple[tuple[float, ...], ...]:
    if path_count < 0:
        raise ValueError("path_count must be non-negative")
    matrix = [[0.0 for _ in range(path_count)] for _ in range(path_count)]
    for record in records:
        left_index = int(record["left_index"])
        right_index = int(record["right_index"])
        if left_index == right_index:
            continue
        if left_index < 0 or right_index < 0 or left_index >= path_count or right_index >= path_count:
            raise ValueError("record index is outside the distance matrix")
        distance = float(record[distance_key])
        matrix[left_index][right_index] = distance
        matrix[right_index][left_index] = distance
    return tuple(tuple(row) for row in matrix)
