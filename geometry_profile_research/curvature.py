from __future__ import annotations

from collections import deque
from statistics import fmean, pstdev
from typing import Iterable, Sequence

from .graphs import SimpleGraph


def undirected_edges(graph: SimpleGraph) -> list[tuple[str, str]]:
    """Return stable undirected edges from a `SimpleGraph`."""
    edges = set()
    for left in graph.nodes:
        for right in graph.neighbors(left):
            edges.add(tuple(sorted((left, right))))
    return sorted(edges)


def forman_ricci_curvature(graph: SimpleGraph) -> dict[tuple[str, str], float]:
    """Compute unweighted Forman-Ricci curvature for each graph edge.

    For a simple unweighted graph, the commonly used edge-level proxy is
    F(u,v) = 4 - deg(u) - deg(v). It is cheap and interpretable enough for
    atlas-scale screening, but it should not replace Ollivier-Ricci curvature
    in final claims about local transport geometry.
    """
    curvatures: dict[tuple[str, str], float] = {}
    for left, right in undirected_edges(graph):
        degree_left = len(graph.neighbors(left))
        degree_right = len(graph.neighbors(right))
        curvatures[(left, right)] = float(4 - degree_left - degree_right)
    return curvatures


def local_probability_measure(
    graph: SimpleGraph,
    node: str,
    *,
    idleness: float = 0.0,
) -> dict[str, float]:
    """Return the lazy random-walk measure used by Ollivier-Ricci curvature."""
    if not 0.0 <= idleness <= 1.0:
        raise ValueError("idleness must be in [0, 1]")
    neighbors = sorted(graph.neighbors(node))
    if not neighbors:
        return {node: 1.0}
    neighbor_mass = (1.0 - idleness) / len(neighbors)
    measure = {neighbor: neighbor_mass for neighbor in neighbors}
    if idleness > 0.0:
        measure[node] = measure.get(node, 0.0) + idleness
    return measure


def _lookup_cost(costs: dict[tuple[str, str], float], left: str, right: str) -> float:
    if left == right:
        return 0.0
    if (left, right) in costs:
        return float(costs[(left, right)])
    if (right, left) in costs:
        return float(costs[(right, left)])
    raise KeyError(f"missing transport cost for {left!r}, {right!r}")


def wasserstein_distance(
    left: dict[str, float],
    right: dict[str, float],
    costs: dict[tuple[str, str], float],
    *,
    tolerance: float = 1e-12,
) -> float:
    """Solve finite W1 transport exactly enough for small local measures.

    The implementation uses a transportation-simplex style loop: start from a
    northwest-corner feasible plan and iteratively improve it along negative
    reduced-cost cycles. Supports in Ollivier-Ricci local measures are small,
    so this dependency-free solver is sufficient for reproducible atlas runs.
    """
    supply_nodes = [node for node, mass in sorted(left.items()) if mass > tolerance]
    demand_nodes = [node for node, mass in sorted(right.items()) if mass > tolerance]
    supply = [float(left[node]) for node in supply_nodes]
    demand = [float(right[node]) for node in demand_nodes]
    if abs(sum(supply) - sum(demand)) > 1e-9:
        raise ValueError("transport distributions must have the same total mass")
    if not supply_nodes and not demand_nodes:
        return 0.0

    cost_matrix = [
        [_lookup_cost(costs, supply_node, demand_node) for demand_node in demand_nodes]
        for supply_node in supply_nodes
    ]
    plan = [[0.0 for _ in demand_nodes] for _ in supply_nodes]
    basic: set[tuple[int, int]] = set()
    remaining_supply = supply[:]
    remaining_demand = demand[:]
    row = 0
    col = 0
    while row < len(supply_nodes) and col < len(demand_nodes):
        amount = min(remaining_supply[row], remaining_demand[col])
        plan[row][col] = amount
        basic.add((row, col))
        remaining_supply[row] -= amount
        remaining_demand[col] -= amount
        supply_empty = remaining_supply[row] <= tolerance
        demand_empty = remaining_demand[col] <= tolerance
        if supply_empty and row + 1 < len(supply_nodes):
            row += 1
        elif demand_empty and col + 1 < len(demand_nodes):
            col += 1
        else:
            row += 1
            col += 1

    # Add zero-flow basics until the basis is connected enough for potentials.
    while len(basic) < len(supply_nodes) + len(demand_nodes) - 1:
        candidates = [
            (i, j)
            for i in range(len(supply_nodes))
            for j in range(len(demand_nodes))
            if (i, j) not in basic
        ]
        basic.add(min(candidates, key=lambda item: cost_matrix[item[0]][item[1]]))

    for _ in range(len(supply_nodes) * len(demand_nodes) * 4):
        u: list[float | None] = [None for _ in supply_nodes]
        v: list[float | None] = [None for _ in demand_nodes]
        u[0] = 0.0
        changed = True
        while changed:
            changed = False
            for i, j in basic:
                if u[i] is not None and v[j] is None:
                    v[j] = cost_matrix[i][j] - u[i]
                    changed = True
                elif v[j] is not None and u[i] is None:
                    u[i] = cost_matrix[i][j] - v[j]
                    changed = True

        entering: tuple[int, int] | None = None
        entering_reduced_cost = 0.0
        for i in range(len(supply_nodes)):
            for j in range(len(demand_nodes)):
                if (i, j) in basic:
                    continue
                reduced_cost = cost_matrix[i][j] - (u[i] or 0.0) - (v[j] or 0.0)
                if reduced_cost < entering_reduced_cost - tolerance:
                    entering = (i, j)
                    entering_reduced_cost = reduced_cost
        if entering is None:
            break

        cycle = _transport_cycle(basic, entering, len(supply_nodes), len(demand_nodes))
        negative_positions = cycle[1::2]
        theta = min(plan[i][j] for i, j in negative_positions)
        for index, (i, j) in enumerate(cycle):
            if index % 2 == 0:
                plan[i][j] += theta
            else:
                plan[i][j] -= theta
        basic.add(entering)
        leaving_candidates = [
            position
            for position in negative_positions
            if plan[position[0]][position[1]] <= tolerance
        ]
        leaving = leaving_candidates[0]
        plan[leaving[0]][leaving[1]] = 0.0
        basic.remove(leaving)

    return sum(
        plan[i][j] * cost_matrix[i][j]
        for i in range(len(supply_nodes))
        for j in range(len(demand_nodes))
    )


def _transport_cycle(
    basic: set[tuple[int, int]],
    entering: tuple[int, int],
    row_count: int,
    col_count: int,
) -> list[tuple[int, int]]:
    """Find the alternating transportation cycle created by an entering cell."""
    cells = set(basic)
    cells.add(entering)

    def search(
        path: list[tuple[int, int]],
        use_row: bool,
    ) -> list[tuple[int, int]] | None:
        current_row, current_col = path[-1]
        if use_row:
            candidates = [
                (current_row, col)
                for col in range(col_count)
                if col != current_col and (current_row, col) in cells
            ]
        else:
            candidates = [
                (row, current_col)
                for row in range(row_count)
                if row != current_row and (row, current_col) in cells
            ]
        for candidate in candidates:
            if candidate == entering and len(path) >= 4:
                return path
            if candidate in path:
                continue
            result = search([*path, candidate], not use_row)
            if result is not None:
                return result
        return None

    cycle = search([entering], use_row=True)
    if cycle is None:
        cycle = search([entering], use_row=False)
    if cycle is None:
        raise ValueError("could not find transportation cycle")
    return cycle


def _single_source_distances(graph: SimpleGraph, source: str) -> dict[str, int]:
    distances = {source: 0}
    queue = deque([source])
    while queue:
        node = queue.popleft()
        for neighbor in graph.neighbors(node):
            if neighbor in distances:
                continue
            distances[neighbor] = distances[node] + 1
            queue.append(neighbor)
    return distances


def _support_distance_costs(graph: SimpleGraph, sources: Iterable[str], targets: Iterable[str]) -> dict[tuple[str, str], float]:
    costs: dict[tuple[str, str], float] = {}
    target_set = set(targets)
    for source in sources:
        distances = _single_source_distances(graph, source)
        for target in target_set:
            if target not in distances:
                raise ValueError(f"graph is disconnected between {source!r} and {target!r}")
            costs[(source, target)] = float(distances[target])
    return costs


def ollivier_ricci_curvature(
    graph: SimpleGraph,
    *,
    idleness: float = 0.0,
) -> dict[tuple[str, str], float]:
    """Compute edge Ollivier-Ricci curvature for an unweighted graph."""
    curvatures: dict[tuple[str, str], float] = {}
    for left, right in undirected_edges(graph):
        left_measure = local_probability_measure(graph, left, idleness=idleness)
        right_measure = local_probability_measure(graph, right, idleness=idleness)
        costs = _support_distance_costs(graph, left_measure, right_measure)
        transport_distance = wasserstein_distance(left_measure, right_measure, costs)
        edge_distance = _lookup_cost(costs, left, right)
        curvatures[(left, right)] = 1.0 - transport_distance / edge_distance
    return curvatures


def _quantile(sorted_values: Sequence[float], probability: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    position = probability * (len(sorted_values) - 1)
    lower = int(position)
    upper = min(len(sorted_values) - 1, lower + 1)
    fraction = position - lower
    return float(sorted_values[lower] * (1.0 - fraction) + sorted_values[upper] * fraction)


def summarize_curvature(
    values: Iterable[float],
    *,
    threshold: float = 0.05,
    prefix: str = "",
) -> dict[str, float | int]:
    """Summarize edge-curvature distribution with signed mass indicators."""
    numeric_values = [float(value) for value in values]
    key = f"{prefix}_" if prefix else ""
    if not numeric_values:
        return {
            f"{key}count": 0,
            f"{key}mean": 0.0,
            f"{key}std": 0.0,
            f"{key}q05": 0.0,
            f"{key}q50": 0.0,
            f"{key}q95": 0.0,
            f"{key}negative_mass": 0.0,
            f"{key}near_zero_mass": 0.0,
            f"{key}positive_mass": 0.0,
        }

    sorted_values = sorted(numeric_values)
    total = len(numeric_values)
    negative = sum(1 for value in numeric_values if value < -threshold)
    positive = sum(1 for value in numeric_values if value > threshold)
    near_zero = total - negative - positive
    return {
        f"{key}count": total,
        f"{key}mean": fmean(numeric_values),
        f"{key}std": pstdev(numeric_values) if total > 1 else 0.0,
        f"{key}q05": _quantile(sorted_values, 0.05),
        f"{key}q50": _quantile(sorted_values, 0.50),
        f"{key}q95": _quantile(sorted_values, 0.95),
        f"{key}negative_mass": negative / total,
        f"{key}near_zero_mass": near_zero / total,
        f"{key}positive_mass": positive / total,
    }


def _single_source_distances_up_to_radius(
    graph: SimpleGraph,
    source: str,
    radius: int,
) -> dict[str, int]:
    distances = {source: 0}
    queue = deque([source])
    while queue:
        node = queue.popleft()
        next_distance = distances[node] + 1
        if next_distance > radius:
            continue
        for neighbor in graph.neighbors(node):
            if neighbor in distances:
                continue
            distances[neighbor] = next_distance
            queue.append(neighbor)
    return distances


def ball_growth_profile(
    graph: SimpleGraph,
    *,
    radii: Sequence[int] = (1, 2, 3),
) -> dict[str, float]:
    """Summarize average graph ball sizes for fixed radii."""
    nodes = sorted(graph.nodes)
    if not nodes:
        return {f"ball_size_mean_r{int(radius)}": 0.0 for radius in radii}

    profile: dict[str, float] = {}
    for radius in radii:
        radius = int(radius)
        ball_sizes = [
            len(_single_source_distances_up_to_radius(graph, node, radius))
            for node in nodes
        ]
        profile[f"ball_size_mean_r{radius}"] = fmean(ball_sizes)
        profile[f"ball_size_std_r{radius}"] = pstdev(ball_sizes) if len(ball_sizes) > 1 else 0.0
    return profile
