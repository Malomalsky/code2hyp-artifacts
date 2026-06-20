from __future__ import annotations

from .curvature import (
    ball_growth_profile,
    forman_ricci_curvature,
    ollivier_ricci_curvature,
    summarize_curvature,
    undirected_edges,
)
from .graphs import SimpleGraph


def code_geometry_profile(
    graph: SimpleGraph,
    *,
    curvature_threshold: float = 0.05,
    growth_radii: tuple[int, ...] = (1, 2, 3),
    include_ollivier: bool = False,
    ollivier_idleness: float = 0.0,
) -> dict[str, float | int]:
    """Compose the first-order intrinsic geometry profile Phi(G).

    This MVP deliberately includes only fast, deterministic graph diagnostics:
    size controls, unweighted Forman curvature and graph-ball growth. Expensive
    metrics such as Ollivier-Ricci curvature and embedding stress should be
    added as separate opt-in layers with their own tests.
    """
    edge_count = len(undirected_edges(graph))
    curvatures = forman_ricci_curvature(graph)
    profile: dict[str, float | int] = {
        "node_count": len(graph.nodes),
        "edge_count": edge_count,
    }
    profile.update(
        summarize_curvature(
            curvatures.values(),
            threshold=curvature_threshold,
            prefix="forman",
        )
    )
    if include_ollivier:
        ollivier_curvatures = ollivier_ricci_curvature(
            graph,
            idleness=ollivier_idleness,
        )
        profile.update(
            summarize_curvature(
                ollivier_curvatures.values(),
                threshold=curvature_threshold,
                prefix="ollivier",
            )
        )
    profile.update(ball_growth_profile(graph, radii=growth_radii))
    return profile
