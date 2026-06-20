from __future__ import annotations

import hashlib
import math

Point2D = tuple[float, float]

_COMMON_TOP_LEVEL_SECTORS = {
    "src": 0,
    "lib": 1,
    "app": 2,
    "tests": 16,
    "test": 16,
    "docs": 32,
    "doc": 32,
    "config": 48,
    "scripts": 56,
}


def _stable_unit_interval(value: str) -> float:
    digest = hashlib.sha256(value.encode("utf-8")).digest()
    integer = int.from_bytes(digest[:8], "big")
    return integer / float(2**64 - 1)


def _stable_signed_offset(value: str, amplitude: float) -> float:
    return (2.0 * _stable_unit_interval(value) - 1.0) * amplitude


def path_components(path: str) -> list[str]:
    normalized = str(path).replace("\\", "/").strip("/")
    if not normalized:
        return []
    return [part for part in normalized.split("/") if part]


def path_angle(path: str, top_level_sectors: int = 64) -> float:
    """Map a path to an angle where early path components dominate the sector."""
    parts = path_components(path)
    if not parts:
        return 0.0

    first = parts[0].lower()
    sector = _COMMON_TOP_LEVEL_SECTORS.get(
        first,
        int(_stable_unit_interval(first) * top_level_sectors) % top_level_sectors,
    )
    angle = (sector + 0.5) * (2.0 * math.pi / top_level_sectors)

    # Later components only refine the sector; they cannot dominate top-level branch.
    for depth, component in enumerate(parts[1:], start=1):
        amplitude = math.pi / (top_level_sectors * (2**depth))
        angle += _stable_signed_offset("/".join(parts[: depth + 1]), amplitude)

    return angle % (2.0 * math.pi)


def path_radius(
    path: str,
    beta: float = 0.45,
    gamma: float = 0.0,
    max_radius: float = 0.999,
) -> float:
    """Monotone radial map from hierarchy depth to the Poincare disk."""
    depth = len(path_components(path))
    if depth == 0 and gamma == 0:
        return 0.0
    return (1.0 - math.exp(-beta * (depth + gamma))) * max_radius


def path_to_poincare(path: str, beta: float = 0.45, gamma: float = 0.0) -> Point2D:
    """Deterministically embed a source-code path into the 2D Poincare disk."""
    radius = path_radius(path, beta=beta, gamma=gamma)
    angle = path_angle(path)
    return (radius * math.cos(angle), radius * math.sin(angle))


def euclidean_distance(left: Point2D, right: Point2D) -> float:
    return math.hypot(left[0] - right[0], left[1] - right[1])


def poincare_distance(left: Point2D, right: Point2D, curvature: float = 1.0) -> float:
    """Poincare-ball distance for curvature c > 0, sectional curvature -c."""
    if curvature <= 0:
        raise ValueError("curvature must be positive")

    c_sqrt = math.sqrt(curvature)
    lu = c_sqrt * math.hypot(*left)
    rv = c_sqrt * math.hypot(*right)
    diff = c_sqrt * euclidean_distance(left, right)

    left_den = max(1.0 - lu * lu, 1e-12)
    right_den = max(1.0 - rv * rv, 1e-12)
    argument = 1.0 + 2.0 * diff * diff / (left_den * right_den)
    return math.acosh(max(argument, 1.0)) / c_sqrt


def angular_distance(left: Point2D, right: Point2D) -> float:
    left_angle = math.atan2(left[1], left[0])
    right_angle = math.atan2(right[1], right[0])
    diff = abs((left_angle - right_angle + math.pi) % (2.0 * math.pi) - math.pi)
    return diff
