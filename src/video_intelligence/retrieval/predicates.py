"""Shared, deterministic predicates used by retrieval and result validation."""

from __future__ import annotations

import math
from collections.abc import Sequence


def within_time(
    timestamp: float, start: float | None = None, end: float | None = None
) -> bool:
    """Apply the canonical half-open interval without rounding its boundaries."""
    return (start is None or timestamp >= start) and (end is None or timestamp < end)


def spatial_relation_holds(
    relation: str,
    first: Sequence[float],
    second: Sequence[float],
    *,
    width: int,
    height: int,
    near_distance_fraction: float | None = None,
) -> bool:
    first_x, first_y = (first[0] + first[2]) / 2, (first[1] + first[3]) / 2
    second_x, second_y = (second[0] + second[2]) / 2, (second[1] + second[3]) / 2
    if relation == "left":
        return first_x < second_x
    if relation == "right":
        return first_x > second_x
    if relation == "overlap":
        return max(first[0], second[0]) < min(first[2], second[2]) and max(
            first[1], second[1]
        ) < min(first[3], second[3])
    if relation != "near":
        raise ValueError("unsupported spatial relation")
    if (
        isinstance(near_distance_fraction, bool)
        or not isinstance(near_distance_fraction, (int, float))
        or not math.isfinite(near_distance_fraction)
        or near_distance_fraction <= 0
    ):
        raise ValueError("near relation requires an explicit finite positive distance policy")
    return math.hypot(first_x - second_x, first_y - second_y) <= (
        max(width, height) * near_distance_fraction
    )
