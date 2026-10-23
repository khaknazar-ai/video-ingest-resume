"""Pure timestamp and frame-selection helpers for decoder backends."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction
from typing import Literal

SelectionPolicy = Literal["at_or_after", "nearest"]


@dataclass(frozen=True, slots=True)
class TimestampObservation:
    """A raw PTS plus its normalized user-timeline timestamp."""

    pts: int
    time_base: Fraction
    timestamp_seconds: float
    decode_ordinal: int


def as_fraction(value: Fraction | str) -> Fraction:
    """Parse a positive rational time base without losing precision."""

    result = value if isinstance(value, Fraction) else Fraction(value)
    if result <= 0:
        raise ValueError("time_base must be positive")
    return result


def pts_to_seconds(pts: int, time_base: Fraction | str) -> float:
    """Convert a raw PTS to seconds using its rational time base."""

    if isinstance(pts, bool) or not isinstance(pts, int):
        raise ValueError("pts must be an integer")
    result = float(pts * as_fraction(time_base))
    if not math.isfinite(result):
        raise ValueError("PTS conversion produced a non-finite timestamp")
    return result


def normalize_pts(
    pts: int,
    time_base: Fraction | str,
    *,
    origin_pts: int,
    decode_ordinal: int,
) -> TimestampObservation:
    """Normalize a raw PTS to the first displayed frame's timeline origin."""

    if isinstance(origin_pts, bool) or not isinstance(origin_pts, int):
        raise ValueError("origin_pts must be an integer")
    if (
        isinstance(decode_ordinal, bool)
        or not isinstance(decode_ordinal, int)
        or decode_ordinal < 0
    ):
        raise ValueError("decode_ordinal must be a non-negative integer")
    rational = as_fraction(time_base)
    timestamp = float((pts - origin_pts) * rational)
    if not math.isfinite(timestamp) or timestamp < 0:
        raise ValueError("normalized PTS must be finite and non-negative")
    return TimestampObservation(pts, rational, timestamp, decode_ordinal)


def select_timestamp_index(
    timestamps: Sequence[float],
    requested_seconds: float,
    *,
    policy: SelectionPolicy = "at_or_after",
) -> int | None:
    """Select an observation without returning a previous frame for at_or_after."""

    if not math.isfinite(requested_seconds) or requested_seconds < 0:
        raise ValueError("requested_seconds must be finite and non-negative")
    if policy not in {"at_or_after", "nearest"}:
        raise ValueError("policy must be 'at_or_after' or 'nearest'")
    if any(not math.isfinite(timestamp) or timestamp < 0 for timestamp in timestamps):
        raise ValueError("timestamps must be finite and non-negative")
    if policy == "at_or_after":
        return next(
            (index for index, timestamp in enumerate(timestamps) if timestamp >= requested_seconds),
            None,
        )
    if not timestamps:
        return None
    return min(
        range(len(timestamps)),
        key=lambda index: (abs(timestamps[index] - requested_seconds), timestamps[index], index),
    )
