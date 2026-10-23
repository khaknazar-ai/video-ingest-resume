"""Timestamp helpers for constant-FPS fixtures and decoder-facing contracts."""

from __future__ import annotations

import math


def _validate_fps(fps: float) -> float:
    value = float(fps)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"fps must be a finite positive number, got {fps!r}")
    return value


def _validate_seconds(seconds: float, *, name: str = "seconds") -> float:
    value = float(seconds)
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be a finite non-negative number, got {seconds!r}")
    return value


def frame_index_to_seconds(frame_index: int, fps: float) -> float:
    """Convert a zero-based frame index to its start timestamp."""

    if isinstance(frame_index, bool) or int(frame_index) != frame_index or frame_index < 0:
        raise ValueError(f"frame_index must be a non-negative integer, got {frame_index!r}")
    return int(frame_index) / _validate_fps(fps)


def seconds_to_frame_index(seconds: float, fps: float, *, rounding: str = "floor") -> int:
    """Map a timestamp to a frame index using an explicit rounding policy."""

    value = _validate_seconds(seconds)
    scaled = value * _validate_fps(fps)
    if rounding == "floor":
        return math.floor(scaled)
    if rounding == "ceil":
        return math.ceil(scaled)
    if rounding == "nearest":
        return math.floor(scaled + 0.5)
    raise ValueError("rounding must be one of: floor, ceil, nearest")


def clamp_timestamp(seconds: float, duration_seconds: float) -> float:
    """Clamp a timestamp to the closed interval [0, duration]."""

    duration = _validate_seconds(duration_seconds, name="duration_seconds")
    return min(_validate_seconds(seconds), duration)
