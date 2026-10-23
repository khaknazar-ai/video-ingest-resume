"""Bounded within-window tracking contracts."""

from video_intelligence.tracking.base import (
    TrackerConfig,
    TrackingFrame,
    TrackingRequest,
    TrackingResult,
    TrackObservation,
    TrackSnapshot,
    TrackState,
    TrackTransition,
)
from video_intelligence.tracking.iou import track_window

__all__ = [
    "TrackerConfig",
    "TrackingFrame",
    "TrackingRequest",
    "TrackingResult",
    "TrackObservation",
    "TrackSnapshot",
    "TrackState",
    "TrackTransition",
    "track_window",
]
