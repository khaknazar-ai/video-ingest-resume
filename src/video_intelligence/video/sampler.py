"""Memory-safe timestamp sampling."""

from __future__ import annotations

import math
from collections.abc import Sequence

from video_intelligence.core.time import seconds_to_frame_index
from video_intelligence.core.types import FrameRef, FrameRequest, Shot, VideoRecord


def _validate_interval(
    video: VideoRecord,
    *,
    start_seconds: float,
    end_seconds: float | None,
) -> tuple[float, float, float]:
    if video.metadata.duration_seconds is None:
        raise ValueError("sampling requires a known duration")
    duration = float(video.metadata.duration_seconds)
    start = float(start_seconds)
    end = duration if end_seconds is None else float(end_seconds)
    if not math.isfinite(start) or not math.isfinite(end):
        raise ValueError("sampling interval must be finite")
    if start < 0 or end < start or end > duration + 1e-9:
        raise ValueError("sampling interval must be inside the video duration")
    return duration, start, min(end, duration)


def uniform_sample_requests(
    video: VideoRecord,
    *,
    target_fps: float,
    max_frames: int | None = None,
    start_seconds: float = 0.0,
    end_seconds: float | None = None,
) -> list[FrameRequest]:
    """Build a half-open request grid at exactly ``1 / target_fps`` seconds."""

    if not math.isfinite(target_fps) or target_fps <= 0:
        raise ValueError("target_fps must be a finite positive number")
    _, start, end = _validate_interval(
        video, start_seconds=start_seconds, end_seconds=end_seconds
    )
    interval = end - start
    if interval <= 0:
        return []
    count = max(1, math.ceil(interval * target_fps - 1e-12))
    if max_frames is not None:
        if isinstance(max_frames, bool) or not isinstance(max_frames, int) or max_frames <= 0:
            raise ValueError("max_frames must be a positive integer when provided")
        count = min(count, max_frames)
    return [
        FrameRequest(video.video_id, start + offset / target_fps)
        for offset in range(count)
    ]


def uniform_budget_requests(
    video: VideoRecord,
    *,
    frame_budget: int,
    start_seconds: float = 0.0,
    end_seconds: float | None = None,
) -> list[FrameRequest]:
    """Build exactly ``frame_budget`` evenly distributed requests in an interval."""

    if isinstance(frame_budget, bool) or not isinstance(frame_budget, int) or frame_budget <= 0:
        raise ValueError("frame_budget must be a positive integer")
    _, start, end = _validate_interval(
        video, start_seconds=start_seconds, end_seconds=end_seconds
    )
    interval = end - start
    if interval <= 0:
        return []
    step = interval / frame_budget
    return [
        FrameRequest(video.video_id, start + offset * step)
        for offset in range(frame_budget)
    ]


def scene_aware_sample_requests(
    video: VideoRecord,
    shots: Sequence[Shot],
    *,
    anchors_per_shot: int = 2,
    max_frames: int | None = None,
) -> list[FrameRequest]:
    """Create chronological coverage anchors across validated shot intervals."""

    if (
        isinstance(anchors_per_shot, bool)
        or not isinstance(anchors_per_shot, int)
        or anchors_per_shot <= 0
    ):
        raise ValueError("anchors_per_shot must be a positive integer")
    if max_frames is not None and (
        isinstance(max_frames, bool) or not isinstance(max_frames, int) or max_frames <= 0
    ):
        raise ValueError("max_frames must be a positive integer when provided")
    if video.metadata.duration_seconds is None:
        raise ValueError("scene-aware sampling requires a known duration")
    ordered_shots = sorted(shots, key=lambda shot: (shot.start_seconds, shot.end_seconds))
    validated_shots: list[tuple[Shot, float]] = []
    previous_end = 0.0
    for shot in ordered_shots:
        if shot.video_id != video.video_id:
            raise ValueError("shot belongs to another video")
        if shot.start_seconds < previous_end:
            raise ValueError("shots must not overlap")
        if shot.end_seconds > video.metadata.duration_seconds:
            raise ValueError("shot interval must be inside video duration")
        interval = shot.end_seconds - shot.start_seconds
        if interval <= 0:
            continue
        validated_shots.append((shot, interval))
        previous_end = shot.end_seconds
    if not validated_shots:
        return []
    selected: list[FrameRequest] = []
    for offset in range(anchors_per_shot):
        for shot, interval in validated_shots:
            selected.append(
                FrameRequest(
                    video.video_id,
                    shot.start_seconds + offset * interval / anchors_per_shot,
                    source="scene_aware",
                )
            )
            if max_frames is not None and len(selected) >= max_frames:
                return sorted(selected, key=lambda request: request.requested_timestamp_seconds)
    return sorted(selected, key=lambda request: request.requested_timestamp_seconds)


def _requests_to_frame_refs(
    video: VideoRecord,
    requests: Sequence[FrameRequest],
    *,
    deduplicate: bool,
) -> list[FrameRef]:
    if any(
        value is None
        for value in (
            video.metadata.fps,
            video.metadata.frame_count,
            video.metadata.width,
            video.metadata.height,
        )
    ):
        raise ValueError("indexed sampling requires fps, frame count and dimensions")
    max_index = max(video.metadata.frame_count - 1, 0)
    seen: set[int] = set()
    frames: list[FrameRef] = []
    for request in requests:
        frame_index = min(
            seconds_to_frame_index(
                request.requested_timestamp_seconds,
                video.metadata.fps,
                rounding="nearest",
            ),
            max_index,
        )
        if deduplicate and frame_index in seen:
            continue
        seen.add(frame_index)
        frames.append(
            FrameRef(
                video_id=video.video_id,
                frame_index=frame_index,
                timestamp_seconds=request.requested_timestamp_seconds,
                width=video.metadata.width,
                height=video.metadata.height,
                source_path=video.source_path,
            )
        )
    return frames


def uniform_sample_frames(
    video: VideoRecord,
    *,
    target_fps: float,
    max_frames: int | None = None,
    start_seconds: float = 0.0,
    end_seconds: float | None = None,
    deduplicate: bool = True,
) -> list[FrameRef]:
    """Return legacy frame references for the exact ``uniform_fps`` request grid.

    ``FrameRequest`` is the authoritative planned output. This wrapper maps requests
    to indexed references for the legacy OpenCV path and removes duplicate indexes by
    default. Use :func:`uniform_budget_requests` when a fixed temporal budget is wanted.
    """

    if not isinstance(deduplicate, bool):
        raise ValueError("deduplicate must be a boolean")
    requests = uniform_sample_requests(
        video,
        target_fps=target_fps,
        max_frames=max_frames,
        start_seconds=start_seconds,
        end_seconds=end_seconds,
    )
    return _requests_to_frame_refs(video, requests, deduplicate=deduplicate)


def uniform_budget_frames(
    video: VideoRecord,
    *,
    frame_budget: int,
    start_seconds: float = 0.0,
    end_seconds: float | None = None,
    deduplicate: bool = True,
) -> list[FrameRef]:
    """Map a fixed-budget request plan to indexed references."""

    if not isinstance(deduplicate, bool):
        raise ValueError("deduplicate must be a boolean")
    requests = uniform_budget_requests(
        video,
        frame_budget=frame_budget,
        start_seconds=start_seconds,
        end_seconds=end_seconds,
    )
    return _requests_to_frame_refs(video, requests, deduplicate=deduplicate)
