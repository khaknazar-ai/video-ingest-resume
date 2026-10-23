"""Resource-light shot detection for CPU fixtures and bounded previews."""

from __future__ import annotations

import math

import numpy as np

from video_intelligence.core.types import DecodeIssue, Shot, VideoRecord
from video_intelligence.video.decoder import DecodeError, VideoDecoder
from video_intelligence.video.sampler import uniform_sample_frames


def _color_histogram(image: np.ndarray) -> np.ndarray:
    histogram, _ = np.histogram(image, bins=32, range=(0, 256))
    normalized = histogram.astype(np.float32)
    total = float(normalized.sum())
    return normalized / total if total > 0 else normalized


def detect_shots(
    video: VideoRecord,
    *,
    sample_fps: float = 2.0,
    cut_threshold: float = 0.35,
    max_frames: int | None = 256,
) -> list[Shot]:
    """Detect coarse cuts from color-histogram distance on decoded preview frames.

    This is a bounded heuristic for sampling allocation. It is not a semantic shot
    detector and should not be used as evidence of an event by itself.
    """

    if not math.isfinite(cut_threshold) or not 0 <= cut_threshold <= 1:
        raise ValueError("cut_threshold must be finite and in [0, 1]")
    if video.metadata.duration_seconds is None:
        raise ValueError("shot detection requires a known duration")
    refs = uniform_sample_frames(
        video,
        target_fps=sample_fps,
        max_frames=max_frames,
    )
    if not refs:
        return []
    boundaries = [0.0]
    decoded_count = 0
    with VideoDecoder(video) as decoder:
        previous_histogram: np.ndarray | None = None
        for decoded in decoder.iter_frames(refs, strict=True):
            decoded_count += 1
            histogram = _color_histogram(decoded.image)
            if previous_histogram is not None:
                distance = 0.5 * float(np.abs(histogram - previous_histogram).sum())
                if distance >= cut_threshold and decoded.actual_timestamp_seconds > boundaries[-1]:
                    boundaries.append(decoded.actual_timestamp_seconds)
            previous_histogram = histogram
        if decoder.issues:
            raise DecodeError(decoder.issues[0])
        if decoded_count == 0:
            raise DecodeError(
                DecodeIssue(video.video_id, "no_decoded_frames", "shot detection decoded no frames")
            )
    boundaries.append(video.metadata.duration_seconds)
    shots: list[Shot] = []
    for index, (start, end) in enumerate(
        zip(boundaries[:-1], boundaries[1:], strict=True), start=1
    ):
        if end <= start:
            continue
        shots.append(
            Shot(
                f"shot_{index:04d}",
                video.video_id,
                start,
                end,
                (start + (end - start) / 2,),
            )
        )
    return shots
