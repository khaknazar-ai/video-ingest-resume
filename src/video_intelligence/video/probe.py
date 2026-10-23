"""Video metadata probing with ffprobe first and OpenCV fallback."""

from __future__ import annotations

import json
import math
import shutil
import subprocess
from fractions import Fraction
from pathlib import Path
from typing import Any

from video_intelligence.core.types import VideoMetadata
from video_intelligence.video.display import (
    UnsupportedVideoTransformError,
    require_identity_display,
)

FFPROBE_TIMEOUT_SECONDS = 30.0
FFPROBE_SAMPLE_PACKETS = 256


def _missing(value: Any) -> bool:
    return value is None or (isinstance(value, str) and value.strip().upper() in {"", "N/A", "NA"})


def _parse_optional_float(value: Any) -> float | None:
    if _missing(value):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if math.isfinite(result) and result > 0 else None


def _parse_rate(value: Any) -> float | None:
    if _missing(value):
        return None
    try:
        result = float(Fraction(str(value)))
    except (ValueError, ZeroDivisionError, OverflowError):
        return None
    return result if math.isfinite(result) and result > 0 else None


def _parse_optional_int(value: Any) -> int | None:
    if _missing(value):
        return None
    try:
        result = int(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if result > 0 else None


def _run_ffprobe(path: Path, *, timeout_seconds: float = FFPROBE_TIMEOUT_SECONDS) -> VideoMetadata:
    executable = shutil.which("ffprobe")
    if executable is None:
        raise FileNotFoundError("ffprobe is not available")
    command = [
        executable,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-read_intervals",
        f"%+#{FFPROBE_SAMPLE_PACKETS}",
        "-show_entries",
        "stream=codec_name,width,height,avg_frame_rate,r_frame_rate,time_base,nb_frames,duration,sample_aspect_ratio:stream_tags=rotate:stream_side_data=rotation,displaymatrix:format=duration:frame=best_effort_timestamp_time",
        "-of",
        "json",
        str(path),
    ]
    completed = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
    )
    payload = json.loads(completed.stdout)
    streams = payload.get("streams") or []
    if not streams:
        raise ValueError(f"No video stream found in {path}")
    stream = streams[0]
    require_identity_display(
        rotation=(stream.get("tags") or {}).get("rotate"),
        sample_aspect_ratio=stream.get("sample_aspect_ratio"),
    )
    for side_data in stream.get("side_data_list") or []:
        require_identity_display(
            rotation=side_data.get("rotation"), matrix=side_data.get("displaymatrix")
        )
    stream_duration = _parse_optional_float(stream.get("duration"))
    format_duration = _parse_optional_float((payload.get("format") or {}).get("duration"))
    duration = stream_duration or format_duration
    average_fps = _parse_rate(stream.get("avg_frame_rate"))
    nominal_fps = _parse_rate(stream.get("r_frame_rate"))
    fps = average_fps or nominal_fps
    header_variable = (
        None
        if average_fps is None or nominal_fps is None
        else not math.isclose(average_fps, nominal_fps, rel_tol=1e-6, abs_tol=1e-9)
    )
    frame_timestamps = []
    for frame in payload.get("frames") or []:
        timestamp = frame.get("best_effort_timestamp_time")
        if _missing(timestamp):
            continue
        try:
            parsed = float(timestamp)
        except (TypeError, ValueError, OverflowError):
            continue
        if math.isfinite(parsed) and parsed >= 0:
            frame_timestamps.append(parsed)
    deltas = [
        current - previous
        for previous, current in zip(frame_timestamps, frame_timestamps[1:], strict=False)
        if current >= previous
    ]
    observed_variable = (
        None
        if len(deltas) < 2
        else not all(
            math.isclose(delta, deltas[0], rel_tol=1e-4, abs_tol=1e-6)
            for delta in deltas[1:]
        )
    )
    frame_count = _parse_optional_int(stream.get("nb_frames"))
    # Equal header rates and a constant prefix cannot certify an unsampled tail.
    variable_frame_rate = (
        True
        if header_variable is True or observed_variable is True
        else False
        if observed_variable is False
        and frame_count is not None
        and len(frame_timestamps) == frame_count
        else None
    )
    width = _parse_optional_int(stream.get("width"))
    height = _parse_optional_int(stream.get("height"))
    if width is None or height is None:
        raise ValueError(f"Incomplete video metadata from ffprobe for {path}")
    return VideoMetadata(
        duration_seconds=duration,
        fps=fps,
        frame_count=frame_count,
        width=width,
        height=height,
        codec=stream.get("codec_name"),
        time_base=stream.get("time_base"),
        variable_frame_rate=variable_frame_rate,
        duration_source=(
            "stream"
            if stream_duration is not None
            else "format"
            if format_duration is not None
            else None
        ),
        fps_source=(
            "avg_frame_rate"
            if average_fps is not None
            else "r_frame_rate"
            if nominal_fps is not None
            else None
        ),
        frame_count_source="stream_nb_frames" if frame_count is not None else None,
    )


def _run_opencv(path: Path) -> VideoMetadata:
    """Return explicitly estimated metadata, without certifying display matrices.

    OpenCV exposes rotation/SAR but cannot prove an identity reflection/skew mapping.
    VideoDecoder requires successful ffprobe certification before emitting frames.
    """

    import cv2

    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise ValueError(f"OpenCV could not open {path}")
        require_identity_display(rotation=capture.get(cv2.CAP_PROP_ORIENTATION_META))
        sar_denominator = capture.get(cv2.CAP_PROP_SAR_DEN)
        sar_numerator = capture.get(cv2.CAP_PROP_SAR_NUM)
        if sar_denominator and sar_numerator:
            require_identity_display(sample_aspect_ratio=sar_numerator / sar_denominator)
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        if fps <= 0 or frame_count <= 0 or width <= 0 or height <= 0:
            raise ValueError(f"Incomplete OpenCV metadata for {path}")
        return VideoMetadata(
            duration_seconds=frame_count / fps,
            fps=fps,
            frame_count=frame_count,
            width=width,
            height=height,
            variable_frame_rate=None,
            duration_source="opencv_estimate",
            fps_source="opencv",
            frame_count_source="opencv",
        )
    finally:
        capture.release()


def probe_video(path: str | Path, *, allow_fallback: bool = True,
                timeout_seconds: float = FFPROBE_TIMEOUT_SECONDS) -> VideoMetadata:
    """Probe video metadata; OpenCV fallback remains estimated and display-unverified."""

    if (isinstance(timeout_seconds, bool) or not isinstance(timeout_seconds, (int, float))
            or not math.isfinite(timeout_seconds) or timeout_seconds <= 0):
        raise ValueError("probe timeout must be finite and positive")
    video_path = Path(path).expanduser().resolve()
    if not video_path.is_file():
        raise FileNotFoundError(video_path)
    try:
        return _run_ffprobe(video_path, timeout_seconds=timeout_seconds)
    except UnsupportedVideoTransformError:
        # Unsupported geometry must not be converted into an apparently valid fallback.
        raise
    except (FileNotFoundError, subprocess.SubprocessError, ValueError, json.JSONDecodeError):
        if not allow_fallback:
            raise
        return _run_opencv(video_path)
