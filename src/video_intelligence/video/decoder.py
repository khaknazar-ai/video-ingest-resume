"""On-demand OpenCV decoder with an explicit timestamp contract."""

from __future__ import annotations

import math
import time
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, replace
from fractions import Fraction
from pathlib import Path

import cv2
import numpy as np

from video_intelligence.core.time import frame_index_to_seconds
from video_intelligence.core.types import (
    DecodeIssue,
    FrameRef,
    FrameRequest,
    TimestampQuality,
    VideoRecord,
)
from video_intelligence.video.display import (
    UnsupportedVideoTransformError,
    require_identity_display,
)
from video_intelligence.video.timestamps import normalize_pts

DEFAULT_MAX_VALIDATION_FRAMES = 100_000
DEFAULT_MAX_VALIDATION_SECONDS = 30.0


@dataclass(frozen=True, slots=True)
class DecodedFrame:
    """A decoded canonical RGB frame and the timestamp observed at decode time."""

    frame_ref: FrameRef
    image: np.ndarray
    actual_timestamp_seconds: float
    requested_timestamp_seconds: float | None = None
    seek_error_seconds: float | None = None


class DecodeError(RuntimeError):
    """Strict decoder failure carrying the structured issue that caused it."""

    def __init__(self, issue: DecodeIssue) -> None:
        super().__init__(f"{issue.code}: {issue.message}")
        self.issue = issue


class VideoDecoder:
    """Seekable decoder that keeps frames in memory only until the caller releases them."""

    def __init__(self, video: VideoRecord | str | Path) -> None:
        if isinstance(video, VideoRecord):
            self.video = video
        else:
            from video_intelligence.video.probe import probe_video

            path = Path(video).expanduser().resolve()
            self.video = VideoRecord(
                video_id=path.stem,
                source_path=path,
                metadata=probe_video(path),
            )
        self._capture: cv2.VideoCapture | None = None
        self._issues: list[DecodeIssue] = []

    @property
    def issues(self) -> tuple[DecodeIssue, ...]:
        """Return issues recorded by tolerant reads."""

        return tuple(self._issues)

    def __enter__(self) -> VideoDecoder:
        self.open()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def open(self) -> None:
        if self._capture is not None:
            return
        capture = cv2.VideoCapture(str(self.video.source_path))
        if not capture.isOpened():
            capture.release()
            raise RuntimeError(f"Could not open video: {self.video.source_path}")
        try:
            # Recheck the source, including legacy records that lack display metadata.
            from video_intelligence.video.probe import probe_video

            # OpenCV exposes an angle but cannot certify reflection/skew matrices.
            # Metadata-only fallback is therefore insufficient for emitting frames.
            probe_video(self.video.source_path, allow_fallback=False)
            require_identity_display(rotation=capture.get(cv2.CAP_PROP_ORIENTATION_META))
        except UnsupportedVideoTransformError as error:
            capture.release()
            issue = DecodeIssue(
                self.video.video_id, "unsupported_display_transform", str(error)
            )
            self._issues.append(issue)
            raise DecodeError(issue) from error
        except Exception as error:
            capture.release()
            issue = DecodeIssue(
                self.video.video_id,
                "display_mapping_unverified",
                f"ffprobe could not certify the display mapping: {error}",
            )
            self._issues.append(issue)
            raise DecodeError(issue) from error
        except BaseException:
            capture.release()
            raise
        self._capture = capture

    def close(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None

    def _record_issue(
        self,
        code: str,
        message: str,
        frame_ref: FrameRef,
        *,
        recoverable: bool = True,
    ) -> DecodeIssue:
        issue = DecodeIssue(
            video_id=self.video.video_id,
            code=code,
            message=message,
            frame_index=frame_ref.frame_index,
            timestamp_seconds=frame_ref.timestamp_seconds,
            recoverable=recoverable,
        )
        self._issues.append(issue)
        return issue

    def _require_open(self) -> cv2.VideoCapture:
        self.open()
        assert self._capture is not None
        return self._capture

    def read_frame(self, frame_ref: FrameRef) -> DecodedFrame | None:
        """Seek to and decode one frame; return None when decoding fails."""

        if frame_ref.video_id != self.video.video_id:
            raise ValueError(
                f"FrameRef belongs to {frame_ref.video_id!r}, decoder owns {self.video.video_id!r}"
            )
        if self.video.metadata.frame_count is None:
            raise ValueError("decoder requires a known frame_count for indexed reads")
        if not 0 <= frame_ref.frame_index < self.video.metadata.frame_count:
            raise IndexError(f"frame index out of range: {frame_ref.frame_index}")

        capture = self._require_open()
        if not capture.set(cv2.CAP_PROP_POS_FRAMES, frame_ref.frame_index):
            self._record_issue(
                "seek_failed",
                "OpenCV rejected the requested frame seek",
                frame_ref,
            )
            return None
        ok, image = capture.read()
        if not ok or image is None:
            self._record_issue(
                "read_failed",
                "OpenCV could not decode the requested frame",
                frame_ref,
            )
            return None

        if image.shape != (frame_ref.height, frame_ref.width, 3) or image.dtype != np.uint8:
            self._record_issue(
                "dimensions_mismatch",
                "decoded RGB dimensions/type do not match the source FrameRef",
                frame_ref,
                recoverable=False,
            )
            return None

        position_msec = float(capture.get(cv2.CAP_PROP_POS_MSEC))
        has_position = (
            math.isfinite(position_msec)
            and position_msec >= 0
            and (position_msec > 0 or frame_ref.frame_index == 0)
        )
        if has_position:
            actual_timestamp = position_msec / 1000.0
            timestamp_origin = "opencv_pos_msec"
        else:
            # OpenCV does not expose reliable timestamps for every codec. The fallback
            # is valid for CFR media and remains explicit in the manifest/report.
            if self.video.metadata.variable_frame_rate is not False:
                self._record_issue(
                    "timestamp_unavailable",
                    "OpenCV did not expose a timestamp and constant frame rate is unverified",
                    frame_ref,
                )
                return None
            if self.video.metadata.fps is None:
                raise RuntimeError("decoder cannot estimate timestamp without fps")
            actual_timestamp = frame_index_to_seconds(
                frame_ref.frame_index, self.video.metadata.fps
            )
            timestamp_origin = "frame_index_fps"
        quality = (
            TimestampQuality.CFR_ESTIMATE
            if self.video.metadata.variable_frame_rate is False
            else TimestampQuality.UNKNOWN
        )
        actual_ref = replace(
            frame_ref,
            timestamp_seconds=actual_timestamp,
            timestamp_quality=quality,
            timestamp_origin=timestamp_origin,
            source_path=self.video.source_path,
            source_sha256=self.video.source_sha256,
            stream_index=0,
            decode_ordinal=frame_ref.frame_index,
            backend="opencv",
        )
        return DecodedFrame(
            frame_ref=actual_ref,
            image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
            actual_timestamp_seconds=actual_timestamp,
            requested_timestamp_seconds=frame_ref.timestamp_seconds,
            seek_error_seconds=abs(actual_timestamp - frame_ref.timestamp_seconds),
        )

    def iter_frames(
        self,
        frame_refs: list[FrameRef] | tuple[FrameRef, ...],
        *,
        strict: bool = False,
    ) -> Iterator[DecodedFrame]:
        """Decode requested references one at a time without writing image files."""

        for frame_ref in frame_refs:
            decoded = self.read_frame(frame_ref)
            if decoded is not None:
                yield decoded
            elif strict:
                raise DecodeError(self._issues[-1])


class PyAVVideoDecoder:
    """Exact-PTS decoder with bounded, full-timeline identity certification.

    Opening makes a sequential metadata pass without RGB conversion or frame cache.
    Duplicate/nonmonotonic PTS and unsupported display transforms fail before any
    successful reference. Limits are cooperative checks between decoded frames;
    an outer process deadline is still needed for a codec that blocks inside decode.
    """

    def __init__(
        self,
        video: VideoRecord | str | Path,
        *,
        max_validation_frames: int = DEFAULT_MAX_VALIDATION_FRAMES,
        max_validation_seconds: float = DEFAULT_MAX_VALIDATION_SECONDS,
        max_request_frames: int = DEFAULT_MAX_VALIDATION_FRAMES,
        max_request_seconds: float = DEFAULT_MAX_VALIDATION_SECONDS,
    ) -> None:
        if (
            isinstance(max_validation_frames, bool)
            or not isinstance(max_validation_frames, int)
            or max_validation_frames <= 0
        ):
            raise ValueError("max_validation_frames must be a positive integer")
        if (
            isinstance(max_validation_seconds, bool)
            or not isinstance(max_validation_seconds, (int, float))
            or not math.isfinite(max_validation_seconds)
            or max_validation_seconds <= 0
        ):
            raise ValueError("max_validation_seconds must be finite and positive")
        self.max_validation_frames = max_validation_frames
        self.max_validation_seconds = float(max_validation_seconds)
        if (isinstance(max_request_frames, bool) or not isinstance(max_request_frames, int)
                or max_request_frames <= 0):
            raise ValueError("max_request_frames must be a positive integer")
        if (isinstance(max_request_seconds, bool)
                or not isinstance(max_request_seconds, (int, float))
                or not math.isfinite(max_request_seconds) or max_request_seconds <= 0):
            raise ValueError("max_request_seconds must be finite and positive")
        self.max_request_frames = max_request_frames
        self.max_request_seconds = float(max_request_seconds)
        if isinstance(video, VideoRecord):
            self.video = video
        else:
            from video_intelligence.video.probe import probe_video

            path = Path(video).expanduser().resolve()
            self.video = VideoRecord(path.stem, path, probe_video(path))
        self._container = None
        self._stream = None
        self._origin_pts: int | None = None
        self._time_base: Fraction | None = None
        self._decode_ordinal = 0
        self._issues: list[DecodeIssue] = []

    @property
    def issues(self) -> tuple[DecodeIssue, ...]:
        return tuple(self._issues)

    def __enter__(self) -> PyAVVideoDecoder:
        self.open()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def open(self) -> None:
        if self._container is not None:
            return
        try:
            import av
        except ImportError as error:
            raise RuntimeError(
                "PyAV is not installed; install the optional 'video' dependency for exact PTS"
            ) from error
        try:
            container = av.open(str(self.video.source_path))
        except Exception as error:
            self._fail("open_failed", str(error).strip() or type(error).__name__)
        streams = container.streams.video
        if not streams:
            container.close()
            self._fail("video_stream_missing", "source contains no video stream")
        self._container = container
        self._stream = streams[0]
        try:
            self._prime_origin()
        except BaseException:
            self.close()
            raise

    def close(self) -> None:
        if self._container is not None:
            self._container.close()
        self._container = None
        self._stream = None
        self._origin_pts = None
        self._time_base = None
        self._decode_ordinal = 0

    def _require_open(self):
        self.open()
        assert self._container is not None and self._stream is not None
        return self._container, self._stream

    def _prime_origin(self) -> None:
        container, stream = self._require_open()
        started = time.monotonic()
        if (
            self.video.metadata.frame_count is not None
            and self.video.metadata.frame_count > self.max_validation_frames
        ):
            self._fail(
                "timeline_validation_limit", "reported frame count exceeds certification cap"
            )
        previous_pts: int | None = None
        count = 0
        decode_iter = container.decode(stream)
        try:
            # A newly opened container starts at its first displayed frame, preserving
            # signed origins; seeking to zero first can discard negative initial PTS.
            for frame in decode_iter:
                count += 1
                if count > self.max_validation_frames or (
                    time.monotonic() - started > self.max_validation_seconds
                ):
                    self._fail(
                        "timeline_validation_limit", "full-timeline PTS certification cap exceeded"
                    )
                self._validate_display(frame, stream)
                if frame.pts is None:
                    self._fail(
                        "missing_pts", "exact timeline certification requires PTS on every frame"
                    )
                rational = Fraction(frame.time_base or stream.time_base)
                if rational <= 0:
                    self._fail("invalid_time_base", "frame time base must be positive")
                if self._origin_pts is None:
                    self._origin_pts, self._time_base = frame.pts, rational
                elif rational != self._time_base:
                    self._fail(
                        "changing_time_base_unsupported", "frame time base changes within stream"
                    )
                if previous_pts is not None and frame.pts <= previous_pts:
                    self._fail(
                        "duplicate_pts_unsupported"
                        if frame.pts == previous_pts
                        else "nonmonotonic_pts_unsupported",
                        "exact decoder requires strictly increasing presentation timestamps",
                    )
                previous_pts = frame.pts
        except DecodeError:
            raise
        except Exception as error:
            self._fail("timeline_validation_failed", str(error) or type(error).__name__)
        finally:
            close = getattr(decode_iter, "close", None)
            if close is not None:
                close()
        if self._origin_pts is None:
            self._fail("missing_pts", "video stream contains no displayed frame with PTS")
        self._restart_stream()

    def _restart_stream(self) -> None:
        """Start with fresh demux/codec state, including signed initial PTS.

        Some valid inter-frame streams cannot decode from demuxer seek points.
        Reopening also avoids stale codec state across requests. Certification is
        retained, and every decoded frame still checks the certified display/time base.
        """
        import av

        if self._container is not None:
            self._container.close()
        self._container = None
        self._stream = None
        container = av.open(str(self.video.source_path))
        try:
            stream = container.streams.video[0]
        except BaseException:
            container.close()
            raise
        self._container, self._stream = container, stream
        self._decode_ordinal = 0

    def _fail(self, code: str, message: str) -> None:
        raise DecodeError(self._record_issue(code, message, recoverable=False))

    def _validate_display(self, frame, stream) -> None:
        try:
            require_identity_display(
                rotation=getattr(frame, "rotation", None),
                sample_aspect_ratio=getattr(stream, "sample_aspect_ratio", None),
            )
            for side_data in getattr(frame, "side_data", ()):
                if str(side_data.type).endswith("DISPLAYMATRIX"):
                    require_identity_display(matrix=bytes(side_data))
        except UnsupportedVideoTransformError as error:
            self._fail("unsupported_display_transform", str(error))

    def _record_issue(
        self,
        code: str,
        message: str,
        *,
        timestamp: float | None = None,
        recoverable: bool = True,
    ) -> DecodeIssue:
        issue = DecodeIssue(
            video_id=self.video.video_id,
            code=code,
            message=message,
            timestamp_seconds=timestamp,
            recoverable=recoverable,
        )
        self._issues.append(issue)
        return issue

    def _convert_frame(self, frame) -> DecodedFrame | None:
        if frame.pts is None:
            self._record_issue("missing_pts", "PyAV returned a frame without PTS")
            return None
        _, stream = self._require_open()
        if self._origin_pts is None:
            raise RuntimeError("decoder timeline origin is not initialized")
        time_base = Fraction(frame.time_base or stream.time_base)
        self._validate_display(frame, stream)
        if time_base != self._time_base:
            self._fail(
                "changing_time_base_unsupported", "frame time base differs from certified stream"
            )
        time_base_text = f"{time_base.numerator}/{time_base.denominator}"
        frame_index = int(frame.pts - self._origin_pts)
        if frame_index < 0:
            self._record_issue(
                "negative_relative_pts",
                "PyAV returned a frame before the initialized timeline origin; "
                "stable frame identity cannot be established",
                recoverable=False,
            )
            return None
        observation = normalize_pts(
            frame.pts,
            time_base,
            origin_pts=self._origin_pts,
            decode_ordinal=self._decode_ordinal,
        )
        image = frame.to_ndarray(format="rgb24")
        if image.shape != (int(frame.height), int(frame.width), 3) or image.dtype != np.uint8:
            self._fail(
                "dimensions_mismatch", "decoded RGB dimensions/type differ from source frame"
            )
        frame_ref = FrameRef(
            video_id=self.video.video_id,
            frame_index=frame_index,
            timestamp_seconds=observation.timestamp_seconds,
            width=int(frame.width),
            height=int(frame.height),
            source_path=self.video.source_path,
            timestamp_quality=TimestampQuality.EXACT_PTS,
            timestamp_origin="first_displayed_pts",
            pts=observation.pts,
            time_base=time_base_text,
            source_sha256=self.video.source_sha256,
            stream_index=stream.index,
            decode_ordinal=self._decode_ordinal,
            backend="pyav",
        )
        self._decode_ordinal += 1
        return DecodedFrame(frame_ref, image, observation.timestamp_seconds)

    def _finalize_request(
        self,
        decoded: DecodedFrame,
        request: FrameRequest,
    ) -> DecodedFrame | None:
        seek_error_seconds = abs(
            decoded.actual_timestamp_seconds - request.requested_timestamp_seconds
        )
        if (
            request.tolerance_seconds is not None
            and seek_error_seconds > request.tolerance_seconds
        ):
            self._record_issue(
                "tolerance_exceeded",
                f"selected frame is {seek_error_seconds:.6f}s from the requested timestamp; "
                f"tolerance is {request.tolerance_seconds:.6f}s",
                timestamp=request.requested_timestamp_seconds,
            )
            return None
        return replace(
            decoded,
            requested_timestamp_seconds=request.requested_timestamp_seconds,
            seek_error_seconds=seek_error_seconds,
        )

    def iter_stream(self, *, max_frames: int | None = None) -> Iterator[DecodedFrame]:
        """Decode the selected stream sequentially with exact PTS when available."""

        container, stream = self._require_open()
        if max_frames is not None and (
            isinstance(max_frames, bool) or not isinstance(max_frames, int) or max_frames <= 0
        ):
            raise ValueError("max_frames must be a positive integer")
        count = 0
        decode_iter = container.decode(stream)
        try:
            while True:
                try:
                    frame = next(decode_iter)
                except StopIteration:
                    return
                decoded = self._convert_frame(frame)
                if decoded is None:
                    continue
                count += 1
                if max_frames is not None and count >= max_frames:
                    close = getattr(decode_iter, "close", None)
                    if close is not None:
                        close()
                    yield decoded
                    return
                yield decoded
        except Exception as error:
            message = str(error).strip() or error.__class__.__name__
            self._record_issue("decode_failed", message, recoverable=False)
            raise
        finally:
            close = getattr(decode_iter, "close", None)
            if close is not None:
                close()

    def read_request(
        self,
        request: FrameRequest,
        *,
        selection_policy: str = "at_or_after",
    ) -> DecodedFrame | None:
        """Bounded decode-forward from the certified origin, without demuxer seek.

        Work is capped by frame count and cooperative time checkpoints. Callers
        requiring preemption of native codec calls must use a process deadline,
        as provided by inspect_source_frames / inspect-frames.
        """

        if request.video_id != self.video.video_id:
            raise ValueError("FrameRequest belongs to another video")
        if selection_policy not in {"at_or_after", "nearest"}:
            raise ValueError("selection_policy must be 'at_or_after' or 'nearest'")
        self._require_open()
        if self._origin_pts is None or self._time_base is None:
            raise RuntimeError("decoder timeline origin is not initialized")
        if self.video.metadata.duration_seconds is not None and (
            request.requested_timestamp_seconds >= self.video.metadata.duration_seconds
        ):
            self._record_issue(
                "out_of_range",
                "requested timestamp is at or beyond video duration",
                timestamp=request.requested_timestamp_seconds,
            )
            return None
        started = time.monotonic()
        previous: DecodedFrame | None = None
        decode_iter = None

        def check_time():
            if time.monotonic() - started > self.max_request_seconds:
                self._fail("decode_timeout", "decode-forward time budget exceeded")

        try:
            self._restart_stream()
            check_time()
            decode_iter = self._container.decode(self._stream)
            count = 0
            while True:
                check_time()
                if count >= self.max_request_frames:
                    self._fail("decode_frame_limit", "decode-forward frame budget exceeded")
                try:
                    frame = next(decode_iter)
                except StopIteration:
                    check_time()
                    break
                count += 1
                check_time()
                decoded = self._convert_frame(frame)
                check_time()
                if decoded is None:
                    continue
                if decoded.actual_timestamp_seconds >= request.requested_timestamp_seconds:
                    selected = decoded
                    if selection_policy == "nearest" and previous is not None:
                        selected = min((previous, decoded), key=lambda item: (
                            abs(item.actual_timestamp_seconds-request.requested_timestamp_seconds),
                            item.actual_timestamp_seconds,
                        ))
                    return self._finalize_request(selected, request)
                previous = decoded
        except DecodeError:
            raise
        except Exception as error:
            self._fail("decode_failed", str(error).strip() or type(error).__name__)
        finally:
            close = getattr(decode_iter, "close", None)
            if close is not None:
                close()
        if selection_policy == "nearest" and previous is not None:
            return self._finalize_request(previous, request)
        self._record_issue(
            "out_of_range",
            "no frame satisfied the requested timestamp before EOF",
            timestamp=request.requested_timestamp_seconds,
        )
        return None

    def iter_requests(
        self,
        requests: Sequence[FrameRequest],
        *,
        selection_policy: str = "at_or_after",
        strict: bool = False,
    ) -> Iterator[DecodedFrame]:
        for request in requests:
            decoded = self.read_request(request, selection_policy=selection_policy)
            if decoded is not None:
                yield decoded
            elif strict:
                raise DecodeError(self._issues[-1])
