"""Read-only, bounded source-frame inspection; no detections or quality claims."""

from __future__ import annotations

import hashlib
import math
import multiprocessing
import subprocess
import time
from dataclasses import asdict
from pathlib import Path

from video_intelligence.core.types import FrameRequest, VideoRecord
from video_intelligence.storage.evidence_frames import (
    EvidenceFrameConfig,
    _Budget,
    _safe,
    _source_hash,
)
from video_intelligence.video.decoder import DecodeError, PyAVVideoDecoder
from video_intelligence.video.display import UnsupportedVideoTransformError
from video_intelligence.video.probe import probe_video


class SourceInspectionError(RuntimeError):
    """Process-boundary failure, including codec timeout or abnormal worker exit."""

    def __init__(self, code, message, issue=None):
        super().__init__(message)
        self.code, self.issue = code, issue


def _inspection_worker(connection, source, timestamps, options):
    try:
        result = _inspect_source_frames(source, timestamps, **options)
        connection.send((True, result))
    except Exception as error:
        issue = asdict(error.issue) if isinstance(error, DecodeError) else None
        connection.send((False, {
            "code": issue["code"] if issue else "inspection_failed",
            "message": str(error), "issue": issue,
        }))
    finally:
        connection.close()


def _run_inspection(target, args, timeout_seconds):
    """Own exactly one spawned worker; timeout kills native calls, then reaps it."""
    context = multiprocessing.get_context("spawn")
    receiver, sender = context.Pipe(duplex=False)
    process = context.Process(target=target, args=(sender, *args), daemon=True)
    deadline = time.monotonic() + timeout_seconds
    try:
        process.start()
        sender.close()
        if not receiver.poll(max(0., deadline-time.monotonic())):
            raise SourceInspectionError("inspection_timeout", "source inspection deadline exceeded")
        try:
            ok, result = receiver.recv()
        except EOFError as error:
            raise SourceInspectionError(
                "inspection_worker_failed", "source worker exited without a result"
            ) from error
        process.join(max(0., deadline-time.monotonic()))
        if process.is_alive() or time.monotonic() > deadline:
            raise SourceInspectionError("inspection_timeout", "source inspection deadline exceeded")
        if process.exitcode != 0:
            raise SourceInspectionError(
                "inspection_worker_failed", "source worker exited abnormally"
            )
        if not ok:
            raise SourceInspectionError(result["code"], result["message"], result["issue"])
        return result
    finally:
        sender.close()
        receiver.close()
        if process.pid is not None:
            if process.is_alive():
                process.terminate()
                process.join(1.)
            if process.is_alive():
                process.kill()
                process.join(1.)
            if process.is_alive():
                raise SourceInspectionError(
                    "inspection_cleanup_failed", "source worker did not stop"
                )
            process.close()


def inspect_source_frames(source: Path, timestamps, *, video_id="source",
                          selection_policy="nearest", max_decode_frames=100_000,
                          timeout_seconds=30.) -> dict:
    """Inspect in a disposable process with an outer deadline covering probe/decode.

    Cleanup has a separate maximum two-second termination/reap allowance. No partial
    frame list escapes a timeout, source mutation, codec error or abnormal exit.
    """
    timestamps = tuple(timestamps)
    _validate_options(timestamps, selection_policy, max_decode_frames, timeout_seconds)
    started = time.monotonic()
    source = Path(source).expanduser().absolute()
    _safe(source, directory=False)
    if source.stat().st_size > 512 * 1024**2:
        raise ValueError("inspection source exceeds 512 MiB")
    # The parent owns ffprobe's subprocess.run timeout/kill/wait lifecycle. It
    # must not become an orphan when a codec worker is forcibly terminated.
    budget = _Budget(EvidenceFrameConfig(max_runtime_seconds=timeout_seconds), None)
    try:
        digest = _source_hash(source, budget)
        metadata = probe_video(source, allow_fallback=False, timeout_seconds=min(
            30., max(.000001, timeout_seconds-(time.monotonic()-started))
        ))
        budget.check()
    except UnsupportedVideoTransformError as error:
        raise SourceInspectionError("unsupported_display_transform", str(error)) from error
    except (subprocess.TimeoutExpired, TimeoutError) as error:
        raise SourceInspectionError(
            "inspection_timeout", "source inspection deadline exceeded"
        ) from error
    remaining = timeout_seconds-(time.monotonic()-started)
    if remaining <= 0:
        raise SourceInspectionError("inspection_timeout", "source inspection deadline exceeded")
    options = dict(video_id=video_id, selection_policy=selection_policy,
                   max_decode_frames=max_decode_frames, timeout_seconds=remaining,
                   metadata=metadata, expected_source_sha256=digest)
    return _run_inspection(_inspection_worker, (source, timestamps, options), remaining)


def _validate_options(timestamps, selection_policy, max_decode_frames, timeout_seconds):
    if not 1 <= len(timestamps) <= 8 or any(
        isinstance(t, bool) or not isinstance(t, (int, float))
        or not math.isfinite(t) or t < 0 for t in timestamps
    ):
        raise ValueError("inspection requires 1..8 finite non-negative timestamps")
    if selection_policy not in {"at_or_after", "nearest"}:
        raise ValueError("invalid frame selection policy")
    if (isinstance(max_decode_frames, bool) or not isinstance(max_decode_frames, int)
            or not 1 <= max_decode_frames <= 100_000):
        raise ValueError("max_decode_frames must be in [1,100000]")
    if (isinstance(timeout_seconds, bool) or not isinstance(timeout_seconds, (int, float))
            or not math.isfinite(timeout_seconds) or not 0 < timeout_seconds <= 60):
        raise ValueError("timeout_seconds must be finite in (0,60]")


def _inspect_source_frames(source: Path, timestamps, *, video_id="source",
                           selection_policy="nearest", max_decode_frames=100_000,
                           timeout_seconds=30., metadata=None, expected_source_sha256=None) -> dict:
    """Report actual source pixels/PTS without relabeling historical observations.

    Worker body: cooperative checkpoints supplement the parent's hard deadline.
    """
    timestamps = tuple(timestamps)
    _validate_options(timestamps, selection_policy, max_decode_frames, timeout_seconds)
    source = Path(source).expanduser().absolute()
    _safe(source, directory=False)
    if source.stat().st_size > 512 * 1024**2:
        raise ValueError("inspection source exceeds 512 MiB")
    budget = _Budget(EvidenceFrameConfig(max_frames=8, max_runtime_seconds=timeout_seconds), None)
    digest = _source_hash(source, budget)
    if expected_source_sha256 is not None and digest != expected_source_sha256:
        raise ValueError("inspection source changed before decode")
    if metadata is None:
        metadata = probe_video(source, allow_fallback=False)
    budget.check()
    if metadata.width * metadata.height > 2_073_600:
        raise ValueError("inspection source exceeds the pixel budget")
    video = VideoRecord(video_id, source, metadata, digest)
    rows = []
    with PyAVVideoDecoder(video, max_validation_frames=max_decode_frames,
                          max_validation_seconds=timeout_seconds,
                          max_request_frames=max_decode_frames,
                          max_request_seconds=timeout_seconds) as decoder:
        for timestamp in timestamps:
            budget.check()
            frame = decoder.read_request(
                FrameRequest(video_id, timestamp, source="source_inspection"),
                selection_policy=selection_policy,
            )
            budget.check()
            if frame is None:
                raise DecodeError(decoder.issues[-1])
            ref = asdict(frame.frame_ref)
            ref["source_path"] = None
            rows.append({
                "requested_seconds": timestamp, "frame_ref": ref,
                "rgb_sha256": hashlib.sha256(frame.image.tobytes(order="C")).hexdigest(),
                "rgb_encoding": "rgb8_hwc", "seek_error_seconds": frame.seek_error_seconds,
            })
    if _source_hash(source, budget) != digest:
        raise ValueError("inspection source changed during decode")
    return {
        "schema_version": "video_intelligence.source_frames.v1",
        "source_sha256": digest, "video_id": video_id, "frames": rows,
        "inference": False, "quality_status": "unmeasured",
        "published_evidence": False,
        "selection_policy": selection_policy, "decode_strategy": "origin_decode_forward.v1",
    }
