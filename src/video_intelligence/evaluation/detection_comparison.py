"""Comparable closed-set adapter runs over identical frame inputs."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import time
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from video_intelligence.detection.base import DetectionBatch, DetectionInput
from video_intelligence.detection.closed_set import ClosedSetCandidate
from video_intelligence.evaluation.detection_metrics import (
    DetectionAnnotation,
    DetectionMetrics,
    evaluate_detection,
)
from video_intelligence.models.contracts import ModelAdapterError, ModelErrorCode


class ComparisonStatus(StrEnum):
    SUCCEEDED = "succeeded"
    UNAVAILABLE = "unavailable"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class DetectionRunReport:
    """One adapter result with independent speed and optional quality fields."""

    model_id: str
    mode: str
    status: ComparisonStatus
    input_frame_count: int
    processed_frame_count: int
    detection_count: int
    duration_seconds: float | None
    fps: float | None
    error_code: str | None
    error_message: str | None
    quality: DetectionMetrics | None
    cleanup_completed: bool | None = None
    cleanup_error_code: str | None = None
    cleanup_error_message: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "mode": self.mode,
            "status": self.status.value,
            "input_frame_count": self.input_frame_count,
            "processed_frame_count": self.processed_frame_count,
            "detection_count": self.detection_count,
            "duration_seconds": self.duration_seconds,
            "fps": self.fps,
            "error_code": self.error_code,
            "error_message": self.error_message,
            "quality": None if self.quality is None else self.quality.to_dict(),
            "cleanup_completed": self.cleanup_completed,
            "cleanup_error_code": self.cleanup_error_code,
            "cleanup_error_message": self.cleanup_error_message,
        }


@dataclass(frozen=True, slots=True)
class DetectionComparisonReport:
    """All adapter runs are tied to one ordered frame-input fingerprint."""

    frame_count: int
    frame_identity_sha256: str
    runs: tuple[DetectionRunReport, ...]

    def __post_init__(self) -> None:
        if isinstance(self.frame_count, bool) or not isinstance(self.frame_count, int):
            raise ValueError("frame_count must be a non-negative integer")
        if self.frame_count < 0:
            raise ValueError("frame_count must be a non-negative integer")
        if (
            not isinstance(self.frame_identity_sha256, str)
            or len(self.frame_identity_sha256) != 64
            or self.frame_identity_sha256 != self.frame_identity_sha256.lower()
            or any(character not in "0123456789abcdef" for character in self.frame_identity_sha256)
        ):
            raise ValueError("frame_identity_sha256 must be a lowercase SHA-256 digest")
        object.__setattr__(self, "runs", tuple(self.runs))
        if any(not isinstance(run, DetectionRunReport) for run in self.runs):
            raise ValueError("runs must contain only DetectionRunReport values")

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame_count": self.frame_count,
            "frame_identity_sha256": self.frame_identity_sha256,
            "runs": [run.to_dict() for run in self.runs],
        }


def _frame_identity(frames: Sequence[DetectionInput]) -> str:
    payload = [
        {
            "video_id": frame.frame_ref.video_id,
            "frame_index": frame.frame_ref.frame_index,
            "timestamp_seconds": frame.frame_ref.timestamp_seconds,
            "width": frame.frame_ref.width,
            "height": frame.frame_ref.height,
            "source_sha256": frame.frame_ref.source_sha256,
            "pts": frame.frame_ref.pts,
            "time_base": frame.frame_ref.time_base,
            "stream_index": frame.frame_ref.stream_index,
        }
        for frame in frames
    ]
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class DetectionComparisonHarness:
    """Run each closed-set candidate over the same bounded frame sequence."""

    def run(
        self,
        frames: Sequence[DetectionInput],
        candidates: Sequence[ClosedSetCandidate],
        *,
        annotations: Sequence[DetectionAnnotation] = (),
    ) -> DetectionComparisonReport:
        if isinstance(frames, (str, bytes)) or not isinstance(frames, Sequence):
            raise ValueError("frames must be a sequence of DetectionInput values")
        frame_values = tuple(frames)
        if not frame_values or any(not isinstance(frame, DetectionInput) for frame in frame_values):
            raise ValueError("frames must contain at least one DetectionInput")
        if isinstance(candidates, (str, bytes)) or not isinstance(candidates, Sequence):
            raise ValueError("candidates must be a sequence of ClosedSetCandidate values")
        candidate_values = tuple(candidates)
        if not candidate_values or any(
            not isinstance(candidate, ClosedSetCandidate) for candidate in candidate_values
        ):
            raise ValueError("candidates must contain at least one ClosedSetCandidate")
        if len({candidate.spec.model_id for candidate in candidate_values}) != len(
            candidate_values
        ):
            raise ValueError("candidate model_ids must be unique")
        if isinstance(annotations, (str, bytes)) or not isinstance(annotations, Sequence):
            raise ValueError("annotations must be a sequence of DetectionAnnotation values")
        annotation_values = tuple(annotations)
        if any(not isinstance(annotation, DetectionAnnotation) for annotation in annotation_values):
            raise ValueError("annotations must contain only DetectionAnnotation values")
        annotation_keys = {
            (annotation.frame_ref.video_id, annotation.frame_ref.frame_index)
            for annotation in annotation_values
        }
        if len(annotation_keys) != len(annotation_values):
            raise ValueError("annotation frame keys must be unique")
        frame_by_key = {
            (frame.frame_ref.video_id, frame.frame_ref.frame_index): frame.frame_ref
            for frame in frame_values
        }
        for annotation in annotation_values:
            expected = frame_by_key.get(
                (annotation.frame_ref.video_id, annotation.frame_ref.frame_index)
            )
            if expected is None or annotation.frame_ref != expected:
                raise ValueError("annotation frame reference does not match comparison input")

        reports: list[DetectionRunReport] = []
        cleanup_blocked = False
        for candidate in candidate_values:
            if cleanup_blocked:
                reports.append(DetectionRunReport(
                    model_id=candidate.spec.model_id, mode=candidate.spec.mode.value,
                    status=ComparisonStatus.FAILED, input_frame_count=len(frame_values),
                    processed_frame_count=0, detection_count=0, duration_seconds=None, fps=None,
                    error_code="blocked_previous_cleanup",
                    error_message="previous adapter cleanup is pending; no next model was loaded",
                    quality=None,
                ))
                continue
            report = self._run_one(frame_values, candidate, annotation_values)
            reports.append(report)
            cleanup_blocked = report.cleanup_completed is not True
        return DetectionComparisonReport(
            frame_count=len(frame_values),
            frame_identity_sha256=_frame_identity(frame_values),
            runs=tuple(reports),
        )

    @staticmethod
    def _run_one(
        frames: tuple[DetectionInput, ...],
        candidate: ClosedSetCandidate,
        annotations: tuple[DetectionAnnotation, ...],
    ) -> DetectionRunReport:
        adapter = candidate.adapter
        spec = candidate.spec
        started = time.perf_counter()
        outputs: list[DetectionBatch] = []
        status = ComparisonStatus.SUCCEEDED
        error_code: str | None = None
        error_message: str | None = None
        cleanup_completed = False
        cleanup_error_code: str | None = None
        cleanup_error_message: str | None = None
        try:
            adapter.load(spec)
            for frame in frames:
                isolated_frame = DetectionInput(
                    frame_ref=frame.frame_ref,
                    image=copy.deepcopy(frame.image),
                )
                result = adapter.infer((isolated_frame,))
                if result.model_id != spec.model_id or result.mode is not spec.mode:
                    raise ValueError("model result provenance does not match candidate spec")
                if len(result.outputs) != 1 or not isinstance(
                    result.outputs[0],
                    DetectionBatch,
                ):
                    raise ValueError("detector adapter must return one DetectionBatch per frame")
                batch = result.outputs[0]
                if (
                    batch.frame_ref != frame.frame_ref
                    or batch.model_id != spec.model_id
                    or batch.model_mode is not spec.mode
                ):
                    raise ValueError("detector output frame or model does not match candidate")
                outputs.append(batch)
        except ModelAdapterError as error:
            status = (
                ComparisonStatus.UNAVAILABLE
                if error.code is ModelErrorCode.MODEL_UNAVAILABLE
                else ComparisonStatus.FAILED
            )
            error_code = error.code.value
            error_message = error.message
        except (TypeError, ValueError) as error:
            status = ComparisonStatus.FAILED
            error_code = "invalid_output"
            error_message = str(error)
        except Exception as error:
            status = ComparisonStatus.FAILED
            error_code = "adapter_failed"
            error_message = str(error)
        finally:
            try:
                cleanup_completed = adapter.close() is True
                if not cleanup_completed:
                    cleanup_error_code = "adapter_cleanup_pending"
                    cleanup_error_message = (
                        "adapter cleanup is pending; resources remain quarantined; "
                        "an external process deadline is required"
                    )
            except Exception as error:
                cleanup_error_code = "adapter_close_failed"
                cleanup_error_message = str(error)
            if cleanup_error_code is not None:
                if status is ComparisonStatus.SUCCEEDED:
                    error_code = cleanup_error_code
                    error_message = cleanup_error_message
                status = ComparisonStatus.FAILED
        duration = max(time.perf_counter() - started, math.ulp(1.0))
        quality = evaluate_detection(outputs, annotations) if annotations and outputs else None
        processed = len(outputs)
        return DetectionRunReport(
            model_id=spec.model_id,
            mode=spec.mode.value,
            status=status,
            input_frame_count=len(frames),
            processed_frame_count=processed,
            detection_count=sum(len(batch.detections) for batch in outputs),
            duration_seconds=duration if status is not ComparisonStatus.UNAVAILABLE else None,
            fps=(
                processed / duration
                if processed and status is not ComparisonStatus.UNAVAILABLE
                else None
            ),
            error_code=error_code,
            error_message=error_message,
            quality=quality,
            cleanup_completed=cleanup_completed,
            cleanup_error_code=cleanup_error_code,
            cleanup_error_message=cleanup_error_message,
        )
