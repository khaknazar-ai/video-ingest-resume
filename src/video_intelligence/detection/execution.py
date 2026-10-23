"""Lazy batch-one grounding execution with typed outputs and bounded cleanup."""

from __future__ import annotations

import copy
import json
import math
import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import asdict, dataclass, replace
from fractions import Fraction
from typing import Any

from video_intelligence.core.types import DecodeIssue, FrameRef, TimestampQuality
from video_intelligence.detection.base import DetectionInput
from video_intelligence.detection.grounding import (
    GroundingBatch,
    GroundingBudget,
    GroundingCandidate,
    GroundingInput,
    GroundingStatus,
)
from video_intelligence.detection.prompts import PromptPlan
from video_intelligence.models.contracts import (
    AdapterState,
    ModelAdapterError,
    ModelErrorCode,
    ModelResult,
    ModelSpec,
)


def _json_copy(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("provenance must be a finite JSON mapping")
    try:
        return json.loads(json.dumps(dict(value), allow_nan=False, ensure_ascii=False))
    except (TypeError, ValueError, RecursionError) as error:
        raise ValueError("provenance must contain finite JSON values") from error


def _portable_frame(frame: FrameRef) -> dict[str, Any]:
    payload = asdict(frame)
    payload["source_path"] = None
    payload["timestamp_quality"] = frame.timestamp_quality.value
    return payload


def _copy_batch(
    batch: GroundingBatch, model_spec: ModelSpec | None = None,
) -> GroundingBatch:
    if not isinstance(batch, GroundingBatch):
        raise TypeError("grounding must return GroundingBatch values")
    detections = []
    for item in batch.detections:
        provenance = _json_copy(item.observation.provenance)
        if model_spec is not None:
            actual_spec = model_spec.to_dict()
            if "model_spec" in provenance and provenance["model_spec"] != actual_spec:
                raise ValueError("observation model_spec conflicts with executed model spec")
            provenance["model_spec"] = actual_spec
        detections.append(replace(
            item, observation=replace(
                item.observation, frame_ref=replace(item.observation.frame_ref),
                provenance=provenance,
            ),
        ))
    return replace(
        batch, frame_ref=replace(batch.frame_ref), prompt=replace(batch.prompt),
        detections=tuple(detections),
    )


def _batch_dict(batch: GroundingBatch) -> dict[str, Any]:
    return {
        "frame_ref": _portable_frame(batch.frame_ref),
        "model_id": batch.model_id, "model_mode": batch.model_mode.value,
        "source_stage": batch.source_stage, "prompt": batch.prompt.to_dict(),
        "is_empty": batch.is_empty,
        "detections": [{
            "detection_id": item.observation.detection_id,
            "frame_ref": _portable_frame(item.observation.frame_ref),
            "label": item.observation.label, "phrase": item.observation.phrase,
            "bbox_xyxy": list(item.observation.bbox_xyxy), "score": item.observation.score,
            "model_id": item.observation.model_id,
            "model_mode": item.observation.model_mode.value,
            "source_stage": item.observation.source_stage,
            "provenance": _json_copy(item.observation.provenance),
            "entity_id": item.entity_id, "matched_phrase": item.matched_phrase,
            "phrase_index": item.phrase_index, "prompt_version": item.prompt_version,
        } for item in batch.detections],
    }


@dataclass(frozen=True, slots=True)
class GroundingExecutionResult:
    """A validated output prefix; only succeeded denotes complete execution."""

    batches: tuple[GroundingBatch, ...]
    attempted_frames: int
    status: GroundingStatus
    model_spec: ModelSpec
    error_code: str | None = None
    error_message: str | None = None
    cleanup_completed: bool = False
    cleanup_error_code: str | None = None
    cleanup_error_message: str | None = None
    duration_seconds: float = 0.0

    def __post_init__(self) -> None:
        if not isinstance(self.model_spec, ModelSpec):
            raise TypeError("model_spec must be a ModelSpec")
        if (
            isinstance(self.attempted_frames, bool) or not isinstance(self.attempted_frames, int)
            or not 0 <= self.attempted_frames <= 1_000_001
        ):
            raise ValueError("attempted_frames must be a bounded non-negative integer")
        if not isinstance(self.batches, tuple) or len(self.batches) > self.attempted_frames:
            raise ValueError("batches must be a tuple no longer than attempted_frames")
        object.__setattr__(self, "status", GroundingStatus(self.status))
        if not isinstance(self.cleanup_completed, bool):
            raise ValueError("cleanup_completed must be a boolean")
        for name in ("error_code", "error_message", "cleanup_error_code", "cleanup_error_message"):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ValueError(f"{name} must be nonempty text or None")
        if (
            isinstance(self.duration_seconds, bool)
            or not isinstance(self.duration_seconds, (int, float))
            or not math.isfinite(self.duration_seconds) or self.duration_seconds < 0
        ):
            raise ValueError("duration_seconds must be finite and non-negative")
        if self.status is GroundingStatus.SUCCEEDED and (
            not self.cleanup_completed or self.error_code is not None
            or self.error_message is not None or self.cleanup_error_code is not None
            or self.cleanup_error_message is not None or not self.batches
            or len(self.batches) != self.attempted_frames
        ):
            raise ValueError("successful execution requires all frames and completed cleanup")
        if self.status is not GroundingStatus.SUCCEEDED and self.error_code is None:
            raise ValueError("failed or unavailable execution requires an explicit error")
        copied = tuple(_copy_batch(batch, self.model_spec) for batch in self.batches)
        if any(
            batch.model_id != self.model_spec.model_id
            or batch.model_mode is not self.model_spec.mode
            for batch in copied
        ):
            raise ValueError("execution batches must match model_spec provenance")
        object.__setattr__(self, "batches", copied)

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value, "model_spec": self.model_spec.to_dict(),
            "batches": [_batch_dict(batch) for batch in self.batches],
            "attempted_frames": self.attempted_frames, "processed_frames": len(self.batches),
            "error_code": self.error_code, "error_message": self.error_message,
            "cleanup_completed": self.cleanup_completed,
            "cleanup_error_code": self.cleanup_error_code,
            "cleanup_error_message": self.cleanup_error_message,
            "duration_seconds": self.duration_seconds,
            "measurement": {
                "phase": "grounding_execution_including_decode_load_infer_cleanup",
                "duration_seconds": self.duration_seconds,
                "attempted_frames": self.attempted_frames, "validated_frames": len(self.batches),
                "peak_rss_bytes": None, "peak_vram_bytes": None,
            },
            "quality_status": "unmeasured", "backend_worker_termination_verified": False,
        }


class _ExecutionFailure(Exception):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def _frame_context(frame: FrameRef) -> tuple[object, ...]:
    if frame.source_sha256 is None:
        raise ValueError("grounding execution requires source SHA-256")
    if frame.timestamp_quality not in {TimestampQuality.EXACT_PTS, TimestampQuality.CFR_ESTIMATE}:
        raise ValueError("grounding execution requires observed exact_pts or cfr_estimate frames")
    _ = frame.identity_key
    return (
        frame.video_id, frame.source_sha256, frame.stream_index, frame.width, frame.height,
        frame.timestamp_quality, frame.timestamp_origin,
        (Fraction(frame.time_base)
         if frame.timestamp_quality is TimestampQuality.EXACT_PTS else None),
    )


def execute_grounding(
    candidate: GroundingCandidate,
    *,
    prompt: PromptPlan,
    frames: Iterable[DetectionInput],
    budget: GroundingBudget,
    cancel_check: Callable[[], bool] | None = None,
    cleanup_timeout_seconds: float = 0.1,
    deadline: float | None = None,
) -> GroundingExecutionResult:
    """Load once and consume frames lazily, inferring a single frame at a time.

    The caller owns the decoder context. An overflow reads at most one extra frame
    to distinguish an exhausted stream from excess input; it never infers that frame.
    Deadlines are cooperative. Synchronous decoding/loading and native model calls
    still require an external process deadline. Cleanup never claims to kill workers.
    """

    if not isinstance(candidate, GroundingCandidate):
        raise TypeError("candidate must be a GroundingCandidate")
    if not isinstance(prompt, PromptPlan) or not isinstance(budget, GroundingBudget):
        raise TypeError("prompt and budget must be typed grounding contracts")
    if isinstance(frames, (str, bytes)) or not isinstance(frames, Iterable):
        raise TypeError("frames must be an iterable of DetectionInput values")
    if cancel_check is not None and not callable(cancel_check):
        raise TypeError("cancel_check must be callable or None")
    if (
        isinstance(cleanup_timeout_seconds, bool)
        or not isinstance(cleanup_timeout_seconds, (int, float))
        or not math.isfinite(cleanup_timeout_seconds) or not 0 <= cleanup_timeout_seconds <= 60
    ):
        raise ValueError("cleanup_timeout_seconds must be finite in [0,60]")
    if deadline is not None and (
        isinstance(deadline, bool) or not isinstance(deadline, (int, float))
        or not math.isfinite(deadline)
    ):
        raise ValueError("deadline must be a finite perf_counter value or None")
    adapter, spec = candidate.adapter, candidate.spec
    if spec.batch_size != 1:
        raise ValueError("grounding execution requires spec.batch_size=1")
    if adapter.capabilities().role != "detection":
        raise ValueError("grounding execution requires a detection adapter")
    if adapter.state is not AdapterState.UNLOADED:
        raise ValueError("grounding execution requires a fresh unloaded adapter")

    def check_interrupt() -> None:
        if deadline is not None and time.perf_counter() >= deadline:
            raise TimeoutError("grounding execution exceeded its cooperative deadline")
        if cancel_check is not None and cancel_check():
            raise ModelAdapterError(ModelErrorCode.CANCELLED, "grounding execution was cancelled")

    started = time.perf_counter()
    batches: list[GroundingBatch] = []
    attempted = 0
    identities: set[tuple[object, ...]] = set()
    detection_ids: set[str] = set()
    first: FrameRef | None = None
    previous: FrameRef | None = None
    source_stage: str | None = None
    status = GroundingStatus.SUCCEEDED
    error_code = error_message = None
    cleanup_completed = False
    cleanup_error_code = cleanup_error_message = None
    phase = "load"
    try:
        check_interrupt()
        adapter.load(spec)
        phase = "frame_stream"
        iterator = iter(frames)
        while True:
            check_interrupt()
            phase = "frame_stream"
            try:
                frame = next(iterator)
            except StopIteration:
                break
            attempted += 1
            phase = "input"
            if attempted > budget.max_candidate_frames:
                raise _ExecutionFailure("input_budget_exceeded", "candidate frame budget exceeded")
            if not isinstance(frame, DetectionInput):
                raise TypeError("frame stream must contain DetectionInput values")
            ref = replace(frame.frame_ref)
            context = _frame_context(ref)
            if first is None:
                first = ref
            elif context != _frame_context(first):
                raise ValueError("grounding source, dimensions and timeline policy must agree")
            if ref.identity_key in identities:
                raise ValueError("duplicate canonical frame identity in grounding stream")
            identities.add(ref.identity_key)
            if previous is not None and ref.timestamp_seconds <= previous.timestamp_seconds:
                raise ValueError("grounding timestamps must be strictly increasing")
            if ref.timestamp_quality is TimestampQuality.EXACT_PTS:
                if previous is not None and ref.pts <= previous.pts:
                    raise ValueError("grounding PTS must be strictly increasing")
                elapsed = ref.timestamp_seconds - first.timestamp_seconds
                pts_elapsed = float((ref.pts - first.pts) * Fraction(ref.time_base))
                if not math.isclose(elapsed, pts_elapsed, abs_tol=1e-6, rel_tol=1e-9):
                    raise ValueError("grounding PTS and timestamp deltas disagree")
            previous = ref
            isolated = DetectionInput(ref, copy.deepcopy(frame.image))
            check_interrupt()
            phase = "inference"
            result = adapter.infer(
                (GroundingInput(frame=isolated, prompt=prompt),), cancel_check=cancel_check,
            )
            check_interrupt()
            phase = "output"
            if not isinstance(result, ModelResult):
                raise TypeError("grounding adapter must return ModelResult")
            provenance = _json_copy(result.provenance)
            if (
                result.model_id != spec.model_id or result.mode is not spec.mode
                or any(provenance.get(key) != value for key, value in spec.to_dict().items())
            ):
                raise ValueError("grounding result must match the complete model spec")
            if result.measurement.input_count != 1:
                raise ValueError("grounding measurement must describe one input frame")
            if len(result.outputs) != 1 or not isinstance(result.outputs[0], GroundingBatch):
                raise ValueError("grounding adapter must return one GroundingBatch per frame")
            batch = result.outputs[0]
            if len(batch.detections) > budget.max_detections_per_frame:
                raise ValueError("grounding output exceeds detection budget")
            batch = _copy_batch(batch, spec)
            if (
                batch.frame_ref != ref or batch.prompt != prompt or batch.model_id != spec.model_id
                or batch.model_mode is not spec.mode
            ):
                raise ValueError("grounding output frame, prompt or model differs from input")
            if source_stage is not None and batch.source_stage != source_stage:
                raise ValueError("grounding source_stage must remain consistent")
            source_stage = batch.source_stage
            current_ids = {item.observation.detection_id for item in batch.detections}
            if detection_ids & current_ids:
                raise ValueError("detection IDs must be unique throughout grounding execution")
            detection_ids.update(current_ids)
            batches.append(batch)
        if not attempted:
            raise _ExecutionFailure("invalid_input", "grounding frame stream must not be empty")
        check_interrupt()
    except ModelAdapterError as error:
        status = (
            GroundingStatus.UNAVAILABLE
            if phase == "load" and error.code is ModelErrorCode.MODEL_UNAVAILABLE
            else GroundingStatus.FAILED
        )
        error_code, error_message = error.code.value, error.message
    except TimeoutError as error:
        status = GroundingStatus.FAILED
        error_code, error_message = "timeout", str(error)
    except _ExecutionFailure as error:
        status = GroundingStatus.FAILED
        error_code, error_message = error.code, str(error)
    except Exception as error:
        status = GroundingStatus.FAILED
        issue = getattr(error, "issue", None)
        error_code = (
            issue.code if isinstance(issue, DecodeIssue)
            else "invalid_output" if phase == "output"
            else "invalid_input" if phase == "input"
            else "frame_stream_failed" if phase == "frame_stream"
            else "adapter_failed"
        )
        error_message = (
            issue.message if isinstance(issue, DecodeIssue)
            else str(error) or type(error).__name__
        )
    finally:
        try:
            cleanup_completed = adapter.close(timeout_seconds=cleanup_timeout_seconds) is True
            if not cleanup_completed:
                cleanup_error_code = "adapter_cleanup_pending"
                cleanup_error_message = (
                    "adapter resources remain quarantined; an external process deadline is required"
                )
        except Exception as error:
            cleanup_error_code = "adapter_close_failed"
            cleanup_error_message = str(error) or type(error).__name__
        if cleanup_error_code is not None:
            if status is GroundingStatus.SUCCEEDED:
                error_code, error_message = cleanup_error_code, cleanup_error_message
            status = GroundingStatus.FAILED
    if (
        status is GroundingStatus.SUCCEEDED and deadline is not None
        and time.perf_counter() >= deadline
    ):
        status = GroundingStatus.FAILED
        error_code = "timeout"
        error_message = "grounding execution exceeded its cooperative deadline during cleanup"
    return GroundingExecutionResult(
        batches=tuple(batches), attempted_frames=attempted, status=status, model_spec=spec,
        error_code=error_code, error_message=error_message,
        cleanup_completed=cleanup_completed, cleanup_error_code=cleanup_error_code,
        cleanup_error_message=cleanup_error_message,
        duration_seconds=max(0.0, time.perf_counter() - started),
    )
