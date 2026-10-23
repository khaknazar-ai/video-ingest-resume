"""CPU-safe open-vocabulary grounding contracts without a model backend."""

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

from video_intelligence.core.types import FrameRef
from video_intelligence.detection.base import DetectionInput, DetectionObservation
from video_intelligence.detection.prompts import PromptPlan
from video_intelligence.models.contracts import (
    ModelAdapter,
    ModelAdapterError,
    ModelErrorCode,
    ModelMode,
    ModelSpec,
    UnavailableModelAdapter,
)


def _identifier(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value.strip() != value
        or any(character.isspace() for character in value)
    ):
        raise ValueError(f"{name} must be a non-empty identifier")
    return value


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ValueError(f"{name} must be a non-empty trimmed string")
    return value


def _raw_text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _positive_int(value: object, name: str, *, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0 or value > maximum:
        raise ValueError(f"{name} must be a positive integer <= {maximum}")
    return value


def _non_negative_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _frame_key(frame_ref: FrameRef) -> tuple[object, ...]:
    return frame_ref.identity_key


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


@dataclass(frozen=True, slots=True)
class GroundingBudget:
    """Hard caps applied before open-vocabulary inference begins."""

    max_candidate_frames: int
    max_detections_per_frame: int = 100

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "max_candidate_frames",
            _positive_int(self.max_candidate_frames, "max_candidate_frames", maximum=1_000_000),
        )
        object.__setattr__(
            self,
            "max_detections_per_frame",
            _positive_int(
                self.max_detections_per_frame,
                "max_detections_per_frame",
                maximum=100_000,
            ),
        )

    def to_dict(self) -> dict[str, int]:
        return {
            "max_candidate_frames": self.max_candidate_frames,
            "max_detections_per_frame": self.max_detections_per_frame,
        }


@dataclass(frozen=True, slots=True)
class GroundingInput:
    """One image plus the exact immutable prompt plan used for grounding."""

    frame: DetectionInput
    prompt: PromptPlan

    def __post_init__(self) -> None:
        if not isinstance(self.frame, DetectionInput):
            raise TypeError("frame must be a DetectionInput")
        if not isinstance(self.prompt, PromptPlan):
            raise TypeError("prompt must be a PromptPlan")


@dataclass(frozen=True, slots=True)
class GroundingRequest:
    """A bounded, duplicate-free request handed to one or more adapters."""

    prompt: PromptPlan
    frames: tuple[DetectionInput, ...]
    budget: GroundingBudget

    def __post_init__(self) -> None:
        if not isinstance(self.prompt, PromptPlan):
            raise TypeError("prompt must be a PromptPlan")
        if not isinstance(self.budget, GroundingBudget):
            raise TypeError("budget must be a GroundingBudget")
        if isinstance(self.frames, (str, bytes)) or not isinstance(self.frames, Sequence):
            raise ValueError("frames must be a sequence of DetectionInput values")
        frames = tuple(self.frames)
        if not frames or any(not isinstance(frame, DetectionInput) for frame in frames):
            raise ValueError("frames must contain at least one DetectionInput")
        if len(frames) > self.budget.max_candidate_frames:
            raise ValueError("candidate frame count exceeds the explicit grounding budget")
        if len({_frame_key(frame.frame_ref) for frame in frames}) != len(frames):
            raise ValueError("frames must not contain duplicate video/frame keys")
        object.__setattr__(self, "frames", frames)

    @property
    def frame_identity_sha256(self) -> str:
        return _frame_identity(self.frames)

    def to_dict(self) -> dict[str, object]:
        return {
            "prompt": self.prompt.to_dict(),
            "budget": self.budget.to_dict(),
            "frame_identity_sha256": self.frame_identity_sha256,
            "frames": [
                {
                    "video_id": frame.frame_ref.video_id,
                    "frame_index": frame.frame_ref.frame_index,
                    "timestamp_seconds": frame.frame_ref.timestamp_seconds,
                }
                for frame in self.frames
            ],
        }


@dataclass(frozen=True, slots=True)
class GroundingDetection:
    """One detection bound to an exact positive phrase in a PromptPlan."""

    observation: DetectionObservation
    entity_id: str
    matched_phrase: str
    phrase_index: int
    prompt_version: str

    def __post_init__(self) -> None:
        if not isinstance(self.observation, DetectionObservation):
            raise TypeError("observation must be a DetectionObservation")
        object.__setattr__(self, "entity_id", _identifier(self.entity_id, "entity_id"))
        object.__setattr__(
            self,
            "matched_phrase",
            _text(self.matched_phrase, "matched_phrase"),
        )
        object.__setattr__(
            self,
            "phrase_index",
            _non_negative_int(self.phrase_index, "phrase_index"),
        )
        object.__setattr__(
            self,
            "prompt_version",
            _identifier(self.prompt_version, "prompt_version"),
        )
        if self.observation.label != self.entity_id:
            raise ValueError("observation label must match grounding entity_id")
        if self.observation.phrase != self.matched_phrase:
            raise ValueError("observation phrase must match grounding matched_phrase")


@dataclass(frozen=True, slots=True)
class GroundingBatch:
    """Per-frame grounding output with query and phrase provenance attached."""

    frame_ref: FrameRef
    model_id: str
    model_mode: ModelMode | str
    source_stage: str
    prompt: PromptPlan
    detections: tuple[GroundingDetection, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.frame_ref, FrameRef):
            raise TypeError("frame_ref must be a FrameRef")
        object.__setattr__(self, "model_id", _identifier(self.model_id, "model_id"))
        object.__setattr__(self, "source_stage", _identifier(self.source_stage, "source_stage"))
        if not isinstance(self.prompt, PromptPlan):
            raise TypeError("prompt must be a PromptPlan")
        try:
            mode = ModelMode(self.model_mode)
        except (TypeError, ValueError) as error:
            raise ValueError(f"unsupported model mode: {self.model_mode!r}") from error
        object.__setattr__(self, "model_mode", mode)
        if isinstance(self.detections, (str, bytes)) or not isinstance(self.detections, Sequence):
            raise ValueError("detections must be a sequence")
        detections = tuple(self.detections)
        if any(not isinstance(item, GroundingDetection) for item in detections):
            raise ValueError("detections must contain only GroundingDetection values")
        if any(
            item.observation.frame_ref != self.frame_ref
            or item.observation.model_id != self.model_id
            or item.observation.model_mode is not mode
            or item.observation.source_stage != self.source_stage
            or item.prompt_version != self.prompt.prompt_version
            or item.phrase_index >= len(self.prompt.positive_phrases)
            or self.prompt.positive_phrases[item.phrase_index] != item.matched_phrase
            or self.prompt.phrase_entity_ids[item.phrase_index] != item.entity_id
            for item in detections
        ):
            raise ValueError("grounding detection provenance does not match batch prompt")
        if len({item.observation.detection_id for item in detections}) != len(detections):
            raise ValueError("detection_ids must be unique within a grounding batch")
        object.__setattr__(self, "detections", detections)

    @property
    def is_empty(self) -> bool:
        return not self.detections

    def to_dict(self) -> dict[str, object]:
        return {
            "video_id": self.frame_ref.video_id,
            "frame_index": self.frame_ref.frame_index,
            "timestamp_seconds": self.frame_ref.timestamp_seconds,
            "model_id": self.model_id,
            "model_mode": self.model_mode.value,
            "source_stage": self.source_stage,
            "prompt": self.prompt.to_dict(),
            "is_empty": self.is_empty,
            "detections": [
                {
                    "detection_id": item.observation.detection_id,
                    "entity_id": item.entity_id,
                    "matched_phrase": item.matched_phrase,
                    "phrase_index": item.phrase_index,
                    "prompt_version": item.prompt_version,
                    "bbox_xyxy": list(item.observation.bbox_xyxy),
                    "score": item.observation.score,
                }
                for item in self.detections
            ],
        }


@dataclass(frozen=True, slots=True)
class GroundingCandidate:
    """An open-vocabulary adapter paired with its immutable ModelSpec."""

    adapter: ModelAdapter[Any, GroundingBatch]
    spec: ModelSpec

    def __post_init__(self) -> None:
        if not isinstance(self.adapter, ModelAdapter):
            raise TypeError("adapter must be a ModelAdapter")
        if not isinstance(self.spec, ModelSpec):
            raise TypeError("spec must be a ModelSpec")
        if self.spec.role != "detection":
            raise ValueError("grounding spec role must be detection")
        capabilities = self.adapter.capabilities()
        if self.spec.model_id != capabilities.model_id:
            raise ValueError("spec model_id does not match adapter capabilities")
        if self.spec.mode is not capabilities.mode:
            raise ValueError("spec mode does not match adapter capabilities")


class UnavailableGroundingAdapter(UnavailableModelAdapter):
    """Explicit grounding fallback that never fabricates phrase-aligned boxes."""

    def __init__(
        self,
        *,
        model_id: str = "grounding_dino_unavailable",
        reason: str = "open-vocabulary grounding backend or weights are unavailable",
    ) -> None:
        super().__init__(model_id=model_id, role="detection", reason=reason)


class GroundingStatus(StrEnum):
    SUCCEEDED = "succeeded"
    UNAVAILABLE = "unavailable"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class GroundingRunReport:
    """Execution report; grounding quality is intentionally evaluated separately."""

    model_id: str
    mode: str
    status: GroundingStatus
    prompt_version: str
    requested_frame_count: int
    processed_frame_count: int
    detection_count: int
    candidate_frame_budget: int
    duration_seconds: float | None
    fps: float | None
    error_code: str | None
    error_message: str | None
    cleanup_completed: bool | None = None
    cleanup_error_code: str | None = None
    cleanup_error_message: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "model_id": self.model_id,
            "mode": self.mode,
            "status": self.status.value,
            "prompt_version": self.prompt_version,
            "requested_frame_count": self.requested_frame_count,
            "processed_frame_count": self.processed_frame_count,
            "detection_count": self.detection_count,
            "candidate_frame_budget": self.candidate_frame_budget,
            "duration_seconds": self.duration_seconds,
            "fps": self.fps,
            "error_code": self.error_code,
            "error_message": self.error_message,
            "cleanup_completed": self.cleanup_completed,
            "cleanup_error_code": self.cleanup_error_code,
            "cleanup_error_message": self.cleanup_error_message,
        }


@dataclass(frozen=True, slots=True)
class GroundingComparisonReport:
    """Comparable runs tied to one prompt and one ordered candidate-frame set."""

    frame_count: int
    frame_identity_sha256: str
    prompt_version: str
    raw_query: str
    candidate_frame_budget: int
    runs: tuple[GroundingRunReport, ...]

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
        object.__setattr__(
            self,
            "prompt_version",
            _identifier(self.prompt_version, "prompt_version"),
        )
        object.__setattr__(self, "raw_query", _raw_text(self.raw_query, "raw_query"))
        object.__setattr__(
            self,
            "candidate_frame_budget",
            _positive_int(
                self.candidate_frame_budget,
                "candidate_frame_budget",
                maximum=1_000_000,
            ),
        )
        object.__setattr__(self, "runs", tuple(self.runs))
        if any(not isinstance(run, GroundingRunReport) for run in self.runs):
            raise ValueError("runs must contain only GroundingRunReport values")

    def to_dict(self) -> dict[str, object]:
        return {
            "frame_count": self.frame_count,
            "frame_identity_sha256": self.frame_identity_sha256,
            "prompt_version": self.prompt_version,
            "raw_query": self.raw_query,
            "candidate_frame_budget": self.candidate_frame_budget,
            "runs": [run.to_dict() for run in self.runs],
        }


class GroundingComparisonHarness:
    """Run grounding candidates over the exact same bounded request."""

    def run(
        self,
        request: GroundingRequest,
        candidates: Sequence[GroundingCandidate],
    ) -> GroundingComparisonReport:
        if not isinstance(request, GroundingRequest):
            raise TypeError("request must be a GroundingRequest")
        if isinstance(candidates, (str, bytes)) or not isinstance(candidates, Sequence):
            raise ValueError("candidates must be a sequence of GroundingCandidate values")
        candidate_values = tuple(candidates)
        if not candidate_values or any(
            not isinstance(candidate, GroundingCandidate) for candidate in candidate_values
        ):
            raise ValueError("candidates must contain at least one GroundingCandidate")
        if len({candidate.spec.model_id for candidate in candidate_values}) != len(
            candidate_values
        ):
            raise ValueError("candidate model_ids must be unique")
        runs: list[GroundingRunReport] = []
        cleanup_blocked = False
        for candidate in candidate_values:
            if cleanup_blocked:
                runs.append(GroundingRunReport(
                    model_id=candidate.spec.model_id, mode=candidate.spec.mode.value,
                    status=GroundingStatus.FAILED, prompt_version=request.prompt.prompt_version,
                    requested_frame_count=len(request.frames), processed_frame_count=0,
                    detection_count=0, candidate_frame_budget=request.budget.max_candidate_frames,
                    duration_seconds=None, fps=None, error_code="blocked_previous_cleanup",
                    error_message="previous adapter cleanup is pending; no next model was loaded",
                ))
                continue
            run = self._run_one(request, candidate)
            runs.append(run)
            cleanup_blocked = run.cleanup_completed is not True
        return GroundingComparisonReport(
            frame_count=len(request.frames),
            frame_identity_sha256=request.frame_identity_sha256,
            prompt_version=request.prompt.prompt_version,
            raw_query=request.prompt.raw_query,
            candidate_frame_budget=request.budget.max_candidate_frames,
            runs=tuple(runs),
        )

    @staticmethod
    def _run_one(
        request: GroundingRequest,
        candidate: GroundingCandidate,
    ) -> GroundingRunReport:
        adapter = candidate.adapter
        spec = candidate.spec
        started = time.perf_counter()
        outputs: list[GroundingBatch] = []
        status = GroundingStatus.SUCCEEDED
        error_code: str | None = None
        error_message: str | None = None
        cleanup_completed = False
        cleanup_error_code: str | None = None
        cleanup_error_message: str | None = None
        try:
            adapter.load(spec)
            for frame in request.frames:
                isolated_frame = DetectionInput(
                    frame_ref=frame.frame_ref,
                    image=copy.deepcopy(frame.image),
                )
                result = adapter.infer(
                    (GroundingInput(frame=isolated_frame, prompt=request.prompt),)
                )
                if result.model_id != spec.model_id or result.mode is not spec.mode:
                    raise ValueError("model result provenance does not match grounding spec")
                if len(result.outputs) != 1 or not isinstance(result.outputs[0], GroundingBatch):
                    raise ValueError("grounding adapter must return one GroundingBatch per frame")
                batch = result.outputs[0]
                if (
                    batch.frame_ref != frame.frame_ref
                    or batch.model_id != spec.model_id
                    or batch.model_mode is not spec.mode
                    or batch.prompt != request.prompt
                ):
                    raise ValueError("grounding output frame, model or prompt does not match input")
                if len(batch.detections) > request.budget.max_detections_per_frame:
                    raise ValueError("grounding output exceeds max_detections_per_frame budget")
                outputs.append(batch)
        except ModelAdapterError as error:
            status = (
                GroundingStatus.UNAVAILABLE
                if error.code is ModelErrorCode.MODEL_UNAVAILABLE
                else GroundingStatus.FAILED
            )
            error_code = error.code.value
            error_message = error.message
        except (TypeError, ValueError) as error:
            status = GroundingStatus.FAILED
            error_code = "invalid_output"
            error_message = str(error)
        except Exception as error:
            status = GroundingStatus.FAILED
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
                if status is GroundingStatus.SUCCEEDED:
                    error_code = cleanup_error_code
                    error_message = cleanup_error_message
                status = GroundingStatus.FAILED
        duration = max(time.perf_counter() - started, math.ulp(1.0))
        processed = len(outputs)
        return GroundingRunReport(
            model_id=spec.model_id,
            mode=spec.mode.value,
            status=status,
            prompt_version=request.prompt.prompt_version,
            requested_frame_count=len(request.frames),
            processed_frame_count=processed,
            detection_count=sum(len(batch.detections) for batch in outputs),
            candidate_frame_budget=request.budget.max_candidate_frames,
            duration_seconds=(duration if status is not GroundingStatus.UNAVAILABLE else None),
            fps=(
                processed / duration
                if processed and status is not GroundingStatus.UNAVAILABLE
                else None
            ),
            error_code=error_code,
            error_message=error_message,
            cleanup_completed=cleanup_completed,
            cleanup_error_code=cleanup_error_code,
            cleanup_error_message=cleanup_error_message,
        )
