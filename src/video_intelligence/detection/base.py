"""Typed detection observations and explicit unavailable adapter."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from video_intelligence.core.types import FrameRef
from video_intelligence.detection.postprocess import BoundingBox
from video_intelligence.models.contracts import ModelMode, UnavailableModelAdapter

if TYPE_CHECKING:
    from video_intelligence.detection.postprocess import RawDetection


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


def _score(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("score must be a finite number in [0, 1]")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError("score must be a finite number in [0, 1]")
    return result


def _json_object(value: Mapping[str, Any], name: str) -> dict[str, Any]:
    copied = dict(value)
    try:
        json.dumps(copied, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must contain finite JSON values") from error
    return copied


@dataclass(frozen=True, slots=True)
class DetectionInput:
    """A decoded canonical frame passed to a detector adapter."""

    frame_ref: FrameRef
    image: object

    def __post_init__(self) -> None:
        if not isinstance(self.frame_ref, FrameRef):
            raise TypeError("frame_ref must be a FrameRef")
        if self.image is None:
            raise ValueError("image must not be null")


@dataclass(frozen=True, slots=True)
class DetectionObservation:
    """One source-pixel detection with model and transform provenance."""

    detection_id: str
    frame_ref: FrameRef
    label: str
    bbox_xyxy: tuple[float, float, float, float]
    score: float
    model_id: str
    model_mode: ModelMode | str
    source_stage: str
    phrase: str | None = None
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "detection_id",
            _identifier(self.detection_id, "detection_id"),
        )
        if not isinstance(self.frame_ref, FrameRef):
            raise TypeError("frame_ref must be a FrameRef")
        object.__setattr__(self, "label", _identifier(self.label, "label"))
        box = BoundingBox(
            self.bbox_xyxy,
            "source_pixels",
            self.frame_ref.width,
            self.frame_ref.height,
        )
        object.__setattr__(self, "bbox_xyxy", box.xyxy)
        object.__setattr__(self, "score", _score(self.score))
        object.__setattr__(self, "model_id", _identifier(self.model_id, "model_id"))
        object.__setattr__(self, "source_stage", _identifier(self.source_stage, "source_stage"))
        try:
            object.__setattr__(self, "model_mode", ModelMode(self.model_mode))
        except (TypeError, ValueError) as error:
            raise ValueError(f"unsupported model mode: {self.model_mode!r}") from error
        if self.phrase is not None:
            object.__setattr__(self, "phrase", _text(self.phrase, "phrase"))
        if not isinstance(self.provenance, Mapping):
            raise ValueError("provenance must be a mapping")
        object.__setattr__(
            self,
            "provenance",
            _json_object(self.provenance, "provenance"),
        )


@dataclass(frozen=True, slots=True)
class DetectionBatch:
    """Per-frame detector output; an empty tuple is an explicit negative result."""

    frame_ref: FrameRef
    model_id: str
    model_mode: ModelMode | str
    source_stage: str
    detections: tuple[DetectionObservation, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.frame_ref, FrameRef):
            raise TypeError("frame_ref must be a FrameRef")
        object.__setattr__(self, "model_id", _identifier(self.model_id, "model_id"))
        object.__setattr__(self, "source_stage", _identifier(self.source_stage, "source_stage"))
        try:
            object.__setattr__(self, "model_mode", ModelMode(self.model_mode))
        except (TypeError, ValueError) as error:
            raise ValueError(f"unsupported model mode: {self.model_mode!r}") from error
        if isinstance(self.detections, (str, bytes)) or not isinstance(
            self.detections, Sequence
        ):
            raise ValueError("detections must be a sequence")
        detections = tuple(self.detections)
        if any(not isinstance(item, DetectionObservation) for item in detections):
            raise ValueError("detections must contain only DetectionObservation values")
        if any(
            item.frame_ref != self.frame_ref
            or item.model_id != self.model_id
            or item.model_mode is not self.model_mode
            or item.source_stage != self.source_stage
            for item in detections
        ):
            raise ValueError("detection observations do not match batch provenance")
        if len({item.detection_id for item in detections}) != len(detections):
            raise ValueError("detection_ids must be unique within a batch")
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
            "is_empty": self.is_empty,
            "detections": [
                {
                    "detection_id": item.detection_id,
                    "label": item.label,
                    "phrase": item.phrase,
                    "bbox_xyxy": list(item.bbox_xyxy),
                    "score": item.score,
                    "provenance": dict(item.provenance),
                }
                for item in self.detections
            ],
        }


def postprocess_detections(
    frame_ref: FrameRef,
    raw_detections: Sequence[RawDetection],
    *,
    model_id: str,
    model_mode: ModelMode | str,
    source_stage: str,
) -> DetectionBatch:
    """Convert detector coordinates to canonical source pixels without clipping."""

    from video_intelligence.detection.postprocess import RawDetection

    if not isinstance(frame_ref, FrameRef):
        raise TypeError("frame_ref must be a FrameRef")
    if isinstance(raw_detections, (str, bytes)) or not isinstance(
        raw_detections, Sequence
    ):
        raise ValueError("raw_detections must be a sequence")
    if any(not isinstance(item, RawDetection) for item in raw_detections):
        raise ValueError("raw_detections must contain only RawDetection values")
    model_id = _identifier(model_id, "model_id")
    source_stage = _identifier(source_stage, "source_stage")
    try:
        mode = ModelMode(model_mode)
    except (TypeError, ValueError) as error:
        raise ValueError(f"unsupported model mode: {model_mode!r}") from error
    observations = []
    for rank, raw in enumerate(raw_detections, start=1):
        source_box = raw.box.to_source_pixels(
            source_width=frame_ref.width,
            source_height=frame_ref.height,
        )
        observations.append(
            DetectionObservation(
                detection_id=(
                    f"det_{frame_ref.video_id}_{frame_ref.frame_index:08d}_{rank:04d}"
                ),
                frame_ref=frame_ref,
                label=raw.label,
                phrase=raw.phrase,
                bbox_xyxy=source_box.xyxy,
                score=raw.score,
                model_id=model_id,
                model_mode=mode,
                source_stage=source_stage,
                provenance={
                    "input_coordinate_space": raw.box.coordinate_space.value,
                    "input_frame_width": raw.box.frame_width,
                    "input_frame_height": raw.box.frame_height,
                    "output_coordinate_space": "source_pixels",
                },
            )
        )
    return DetectionBatch(
        frame_ref=frame_ref,
        model_id=model_id,
        model_mode=mode,
        source_stage=source_stage,
        detections=tuple(observations),
    )


class UnavailableDetectorAdapter(UnavailableModelAdapter):
    """Typed detector fallback that never fabricates boxes for missing weights."""

    def __init__(
        self,
        *,
        model_id: str = "grounding_dino_unavailable",
        reason: str = "detector backend or weights are unavailable",
    ) -> None:
        super().__init__(
            model_id=model_id,
            role="detection",
            reason=reason,
        )
