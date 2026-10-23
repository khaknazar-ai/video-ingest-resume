"""Transparent one-to-one detection metrics for annotated frames."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from numbers import Real
from typing import Any

from video_intelligence.core.types import FrameRef
from video_intelligence.detection.base import DetectionBatch
from video_intelligence.detection.postprocess import BoundingBox


def _identifier(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value.strip() != value
        or any(character.isspace() for character in value)
    ):
        raise ValueError(f"{name} must be a non-empty identifier")
    return value


def _score(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite number")
    return result


def _frame_key(frame_ref: FrameRef) -> tuple[str, int]:
    return frame_ref.video_id, frame_ref.frame_index


def _same_frame_identity(first: FrameRef, second: FrameRef) -> bool:
    if (
        first.video_id != second.video_id
        or first.frame_index != second.frame_index
        or first.width != second.width
        or first.height != second.height
        or first.source_sha256 != second.source_sha256
        or first.stream_index != second.stream_index
        or first.pts != second.pts
        or first.time_base != second.time_base
    ):
        return False
    return math.isclose(
        first.timestamp_seconds,
        second.timestamp_seconds,
        rel_tol=1e-9,
        abs_tol=1e-6,
    )


@dataclass(frozen=True, slots=True)
class GroundTruthBox:
    """One annotation box in canonical source pixels."""

    label: str
    bbox_xyxy: tuple[float, float, float, float]

    def __post_init__(self) -> None:
        object.__setattr__(self, "label", _identifier(self.label, "label"))
        if isinstance(self.bbox_xyxy, (str, bytes)):
            raise ValueError("bbox_xyxy must contain four finite numbers")
        values = tuple(_score(value, "bbox_xyxy") for value in self.bbox_xyxy)
        if len(values) != 4:
            raise ValueError("bbox_xyxy must contain four finite numbers")
        if values[0] >= values[2] or values[1] >= values[3]:
            raise ValueError("bbox_xyxy must have positive area")
        object.__setattr__(self, "bbox_xyxy", values)


@dataclass(frozen=True, slots=True)
class DetectionAnnotation:
    """Annotation for one frame; an empty box tuple is an explicit negative frame."""

    frame_ref: FrameRef
    boxes: tuple[GroundTruthBox, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not isinstance(self.frame_ref, FrameRef):
            raise TypeError("frame_ref must be a FrameRef")
        if isinstance(self.boxes, (str, bytes)) or not isinstance(self.boxes, Sequence):
            raise ValueError("boxes must be a sequence of GroundTruthBox values")
        boxes = tuple(self.boxes)
        if any(not isinstance(box, GroundTruthBox) for box in boxes):
            raise ValueError("boxes must contain only GroundTruthBox values")
        for box in boxes:
            BoundingBox(
                box.bbox_xyxy,
                "source_pixels",
                self.frame_ref.width,
                self.frame_ref.height,
            )
        object.__setattr__(self, "boxes", boxes)


@dataclass(frozen=True, slots=True)
class DetectionMetrics:
    """Frame-level aggregate with nulls where precision/recall is undefined."""

    iou_threshold: float
    annotated_frame_count: int
    evaluated_frame_count: int
    missing_frame_count: int
    true_positive: int
    false_positive: int
    false_negative: int
    precision: float | None
    recall: float | None
    mean_iou: float | None

    def __post_init__(self) -> None:
        if (
            isinstance(self.iou_threshold, bool)
            or not isinstance(self.iou_threshold, Real)
            or not 0.0 <= float(self.iou_threshold) <= 1.0
            or not math.isfinite(float(self.iou_threshold))
        ):
            raise ValueError("iou_threshold must be a finite number in [0, 1]")
        object.__setattr__(self, "iou_threshold", float(self.iou_threshold))
        for name in (
            "annotated_frame_count",
            "evaluated_frame_count",
            "missing_frame_count",
            "true_positive",
            "false_positive",
            "false_negative",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if self.evaluated_frame_count + self.missing_frame_count != self.annotated_frame_count:
            raise ValueError("evaluated and missing frame counts must sum to annotated count")
        for name in ("precision", "recall", "mean_iou"):
            value = getattr(self, name)
            if value is not None:
                value = _score(value, name)
                if not 0.0 <= value <= 1.0:
                    raise ValueError(f"{name} must be in [0, 1] or null")
                object.__setattr__(self, name, value)

    def to_dict(self) -> dict[str, Any]:
        return {
            "iou_threshold": self.iou_threshold,
            "annotated_frame_count": self.annotated_frame_count,
            "evaluated_frame_count": self.evaluated_frame_count,
            "missing_frame_count": self.missing_frame_count,
            "true_positive": self.true_positive,
            "false_positive": self.false_positive,
            "false_negative": self.false_negative,
            "precision": self.precision,
            "recall": self.recall,
            "mean_iou": self.mean_iou,
        }


def _iou(
    first: tuple[float, float, float, float],
    second: tuple[float, float, float, float],
) -> float:
    left = max(first[0], second[0])
    top = max(first[1], second[1])
    right = min(first[2], second[2])
    bottom = min(first[3], second[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    first_area = (first[2] - first[0]) * (first[3] - first[1])
    second_area = (second[2] - second[0]) * (second[3] - second[1])
    union = first_area + second_area - intersection
    return intersection / union if union > 0.0 else 0.0


def evaluate_detection(
    predictions: Sequence[DetectionBatch],
    annotations: Sequence[DetectionAnnotation],
    *,
    iou_threshold: float = 0.5,
) -> DetectionMetrics:
    """Match same-label boxes one-to-one by descending IoU."""

    if isinstance(predictions, (str, bytes)) or not isinstance(predictions, Sequence):
        raise ValueError("predictions must be a sequence of DetectionBatch values")
    if isinstance(annotations, (str, bytes)) or not isinstance(annotations, Sequence):
        raise ValueError("annotations must be a sequence of DetectionAnnotation values")
    prediction_values = tuple(predictions)
    annotation_values = tuple(annotations)
    if any(not isinstance(item, DetectionBatch) for item in prediction_values):
        raise ValueError("predictions must contain only DetectionBatch values")
    if any(not isinstance(item, DetectionAnnotation) for item in annotation_values):
        raise ValueError("annotations must contain only DetectionAnnotation values")
    annotation_by_frame = {_frame_key(item.frame_ref): item for item in annotation_values}
    if len(annotation_by_frame) != len(annotation_values):
        raise ValueError("annotation frame keys must be unique")
    prediction_by_frame = {_frame_key(item.frame_ref): item for item in prediction_values}
    if len(prediction_by_frame) != len(prediction_values):
        raise ValueError("prediction frame keys must be unique")
    if (
        isinstance(iou_threshold, bool)
        or not isinstance(iou_threshold, Real)
        or not math.isfinite(float(iou_threshold))
        or not 0.0 <= float(iou_threshold) <= 1.0
    ):
        raise ValueError("iou_threshold must be a finite number in [0, 1]")
    iou_threshold = float(iou_threshold)

    true_positive = false_positive = false_negative = 0
    matched_ious: list[float] = []
    missing_frame_count = 0
    evaluated_frame_count = 0
    for key, annotation in annotation_by_frame.items():
        prediction = prediction_by_frame.get(key)
        if prediction is None:
            missing_frame_count += 1
            false_negative += len(annotation.boxes)
            continue
        if not _same_frame_identity(prediction.frame_ref, annotation.frame_ref):
            raise ValueError("prediction and annotation frame identities do not match")
        if (
            prediction.frame_ref.width != annotation.frame_ref.width
            or prediction.frame_ref.height != annotation.frame_ref.height
        ):
            raise ValueError("prediction and annotation frame dimensions do not match")
        evaluated_frame_count += 1
        pairs = sorted(
            [
                (
                    _iou(candidate.bbox_xyxy, truth.bbox_xyxy),
                    candidate_index,
                    truth_index,
                )
                for candidate_index, candidate in enumerate(prediction.detections)
                for truth_index, truth in enumerate(annotation.boxes)
                if candidate.label == truth.label
            ],
            key=lambda item: (-item[0], item[1], item[2]),
        )
        used_candidates: set[int] = set()
        used_truth: set[int] = set()
        for overlap, candidate_index, truth_index in pairs:
            if (
                overlap < iou_threshold
                or candidate_index in used_candidates
                or truth_index in used_truth
            ):
                continue
            used_candidates.add(candidate_index)
            used_truth.add(truth_index)
            true_positive += 1
            matched_ious.append(overlap)
        false_positive += len(prediction.detections) - len(used_candidates)
        false_negative += len(annotation.boxes) - len(used_truth)
    precision_denominator = true_positive + false_positive
    recall_denominator = true_positive + false_negative
    return DetectionMetrics(
        iou_threshold=iou_threshold,
        annotated_frame_count=len(annotation_values),
        evaluated_frame_count=evaluated_frame_count,
        missing_frame_count=missing_frame_count,
        true_positive=true_positive,
        false_positive=false_positive,
        false_negative=false_negative,
        precision=(true_positive / precision_denominator if precision_denominator else None),
        recall=true_positive / recall_denominator if recall_denominator else None,
        mean_iou=sum(matched_ious) / len(matched_ious) if matched_ious else None,
    )
