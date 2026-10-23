"""Finite bounding boxes and explicit coordinate transforms."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from numbers import Real


def _dimension(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ValueError(f"{name} must be a non-empty trimmed string")
    return value


def _number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite real number")
    return result


def _xyxy(value: object) -> tuple[float, float, float, float]:
    if not isinstance(value, (tuple, list)) or len(value) != 4:
        raise ValueError("xyxy must contain four coordinates")
    return tuple(_number(item, f"xyxy[{index}]") for index, item in enumerate(value))


class CoordinateSpace(StrEnum):
    """Coordinate systems accepted at the detector boundary."""

    SOURCE_PIXELS = "source_pixels"
    RESIZED_PIXELS = "resized_pixels"
    NORMALIZED = "normalized"


class BoxFormat(StrEnum):
    """Box layouts accepted from model adapters; no implicit format guessing."""

    XYXY = "xyxy"
    CENTER_XYWH = "cxcywh"


@dataclass(frozen=True, slots=True)
class BoundingBox:
    """An xyxy box with dimensions for the coordinate space it belongs to."""

    xyxy: tuple[float, float, float, float]
    coordinate_space: CoordinateSpace | str
    frame_width: int
    frame_height: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "xyxy", _xyxy(self.xyxy))
        object.__setattr__(self, "frame_width", _dimension(self.frame_width, "frame_width"))
        object.__setattr__(self, "frame_height", _dimension(self.frame_height, "frame_height"))
        try:
            space = CoordinateSpace(self.coordinate_space)
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"unsupported coordinate space: {self.coordinate_space!r}"
            ) from error
        object.__setattr__(self, "coordinate_space", space)
        x1, y1, x2, y2 = self.xyxy
        if x2 <= x1 or y2 <= y1:
            raise ValueError("xyxy must have positive width and height")
        upper_x = 1.0 if space is CoordinateSpace.NORMALIZED else float(self.frame_width)
        upper_y = 1.0 if space is CoordinateSpace.NORMALIZED else float(self.frame_height)
        if x1 < 0.0 or y1 < 0.0 or x2 > upper_x or y2 > upper_y:
            raise ValueError("xyxy must be inside its declared coordinate space")

    def to_source_pixels(self, *, source_width: int, source_height: int) -> BoundingBox:
        """Invert resize/normalization into canonical source pixels without clipping."""

        source_width = _dimension(source_width, "source_width")
        source_height = _dimension(source_height, "source_height")
        x1, y1, x2, y2 = self.xyxy
        if self.coordinate_space is CoordinateSpace.SOURCE_PIXELS:
            if (self.frame_width, self.frame_height) != (source_width, source_height):
                raise ValueError("source-pixel box dimensions do not match source frame")
            values = self.xyxy
        elif self.coordinate_space is CoordinateSpace.NORMALIZED:
            values = (
                x1 * source_width,
                y1 * source_height,
                x2 * source_width,
                y2 * source_height,
            )
        else:
            values = (
                x1 * source_width / self.frame_width,
                y1 * source_height / self.frame_height,
                x2 * source_width / self.frame_width,
                y2 * source_height / self.frame_height,
            )
        return BoundingBox(
            xyxy=values,
            coordinate_space=CoordinateSpace.SOURCE_PIXELS,
            frame_width=source_width,
            frame_height=source_height,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "xyxy": list(self.xyxy),
            "coordinate_space": self.coordinate_space.value,
            "frame_width": self.frame_width,
            "frame_height": self.frame_height,
        }


def decode_box(
    values: Sequence[float],
    *,
    box_format: BoxFormat | str,
    coordinate_space: CoordinateSpace | str,
    frame_width: int,
    frame_height: int,
) -> BoundingBox:
    """Decode an explicitly declared model layout into a validated box."""

    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise ValueError("values must be a sequence of four coordinates")
    if len(values) != 4:
        raise ValueError("values must contain four coordinates")
    try:
        layout = BoxFormat(box_format)
    except (TypeError, ValueError) as error:
        raise ValueError(f"unsupported box format: {box_format!r}") from error
    coordinates = tuple(_number(item, f"values[{index}]") for index, item in enumerate(values))
    if layout is BoxFormat.XYXY:
        xyxy = coordinates
    else:
        center_x, center_y, width, height = coordinates
        if width <= 0.0 or height <= 0.0:
            raise ValueError("cxcywh width and height must be positive")
        xyxy = (
            center_x - width / 2.0,
            center_y - height / 2.0,
            center_x + width / 2.0,
            center_y + height / 2.0,
        )
    return BoundingBox(
        xyxy=xyxy,
        coordinate_space=coordinate_space,
        frame_width=frame_width,
        frame_height=frame_height,
    )


@dataclass(frozen=True, slots=True)
class EncodedDetection:
    """Raw model output with mandatory layout and coordinate declarations."""

    label: str
    score: float
    values: tuple[float, float, float, float]
    box_format: BoxFormat | str
    coordinate_space: CoordinateSpace | str
    frame_width: int
    frame_height: int
    phrase: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "label", _text(self.label, "label"))
        if isinstance(self.score, bool) or not isinstance(self.score, Real):
            raise ValueError("score must be a finite number in [0, 1]")
        score = float(self.score)
        if not math.isfinite(score) or not 0.0 <= score <= 1.0:
            raise ValueError("score must be a finite number in [0, 1]")
        object.__setattr__(self, "score", score)
        object.__setattr__(
            self,
            "values",
            tuple(_number(item, f"values[{index}]") for index, item in enumerate(self.values)),
        )
        if len(self.values) != 4:
            raise ValueError("values must contain four coordinates")
        try:
            object.__setattr__(self, "box_format", BoxFormat(self.box_format))
        except (TypeError, ValueError) as error:
            raise ValueError(f"unsupported box format: {self.box_format!r}") from error
        try:
            object.__setattr__(
                self,
                "coordinate_space",
                CoordinateSpace(self.coordinate_space),
            )
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"unsupported coordinate space: {self.coordinate_space!r}"
            ) from error
        object.__setattr__(self, "frame_width", _dimension(self.frame_width, "frame_width"))
        object.__setattr__(self, "frame_height", _dimension(self.frame_height, "frame_height"))
        if self.phrase is not None:
            object.__setattr__(self, "phrase", _text(self.phrase, "phrase"))

    def decode(self) -> RawDetection:
        return RawDetection(
            label=self.label,
            score=self.score,
            phrase=self.phrase,
            box=decode_box(
                self.values,
                box_format=self.box_format,
                coordinate_space=self.coordinate_space,
                frame_width=self.frame_width,
                frame_height=self.frame_height,
            ),
        )


def decode_model_detections(
    detections: Sequence[EncodedDetection],
    *,
    score_threshold: float = 0.0,
    max_detections: int | None = None,
) -> tuple[RawDetection, ...]:
    """Decode, threshold and deterministically cap model detections."""

    if isinstance(detections, (str, bytes)) or not isinstance(detections, Sequence):
        raise ValueError("detections must be a sequence")
    if any(not isinstance(item, EncodedDetection) for item in detections):
        raise ValueError("detections must contain only EncodedDetection values")
    if isinstance(score_threshold, bool) or not isinstance(score_threshold, Real):
        raise ValueError("score_threshold must be a finite number in [0, 1]")
    score_threshold = float(score_threshold)
    if not math.isfinite(score_threshold) or not 0.0 <= score_threshold <= 1.0:
        raise ValueError("score_threshold must be a finite number in [0, 1]")
    if max_detections is not None:
        max_detections = _dimension(max_detections, "max_detections")
    decoded = tuple(item.decode() for item in detections)
    decoded = tuple(item for item in decoded if item.score >= score_threshold)
    ordered = tuple(
        sorted(
            enumerate(decoded),
            key=lambda item: (-item[1].score, item[1].label, item[0]),
        )
    )
    return tuple(item[1] for item in ordered[:max_detections])


@dataclass(frozen=True, slots=True)
class RawDetection:
    """A detector output before source-coordinate inversion."""

    label: str
    score: float
    box: BoundingBox
    phrase: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "label", _text(self.label, "label"))
        if isinstance(self.score, bool) or not isinstance(self.score, Real):
            raise ValueError("score must be a finite number in [0, 1]")
        score = float(self.score)
        if not math.isfinite(score) or not 0.0 <= score <= 1.0:
            raise ValueError("score must be a finite number in [0, 1]")
        object.__setattr__(self, "score", score)
        if not isinstance(self.box, BoundingBox):
            raise TypeError("box must be a BoundingBox")
        if self.phrase is not None:
            object.__setattr__(self, "phrase", _text(self.phrase, "phrase"))
