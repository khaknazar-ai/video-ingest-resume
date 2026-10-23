"""Serializable data contracts shared by pipeline stages."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from fractions import Fraction
from numbers import Real
from pathlib import Path
from typing import Any


def _number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a real number, got {value!r}")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return result


def _non_negative(value: float, name: str) -> float:
    result = _number(value, name)
    if result < 0:
        raise ValueError(f"{name} must be non-negative")
    return result


def _positive(value: float, name: str) -> float:
    result = _number(value, name)
    if result <= 0:
        raise ValueError(f"{name} must be positive")
    return result


def _integer(value: object, name: str, *, minimum: int | None = 0) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or (minimum is not None and value < minimum)
    ):
        raise ValueError(f"{name} must be an integer >= {minimum}, got {value!r}")
    return value


def _identifier(value: object, name: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ValueError(f"{name} must be a non-empty trimmed string")
    if any(character.isspace() for character in value):
        raise ValueError(f"{name} must not contain whitespace")
    return value


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ValueError(f"{name} must be a non-empty trimmed string")
    return value


def _optional_identifier(value: object, name: str) -> str | None:
    if value is None:
        return None
    return _identifier(value, name)


def _optional_non_negative(value: object, name: str) -> float | None:
    if value is None:
        return None
    return _non_negative(value, name)


def _optional_positive(value: object, name: str) -> float | None:
    if value is None:
        return None
    return _positive(value, name)


def _validate_sha256(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or value != value.lower()
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 hex digest")
    return value


def _bbox(value: object, name: str = "bbox_xyxy") -> tuple[float, float, float, float]:
    if not isinstance(value, (tuple, list)) or len(value) != 4:
        raise ValueError(f"{name} must contain four coordinates")
    coordinates = tuple(_number(item, f"{name}[{index}]") for index, item in enumerate(value))
    x1, y1, x2, y2 = coordinates
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"{name} must have positive width and height")
    return coordinates


def _strict_mapping(value: object, name: str, keys: set[str]) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    unknown = set(value) - keys
    if unknown:
        raise ValueError(f"{name} contains unknown fields: {sorted(unknown)!r}")
    return value


class TimestampQuality(StrEnum):
    """How a frame timestamp was established."""

    REQUESTED = "requested"
    EXACT_PTS = "exact_pts"
    CFR_ESTIMATE = "cfr_estimate"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class FrameRequest:
    """A request for a frame, before a decoder establishes actual identity."""

    video_id: str
    requested_timestamp_seconds: float
    source: str = "sampler"
    tolerance_seconds: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "video_id", _identifier(self.video_id, "video_id"))
        object.__setattr__(
            self,
            "requested_timestamp_seconds",
            _non_negative(self.requested_timestamp_seconds, "requested_timestamp_seconds"),
        )
        object.__setattr__(self, "source", _identifier(self.source, "source"))
        object.__setattr__(
            self,
            "tolerance_seconds",
            _optional_non_negative(self.tolerance_seconds, "tolerance_seconds"),
        )


@dataclass(frozen=True, slots=True)
class VideoMetadata:
    duration_seconds: float | None
    fps: float | None
    frame_count: int | None
    width: int | None
    height: int | None
    codec: str | None = None
    time_base: str | None = None
    variable_frame_rate: bool | None = None
    duration_source: str | None = None
    fps_source: str | None = None
    frame_count_source: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "duration_seconds",
            _optional_non_negative(self.duration_seconds, "duration_seconds"),
        )
        object.__setattr__(self, "fps", _optional_positive(self.fps, "fps"))
        if self.frame_count is not None:
            object.__setattr__(self, "frame_count", _integer(self.frame_count, "frame_count"))
        for name in ("width", "height"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _integer(value, name, minimum=1))
        for name in (
            "codec",
            "time_base",
            "duration_source",
            "fps_source",
            "frame_count_source",
        ):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _identifier(value, name))
        if self.variable_frame_rate is not None and not isinstance(
            self.variable_frame_rate, bool
        ):
            raise ValueError("variable_frame_rate must be a boolean or null")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> VideoMetadata:
        allowed = {
            "duration_seconds",
            "fps",
            "frame_count",
            "width",
            "height",
            "codec",
            "time_base",
            "variable_frame_rate",
            "duration_source",
            "fps_source",
            "frame_count_source",
        }
        return cls(**dict(_strict_mapping(value, "metadata", allowed)))


@dataclass(frozen=True, slots=True)
class VideoRecord:
    video_id: str
    source_path: Path
    metadata: VideoMetadata
    source_sha256: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "video_id", _identifier(self.video_id, "video_id"))
        if not isinstance(self.metadata, VideoMetadata):
            raise ValueError("metadata must be a VideoMetadata instance")
        if not isinstance(self.source_path, Path):
            if not isinstance(self.source_path, str) or not self.source_path:
                raise ValueError("source_path must be a non-empty path")
            object.__setattr__(self, "source_path", Path(self.source_path))
        if self.source_sha256 is not None:
            object.__setattr__(
                self, "source_sha256", _validate_sha256(self.source_sha256, "source_sha256")
            )

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["source_path"] = str(self.source_path)
        return value

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> VideoRecord:
        allowed = {"video_id", "source_path", "metadata", "source_sha256"}
        mapping = _strict_mapping(value, "video", allowed)
        required = {"video_id", "source_path", "metadata"}
        missing = required - set(mapping)
        if missing:
            raise ValueError(f"video is missing fields: {sorted(missing)!r}")
        metadata = mapping["metadata"]
        return cls(
            video_id=mapping["video_id"],
            source_path=Path(mapping["source_path"])
            if isinstance(mapping["source_path"], (str, Path))
            else mapping["source_path"],
            metadata=(
                metadata
                if isinstance(metadata, VideoMetadata)
                else VideoMetadata.from_dict(metadata)
            ),
            source_sha256=mapping.get("source_sha256"),
        )


@dataclass(frozen=True, slots=True)
class FrameRef:
    video_id: str
    frame_index: int
    timestamp_seconds: float
    width: int
    height: int
    source_path: Path | None = None
    timestamp_quality: TimestampQuality | str = TimestampQuality.REQUESTED
    timestamp_origin: str = "request"
    pts: int | None = None
    time_base: str | None = None
    source_sha256: str | None = None
    stream_index: int = 0
    decode_ordinal: int | None = None
    backend: str = "opencv"

    def __post_init__(self) -> None:
        object.__setattr__(self, "video_id", _identifier(self.video_id, "video_id"))
        object.__setattr__(self, "frame_index", _integer(self.frame_index, "frame_index"))
        object.__setattr__(
            self, "timestamp_seconds", _non_negative(self.timestamp_seconds, "timestamp_seconds")
        )
        object.__setattr__(self, "width", _integer(self.width, "width", minimum=1))
        object.__setattr__(self, "height", _integer(self.height, "height", minimum=1))
        try:
            quality = TimestampQuality(self.timestamp_quality)
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"unsupported timestamp_quality: {self.timestamp_quality!r}"
            ) from error
        object.__setattr__(self, "timestamp_quality", quality)
        object.__setattr__(
            self,
            "timestamp_origin",
            _identifier(self.timestamp_origin, "timestamp_origin"),
        )
        if self.pts is not None:
            if isinstance(self.pts, bool) or not isinstance(self.pts, int):
                raise ValueError(f"pts must be an integer, got {self.pts!r}")
            object.__setattr__(self, "pts", self.pts)
        if self.time_base is not None:
            object.__setattr__(self, "time_base", _identifier(self.time_base, "time_base"))
        if self.source_sha256 is not None:
            object.__setattr__(
                self,
                "source_sha256",
                _validate_sha256(self.source_sha256, "source_sha256"),
            )
        object.__setattr__(self, "stream_index", _integer(self.stream_index, "stream_index"))
        if self.decode_ordinal is not None:
            object.__setattr__(
                self,
                "decode_ordinal",
                _integer(self.decode_ordinal, "decode_ordinal"),
            )
        object.__setattr__(self, "backend", _identifier(self.backend, "backend"))

    @property
    def identity_key(self) -> tuple[object, ...]:
        """Portable identity; duplicate PTS are unsupported by the exact decoder.

        Location, backend and decode ordinal describe access, not source identity.
        The PTS time base is canonicalized so equivalent rationals compare equally.
        """

        source = (self.video_id, self.source_sha256, self.stream_index)
        if self.pts is not None and self.time_base is not None:
            rational = Fraction(self.time_base)
            if rational <= 0:
                raise ValueError("frame identity time_base must be positive")
            return (*source, "pts", self.pts, rational)
        if self.timestamp_quality is TimestampQuality.EXACT_PTS:
            raise ValueError("exact frame identity requires pts and time_base")
        return (*source, "index", self.frame_index)


@dataclass(frozen=True, slots=True)
class DecodeIssue:
    """A structured decode failure that must not disappear in an iterator."""

    video_id: str
    code: str
    message: str
    frame_index: int | None = None
    timestamp_seconds: float | None = None
    recoverable: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "video_id", _identifier(self.video_id, "video_id"))
        object.__setattr__(self, "code", _identifier(self.code, "code"))
        object.__setattr__(self, "message", _text(self.message, "message"))
        if self.frame_index is not None:
            object.__setattr__(self, "frame_index", _integer(self.frame_index, "frame_index"))
        object.__setattr__(
            self,
            "timestamp_seconds",
            _optional_non_negative(self.timestamp_seconds, "timestamp_seconds"),
        )
        if not isinstance(self.recoverable, bool):
            raise ValueError("recoverable must be a boolean")


@dataclass(frozen=True, slots=True)
class Shot:
    shot_id: str
    video_id: str
    start_seconds: float
    end_seconds: float
    representative_timestamps: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "shot_id", _identifier(self.shot_id, "shot_id"))
        object.__setattr__(self, "video_id", _identifier(self.video_id, "video_id"))
        start = _non_negative(self.start_seconds, "start_seconds")
        end = _non_negative(self.end_seconds, "end_seconds")
        if end < start:
            raise ValueError("end_seconds must be greater than or equal to start_seconds")
        timestamps = tuple(
            _non_negative(item, "representative_timestamps")
            for item in self.representative_timestamps
        )
        if any(item < start or item > end for item in timestamps):
            raise ValueError("representative_timestamps must be inside the shot interval")
        object.__setattr__(self, "start_seconds", start)
        object.__setattr__(self, "end_seconds", end)
        object.__setattr__(self, "representative_timestamps", timestamps)


@dataclass(frozen=True, slots=True)
class Detection:
    detection_id: str
    video_id: str
    frame_index: int
    timestamp_seconds: float
    label: str
    score: float
    bbox_xyxy: tuple[float, float, float, float]
    model_id: str
    track_id: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "detection_id", _identifier(self.detection_id, "detection_id"))
        object.__setattr__(self, "video_id", _identifier(self.video_id, "video_id"))
        object.__setattr__(self, "frame_index", _integer(self.frame_index, "frame_index"))
        object.__setattr__(
            self, "timestamp_seconds", _non_negative(self.timestamp_seconds, "timestamp_seconds")
        )
        object.__setattr__(self, "label", _identifier(self.label, "label"))
        score = _number(self.score, "score")
        if not 0.0 <= score <= 1.0:
            raise ValueError("score must be between 0 and 1")
        object.__setattr__(self, "score", score)
        object.__setattr__(self, "bbox_xyxy", _bbox(self.bbox_xyxy))
        object.__setattr__(self, "model_id", _identifier(self.model_id, "model_id"))
        object.__setattr__(self, "track_id", _optional_identifier(self.track_id, "track_id"))


@dataclass(frozen=True, slots=True)
class Track:
    track_id: str
    video_id: str
    label: str | None
    start_seconds: float
    end_seconds: float
    observation_count: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "track_id", _identifier(self.track_id, "track_id"))
        object.__setattr__(self, "video_id", _identifier(self.video_id, "video_id"))
        object.__setattr__(self, "label", _optional_identifier(self.label, "label"))
        start = _non_negative(self.start_seconds, "start_seconds")
        end = _non_negative(self.end_seconds, "end_seconds")
        if end < start:
            raise ValueError("end_seconds must be greater than or equal to start_seconds")
        object.__setattr__(self, "start_seconds", start)
        object.__setattr__(self, "end_seconds", end)
        object.__setattr__(
            self,
            "observation_count",
            _integer(self.observation_count, "observation_count"),
        )


@dataclass(frozen=True, slots=True)
class EvidenceItem:
    evidence_id: str
    video_id: str
    timestamp_seconds: float
    source: str
    score: float | None = None
    frame_index: int | None = None
    bbox_xyxy: tuple[float, float, float, float] | None = None
    mask_ref: str | None = None
    track_id: str | None = None
    provenance: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "evidence_id", _identifier(self.evidence_id, "evidence_id"))
        object.__setattr__(self, "video_id", _identifier(self.video_id, "video_id"))
        object.__setattr__(
            self, "timestamp_seconds", _non_negative(self.timestamp_seconds, "timestamp_seconds")
        )
        object.__setattr__(self, "source", _identifier(self.source, "source"))
        if self.score is not None:
            score = _number(self.score, "score")
            if not 0.0 <= score <= 1.0:
                raise ValueError("score must be between 0 and 1")
            object.__setattr__(self, "score", score)
        if self.frame_index is not None:
            object.__setattr__(self, "frame_index", _integer(self.frame_index, "frame_index"))
        if self.bbox_xyxy is not None:
            object.__setattr__(self, "bbox_xyxy", _bbox(self.bbox_xyxy))
        object.__setattr__(self, "track_id", _optional_identifier(self.track_id, "track_id"))
        if not isinstance(self.provenance, dict):
            raise ValueError("provenance must be a JSON object")


@dataclass(frozen=True, slots=True)
class Provenance:
    """Required lineage fields for evidence-producing stages."""

    run_id: str
    source_sha256: str
    stage: str
    component: str
    config_sha256: str | None = None
    model_id: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "run_id", _identifier(self.run_id, "run_id"))
        object.__setattr__(
            self,
            "source_sha256",
            _validate_sha256(self.source_sha256, "source_sha256"),
        )
        object.__setattr__(self, "stage", _identifier(self.stage, "stage"))
        object.__setattr__(self, "component", _identifier(self.component, "component"))
        if self.config_sha256 is not None:
            object.__setattr__(
                self, "config_sha256", _validate_sha256(self.config_sha256, "config_sha256")
            )
        object.__setattr__(self, "model_id", _optional_identifier(self.model_id, "model_id"))
