"""Bounded tracking inputs and observed-only, portable association results."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
from enum import StrEnum
from fractions import Fraction
from typing import Any

from video_intelligence.core.types import FrameRef, TimestampQuality
from video_intelligence.detection.base import DetectionBatch, DetectionObservation

_MAX_FRAMES = 256
_MAX_OBJECTS = 256
_MAX_TRACKS = 4096
_MAX_SECONDS = 60.0


def _identifier(value: object, name: str) -> str:
    if (
        not isinstance(value, str) or not value or value.strip() != value
        or any(character.isspace() for character in value)
    ):
        raise ValueError(f"{name} must be a non-empty identifier")
    return value


def _integer(value: object, name: str, *, minimum: int = 1, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
        raise ValueError(f"{name} must be an integer in [{minimum}, {maximum}]")
    return value


def _number(value: object, name: str, *, maximum: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    try:
        result = float(value)
    except OverflowError as error:
        raise ValueError(f"{name} must be a finite number") from error
    if not math.isfinite(result) or not 0 < result <= maximum:
        raise ValueError(f"{name} must be positive and at most {maximum}")
    return result


def _sequence(value: object, name: str, *, maximum: int) -> tuple[Any, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{name} must be a bounded sequence")
    if len(value) > maximum:
        raise ValueError(f"{name} exceeds its {maximum} item cap")
    return tuple(value)


def _json_copy(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("detection provenance must be a JSON mapping")
    try:
        return json.loads(json.dumps(dict(value), ensure_ascii=False, allow_nan=False))
    except (TypeError, ValueError, RecursionError) as error:
        raise ValueError("detection provenance must contain finite JSON values") from error


def _validate_frame(frame: FrameRef) -> None:
    if not isinstance(frame, FrameRef):
        raise TypeError("frame_ref must be a FrameRef")
    if frame.source_sha256 is None:
        raise ValueError("tracking requires a source SHA-256")
    if frame.timestamp_quality not in {TimestampQuality.EXACT_PTS, TimestampQuality.CFR_ESTIMATE}:
        raise ValueError("tracking requires exact_pts or cfr_estimate observations")
    if frame.timestamp_quality is TimestampQuality.EXACT_PTS:
        if frame.pts is None or frame.time_base is None:
            raise ValueError("exact tracking frame requires PTS and time_base")
        try:
            valid = Fraction(frame.time_base) > 0
        except (ValueError, ZeroDivisionError) as error:
            raise ValueError("exact tracking time_base must be a positive rational") from error
        if not valid:
            raise ValueError("exact tracking time_base must be a positive rational")
    # Also validate any optional PTS identity on estimated frames.
    _ = frame.identity_key


def _portable_frame(frame: FrameRef) -> dict[str, Any]:
    payload = asdict(frame)
    payload["source_path"] = None
    payload["timestamp_quality"] = frame.timestamp_quality.value
    return payload


def _copy_detection(detection: DetectionObservation) -> DetectionObservation:
    if not isinstance(detection, DetectionObservation):
        raise TypeError("detection must be a DetectionObservation")
    _validate_frame(detection.frame_ref)
    return replace(
        detection, frame_ref=replace(detection.frame_ref),
        provenance=_json_copy(detection.provenance),
    )


def _frame_source(frame: FrameRef) -> tuple[object, ...]:
    return (
        frame.video_id, frame.source_sha256, frame.stream_index, frame.width, frame.height,
        frame.timestamp_quality, frame.timestamp_origin,
    )


def _detection_source(detection: DetectionObservation) -> tuple[object, ...]:
    return (
        *_frame_source(detection.frame_ref), detection.model_id,
        detection.model_mode, detection.source_stage,
    )


def gap_exceeds(earlier: FrameRef, later: FrameRef, limit: float) -> bool:
    """Compare seconds without turning a decimal equality into an expired gap.

    Exact frames use rational PTS deltas. CFR estimates allow four ULPs of the
    supplied clock values; this handles floating subtraction, not timestamp quality.
    """

    _validate_frame(earlier)
    _validate_frame(later)
    limit = _number(limit, "time limit", maximum=_MAX_SECONDS)
    if earlier.timestamp_quality is not later.timestamp_quality:
        raise ValueError("time comparison requires a consistent timestamp quality")
    if earlier.timestamp_quality is TimestampQuality.EXACT_PTS:
        time_base = Fraction(earlier.time_base)
        if Fraction(later.time_base) != time_base:
            raise ValueError("time comparison requires a consistent exact time_base")
        return (later.pts - earlier.pts) * time_base > Fraction(str(limit))
    delta = later.timestamp_seconds - earlier.timestamp_seconds
    tolerance = 4 * max(
        math.ulp(earlier.timestamp_seconds), math.ulp(later.timestamp_seconds), math.ulp(limit),
    )
    return delta > limit and delta - limit > tolerance


@dataclass(frozen=True, slots=True)
class TrackerConfig:
    """CPU association limits; timing thresholds are actual seconds."""

    iou_threshold: float = 0.3
    min_hits: int = 2
    max_age_seconds: float = 0.5
    max_observation_gap_seconds: float = 0.25
    max_frames: int = 64
    max_detections_per_frame: int = 32
    max_active_tracks: int = 64
    max_tracks: int = 256
    max_window_seconds: float = 5.0
    max_runtime_seconds: float = 5.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "iou_threshold", _number(
            self.iou_threshold, "iou_threshold", maximum=1.0,
        ))
        for name, maximum in (
            ("min_hits", _MAX_FRAMES), ("max_frames", _MAX_FRAMES),
            ("max_detections_per_frame", _MAX_OBJECTS),
            ("max_active_tracks", _MAX_OBJECTS), ("max_tracks", _MAX_TRACKS),
        ):
            object.__setattr__(self, name, _integer(getattr(self, name), name, maximum=maximum))
        for name in (
            "max_age_seconds", "max_observation_gap_seconds",
            "max_window_seconds", "max_runtime_seconds",
        ):
            object.__setattr__(self, name, _number(getattr(self, name), name, maximum=_MAX_SECONDS))
        if self.max_active_tracks > self.max_tracks:
            raise ValueError("max_active_tracks must not exceed max_tracks")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class TrackingFrame:
    """One processed frame, including an explicit empty detection batch."""

    batch: DetectionBatch
    shot_id: str
    scene_cut: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.batch, DetectionBatch):
            raise TypeError("batch must be a DetectionBatch")
        object.__setattr__(self, "shot_id", _identifier(self.shot_id, "shot_id"))
        if not isinstance(self.scene_cut, bool):
            raise ValueError("scene_cut must be a boolean")
        _validate_frame(self.batch.frame_ref)
        detections = _sequence(self.batch.detections, "detections", maximum=_MAX_OBJECTS)
        copied = tuple(_copy_detection(item) for item in detections)
        object.__setattr__(self, "batch", replace(
            self.batch, frame_ref=replace(self.batch.frame_ref), detections=copied,
        ))

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame_ref": _portable_frame(self.batch.frame_ref),
            "shot_id": self.shot_id, "scene_cut": self.scene_cut,
            "model_id": self.batch.model_id, "model_mode": self.batch.model_mode.value,
            "source_stage": self.batch.source_stage,
            "detections": [TrackObservation(item).to_dict() for item in self.batch.detections],
        }


@dataclass(frozen=True, slots=True)
class TrackingRequest:
    """Complete validated window; this contract does not certify dense coverage."""

    run_id: str
    window_id: str
    frames: tuple[TrackingFrame, ...]
    config: TrackerConfig = field(default_factory=TrackerConfig)

    def __post_init__(self) -> None:
        object.__setattr__(self, "run_id", _identifier(self.run_id, "run_id"))
        object.__setattr__(self, "window_id", _identifier(self.window_id, "window_id"))
        if not isinstance(self.config, TrackerConfig):
            raise TypeError("config must be a TrackerConfig")
        frames = _sequence(self.frames, "frames", maximum=self.config.max_frames)
        if not frames:
            raise ValueError("tracking requires at least one processed frame")
        if any(not isinstance(item, TrackingFrame) for item in frames):
            raise TypeError("frames must contain only TrackingFrame values")
        identities: set[tuple[object, ...]] = set()
        detection_ids: set[str] = set()
        first = frames[0].batch
        source = (
            *_frame_source(first.frame_ref), first.model_id, first.model_mode, first.source_stage,
        )
        exact = first.frame_ref.timestamp_quality is TimestampQuality.EXACT_PTS
        time_base = Fraction(first.frame_ref.time_base) if exact else None
        previous_time: float | None = None
        previous_pts: int | None = None
        copied = []
        for frame in frames:
            batch = frame.batch
            if len(batch.detections) > self.config.max_detections_per_frame:
                raise ValueError("detections exceed max_detections_per_frame")
            current = TrackingFrame(batch, frame.shot_id, frame.scene_cut)
            ref = current.batch.frame_ref
            current_source = (
                *_frame_source(ref), batch.model_id, batch.model_mode, batch.source_stage,
            )
            if current_source != source:
                raise ValueError("tracking source, dimensions and model provenance must agree")
            if ref.identity_key in identities:
                raise ValueError("tracking frames contain duplicate canonical identity")
            identities.add(ref.identity_key)
            if previous_time is not None and ref.timestamp_seconds <= previous_time:
                raise ValueError("tracking frame timestamps must be strictly increasing")
            previous_time = ref.timestamp_seconds
            elapsed = ref.timestamp_seconds - first.frame_ref.timestamp_seconds
            if exact:
                if Fraction(ref.time_base) != time_base:
                    raise ValueError("exact tracking time_base must be consistent within window")
                if previous_pts is not None and ref.pts <= previous_pts:
                    raise ValueError("exact tracking PTS must be strictly increasing")
                previous_pts = ref.pts
                try:
                    pts_elapsed = float((ref.pts - first.frame_ref.pts) * time_base)
                except OverflowError as error:
                    raise ValueError("exact tracking PTS delta exceeds finite range") from error
                if not math.isclose(elapsed, pts_elapsed, abs_tol=1e-6, rel_tol=1e-9):
                    raise ValueError("exact tracking PTS and timestamp deltas disagree")
            if gap_exceeds(first.frame_ref, ref, self.config.max_window_seconds):
                raise ValueError("tracking frames exceed max_window_seconds")
            for detection in current.batch.detections:
                if detection.detection_id in detection_ids:
                    raise ValueError("detection_ids must be unique throughout a tracking window")
                detection_ids.add(detection.detection_id)
            copied.append(current)
        object.__setattr__(self, "frames", tuple(copied))

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id, "window_id": self.window_id,
            "config": self.config.to_dict(), "frames": [frame.to_dict() for frame in self.frames],
        }


@dataclass(frozen=True, slots=True)
class TrackObservation:
    """A measured detection; association never synthesizes or interpolates its box."""

    detection: DetectionObservation

    def __post_init__(self) -> None:
        object.__setattr__(self, "detection", _copy_detection(self.detection))

    @property
    def frame_ref(self) -> FrameRef:
        return self.detection.frame_ref

    def to_dict(self) -> dict[str, Any]:
        detection = self.detection
        return {
            "detection_id": detection.detection_id,
            "frame_ref": _portable_frame(detection.frame_ref),
            "label": detection.label, "bbox_xyxy": list(detection.bbox_xyxy),
            "score": detection.score, "model_id": detection.model_id,
            "model_mode": detection.model_mode.value, "source_stage": detection.source_stage,
            "phrase": detection.phrase, "provenance": _json_copy(detection.provenance),
        }


class TrackState(StrEnum):
    TENTATIVE = "tentative"
    CONFIRMED = "confirmed"
    LOST = "lost"
    TERMINATED = "terminated"


@dataclass(frozen=True, slots=True)
class TrackSnapshot:
    """Association state as of the last processed frame, without a forced finish."""

    track_id: str
    shot_id: str
    scene_epoch: int
    label: str
    state: TrackState
    observations: tuple[TrackObservation, ...]
    termination_reason: str | None = None

    def __post_init__(self) -> None:
        for name in ("track_id", "shot_id", "label"):
            object.__setattr__(self, name, _identifier(getattr(self, name), name))
        object.__setattr__(self, "scene_epoch", _integer(
            self.scene_epoch, "scene_epoch", minimum=0, maximum=_MAX_FRAMES,
        ))
        object.__setattr__(self, "state", TrackState(self.state))
        observations = _sequence(self.observations, "observations", maximum=_MAX_FRAMES)
        if not observations:
            raise ValueError("a track must contain at least one measured observation")
        if any(not isinstance(item, TrackObservation) for item in observations):
            raise TypeError("observations must contain only TrackObservation values")
        copied = tuple(TrackObservation(item.detection) for item in observations)
        source = _detection_source(copied[0].detection)
        seen_ids: set[str] = set()
        previous_time: float | None = None
        for observation in copied:
            detection = observation.detection
            if detection.label != self.label or _detection_source(detection) != source:
                raise ValueError("track observations must share label, source and model provenance")
            if detection.detection_id in seen_ids:
                raise ValueError("a track cannot repeat a detection")
            seen_ids.add(detection.detection_id)
            timestamp = observation.frame_ref.timestamp_seconds
            if previous_time is not None and timestamp <= previous_time:
                raise ValueError("track observation timestamps must be strictly increasing")
            previous_time = timestamp
        if self.termination_reason is not None:
            _identifier(self.termination_reason, "termination_reason")
        if (self.state is TrackState.TERMINATED) != (self.termination_reason is not None):
            raise ValueError("only terminated tracks require a termination_reason")
        object.__setattr__(self, "observations", copied)

    @property
    def last_seen_seconds(self) -> float:
        return self.observations[-1].frame_ref.timestamp_seconds

    def to_dict(self) -> dict[str, Any]:
        return {
            "track_id": self.track_id, "shot_id": self.shot_id, "scene_epoch": self.scene_epoch,
            "label": self.label, "state": self.state.value,
            "observations": [item.to_dict() for item in self.observations],
            "last_seen_seconds": self.last_seen_seconds,
            "termination_reason": self.termination_reason,
        }


@dataclass(frozen=True, slots=True)
class TrackTransition:
    track_id: str
    frame_ref: FrameRef
    previous_state: TrackState | None
    state: TrackState
    reason: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "track_id", _identifier(self.track_id, "track_id"))
        _validate_frame(self.frame_ref)
        object.__setattr__(self, "frame_ref", replace(self.frame_ref))
        if self.previous_state is not None:
            object.__setattr__(self, "previous_state", TrackState(self.previous_state))
        object.__setattr__(self, "state", TrackState(self.state))
        object.__setattr__(self, "reason", _identifier(self.reason, "reason"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "track_id": self.track_id, "frame_ref": _portable_frame(self.frame_ref),
            "previous_state": None if self.previous_state is None else self.previous_state.value,
            "state": self.state.value, "reason": self.reason,
        }


@dataclass(frozen=True, slots=True)
class TrackingResult:
    """Association result with full negative-frame lineage and no quality claim."""

    run_id: str
    window_id: str
    algorithm_id: str
    config: TrackerConfig
    tracks: tuple[TrackSnapshot, ...]
    transitions: tuple[TrackTransition, ...]
    processed_frames: int
    frames: tuple[TrackingFrame, ...]

    def __post_init__(self) -> None:
        for name in ("run_id", "window_id", "algorithm_id"):
            object.__setattr__(self, name, _identifier(getattr(self, name), name))
        validated = TrackingRequest(self.run_id, self.window_id, self.frames, self.config)
        object.__setattr__(self, "frames", validated.frames)
        _integer(self.processed_frames, "processed_frames", maximum=self.config.max_frames)
        if self.processed_frames != len(self.frames):
            raise ValueError("processed_frames must equal the complete frame lineage length")
        tracks = _sequence(self.tracks, "tracks", maximum=self.config.max_tracks)
        if any(not isinstance(item, TrackSnapshot) for item in tracks):
            raise TypeError("tracks must contain only TrackSnapshot values")
        copied = tuple(replace(item) for item in tracks)
        if len({item.track_id for item in copied}) != len(copied):
            raise ValueError("track_ids must be unique in a tracking result")
        transitions = _sequence(
            self.transitions, "transitions",
            maximum=self.config.max_tracks * self.config.max_frames,
        )
        if any(not isinstance(item, TrackTransition) for item in transitions):
            raise TypeError("transitions must contain only TrackTransition values")
        input_frames = {frame.batch.frame_ref.identity_key: frame for frame in self.frames}
        input_detections = {
            detection.detection_id: (detection, frame.shot_id)
            for frame in self.frames for detection in frame.batch.detections
        }
        assigned: set[str] = set()
        for track in copied:
            for observation in track.observations:
                detection = observation.detection
                if input_detections.get(detection.detection_id) != (detection, track.shot_id):
                    raise ValueError("track observation must match an input detection and shot")
                if detection.detection_id in assigned:
                    raise ValueError("a detection cannot be assigned to multiple tracks")
                assigned.add(detection.detection_id)
        if assigned != set(input_detections):
            raise ValueError("tracking results cannot omit processed input detections")
        known_tracks = {track.track_id for track in copied}
        frame_positions = {
            frame.batch.frame_ref.identity_key: index for index, frame in enumerate(self.frames)
        }
        states: dict[str, TrackState] = {}
        last_transition: dict[str, TrackTransition] = {}
        previous_position = -1
        for transition in transitions:
            if transition.track_id not in known_tracks:
                raise ValueError("transition must refer to a known track")
            frame = input_frames.get(transition.frame_ref.identity_key)
            if frame is None or frame.batch.frame_ref != transition.frame_ref:
                raise ValueError("transition must refer to an actual processed frame")
            position = frame_positions[transition.frame_ref.identity_key]
            if position < previous_position:
                raise ValueError("transitions must follow processed frame order")
            previous_position = position
            current = states.get(transition.track_id)
            if current is None:
                if transition.previous_state is not None or transition.state not in {
                    TrackState.TENTATIVE, TrackState.CONFIRMED,
                }:
                    raise ValueError("a track must begin with tentative or confirmed creation")
            elif current is TrackState.TERMINATED or transition.previous_state is not current:
                raise ValueError("transition chain must match the prior nonterminated state")
            states[transition.track_id] = transition.state
            last_transition[transition.track_id] = transition
        for track in copied:
            if states.get(track.track_id) is not track.state:
                raise ValueError("each snapshot requires a creation and matching final transition")
            if (
                track.state is TrackState.TERMINATED
                and last_transition[track.track_id].reason != track.termination_reason
            ):
                raise ValueError("termination_reason must match the terminal transition")
        object.__setattr__(self, "tracks", copied)
        object.__setattr__(self, "transitions", tuple(replace(item) for item in transitions))

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id, "window_id": self.window_id, "algorithm_id": self.algorithm_id,
            "quality_status": "unmeasured", "identity_scope": "within_window_association",
            "config": self.config.to_dict(), "processed_frames": self.processed_frames,
            "frames": [frame.to_dict() for frame in self.frames],
            "tracks": [track.to_dict() for track in self.tracks],
            "transitions": [transition.to_dict() for transition in self.transitions],
        }
