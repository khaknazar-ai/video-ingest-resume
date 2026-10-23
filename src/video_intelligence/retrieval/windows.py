"""Bounded, source-bound candidate windows; similarity is never visual evidence."""

from __future__ import annotations

import hashlib
import json
import math
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from fractions import Fraction
from typing import Any

from video_intelligence.core.types import FrameRequest, Shot, VideoRecord
from video_intelligence.models.contracts import ModelMode
from video_intelligence.retrieval.candidate_search import CandidateSet
from video_intelligence.retrieval.schemas import ParseStatus, QuerySpec

_MAX_WINDOWS = 64
_MAX_CANDIDATES = 256
_MAX_FRAMES = 4096
_MAX_SHOTS = 4096
_MAX_SECONDS = 60.0


def _identifier(value: object, name: str) -> str:
    if (
        not isinstance(value, str) or not value or value.strip() != value
        or len(value) > 256 or any(character.isspace() for character in value)
    ):
        raise ValueError(f"{name} must be a bounded non-empty identifier")
    return value


def _number(value: object, name: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    try:
        result = float(value)
    except OverflowError as error:
        raise ValueError(f"{name} must be a finite number") from error
    if not math.isfinite(result) or result < 0 or (positive and result == 0):
        raise ValueError(f"{name} must be finite and {'positive' if positive else 'non-negative'}")
    return result


def _integer(value: object, name: str, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 < value <= maximum:
        raise ValueError(f"{name} must be an integer in [1,{maximum}]")
    return value


def _sequence(value: object, name: str, maximum: int) -> tuple:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{name} must be a bounded sequence")
    if len(value) > maximum:
        raise ValueError(f"{name} exceeds its item budget")
    return tuple(value)


def _fraction(value: float) -> Fraction:
    return Fraction(str(value))


def _deadline(deadline: float) -> None:
    if time.perf_counter() >= deadline:
        raise TimeoutError("candidate planning exceeded its cooperative deadline")


@dataclass(frozen=True, slots=True)
class WindowConfig:
    context_before_seconds: float = 0.2
    context_after_seconds: float = 0.3
    sample_fps: float = 10.0
    max_window_seconds: float = 5.0
    max_windows: int = 4
    max_candidates: int = 32
    max_total_frames: int = 64
    max_runtime_seconds: float = 5.0

    def __post_init__(self) -> None:
        for name in (
            "context_before_seconds", "context_after_seconds", "sample_fps",
            "max_window_seconds", "max_runtime_seconds",
        ):
            value = _number(getattr(self, name), name, positive=name != "context_before_seconds")
            maximum = 120.0 if name == "sample_fps" else _MAX_SECONDS
            if value > maximum:
                raise ValueError(f"{name} exceeds the hard limit of {maximum}")
            object.__setattr__(self, name, value)
        for name, maximum in (
            ("max_windows", _MAX_WINDOWS), ("max_candidates", _MAX_CANDIDATES),
            ("max_total_frames", _MAX_FRAMES),
        ):
            object.__setattr__(self, name, _integer(getattr(self, name), name, maximum))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class WindowSeed:
    entity_id: str
    row_id: str
    rank: int
    score: float
    timestamp_seconds: float

    def __post_init__(self) -> None:
        for name in ("entity_id", "row_id"):
            object.__setattr__(self, name, _identifier(getattr(self, name), name))
        _integer(self.rank, "rank", _MAX_CANDIDATES)
        if isinstance(self.score, bool) or not isinstance(self.score, (int, float)):
            raise ValueError("seed score must be finite")
        if not math.isfinite(self.score):
            raise ValueError("seed score must be finite")
        object.__setattr__(self, "score", float(self.score))
        object.__setattr__(self, "timestamp_seconds", _number(
            self.timestamp_seconds, "timestamp_seconds"
        ))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class CandidateWindow:
    window_id: str
    shot_id: str
    start_seconds: float
    end_seconds: float
    seeds: tuple[WindowSeed, ...]
    requests: tuple[FrameRequest, ...]

    def __post_init__(self) -> None:
        for name in ("window_id", "shot_id"):
            object.__setattr__(self, name, _identifier(getattr(self, name), name))
        start = _number(self.start_seconds, "window start")
        end = _number(self.end_seconds, "window end", positive=True)
        if end <= start:
            raise ValueError("candidate window must have a positive representable duration")
        object.__setattr__(self, "start_seconds", start)
        object.__setattr__(self, "end_seconds", end)
        seeds = _sequence(self.seeds, "seeds", _MAX_CANDIDATES)
        requests = _sequence(self.requests, "requests", _MAX_FRAMES)
        if not seeds or any(not isinstance(seed, WindowSeed) for seed in seeds):
            raise ValueError("candidate window requires typed seeds")
        if not requests or any(not isinstance(item, FrameRequest) for item in requests):
            raise ValueError("candidate window requires typed frame requests")
        seeds = tuple(replace(seed) for seed in seeds)
        requests = tuple(replace(item) for item in requests)
        times = tuple(item.requested_timestamp_seconds for item in requests)
        if len({item.video_id for item in requests}) != 1:
            raise ValueError("window requests must reference one video")
        if any(not start <= value < end for value in times):
            raise ValueError("window requests must stay in its half-open bounds")
        if any(left >= right for left, right in zip(times[:-1], times[1:], strict=True)):
            raise ValueError("window requests must be unique and chronological")
        if any(seed.timestamp_seconds not in times for seed in seeds):
            raise ValueError("every candidate seed must be requested inside its window")
        if len({(seed.entity_id, seed.row_id) for seed in seeds}) != len(seeds):
            raise ValueError("window seeds must be unique per entity and row")
        object.__setattr__(self, "seeds", seeds)
        object.__setattr__(self, "requests", requests)

    def to_dict(self) -> dict[str, Any]:
        return {
            "window_id": self.window_id, "shot_id": self.shot_id,
            "start_seconds": self.start_seconds, "end_seconds": self.end_seconds,
            "seeds": [seed.to_dict() for seed in self.seeds],
            "requests": [asdict(request) for request in self.requests],
        }


@dataclass(frozen=True, slots=True)
class CandidateWindowPlan:
    query_id: str
    video_id: str
    source_sha256: str
    fingerprint_id: str
    model_id: str
    model_mode: ModelMode
    config: WindowConfig
    windows: tuple[CandidateWindow, ...]
    stream_index: int | None = None

    def __post_init__(self) -> None:
        for name in ("query_id", "video_id", "fingerprint_id", "model_id"):
            object.__setattr__(self, name, _identifier(getattr(self, name), name))
        if (
            not isinstance(self.source_sha256, str) or len(self.source_sha256) != 64
            or any(character not in "0123456789abcdef" for character in self.source_sha256)
        ):
            raise ValueError("plan source requires a lowercase SHA-256")
        object.__setattr__(self, "model_mode", ModelMode(self.model_mode))
        if not isinstance(self.config, WindowConfig):
            raise TypeError("config must be a WindowConfig")
        if self.stream_index is not None and (
            isinstance(self.stream_index, bool) or not isinstance(self.stream_index, int)
            or self.stream_index < 0
        ):
            raise ValueError("stream_index must be a non-negative integer or null")
        windows = _sequence(self.windows, "windows", self.config.max_windows)
        if any(not isinstance(window, CandidateWindow) for window in windows):
            raise TypeError("windows must contain CandidateWindow values")
        windows = tuple(replace(window) for window in windows)
        if len({window.window_id for window in windows}) != len(windows):
            raise ValueError("window ids must be unique")
        if any(left.end_seconds > right.start_seconds
               for left, right in zip(windows[:-1], windows[1:], strict=True)):
            raise ValueError("planned windows must be chronological and non-overlapping")
        for window in windows:
            if _fraction(window.end_seconds) - _fraction(window.start_seconds) > _fraction(
                self.config.max_window_seconds
            ):
                raise ValueError("window exceeds max_window_seconds")
            if any(request.video_id != self.video_id for request in window.requests):
                raise ValueError("window request belongs to another video")
        object.__setattr__(self, "windows", windows)
        if self.total_frames > self.config.max_total_frames:
            raise ValueError("planned requests exceed max_total_frames")

    @property
    def total_frames(self) -> int:
        return sum(len(window.requests) for window in self.windows)

    def to_dict(self) -> dict[str, Any]:
        return {
            "query_id": self.query_id, "video_id": self.video_id,
            "source_sha256": self.source_sha256, "fingerprint_id": self.fingerprint_id,
            "model_id": self.model_id, "model_mode": self.model_mode.value,
            "stream_index": self.stream_index, "config": self.config.to_dict(),
            "windows": [window.to_dict() for window in self.windows],
            "total_frames": self.total_frames,
        }


_DEFAULT_CONFIG = WindowConfig()


def plan_candidate_windows(
    video: VideoRecord,
    query: QuerySpec,
    candidates_by_entity: Mapping[str, CandidateSet],
    *,
    shots: Sequence[Shot],
    config: WindowConfig = _DEFAULT_CONFIG,
) -> CandidateWindowPlan:
    """Clip, merge and densely sample candidates without inferring their semantics.

    All submitted seeds survive or planning fails before execution. Empty retrieval
    produces an empty plan, not a negative answer. Context after a seed must be
    positive so that the seed remains inside a representable half-open window.
    """
    if not isinstance(config, WindowConfig):
        raise TypeError("config must be a WindowConfig")
    config = replace(config)
    deadline = time.perf_counter() + config.max_runtime_seconds
    _deadline(deadline)
    if not isinstance(video, VideoRecord) or not isinstance(query, QuerySpec):
        raise TypeError("planner requires VideoRecord and QuerySpec")
    if video.source_sha256 is None:
        raise ValueError("candidate planning requires registered source identity")
    duration = _number(video.metadata.duration_seconds, "video duration", positive=True)
    query = replace(query)
    if query.parse_status is not ParseStatus.SUPPORTED:
        raise ValueError("candidate planning requires a supported query")
    entities = (query.entity,) if query.secondary_entity is None else (
        query.entity, query.secondary_entity
    )
    if not isinstance(candidates_by_entity, Mapping) or set(candidates_by_entity) != set(entities):
        raise ValueError("candidate entity roles must exactly match the query")
    role_sets = []
    for entity in entities:
        _deadline(deadline)
        value = candidates_by_entity[entity]
        if not isinstance(value, CandidateSet):
            raise TypeError("each entity requires a CandidateSet")
        if len(value.candidates) > config.max_candidates:
            raise ValueError("retrieval exceeds max_candidates")
        value = replace(value)
        if value.query_id != query.query_id:
            raise ValueError("candidate query identity does not match request")
        if value.model_id is None or value.model_mode is None:
            raise ValueError("candidate model and mode provenance are required")
        role_sets.append((entity, value))
    if sum(len(value.candidates) for _, value in role_sets) > config.max_candidates:
        raise ValueError("retrieval exceeds the combined max_candidates budget")
    first = role_sets[0][1]
    provenance = (first.fingerprint_id, first.model_id, first.model_mode)
    if any((value.fingerprint_id, value.model_id, value.model_mode) != provenance
           for _, value in role_sets):
        raise ValueError("entity retrieval model, mode and fingerprint must agree")
    ordered_shots = _sequence(shots, "shots", _MAX_SHOTS)
    if not ordered_shots or any(not isinstance(shot, Shot) for shot in ordered_shots):
        raise ValueError("planning requires non-empty typed shot boundaries")
    ordered_shots = tuple(sorted((replace(shot) for shot in ordered_shots), key=lambda item: (
        item.start_seconds, item.end_seconds, item.shot_id
    )))
    previous_end = Fraction(0)
    seen_shots = set()
    for shot in ordered_shots:
        _deadline(deadline)
        if (
            shot.video_id != video.video_id or shot.shot_id in seen_shots
            or not 0 <= shot.start_seconds < shot.end_seconds <= duration
            or _fraction(shot.start_seconds) < previous_end
        ):
            raise ValueError("shots must be positive, non-overlapping and bound to this video")
        seen_shots.add(shot.shot_id)
        previous_end = _fraction(shot.end_seconds)
    lower = _fraction(query.time_start_seconds or 0.0)
    upper = min(_fraction(duration), _fraction(
        duration if query.time_end_seconds is None else query.time_end_seconds
    ))
    segments = []
    streams = set()
    for entity, values in role_sets:
        for candidate in values.candidates:
            _deadline(deadline)
            candidate = replace(candidate)
            metadata = dict(candidate.metadata)
            if metadata.get("video_id") != video.video_id or metadata.get(
                "source_sha256"
            ) != video.source_sha256:
                raise ValueError("candidate source identity does not match registered video")
            timestamp = _number(metadata.get("timestamp_seconds"), "candidate timestamp")
            point = _fraction(timestamp)
            if not lower <= point < upper:
                raise ValueError("candidate lies outside video or query half-open time filter")
            for key, expected in (
                ("embedding_fingerprint_id", first.fingerprint_id),
                ("embedding_model_id", first.model_id),
                ("embedding_model_mode", first.model_mode.value),
            ):
                if key in metadata and metadata[key] != expected:
                    raise ValueError("candidate metadata conflicts with embedding provenance")
            if "stream_index" in metadata:
                stream = metadata["stream_index"]
                if isinstance(stream, bool) or not isinstance(stream, int) or stream < 0:
                    raise ValueError("candidate stream_index must be a non-negative integer")
                streams.add(stream)
                if len(streams) > 1:
                    raise ValueError("candidate streams must agree")
            matching = [shot for shot in ordered_shots
                        if _fraction(shot.start_seconds) <= point < _fraction(shot.end_seconds)]
            if len(matching) != 1:
                raise ValueError("candidate must lie inside exactly one supplied shot")
            shot = matching[0]
            if "shot_id" in metadata and metadata["shot_id"] != shot.shot_id:
                raise ValueError("candidate shot metadata conflicts with supplied boundary")
            start = max(lower, _fraction(shot.start_seconds), point - _fraction(
                config.context_before_seconds
            ))
            end = min(upper, _fraction(shot.end_seconds), point + _fraction(
                config.context_after_seconds
            ))
            if not start <= point < end:
                raise ValueError("candidate cannot be retained inside the clipped window")
            if end - start > _fraction(config.max_window_seconds):
                raise ValueError("candidate context exceeds max_window_seconds")
            seed = WindowSeed(entity, candidate.row_id, candidate.rank, candidate.score, timestamp)
            segments.append((shot.shot_id, start, end, seed))
    segments.sort(key=lambda item: (item[1], item[2], item[0], item[3].entity_id, item[3].row_id))
    merged = []
    for shot_id, start, end, seed in segments:
        _deadline(deadline)
        if merged and merged[-1][0] == shot_id and start <= merged[-1][2]:
            previous = merged[-1]
            combined_end = max(previous[2], end)
            if combined_end - previous[1] > _fraction(config.max_window_seconds):
                raise ValueError("merged candidate context exceeds max_window_seconds")
            merged[-1] = (shot_id, previous[1], combined_end, (*previous[3], seed))
        else:
            merged.append((shot_id, start, end, (seed,)))
            if len(merged) > config.max_windows:
                raise ValueError("candidate windows exceed max_windows")
    windows = []
    total = 0
    for shot_id, start, end, seeds in merged:
        _deadline(deadline)
        fps = _fraction(config.sample_fps)
        count_fraction = (end - start) * fps
        count = (count_fraction.numerator + count_fraction.denominator - 1) // (
            count_fraction.denominator
        )
        if total + count > config.max_total_frames:
            raise ValueError("dense sampling exceeds max_total_frames")
        points = set()
        for index in range(count):
            _deadline(deadline)
            points.add(start + Fraction(index, 1) / fps)
        points.update(_fraction(seed.timestamp_seconds) for seed in seeds)
        total += len(points)
        if total > config.max_total_frames:
            raise ValueError("seed-preserving sampling exceeds max_total_frames")
        times = tuple(float(point) for point in sorted(points))
        if len(set(times)) != len(points):
            raise ValueError("sampling timeline loses precision when represented as seconds")
        requests = tuple(FrameRequest(video.video_id, point, "candidate_window") for point in times)
        seeds = tuple(sorted(seeds, key=lambda item: (
            item.timestamp_seconds, item.entity_id, item.rank, item.row_id
        )))
        identity = {
            "query": query.query_id, "video": video.video_id, "source": video.source_sha256,
            "embedding": provenance[:2] + (first.model_mode.value,),
            "stream_index": next(iter(streams)) if streams else None,
            "shot": shot_id, "start": float(start), "end": float(end),
            "config": config.to_dict(), "seeds": [seed.to_dict() for seed in seeds],
            "requests": [asdict(request) for request in requests],
        }
        digest = hashlib.sha256(json.dumps(
            identity, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")).hexdigest()
        windows.append(CandidateWindow(
            "window_" + digest, shot_id, float(start), float(end), seeds, requests
        ))
    _deadline(deadline)
    plan = CandidateWindowPlan(
        query.query_id, video.video_id, video.source_sha256, *provenance, config, tuple(windows),
        stream_index=next(iter(streams)) if streams else None,
    )
    _deadline(deadline)
    return plan
