"""Bounded educational tracking over measured P7 detection batches."""

from __future__ import annotations

import hashlib
import json
import math
import sys
import time
from dataclasses import asdict, dataclass, field

from video_intelligence.detection.base import DetectionObservation
from video_intelligence.tracking.assignment import _check_deadline, maximum_iou_assignment
from video_intelligence.tracking.base import (
    TrackingRequest,
    TrackingResult,
    TrackObservation,
    TrackSnapshot,
    TrackState,
    TrackTransition,
    gap_exceeds,
)

ALGORITHM_ID = "educational_iou_hungarian.v1"


@dataclass
class _Track:
    track_id: str
    shot_id: str
    scene_epoch: int
    label: str
    state: TrackState
    observations: list[TrackObservation] = field(default_factory=list)
    consecutive_hits: int = 1
    termination_reason: str | None = None

    @property
    def last_seen_seconds(self) -> float:
        return self.observations[-1].frame_ref.timestamp_seconds


def _iou(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    width = max(0.0, min(left[2], right[2]) - max(left[0], right[0]))
    height = max(0.0, min(left[3], right[3]) - max(left[1], right[1]))
    if not width or not height:
        return 0.0
    left_width, left_height = left[2] - left[0], left[3] - left[1]
    right_width, right_height = right[2] - right[0], right[3] - right[1]
    intersection = width * height
    area_left, area_right = left_width * left_height, right_width * right_height
    union = area_left + area_right - intersection
    if (
        math.isfinite(union)
        and min(area_left, area_right, intersection, union) >= sys.float_info.min
    ):
        return intersection / union
    # Finite source coordinates can still under/overflow when their extents are
    # multiplied. Scale areas in log space only for those exceptional geometries.
    log_left = math.log(left_width) + math.log(left_height)
    log_right = math.log(right_width) + math.log(right_height)
    log_intersection = math.log(width) + math.log(height)
    scale = max(log_left, log_right)
    scaled_intersection = math.exp(log_intersection - scale)
    scaled_union = (
        math.exp(log_left - scale) + math.exp(log_right - scale) - scaled_intersection
    )
    return min(1.0, max(0.0, scaled_intersection / scaled_union))


def _namespace(request: TrackingRequest) -> str:
    first, last = request.frames[0].batch.frame_ref, request.frames[-1].batch.frame_ref
    material = {
        "algorithm": ALGORITHM_ID, "run": request.run_id, "window": request.window_id,
        "source": (first.video_id, first.source_sha256, first.stream_index),
        "geometry": (first.width, first.height), "config": asdict(request.config),
        "span": (first.timestamp_seconds, last.timestamp_seconds),
        "shots": [(frame.shot_id, frame.scene_cut) for frame in request.frames],
        "detector": (
            request.frames[0].batch.model_id, request.frames[0].batch.model_mode.value,
            request.frames[0].batch.source_stage,
        ),
    }
    encoded = json.dumps(material, sort_keys=True, allow_nan=False, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:32]


def track_window(request: TrackingRequest) -> TrackingResult:
    """Associate one validated window; returned states apply at its final frame.

    There is no retained tracker state and no cross-window identity. Source
    detections are the only emitted observations. Confirmed tracks may be lost
    and recover; tentative tracks must collect consecutive observed hits. An
    observed-frame gap exceeding policy closes all associations before matching.
    The CPU deadline is cooperative; failure returns no partial success result.
    """

    if not isinstance(request, TrackingRequest):
        raise TypeError("request must be a TrackingRequest")
    started = time.perf_counter()
    # Revalidate nested mutable provenance and detach it again at execution entry.
    request = TrackingRequest(
        run_id=request.run_id, window_id=request.window_id, frames=request.frames,
        config=request.config,
    )
    config = request.config
    deadline = started + config.max_runtime_seconds
    _check_deadline(deadline)
    namespace = _namespace(request)
    tracks: list[_Track] = []
    transitions: list[TrackTransition] = []
    epoch = 0
    previous_frame = None

    def change(track: _Track, state: TrackState, frame_ref, reason: str) -> None:
        if track.state is state:
            return
        transitions.append(TrackTransition(
            track_id=track.track_id, frame_ref=frame_ref, previous_state=track.state,
            state=state, reason=reason,
        ))
        track.state = state
        if state is TrackState.TERMINATED:
            track.termination_reason = reason

    for frame in request.frames:
        _check_deadline(deadline)
        ref = frame.batch.frame_ref
        active = [track for track in tracks if track.state is not TrackState.TERMINATED]
        boundary = previous_frame is not None and (
            frame.scene_cut or frame.shot_id != previous_frame.shot_id
        )
        sparse_gap = previous_frame is not None and gap_exceeds(
            previous_frame.batch.frame_ref, ref, config.max_observation_gap_seconds,
        )
        if boundary or sparse_gap:
            reason = "scene_boundary" if boundary else "observation_gap"
            for track in active:
                change(track, TrackState.TERMINATED, ref, reason)
            epoch += 1
        else:
            for track in active:
                if gap_exceeds(
                    track.observations[-1].frame_ref, ref, config.max_age_seconds,
                ):
                    change(track, TrackState.TERMINATED, ref, "max_age")
        active = [track for track in tracks if track.state is not TrackState.TERMINATED]
        if len(active) > config.max_active_tracks:
            raise ValueError("tracking exceeds active track cap")
        detections = sorted(frame.batch.detections, key=lambda item: item.detection_id)

        def match_score(track: _Track, detection: DetectionObservation) -> float | None:
            if track.label != detection.label:
                return None
            score = _iou(track.observations[-1].detection.bbox_xyxy, detection.bbox_xyxy)
            return score if score >= config.iou_threshold else None

        scores = []
        for track in active:
            _check_deadline(deadline)
            scores.append([match_score(track, detection) for detection in detections])
        pairs = maximum_iou_assignment(scores, deadline=deadline)
        matched_tracks = {row for row, _ in pairs}
        matched_detections = {column for _, column in pairs}
        for row, column in pairs:
            track = active[row]
            track.observations.append(TrackObservation(detections[column]))
            track.consecutive_hits += 1
            if track.state is TrackState.LOST or track.consecutive_hits >= config.min_hits:
                change(track, TrackState.CONFIRMED, ref, "observed_match")
        for index, track in enumerate(active):
            if index in matched_tracks:
                continue
            track.consecutive_hits = 0
            state = (
                TrackState.TERMINATED if track.state is TrackState.TENTATIVE else TrackState.LOST
            )
            change(track, state, ref, "missed_detection")
        live_count = sum(track.state is not TrackState.TERMINATED for track in tracks)
        unmatched = [
            item for index, item in enumerate(detections) if index not in matched_detections
        ]
        if live_count + len(unmatched) > config.max_active_tracks:
            raise ValueError("tracking exceeds active track cap")
        if len(tracks) + len(unmatched) > config.max_tracks:
            raise ValueError("tracking exceeds total track cap")
        for detection in unmatched:
            state = TrackState.CONFIRMED if config.min_hits == 1 else TrackState.TENTATIVE
            track = _Track(
                track_id=f"trk_{namespace}_{epoch}_{len(tracks):06d}", shot_id=frame.shot_id,
                scene_epoch=epoch, label=detection.label, state=state,
                observations=[TrackObservation(detection)],
            )
            tracks.append(track)
            transitions.append(TrackTransition(
                track_id=track.track_id, frame_ref=ref, previous_state=None,
                state=state, reason="creation",
            ))
        previous_frame = frame
    _check_deadline(deadline)
    result = TrackingResult(
        run_id=request.run_id, window_id=request.window_id, algorithm_id=ALGORITHM_ID,
        config=config, frames=request.frames, processed_frames=len(request.frames),
        tracks=tuple(TrackSnapshot(
            track_id=track.track_id, shot_id=track.shot_id, scene_epoch=track.scene_epoch,
            label=track.label, state=track.state, observations=tuple(track.observations),
            termination_reason=track.termination_reason,
        ) for track in tracks), transitions=tuple(transitions),
    )
    _check_deadline(deadline)
    return result
