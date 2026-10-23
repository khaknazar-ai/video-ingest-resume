"""Convert measured grounding observations into source-bound typed evidence."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import replace
from fractions import Fraction
from typing import Any

from video_intelligence.core.types import FrameRef, TimestampQuality, VideoRecord
from video_intelligence.detection.grounding import GroundingBatch
from video_intelligence.models.contracts import ModelMode, ModelSpec
from video_intelligence.retrieval.schemas import EvidenceItem
from video_intelligence.tracking.base import TrackingResult, TrackSnapshot

MAX_OBSERVATION_BATCHES = 256
MAX_OBSERVATIONS_PER_BATCH = 256


def _identifier(value: object, name: str) -> str:
    if (
        not isinstance(value, str) or not value or value.strip() != value
        or any(character.isspace() for character in value)
    ):
        raise ValueError(f"{name} must be a non-empty identifier")
    return value


def _sha256(value: object, name: str) -> str:
    if (
        not isinstance(value, str) or len(value) != 64 or value != value.lower()
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 hex digest")
    return value


def _json_copy(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("observation provenance must be a JSON mapping")
    try:
        copied = deepcopy(dict(value))
        json.dumps(copied, allow_nan=False)
    except (TypeError, ValueError, RecursionError) as error:
        raise ValueError("observation provenance must contain finite JSON values") from error
    return copied


def _copy_batch(batch: GroundingBatch) -> GroundingBatch:
    if not isinstance(batch, GroundingBatch):
        raise TypeError("batches must contain only GroundingBatch values")
    if len(batch.detections) > MAX_OBSERVATIONS_PER_BATCH:
        raise ValueError("grounding batch exceeds its observation cap")
    return replace(
        batch, frame_ref=replace(batch.frame_ref), prompt=replace(batch.prompt),
        detections=tuple(replace(
            item, observation=replace(
                item.observation, frame_ref=replace(item.observation.frame_ref),
                provenance=_json_copy(item.observation.provenance),
            ),
        ) for item in batch.detections),
    )


def _model_spec(batch: GroundingBatch, provenance: Mapping[str, Any]) -> dict[str, Any] | None:
    if "model_spec" not in provenance:
        return None
    declared = provenance["model_spec"]
    if not isinstance(declared, dict):
        raise ValueError("grounding model_spec must contain a complete ModelSpec")
    try:
        spec = ModelSpec(**declared)
    except (TypeError, ValueError) as error:
        raise ValueError("grounding model_spec must contain a complete ModelSpec") from error
    if (
        spec.to_dict() != declared or spec.model_id != batch.model_id
        or spec.mode is not batch.model_mode or spec.role != "detection"
    ):
        raise ValueError("grounding model_spec contradicts the measured observation")
    return deepcopy(declared)


def _validate_frame(frame: FrameRef, video: VideoRecord) -> Fraction:
    if frame.timestamp_quality is not TimestampQuality.EXACT_PTS:
        raise ValueError("observation evidence requires actual exact_pts references")
    if frame.backend == "legacy":
        raise ValueError("legacy frame references cannot certify exact observation identity")
    if frame.pts is None or frame.time_base is None:
        raise ValueError("observation evidence requires PTS and time_base")
    try:
        time_base = Fraction(frame.time_base)
    except (ValueError, ZeroDivisionError) as error:
        raise ValueError("observation time_base must be a positive rational") from error
    if time_base <= 0:
        raise ValueError("observation time_base must be a positive rational")
    if frame.video_id != video.video_id or frame.source_sha256 != video.source_sha256:
        raise ValueError("grounding observation belongs to another source")
    metadata = video.metadata
    if (
        metadata.width is not None and frame.width != metadata.width
    ) or (metadata.height is not None and frame.height != metadata.height):
        raise ValueError("grounding observation dimensions differ from registered video")
    if metadata.duration_seconds is not None and (
        frame.timestamp_seconds >= metadata.duration_seconds
    ):
        raise ValueError("grounding observation is outside registered video duration")
    return time_base


def _track_bindings(
    batches: tuple[GroundingBatch, ...], tracking: TrackingResult | None, run_id: str,
) -> tuple[TrackingResult | None, dict[str, TrackSnapshot]]:
    if tracking is None:
        return None, {}
    if not isinstance(tracking, TrackingResult):
        raise TypeError("tracking must be a TrackingResult or null")
    tracking = replace(tracking)
    if tracking.run_id != run_id:
        raise ValueError("tracking belongs to another run")
    if len(tracking.frames) != len(batches):
        raise ValueError("tracking must cover every grounding frame, including negatives")
    for batch, frame in zip(batches, tracking.frames, strict=True):
        supplied = {item.observation.detection_id: item.observation for item in batch.detections}
        tracked = {item.detection_id: item for item in frame.batch.detections}
        if (
            frame.batch.frame_ref != batch.frame_ref or tracked != supplied
            or frame.batch.model_id != batch.model_id
            or frame.batch.model_mode is not batch.model_mode
            or frame.batch.source_stage != batch.source_stage
        ):
            raise ValueError("tracking inputs must exactly match grounding frames and observations")
    bindings = {
        observation.detection.detection_id: track
        for track in tracking.tracks for observation in track.observations
    }
    return tracking, bindings


def evidence_from_observations(
    batches: Sequence[GroundingBatch],
    *,
    video: VideoRecord,
    run_id: str,
    snapshot_id: str,
    artifact_id: str,
    mode: str,
    entity_roles: Mapping[str, str],
    tracking: TrackingResult | None = None,
) -> tuple[EvidenceItem, ...]:
    """Bind observed boxes to a caller-verified source and retrieval snapshot.

    The caller verifies source bytes and the real retrieval snapshot before use.
    ``artifact_id`` identifies the caller's in-memory observation content, not an
    exported image or persisted file. This converter performs no I/O and makes no
    coverage or answer-support decision. Empty detections yield an empty tuple.
    Tracking state describes the final window snapshot, not state at each image.
    """

    if not isinstance(video, VideoRecord):
        raise TypeError("video must be a VideoRecord")
    video = replace(video, metadata=replace(video.metadata))
    _sha256(video.source_sha256, "registered source_sha256")
    run_id = _identifier(run_id, "run_id")
    snapshot_id = _sha256(snapshot_id, "snapshot_id")
    artifact_id = _identifier(artifact_id, "artifact_id")
    if not isinstance(mode, str) or mode not in {ModelMode.MOCK.value, ModelMode.INFERENCE.value}:
        raise ValueError("observation evidence mode must be mock or inference")
    if not isinstance(entity_roles, Mapping) or not entity_roles:
        raise ValueError("entity_roles must be a non-empty mapping")
    roles = {_identifier(entity, "entity"): role for entity, role in entity_roles.items()}
    if any(
        not isinstance(role, str) or role not in {"entity", "primary", "secondary"}
        for role in roles.values()
    ):
        raise ValueError("entity_roles must use entity, primary or secondary roles")
    if isinstance(batches, (str, bytes)) or not isinstance(batches, Sequence):
        raise ValueError("batches must be a bounded sequence of GroundingBatch values")
    if len(batches) > MAX_OBSERVATION_BATCHES:
        raise ValueError("grounding batches exceed the evidence conversion cap")
    values = tuple(_copy_batch(batch) for batch in batches)
    seen_frames: set[tuple[object, ...]] = set()
    seen_detections: set[str] = set()
    previous: FrameRef | None = None
    for batch in values:
        frame = batch.frame_ref
        time_base = _validate_frame(frame, video)
        if batch.model_mode not in {ModelMode.MOCK, ModelMode.INFERENCE}:
            raise ValueError("grounding must be a measured mock or inference execution")
        if mode == ModelMode.INFERENCE.value and batch.model_mode is ModelMode.MOCK:
            raise ValueError("mock grounding cannot become inference evidence")
        if set(batch.prompt.entity_ids) != set(roles):
            raise ValueError("entity roles must exactly match the grounding prompt")
        if values and (
            batch.prompt != values[0].prompt or batch.model_id != values[0].model_id
            or batch.model_mode is not values[0].model_mode
            or batch.source_stage != values[0].source_stage
            or frame.stream_index != values[0].frame_ref.stream_index
            or (frame.width, frame.height, frame.timestamp_origin)
            != (values[0].frame_ref.width, values[0].frame_ref.height,
                values[0].frame_ref.timestamp_origin)
            or time_base != Fraction(values[0].frame_ref.time_base)
        ):
            raise ValueError("grounding frame, prompt and model provenance must agree")
        if frame.identity_key in seen_frames:
            raise ValueError("grounding frames contain duplicate canonical identity")
        seen_frames.add(frame.identity_key)
        if previous is not None and (
            frame.timestamp_seconds <= previous.timestamp_seconds or frame.pts <= previous.pts
        ):
            raise ValueError("grounding frame timestamps and PTS must strictly increase")
        try:
            elapsed = float((frame.pts - values[0].frame_ref.pts) * time_base)
        except OverflowError as error:
            raise ValueError("grounding PTS delta exceeds finite range") from error
        if not math.isclose(
            frame.timestamp_seconds - values[0].frame_ref.timestamp_seconds,
            elapsed, abs_tol=1e-6, rel_tol=1e-9,
        ):
            raise ValueError("grounding PTS and timestamp deltas disagree")
        previous = frame
        for item in batch.detections:
            _model_spec(batch, item.observation.provenance)
            identifier = item.observation.detection_id
            if identifier in seen_detections:
                raise ValueError("detection_ids must be unique across grounding batches")
            seen_detections.add(identifier)
    tracking, bindings = _track_bindings(values, tracking, run_id)
    items = []
    for batch in values:
        for grounded in batch.detections:
            observation = grounded.observation
            track = bindings.get(observation.detection_id)
            tracking_provenance = None if tracking is None else {
                "track_id": track.track_id, "state": track.state.value,
                "state_scope": "window_final_snapshot", "shot_id": track.shot_id,
                "scene_epoch": track.scene_epoch, "window_id": tracking.window_id,
                "run_id": tracking.run_id, "algorithm_id": tracking.algorithm_id,
                "config": tracking.config.to_dict(),
                "termination_reason": track.termination_reason,
                "identity_scope": "within_window_association", "quality_status": "unmeasured",
            }
            items.append(EvidenceItem(
                evidence_id=observation.detection_id, frame_ref=observation.frame_ref,
                label=grounded.entity_id, bbox_xyxy=observation.bbox_xyxy,
                score=observation.score, source_sha256=video.source_sha256,
                snapshot_id=snapshot_id, artifact_id=artifact_id,
                source_stage=observation.source_stage, role=roles[grounded.entity_id],
                provenance={
                    "mode": mode, "run_id": run_id, "source_sha256": video.source_sha256,
                    "snapshot_id": snapshot_id, "snapshot_kind": "retrieval_index",
                    "artifact_kind": "in_memory_grounding_observations",
                    "artifact_persisted": False, "image_persisted": False,
                    "grounding": {
                        "detection_id": observation.detection_id,
                        "model_id": observation.model_id,
                        "model_mode": observation.model_mode.value,
                        "source_stage": observation.source_stage,
                        "entity_id": grounded.entity_id, "matched_phrase": grounded.matched_phrase,
                        "phrase_index": grounded.phrase_index,
                        "prompt_version": grounded.prompt_version, "prompt": batch.prompt.to_dict(),
                        "model_spec": _model_spec(batch, observation.provenance),
                        "observation_provenance": _json_copy(observation.provenance),
                    },
                    "tracking": tracking_provenance,
                },
            ))
    return tuple(items)
