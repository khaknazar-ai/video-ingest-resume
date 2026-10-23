"""Resource-bounded retrieval, grounding, tracking and observed evidence orchestration."""

from __future__ import annotations

import hashlib
import json
import math
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np

from video_intelligence.core.types import Shot, VideoRecord
from video_intelligence.detection.base import DetectionBatch, DetectionInput
from video_intelligence.detection.execution import GroundingExecutionResult, execute_grounding
from video_intelligence.detection.grounding import (
    GroundingBudget,
    GroundingCandidate,
    GroundingStatus,
)
from video_intelligence.detection.prompts import PromptRegistry
from video_intelligence.embeddings.base import EmbeddingFingerprint, EmbeddingIndex, EmbeddingQuery
from video_intelligence.index.snapshot import SnapshotStore
from video_intelligence.models.contracts import ModelMode
from video_intelligence.retrieval.candidate_search import CandidateSet, search_candidates
from video_intelligence.retrieval.observation_evidence import evidence_from_observations
from video_intelligence.retrieval.predicates import spatial_relation_holds, within_time
from video_intelligence.retrieval.schemas import (
    OBSERVATION_SUPPORT_POLICY,
    RELATION_POLICY_VERSION,
    EvidenceBundle,
    ParseStatus,
    ProcessingStatus,
    QueryResult,
    QuerySpec,
    SupportStatus,
    requires_semantic_verification,
)
from video_intelligence.retrieval.windows import (
    CandidateWindowPlan,
    WindowConfig,
    plan_candidate_windows,
)
from video_intelligence.tracking import (
    TrackerConfig,
    TrackingFrame,
    TrackingRequest,
    TrackingResult,
    track_window,
)
from video_intelligence.video.decoder import PyAVVideoDecoder


@dataclass(frozen=True, slots=True)
class ObservationPipelineConfig:
    windows: WindowConfig = field(default_factory=WindowConfig)
    tracking: TrackerConfig = field(default_factory=TrackerConfig)
    top_k_per_entity: int = 3
    max_detections_per_frame: int = 32
    max_evidence_items: int = 2048
    max_input_pixels: int = 2_073_600
    max_source_bytes: int = 512 * 1024 * 1024
    max_runtime_seconds: float = 30.0
    seek_tolerance_seconds: float = 0.2
    cleanup_timeout_seconds: float = 0.1
    near_distance_fraction: float = 0.15

    def __post_init__(self) -> None:
        if (
            not isinstance(self.windows, WindowConfig)
            or not isinstance(self.tracking, TrackerConfig)
        ):
            raise TypeError("windows and tracking must be typed configurations")
        for name, cap in (
            ("top_k_per_entity", 32), ("max_detections_per_frame", 256),
            ("max_evidence_items", 8192), ("max_input_pixels", 8_294_400),
            ("max_source_bytes", 2 * 1024**3),
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= cap:
                raise ValueError(f"{name} must be an integer in [1,{cap}]")
        for name in (
            "max_runtime_seconds", "seek_tolerance_seconds", "cleanup_timeout_seconds",
            "near_distance_fraction",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) or not 0 < value <= 60
            ):
                raise ValueError(f"{name} must be finite, positive and at most 60")
        if self.windows.max_total_frames > 256 or self.windows.max_windows > 16:
            raise ValueError("pipeline permits at most 256 requests and 16 windows")
        if self.max_detections_per_frame > self.tracking.max_detections_per_frame:
            raise ValueError("grounding output cap exceeds tracking admission cap")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ObservationPipelineResult:
    query_result: QueryResult
    plan: CandidateWindowPlan | None
    execution: GroundingExecutionResult | None
    tracking: tuple[TrackingResult, ...]
    candidates: Mapping[str, CandidateSet]
    duration_seconds: float
    error_message: str | None = None
    decode_aliases: int = 0
    segmentation: tuple[dict, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "video_intelligence.observation_pipeline.v1",
            "quality_status": "unmeasured", "query_result": self.query_result.to_dict(),
            "plan": self.plan.to_dict() if self.plan is not None else None,
            "execution": self.execution.to_dict() if self.execution is not None else None,
            "tracking": [result.to_dict() for result in self.tracking],
            "candidates": {key: value.to_dict() for key, value in self.candidates.items()},
            "measurements": {
                "wall_seconds": self.duration_seconds, "peak_ram_bytes": None,
                "peak_vram_bytes": None, "scope": "this orchestration call; device in model_spec",
            },
            "error_message": self.error_message, "decode_aliases": self.decode_aliases,
            **({"segmentation": list(self.segmentation)} if self.segmentation else {}),
        }


class _PipelineError(RuntimeError):
    def __init__(self, code: str, message: str):
        self.code = code
        super().__init__(message)


def _check(deadline: float, cancel_check: Callable[[], bool] | None) -> None:
    if cancel_check is not None and cancel_check():
        raise _PipelineError("cancelled", "observation pipeline was cancelled")
    if time.perf_counter() >= deadline:
        raise _PipelineError("deadline", "observation pipeline cooperative deadline exceeded")


def _hash_source(path: Path, deadline: float, cancel_check) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            _check(deadline, cancel_check)
            digest.update(chunk)
    return digest.hexdigest()


def _selected_evidence(query: QuerySpec, items, near_distance_fraction: float):
    if query.relation is None:
        return tuple(item for item in items if item.label == query.entity)
    grouped = {}
    for item in items:
        grouped.setdefault(item.frame_ref.identity_key, []).append(item)
    selected = []
    for values in grouped.values():
        primary = sorted((item for item in values if item.role == "primary"),
                         key=lambda item: item.evidence_id)
        secondary = sorted((item for item in values if item.role == "secondary"),
                           key=lambda item: item.evidence_id)
        pair = next(((first, second) for first in primary for second in secondary if
                     spatial_relation_holds(
                         query.relation, first.bbox_xyxy, second.bbox_xyxy,
                         width=first.frame_ref.width, height=first.frame_ref.height,
                         near_distance_fraction=near_distance_fraction,
                     )), None)
        if pair is not None:
            selected.extend(pair)
    return tuple(selected)


class ObservationPipeline:
    """Connect a verified snapshot to observations; the caller supplies typed text vectors.

    Decoder injection is an explicit contract-test/adapter seam. The default path
    uses real PyAV decoding and verifies source bytes before and after execution.
    There is no fixture or missing-model fallback.
    """

    def __init__(self, snapshots: SnapshotStore, *, decoder_factory=PyAVVideoDecoder):
        if not isinstance(snapshots, SnapshotStore):
            raise TypeError("snapshots must be a SnapshotStore")
        if not callable(decoder_factory):
            raise TypeError("decoder_factory must be callable")
        self.snapshots = snapshots
        self.decoder_factory = decoder_factory

    def run(
        self, *, video: VideoRecord, query: QuerySpec, fingerprint: EmbeddingFingerprint,
        query_embeddings: Mapping[str, EmbeddingQuery], shots: Sequence[Shot],
        prompts: PromptRegistry, grounding: GroundingCandidate, run_id: str,
        config: ObservationPipelineConfig | None = None,
        cancel_check: Callable[[], bool] | None = None, snapshot_id: str,
        sam2_backend=None, segmentation_root=None,
    ) -> ObservationPipelineResult:
        if not isinstance(video, VideoRecord) or not isinstance(query, QuerySpec):
            raise TypeError("video and query must be typed contracts")
        if (
            not isinstance(run_id, str) or not run_id or run_id.strip() != run_id
            or any(character.isspace() for character in run_id)
        ):
            raise ValueError("run_id must be a non-empty identifier")
        config = ObservationPipelineConfig() if config is None else config
        if not isinstance(config, ObservationPipelineConfig):
            raise TypeError("config must be an ObservationPipelineConfig")
        if not isinstance(fingerprint, EmbeddingFingerprint):
            raise TypeError("fingerprint must be an EmbeddingFingerprint")
        if not isinstance(grounding, GroundingCandidate) or not isinstance(prompts, PromptRegistry):
            raise TypeError("grounding and prompts must use the existing typed interfaces")
        started = time.perf_counter()
        deadline = started + config.max_runtime_seconds
        mode = "inference" if (
            fingerprint.model_mode is ModelMode.INFERENCE
            and grounding.spec.mode is ModelMode.INFERENCE
        ) else "mock"
        plan, execution = None, None
        tracking, candidates = [], {}
        segmentation = []
        query_config = {
            "version": "observation_pipeline.v2", "source_sha256": video.source_sha256,
            "support_policy": OBSERVATION_SUPPORT_POLICY,
            "snapshot_id": snapshot_id, "budget": config.to_dict(),
            "max_evidence_items": config.max_evidence_items,
            "near_distance_fraction": config.near_distance_fraction,
            "grounding_model": grounding.spec.to_dict(),
            "embedding_fingerprint": fingerprint.to_dict(),
        }
        aliases = 0
        if sam2_backend is not None:
            from video_intelligence.segmentation.sam2 import SAM2Backend

            if not isinstance(sam2_backend, SAM2Backend) or segmentation_root is None:
                raise ValueError("SAM2 requires a typed backend and explicit artifact root")
            query_config["sam2_backend"] = sam2_backend.to_dict()

        def finish(result, message=None):
            result.validate_video(video)
            return ObservationPipelineResult(
                result.with_revision(query_config), plan, execution, tuple(tracking),
                dict(candidates), time.perf_counter() - started, message, aliases,
                tuple(segmentation),
            )

        def failed(code, message):
            return finish(QueryResult(
                query, video.video_id, run_id, ProcessingStatus.FAILED,
                SupportStatus.INSUFFICIENT_EVIDENCE, mode, error_code=code,
            ), message)

        stage = "admission"
        try:
            _check(deadline, cancel_check)
            if query.parse_status is not ParseStatus.SUPPORTED:
                return finish(QueryResult(
                    query, video.video_id, run_id, ProcessingStatus.COMPLETE,
                    SupportStatus.UNSUPPORTED_QUERY, mode,
                ))
            allowed_modes = {ModelMode.MOCK, ModelMode.INFERENCE}
            if (
                fingerprint.model_mode not in allowed_modes
                or grounding.spec.mode not in allowed_modes
            ):
                raise ValueError("pipeline accepts explicit mock or inference stages only")
            entities = (query.entity,) if query.secondary_entity is None else (
                query.entity, query.secondary_entity,
            )
            if not isinstance(query_embeddings, Mapping) or set(query_embeddings) != set(entities):
                raise ValueError("text embeddings must cover every query entity exactly")
            prompt = prompts.plan(query.raw_text, entities)
            query_config["prompt"] = prompt.to_dict()
            if video.source_sha256 is None or not video.source_path.is_file():
                raise ValueError("pipeline requires existing registered source bytes")
            if video.source_path.stat().st_size > config.max_source_bytes:
                raise ValueError("source exceeds configured byte cap")
            width, height = video.metadata.width, video.metadata.height
            if width is None or height is None or width * height > config.max_input_pixels:
                raise ValueError("source dimensions are unknown or exceed input pixel cap")
            if _hash_source(video.source_path, deadline, cancel_check) != video.source_sha256:
                raise _PipelineError("source_changed", "registered source bytes changed")
            stage = "retrieval"
            loaded = self.snapshots.load(snapshot_id)
            index = EmbeddingIndex.from_loaded_snapshot(loaded, fingerprint)
            for entity in entities:
                _check(deadline, cancel_check)
                candidates[entity] = search_candidates(
                    index, query, query_embeddings[entity], top_k=config.top_k_per_entity,
                    video_id=video.video_id,
                )
            stage = "planning"
            plan = plan_candidate_windows(
                video, query, candidates, shots=shots, config=config.windows,
            )
            query_config["window_plan"] = plan.to_dict()
            if not plan.windows:
                return finish(QueryResult(
                    query, video.video_id, run_id, ProcessingStatus.COMPLETE,
                    SupportStatus.INSUFFICIENT_EVIDENCE, mode, error_code="no_candidates",
                ))
            for window in plan.windows:
                if len(window.requests) > config.tracking.max_frames or (
                    Fraction(str(window.end_seconds)) - Fraction(str(window.start_seconds))
                    > Fraction(str(config.tracking.max_window_seconds))
                ):
                    raise ValueError("planned window exceeds tracking admission budget")
            assignments, rgb_hashes, decoded_semantics = {}, {}, {}
            decode_error = None
            stage = "decode_grounding"
            _check(deadline, cancel_check)
            with self.decoder_factory(video) as decoder:
                def frames():
                    nonlocal aliases, decode_error
                    previous_time = None
                    actual_stream = getattr(plan, "stream_index", None)
                    try:
                        for window in plan.windows:
                            for request in window.requests:
                                _check(deadline, cancel_check)
                                decoded = decoder.read_request(
                                    replace(request,
                                            tolerance_seconds=config.seek_tolerance_seconds),
                                    selection_policy="at_or_after",
                                )
                                if decoded is None:
                                    raise _PipelineError("decode_failed", "frame unavailable")
                                ref, image = decoded.frame_ref, decoded.image
                                if any(not issue.recoverable for issue in
                                       getattr(decoder, "issues", ())):
                                    raise _PipelineError("decode_failed", "fatal decoder issue")
                                if (
                                    ref.video_id != video.video_id
                                    or ref.source_sha256 != video.source_sha256
                                    or (ref.width, ref.height) != (width, height)
                                ):
                                    raise _PipelineError("decode_identity", "source mismatch")
                                if actual_stream is None:
                                    actual_stream = ref.stream_index
                                if ref.stream_index != actual_stream:
                                    raise _PipelineError("decode_stream", "decoded stream mismatch")
                                if not (
                                    window.start_seconds <= ref.timestamp_seconds
                                    < window.end_seconds
                                ):
                                    raise _PipelineError("decode_bounds", "frame outside window")
                                if not within_time(ref.timestamp_seconds, query.time_start_seconds,
                                                   query.time_end_seconds):
                                    raise _PipelineError("decode_bounds", "frame outside query")
                                actual = decoded.actual_timestamp_seconds
                                requested = request.requested_timestamp_seconds
                                if (
                                    isinstance(actual, bool)
                                    or not isinstance(actual, (int, float))
                                    or not math.isfinite(actual)
                                ):
                                    raise _PipelineError("decode_time", "invalid actual timestamp")
                                rounding = 8 * max(
                                    math.ulp(actual), math.ulp(ref.timestamp_seconds),
                                    math.ulp(requested), math.ulp(config.seek_tolerance_seconds),
                                )
                                if abs(actual - ref.timestamp_seconds) > rounding:
                                    raise _PipelineError("decode_time", "actual timestamp mismatch")
                                delta = ref.timestamp_seconds - requested
                                if delta < -rounding or delta > (
                                    config.seek_tolerance_seconds + rounding
                                ):
                                    raise _PipelineError(
                                        "decode_seek", "at_or_after seek violation",
                                    )
                                if (
                                    not isinstance(image, np.ndarray) or image.dtype != np.uint8
                                    or image.shape != (height, width, 3)
                                ):
                                    raise _PipelineError("decode_rgb", "expected canonical RGB8")
                                key = ref.identity_key
                                rgb_hash = hashlib.sha256(image.tobytes()).hexdigest()
                                semantic = (ref.timestamp_seconds, ref.width, ref.height, rgb_hash)
                                if key in decoded_semantics:
                                    if (decoded_semantics[key] != semantic or
                                            assignments[key] != window.window_id):
                                        raise _PipelineError("decode_alias", "contradictory alias")
                                    aliases += 1
                                    continue
                                if (previous_time is not None and
                                        ref.timestamp_seconds <= previous_time):
                                    raise _PipelineError("decode_order", "timestamp order")
                                previous_time = ref.timestamp_seconds
                                decoded_semantics[key] = semantic
                                rgb_hashes[key] = rgb_hash
                                assignments[key] = window.window_id
                                yield DetectionInput(ref, image)
                    except Exception as error:
                        decode_error = error
                        raise

                stream = frames()
                try:
                    execution = execute_grounding(
                        grounding, prompt=prompt, frames=stream,
                        budget=GroundingBudget(plan.total_frames, config.max_detections_per_frame),
                        cancel_check=cancel_check, deadline=deadline,
                        cleanup_timeout_seconds=config.cleanup_timeout_seconds,
                    )
                finally:
                    stream.close()
            if execution.status is not GroundingStatus.SUCCEEDED:
                code = (
                    getattr(decode_error, "code", None) or execution.error_code
                    or "grounding_failed"
                )
                return failed(code, execution.error_message or str(decode_error))
            if _hash_source(video.source_path, deadline, cancel_check) != video.source_sha256:
                raise _PipelineError("source_changed", "video changed during inference")
            stage = "tracking_evidence"
            items = []
            roles = {query.entity: "entity"} if query.relation is None else {
                query.entity: "primary", query.secondary_entity: "secondary",
            }
            for window in plan.windows:
                _check(deadline, cancel_check)
                batches = tuple(batch for batch in execution.batches if
                                assignments[batch.frame_ref.identity_key] == window.window_id)
                if not batches:
                    raise _PipelineError("empty_window", "planned window had no decoded frames")
                tracked = track_window(TrackingRequest(
                    run_id, window.window_id, tuple(TrackingFrame(DetectionBatch(
                        batch.frame_ref, batch.model_id, batch.model_mode, batch.source_stage,
                        tuple(detection.observation for detection in batch.detections),
                    ), window.shot_id) for batch in batches), config.tracking,
                ))
                tracking.append(tracked)
                masks = None
                if sam2_backend is not None:
                    details, masks = sam2_backend.segment_window(
                        video=video, batches=batches, tracking=tracked, rgb_hashes=rgb_hashes,
                        root=segmentation_root, checkpoint=lambda: _check(deadline, cancel_check),
                        timeout=deadline - time.perf_counter(),
                    )
                    segmentation.append(details)
                    if sum(len(json.dumps(x, separators=(",", ":"))) for x in segmentation) > (
                        sam2_backend.config.max_mask_bytes * 2 + 128 * 1024
                    ):
                        raise ValueError("SAM2 aggregate output budget exceeded")
                artifact_hash = hashlib.sha256(json.dumps(
                    {"grounding": [batch.to_dict() for batch in batches],
                     "tracking": tracked.to_dict()},
                    sort_keys=True, separators=(",", ":"), allow_nan=False,
                ).encode("utf-8")).hexdigest()
                observed = evidence_from_observations(
                    batches, video=video, run_id=run_id, snapshot_id=snapshot_id,
                    artifact_id=f"observations_{artifact_hash}", mode=mode,
                    entity_roles=roles, tracking=tracked,
                )
                items.extend(replace(item, provenance={
                    **item.provenance,
                    "frame_rgb_sha256": rgb_hashes[item.frame_ref.identity_key],
                    "frame_rgb_encoding": "rgb8_hwc",
                    **({"segmentation": masks[item.evidence_id]} if masks is not None else {}),
                }) for item in observed if masks is None or item.evidence_id in masks)
                if len(items) > config.max_evidence_items:
                    raise ValueError("observations exceed evidence budget")
            selected = _selected_evidence(query, items, config.near_distance_fraction)
            unverified = requires_semantic_verification(mode, query_config, selected)
            bundle = None
            if selected:
                timestamps = [item.frame_ref.timestamp_seconds for item in selected]
                bundle = EvidenceBundle(
                    query.query_id, run_id, video.video_id, video.source_sha256, snapshot_id,
                    selected, min(timestamps), max(timestamps), False,
                    ("candidate_window_scope", "unobserved_between_frames")
                    + (("semantic_verification_required",) if unverified else ()),
                    {"version": RELATION_POLICY_VERSION,
                     "near_distance_fraction": config.near_distance_fraction}
                    if query.relation is not None else None,
                )
            _check(deadline, cancel_check)
            supported = bool(selected) and query.temporal is None and not unverified
            answer = {"evidence_count": len(selected)} if supported else None
            if supported and query.relation is not None:
                answer["relation"] = query.relation
            return finish(QueryResult(
                query, video.video_id, run_id, ProcessingStatus.COMPLETE,
                SupportStatus.SUPPORTED if supported else SupportStatus.INSUFFICIENT_EVIDENCE,
                mode, answer=answer, evidence=bundle,
                error_code=None if supported else (
                    "temporal_coverage_required" if query.temporal is not None
                    else "semantic_verification_required" if selected and unverified
                    else "no_grounded_support"
                ),
            ))
        except Exception as error:
            return failed(getattr(error, "code", None) or f"{stage}_failed", str(error))
