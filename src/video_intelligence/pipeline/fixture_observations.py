"""Explicit synthetic-contract preparation; RGB mock features, never annotations."""

from __future__ import annotations

import hashlib
import math
from dataclasses import asdict

import numpy as np

from video_intelligence.core.types import Shot, TimestampQuality
from video_intelligence.data.dataset import require_synthetic_contract
from video_intelligence.detection.prompts import PhraseSpec, PromptRegistry
from video_intelligence.embeddings import EmbeddingFingerprint, EmbeddingIndex, EmbeddingRow
from video_intelligence.index.snapshot import SnapshotStore
from video_intelligence.models.contracts import ModelMode
from video_intelligence.pipeline.index_registry import IndexRegistry
from video_intelligence.pipeline.observation_jobs import (
    ObservationJobRequest,
    fixture_rgb_features,
)
from video_intelligence.pipeline.observations import ObservationPipelineConfig
from video_intelligence.retrieval.query_parser import parse_query
from video_intelligence.retrieval.schemas import ParseStatus
from video_intelligence.storage.sqlite import SQLiteStore
from video_intelligence.video.decoder import PyAVVideoDecoder
from video_intelligence.video.manifest import sha256_file
from video_intelligence.video.sampler import uniform_budget_requests

FIXTURE_OBSERVATION_INDEX_PREFIX = "fixture_observation_rgb:"
FIXTURE_EMBEDDING_MODEL_ID = "fixture.rgb_mask_features_v1"
FIXTURE_EMBEDDING_REVISION = "rgb_mask_counts_v1"
_ENTITIES = ("red_square", "blue_circle", "green_object")


def prepare_fixture_observation_request(
    store: SQLiteStore,
    snapshots: SnapshotStore,
    *,
    video_id: str,
    query_text: str,
    run_id: str = "run_cli",
    frame_budget: int = 16,
) -> ObservationJobRequest:
    """Prepare an actual mock snapshot, without creating or running a job.

    Decode at most 64 uniform nearest-frame requests, one RGB image at a time.
    Coalesce actual PTS duplicates and explicitly exclude featureless frames from
    the cosine index. The full-video sampling boundary is a fixture heuristic,
    not a detected or annotated shot. Verify source and sidecar before publication.
    """
    if not isinstance(store, SQLiteStore) or not isinstance(snapshots, SnapshotStore):
        raise TypeError("store and snapshots must be typed local stores")
    if (isinstance(frame_budget, bool) or not isinstance(frame_budget, int)
            or not 1 <= frame_budget <= 64):
        raise ValueError("frame_budget must be an integer in [1,64]")
    if (not isinstance(run_id, str) or not run_id or run_id.strip() != run_id
            or any(character.isspace() for character in run_id)):
        raise ValueError("run_id must be a non-empty identifier")
    query = parse_query(query_text)
    if query.parse_status is not ParseStatus.SUPPORTED:
        raise ValueError("fixture observation query must use the supported explicit grammar")
    entities = (query.entity,) if query.secondary_entity is None else (
        query.entity, query.secondary_entity,
    )
    if any(entity not in _ENTITIES for entity in entities):
        raise ValueError("fixture observation query contains an unsupported entity")
    video = store.get_video(video_id)
    if video is None:
        raise ValueError("fixture observation video is not registered")
    config = ObservationPipelineConfig()
    width, height = video.metadata.width, video.metadata.height
    duration = video.metadata.duration_seconds
    if duration is None or not math.isfinite(duration) or duration <= 0:
        raise ValueError("fixture observation source requires a known positive duration")
    if width is None or height is None or width * height > config.max_input_pixels:
        raise ValueError("fixture observation dimensions are unknown or exceed pixel cap")
    if video.source_path.stat().st_size > config.max_source_bytes:
        raise ValueError("fixture observation source exceeds byte cap")
    initial_manifest = require_synthetic_contract(video.source_path)
    if video.source_sha256 is None or sha256_file(video.source_path) != video.source_sha256:
        raise ValueError("source hash no longer matches the registered fixture video")
    fingerprint = EmbeddingFingerprint(
        FIXTURE_EMBEDDING_MODEL_ID, FIXTURE_EMBEDDING_REVISION, FIXTURE_EMBEDDING_REVISION,
        "image", 3, model_mode=ModelMode.MOCK,
    )
    shot = Shot(f"fixture_full_video_{video.video_id}", video.video_id, 0, duration)
    prompts = PromptRegistry([
        PhraseSpec("red_square", ("a red square",)),
        PhraseSpec("blue_circle", ("a blue circle",)),
        PhraseSpec("green_object", ("a green object",)),
    ], prompt_version="fixture_rgb_prompt_v1")
    payload = {
        "schema_version": "video_intelligence.observation_job.v1",
        "video_id": video.video_id, "snapshot_id": "0" * 64,
        "query_text": query_text, "run_id": run_id,
        "embedding_fingerprint": fingerprint.to_dict(),
        "query_embeddings": {
            entity: [float(index == _ENTITIES.index(entity)) for index in range(3)]
            for entity in entities
        },
        "shots": [asdict(shot)], "prompts": prompts.to_dict(),
        "grounding_backend": "fixture_pixel",
    }
    # This valid placeholder is never enqueued or persisted as a job/result.
    prepared = ObservationJobRequest.from_dict(payload)
    requests = uniform_budget_requests(video, frame_budget=frame_budget)
    rows, vectors, seen = [], [], {}
    decoded_count = featureless_count = aliases = 0
    with PyAVVideoDecoder(
        video, max_validation_frames=10_000, max_validation_seconds=5.0,
    ) as decoder:
        for request in requests:
            decoded = decoder.read_request(request, selection_policy="nearest")
            if decoded is None or any(not issue.recoverable for issue in decoder.issues):
                raise ValueError("fixture observation frame could not be decoded")
            ref, image = decoded.frame_ref, decoded.image
            if (
                ref.video_id != video.video_id or ref.source_sha256 != video.source_sha256
                or (ref.width, ref.height) != (width, height)
                or ref.timestamp_quality is not TimestampQuality.EXACT_PTS
                or not 0 <= ref.timestamp_seconds < duration
                or not math.isfinite(decoded.actual_timestamp_seconds)
                or decoded.actual_timestamp_seconds != ref.timestamp_seconds
            ):
                raise ValueError("fixture decoder returned inconsistent source identity")
            if (not isinstance(image, np.ndarray) or image.dtype != np.uint8
                    or image.shape != (height, width, 3)):
                raise ValueError("fixture observation decoder must return canonical RGB8")
            key = ref.identity_key
            rgb_hash = hashlib.sha256(image.tobytes()).hexdigest()
            semantic = (ref.timestamp_seconds, ref.width, ref.height, rgb_hash)
            if key in seen:
                if seen[key] != semantic:
                    raise ValueError("fixture decoder returned contradictory PTS aliases")
                aliases += 1
                continue
            seen[key] = semantic
            decoded_count += 1
            features = np.asarray(fixture_rgb_features(image), dtype=np.float32)
            if features.shape != (3,) or not np.isfinite(features).all() or (features < 0).any():
                raise ValueError("fixture RGB feature policy returned invalid counts")
            if not features.any():
                featureless_count += 1
                continue
            identity = hashlib.sha256(repr(key).encode("utf-8")).hexdigest()
            rows.append(EmbeddingRow(f"fixture_rgb_{identity}", {
                "video_id": video.video_id, "source_sha256": video.source_sha256,
                "timestamp_seconds": ref.timestamp_seconds, "stream_index": ref.stream_index,
                "frame_index": ref.frame_index, "pts": ref.pts, "time_base": ref.time_base,
                "timestamp_quality": ref.timestamp_quality.value,
                "timestamp_origin": ref.timestamp_origin, "width": ref.width, "height": ref.height,
                "shot_id": shot.shot_id, "frame_rgb_sha256": rgb_hash,
                "rgb_feature_counts": features.tolist(),
            }))
            vectors.append(features)
    if not rows:
        raise ValueError("fixture observation sampling found no nonzero RGB features")
    if (sha256_file(video.source_path) != video.source_sha256
            or require_synthetic_contract(video.source_path) != initial_manifest):
        raise ValueError("fixture source bytes or synthetic contract changed during preparation")
    index = EmbeddingIndex(fingerprint, vectors, rows)
    attributes = {
        **index.snapshot_attributes(),
        "fixture_preparation": {
            "mode": "mock", "quality_status": "unmeasured",
            "feature_policy": FIXTURE_EMBEDDING_REVISION,
            "source_sha256": video.source_sha256,
            "dataset_kind": "synthetic", "dataset_purpose": "contract",
            "frame_budget": frame_budget, "requested_frames": len(requests),
            "decoded_unique_frames": decoded_count, "decode_aliases": aliases,
            "featureless_frames_excluded": featureless_count,
            "sampling_policy": "uniform_budget_nearest_actual_pts",
            "boundary_policy": "fixture_full_video_sampling_heuristic",
        },
    }
    video_identity = hashlib.sha256(video.video_id.encode("utf-8")).hexdigest()
    active = IndexRegistry(store, snapshots).publish_and_activate(
        f"{FIXTURE_OBSERVATION_INDEX_PREFIX}{video_identity}", index.as_exact_index(),
        attributes=attributes,
    )
    return ObservationJobRequest.from_dict({
        **prepared.to_dict(), "snapshot_id": active.descriptor.snapshot_id,
    })
