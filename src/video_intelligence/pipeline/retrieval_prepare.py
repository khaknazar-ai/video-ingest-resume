"""Learned retrieval preparation, then the existing durable observation worker."""

from __future__ import annotations

import hashlib
import math
import time
from dataclasses import asdict

import numpy as np

from video_intelligence.core.types import Shot, TimestampQuality
from video_intelligence.detection.backends import backend_digest
from video_intelligence.embeddings.base import EmbeddingIndex, EmbeddingRow
from video_intelligence.embeddings.local_siglip2 import LocalRetrievalBackend, RetrievalConfig
from video_intelligence.embeddings.siglip2 import SigLIP2Input
from video_intelligence.index.build import EmbeddingBuildBatch, build_embedding_index
from video_intelligence.models.contracts import ModelAdapterError, ModelErrorCode, ModelSpec
from video_intelligence.pipeline.index_registry import IndexRegistry
from video_intelligence.pipeline.observation_jobs import (
    ObservationJobRequest,
    _identifier,
    _pipeline_config,
    _prompts,
)
from video_intelligence.retrieval.query_parser import parse_query
from video_intelligence.retrieval.schemas import ParseStatus
from video_intelligence.storage.evidence_frames import EvidenceFrameConfig
from video_intelligence.video.decoder import PyAVVideoDecoder
from video_intelligence.video.manifest import sha256_file
from video_intelligence.video.sampler import uniform_budget_requests

PLAN_SCHEMA = "video_intelligence.retrieval_observation.v1"
LINEAGE_SCHEMA = "video_intelligence.learned_retrieval_provenance.v1"


def normalize_plan(payload):
    required = {"schema_version", "video_id", "query_text", "run_id", "shots", "prompts",
                "grounding_backend"}
    if (not isinstance(payload, dict) or not required <= set(payload)
            or set(payload) - required - {"config", "evidence_config", "grounding_backend_sha256",
                                         "sam2_backend_sha256", "vlm_backend_sha256"}
            or payload["schema_version"] != PLAN_SCHEMA):
        raise ValueError("invalid retrieval-observation plan schema/fields")
    try:
        for name in ("video_id", "run_id", "grounding_backend"):
            _identifier(payload[name], name)
        if payload["grounding_backend"] == "fixture_pixel":
            raise ValueError("learned retrieval cannot use fixture grounding")
        registry = _prompts(payload["prompts"])
        query = parse_query(payload["query_text"], registry=registry)
        if query.parse_status is not ParseStatus.SUPPORTED:
            raise ValueError("retrieval preparation requires the supported query grammar")
        entities = (query.entity,) if query.secondary_entity is None else (
            query.entity, query.secondary_entity,
        )
        plan = registry.plan(payload["query_text"], entities)
        if plan.negative_phrases:
            raise ValueError("learned retrieval positive-only policy rejects negative phrases")
        _pipeline_config(payload.get("config", {}))
        EvidenceFrameConfig(**payload.get("evidence_config", {}))
        if not isinstance(payload["shots"], list) or not 1 <= len(payload["shots"]) <= 4096:
            raise ValueError("shots must contain 1..4096 explicit boundaries")
        shots = sorted((Shot(**row) for row in payload["shots"]), key=lambda x: x.start_seconds)
        if any(shot.video_id != payload["video_id"] for shot in shots):
            raise ValueError("retrieval shot video_id differs")
        # An explicit first phrase per role avoids silent averaging of prompt variants.
        texts = {}
        for entity in entities:
            phrase = next(phrase for phrase, owner in zip(
                plan.positive_phrases, plan.phrase_entity_ids, strict=True,
            ) if owner == entity)
            texts[entity] = f"this is a photo of {phrase}.".lower()
        if any(len(text) > 1024 for text in texts.values()):
            raise ValueError("retrieval text exceeds character budget")
        return {**payload, "shots": [asdict(shot) for shot in shots]}, texts
    except (TypeError, KeyError, AttributeError, OverflowError, StopIteration) as error:
        raise ValueError("malformed retrieval observation plan") from error


def validate_request_lineage(payload):
    """Validate detached portable bindings without importing any ML runtime."""
    value = payload["retrieval_provenance"]
    if (not isinstance(value, dict) or set(value) != {"schema_version", "backend_sha256",
            "backend", "snapshot_id", "source_sha256", "shots_sha256", "text_inputs",
            "query_embeddings_sha256"} or value["schema_version"] != LINEAGE_SCHEMA):
        raise ValueError("invalid learned retrieval provenance fields")
    if (backend_digest(value["backend"]) != value["backend_sha256"]
            or value["snapshot_id"] != payload["snapshot_id"]
            or backend_digest(payload["shots"]) != value["shots_sha256"]
            or backend_digest(payload["query_embeddings"]) != value["query_embeddings_sha256"]):
        raise ValueError("learned retrieval config/snapshot/shot/vector binding differs")
    backend = LocalRetrievalBackend(ModelSpec(**value["backend"]["model_spec"]),
                                    RetrievalConfig(**value["backend"]["config"]))
    if (backend.to_dict() != value["backend"]
            or backend.fingerprint.to_dict() != payload["embedding_fingerprint"]):
        raise ValueError("learned retrieval fingerprint/backend policy binding differs")
    plan = {key: payload[key] for key in (
        "video_id", "query_text", "run_id", "shots", "prompts", "grounding_backend",
    )}
    _, texts = normalize_plan({**plan, "schema_version": PLAN_SCHEMA})
    if value["text_inputs"] != texts:
        raise ValueError("learned retrieval text differs from query prompt roles")


def verify_snapshot_lineage(request, loaded, video):
    payload = request.to_dict()
    if "retrieval_provenance" not in payload:
        return
    validate_request_lineage(payload)
    value = payload["retrieval_provenance"]
    expected = loaded.descriptor.attributes.get("learned_retrieval", {})
    if (expected.get("backend") != value["backend"]
            or expected.get("source_sha256") != video.source_sha256
            or value["source_sha256"] != video.source_sha256
            or expected.get("shots_sha256") != value["shots_sha256"]
            or loaded.descriptor.attributes.get("embedding_fingerprint")
            != payload["embedding_fingerprint"]):
        raise ValueError("learned retrieval snapshot lineage differs from admitted request")


def prepare_retrieval_request(
    store, snapshots, *, plan, backend: LocalRetrievalBackend, snapshot_id=None,
    checkpoint=None, decoder_factory=PyAVVideoDecoder,
):
    """Decode and infer synchronously; release encoder before publishing/grounding.

    Only frame metadata and bounded vectors survive preparation, not RGB arrays.
    A supplied snapshot is reused only with identical source/model/sampling/shots.
    """
    plan, texts = normalize_plan(plan)
    if not isinstance(backend, LocalRetrievalBackend):
        raise TypeError("backend must be a LocalRetrievalBackend")
    config, fingerprint = backend.config, backend.fingerprint
    started = time.perf_counter()

    def check():
        if checkpoint is not None:
            checkpoint()
        if time.perf_counter() - started > config.max_runtime_seconds:
            raise TimeoutError("retrieval preparation runtime budget exceeded")

    check()
    video = store.get_video(plan["video_id"])
    if video is None:
        raise ValueError("retrieval video is not registered")
    duration, width, height = (video.metadata.duration_seconds, video.metadata.width,
                               video.metadata.height)
    if (duration is None or not math.isfinite(duration) or duration <= 0
            or width is None or height is None or width * height > config.max_input_pixels
            or video.source_path.stat().st_size > config.max_source_bytes):
        raise ValueError("retrieval source metadata/size exceeds admission budget")
    shots = tuple(Shot(**row) for row in plan["shots"])
    if (shots[0].start_seconds != 0 or shots[-1].end_seconds != duration
            or any(first.end_seconds != second.start_seconds
                   for first, second in zip(shots, shots[1:], strict=False))):
        raise ValueError("explicit retrieval shots must cover the source without gaps/overlap")
    if not video.source_sha256 or sha256_file(video.source_path) != video.source_sha256:
        raise ValueError("retrieval source bytes differ from registration")
    lineage = {"backend": backend.to_dict(), "source_sha256": video.source_sha256,
               "shots_sha256": backend_digest(plan["shots"]),
               "sampling_policy": "uniform_budget_nearest_actual_pts.v1",
               "quality_status": "unmeasured"}
    reused = None
    if snapshot_id is not None:
        reused = snapshots.load(snapshot_id)
        row = store.get_index_snapshot(snapshot_id)
        if row is None or any(row[key] != getattr(reused.descriptor, key) for key in (
            "matrix_sha256", "metadata_sha256", "rows", "dimension", "size_bytes",
        )) or row["relative_path"] != f"{snapshots.root.name}/{snapshot_id}":
            raise ValueError("reused retrieval snapshot is not registered with matching content")
        recorded = reused.descriptor.attributes.get("learned_retrieval", {})
        if any(recorded.get(key) != value for key, value in lineage.items()):
            raise ValueError("reused retrieval snapshot has different source/backend/shots")
        EmbeddingIndex.from_loaded_snapshot(reused, fingerprint)
    adapter = backend.adapter()
    batches, seen, aliases = [], {}, 0
    vectors = {}
    try:
        check()
        adapter.load(backend.spec)
        check()
        if reused is None:
            with decoder_factory(video, max_validation_frames=10_000,
                                 max_validation_seconds=5.0) as decoder:
                for request in uniform_budget_requests(video, frame_budget=config.frame_budget):
                    check()
                    decoded = decoder.read_request(request, selection_policy="nearest")
                    if decoded is None or any(not issue.recoverable for issue in decoder.issues):
                        raise ValueError("retrieval frame cannot be decoded")
                    ref, rgb = decoded.frame_ref, decoded.image
                    if (ref.video_id != video.video_id or ref.source_sha256 != video.source_sha256
                            or (ref.width, ref.height) != (width, height)
                            or ref.timestamp_quality is not TimestampQuality.EXACT_PTS
                            or ref.pts is None or ref.time_base is None or ref.stream_index is None
                            or ref.timestamp_origin != "first_displayed_pts"
                            or decoded.requested_timestamp_seconds
                            != request.requested_timestamp_seconds
                            or not 0 <= ref.timestamp_seconds < duration
                            or decoded.actual_timestamp_seconds != ref.timestamp_seconds
                            or abs(ref.timestamp_seconds - request.requested_timestamp_seconds)
                            > config.seek_tolerance_seconds
                            or not isinstance(rgb, np.ndarray) or rgb.dtype != np.uint8
                            or rgb.shape != (height, width, 3)):
                        raise ValueError("retrieval decoder source/time/RGB identity differs")
                    rgb_sha = hashlib.sha256(rgb.tobytes()).hexdigest()
                    semantic = (ref.timestamp_seconds, ref.width, ref.height, rgb_sha)
                    if ref.identity_key in seen:
                        if seen[ref.identity_key] != semantic:
                            raise ValueError("retrieval decoder returned contradictory PTS aliases")
                        aliases += 1
                        continue
                    seen[ref.identity_key] = semantic
                    shot, = (s for s in shots if s.start_seconds <= ref.timestamp_seconds
                             < s.end_seconds)
                    result = adapter.infer([SigLIP2Input("image", rgb)])
                    check()
                    identity = hashlib.sha256(repr(ref.identity_key).encode("utf-8")).hexdigest()
                    row = EmbeddingRow(f"siglip2_{identity}", {
                        "video_id": video.video_id, "source_sha256": video.source_sha256,
                        "timestamp_seconds": ref.timestamp_seconds, "frame_index": ref.frame_index,
                        "stream_index": ref.stream_index, "pts": ref.pts,
                        "time_base": ref.time_base,
                        "timestamp_quality": ref.timestamp_quality.value,
                        "timestamp_origin": ref.timestamp_origin, "width": width, "height": height,
                        "shot_id": shot.shot_id, "frame_rgb_sha256": rgb_sha,
                        "embedding_model_spec": result.provenance,
                        "retrieval_backend_sha256": backend.identity_sha256,
                        "actual_dtype": adapter.actual_dtype,
                    })
                    batches.append(EmbeddingBuildBatch(result, (row,)))
                    # Do not retain decoded RGB after each inference.
                    del decoded, rgb, result
        for entity, text in texts.items():
            check()
            result = adapter.infer([SigLIP2Input("text", text)])
            # Reuse the same model-result provenance/dimension validator for text.
            checked, _ = build_embedding_index(fingerprint, [EmbeddingBuildBatch(
                result, (EmbeddingRow(f"text_{entity}"),),
            )])
            vectors[entity] = list(result.outputs[0])
            if checked.fingerprint != fingerprint:
                raise ValueError("text embedding space changed")
        check()
        if sha256_file(video.source_path) != video.source_sha256:
            raise ValueError("retrieval source bytes changed during inference")
    finally:
        if not adapter.close(timeout_seconds=0.1):
            raise ModelAdapterError(ModelErrorCode.MODEL_UNAVAILABLE,
                                    "retrieval cleanup unfinished; grounding is blocked",
                                    model_id=backend.spec.model_id)
    check()
    if sha256_file(video.source_path) != video.source_sha256:
        raise ValueError("retrieval source bytes changed during cleanup")
    if reused is None:
        index, manifest = build_embedding_index(fingerprint, batches)
        attributes = {**index.snapshot_attributes(), "embedding_build": manifest.to_dict(),
                      "learned_retrieval": {**lineage, "decode_aliases": aliases,
                                            "decoded_unique_frames": len(batches)}}
        active = IndexRegistry(store, snapshots).publish_and_activate(
            f"siglip2:{backend_digest(video.video_id)}:{backend.identity_sha256}",
            index.as_exact_index(), attributes=attributes, checkpoint=check,
        )
        snapshot_id = active.descriptor.snapshot_id
    payload = {**plan, "schema_version": "video_intelligence.observation_job.v1",
               "snapshot_id": snapshot_id, "embedding_fingerprint": fingerprint.to_dict(),
               "query_embeddings": vectors, "retrieval_provenance": {
                   "schema_version": LINEAGE_SCHEMA, "backend": backend.to_dict(),
                   "backend_sha256": backend.identity_sha256, "snapshot_id": snapshot_id,
                   "source_sha256": video.source_sha256, "shots_sha256": lineage["shots_sha256"],
                   "text_inputs": texts, "query_embeddings_sha256": backend_digest(vectors),
               }}
    return ObservationJobRequest.from_dict(payload)
