"""Explicit leased observation jobs and verified, process-independent result replay."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np

from video_intelligence.core.paths import ProjectPaths
from video_intelligence.core.types import Shot, VideoRecord
from video_intelligence.data.dataset import require_synthetic_contract
from video_intelligence.detection.base import DetectionObservation
from video_intelligence.detection.grounding import (
    GroundingBatch,
    GroundingCandidate,
    GroundingDetection,
)
from video_intelligence.detection.prompts import PhraseSpec, PromptRegistry
from video_intelligence.embeddings import EmbeddingFingerprint, EmbeddingQuery
from video_intelligence.index.snapshot import SnapshotStore
from video_intelligence.models.contracts import (
    ModelAdapter,
    ModelCapabilities,
    ModelMode,
    ModelSpec,
)
from video_intelligence.pipeline.jobs import Job, JobQueue
from video_intelligence.pipeline.observations import ObservationPipeline, ObservationPipelineConfig
from video_intelligence.pipeline.worker import (
    CommittedJobOutput,
    JobContext,
    JobHandlerFailure,
    JobLeaseLost,
)
from video_intelligence.retrieval.query_parser import parse_query
from video_intelligence.retrieval.schemas import ParseStatus, ProcessingStatus, QueryResult
from video_intelligence.retrieval.windows import WindowConfig
from video_intelligence.storage.evidence_frames import EvidenceFrameConfig, EvidenceFrameStore
from video_intelligence.storage.sqlite import SQLiteStore
from video_intelligence.tracking import TrackerConfig

SCHEMA_VERSION = "video_intelligence.observation_job.v1"
RESULT_SCHEMA_VERSION = "video_intelligence.observation_job_result.v1"
FIXTURE_ENTITIES = ("red_square", "blue_circle", "green_object")
MAX_REQUEST_BYTES = 1024 * 1024


def _identifier(value: object, name: str) -> str:
    if (
        not isinstance(value, str) or not value or len(value) > 256
        or value.strip() != value or any(character.isspace() for character in value)
    ):
        raise ValueError(f"{name} must be a bounded identifier")
    return value


def _sha256(value: object) -> str:
    if (
        not isinstance(value, str) or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError("expected a lowercase SHA-256")
    return value


def _canonical(payload: object) -> str:
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False)
    if len(text.encode("utf-8")) > MAX_REQUEST_BYTES:
        raise ValueError("observation request exceeds its 1 MiB payload cap")
    return text


def _fingerprint(value: dict) -> EmbeddingFingerprint:
    value = dict(value)
    declared = value.pop("fingerprint_id")
    fingerprint = EmbeddingFingerprint(**value)
    if fingerprint.modality.value != "image" or fingerprint.model_mode not in {
        ModelMode.MOCK, ModelMode.INFERENCE,
    }:
        raise ValueError("observation retrieval fingerprint must be image mock or inference")
    if declared != fingerprint.fingerprint_id:
        raise ValueError("embedding fingerprint_id differs from its declared fields")
    return fingerprint


def _prompts(value: dict) -> PromptRegistry:
    if not isinstance(value, dict) or set(value) != {"prompt_version", "specs"}:
        raise ValueError("prompts must contain prompt_version and specs")
    specs = value["specs"]
    if not isinstance(specs, dict) or not 1 <= len(specs) <= 64:
        raise ValueError("prompt registry requires 1..64 explicit entities")
    values = []
    for entity, phrases in specs.items():
        if not isinstance(phrases, dict) or set(phrases) != {
            "positive_phrases", "negative_phrases",
        }:
            raise ValueError("prompt specs must contain positive_phrases and negative_phrases")
        values.append(PhraseSpec(entity, **phrases))
    return PromptRegistry(values, prompt_version=value["prompt_version"])


def _pipeline_config(value: dict) -> ObservationPipelineConfig:
    if not isinstance(value, dict):
        raise ValueError("config must be a JSON object")
    value = dict(value)
    return ObservationPipelineConfig(
        windows=WindowConfig(**value.pop("windows", {})),
        tracking=TrackerConfig(**value.pop("tracking", {})), **value,
    )


def _normalize(value: dict) -> dict:
    required = {
        "schema_version", "video_id", "snapshot_id", "query_text", "run_id",
        "embedding_fingerprint", "query_embeddings", "shots", "prompts", "grounding_backend",
    }
    if not isinstance(value, dict) or not required <= set(value) or set(value) - required - {
        "config", "evidence_config", "grounding_backend_sha256", "retrieval_provenance",
        "sam2_backend_sha256", "vlm_backend_sha256",
    }:
        raise ValueError("observation job request has missing or unknown fields")
    if value["schema_version"] != SCHEMA_VERSION:
        raise ValueError("unsupported observation job request schema")
    for name in ("video_id", "run_id", "grounding_backend"):
        _identifier(value[name], name)
    _sha256(value["snapshot_id"])
    if "grounding_backend_sha256" in value:
        _sha256(value["grounding_backend_sha256"])
    if "sam2_backend_sha256" in value:
        _sha256(value["sam2_backend_sha256"])
        if value["grounding_backend"] == "fixture_pixel":
            raise ValueError("fixture jobs cannot claim a SAM2 backend")
    if "vlm_backend_sha256" in value:
        _sha256(value["vlm_backend_sha256"])
        if value["grounding_backend"] == "fixture_pixel":
            raise ValueError("fixture jobs cannot claim a VLM backend")
    text = value["query_text"]
    if not isinstance(text, str) or not text.strip() or len(text) > 16_384:
        raise ValueError("query_text must contain 1..16384 characters")
    fingerprint = _fingerprint(value["embedding_fingerprint"])
    registry = _prompts(value["prompts"])
    query = parse_query(text, query_namespace=f"observation:{value['video_id']}:{value['run_id']}",
                        registry=registry)
    entities = (() if query.parse_status is not ParseStatus.SUPPORTED else
                (query.entity,) if query.secondary_entity is None else
                (query.entity, query.secondary_entity))
    vectors = value["query_embeddings"]
    if not isinstance(vectors, dict) or set(vectors) != set(entities):
        raise ValueError("query_embeddings must match the parsed query roles exactly")
    canonical_vectors = {}
    for entity, vector in vectors.items():
        if not isinstance(vector, list) or any(
            isinstance(number, bool) or not isinstance(number, (int, float)) for number in vector
        ):
            raise ValueError("query embeddings must be finite numeric lists")
        EmbeddingQuery(replace(fingerprint, modality="text"), vector)
        canonical_vectors[entity] = [float(number) for number in vector]
    if entities:
        registry.plan(text, entities)
    raw_shots = value["shots"]
    if not isinstance(raw_shots, list) or not 1 <= len(raw_shots) <= 4096:
        raise ValueError("shots must contain 1..4096 explicit boundaries")
    shots = tuple(sorted((Shot(**row) for row in raw_shots), key=lambda shot: (
        shot.start_seconds, shot.end_seconds, shot.shot_id,
    )))
    if any(shot.video_id != value["video_id"] for shot in shots):
        raise ValueError("shot video_id must match the request")
    config = _pipeline_config(value.get("config", {}))
    evidence_config = EvidenceFrameConfig(**value.get("evidence_config", {}))
    if "retrieval_provenance" in value:
        from video_intelligence.pipeline.retrieval_prepare import validate_request_lineage

        validate_request_lineage({**value, "shots": [asdict(shot) for shot in shots],
                                  "query_embeddings": canonical_vectors})
    return {
        **value, "embedding_fingerprint": fingerprint.to_dict(),
        "query_embeddings": canonical_vectors, "shots": [asdict(shot) for shot in shots],
        "prompts": registry.to_dict(), "config": config.to_dict(),
        "evidence_config": evidence_config.to_dict(),
    }


@dataclass(frozen=True, slots=True)
class ObservationJobRequest:
    """Store canonical JSON privately; every exposed payload is freshly detached."""

    _payload_json: str

    def __post_init__(self) -> None:
        try:
            normalized = _normalize(json.loads(self._payload_json))
        except (KeyError, TypeError, AttributeError, OverflowError) as error:
            raise ValueError("observation job request has malformed field types") from error
        object.__setattr__(self, "_payload_json", _canonical(normalized))

    @classmethod
    def from_dict(cls, payload: dict) -> ObservationJobRequest:
        return cls(_canonical(payload))

    def to_dict(self) -> dict:
        return json.loads(self._payload_json)

    @property
    def config_sha256(self) -> str:
        return hashlib.sha256(self._payload_json.encode("utf-8")).hexdigest()

    @property
    def video_id(self) -> str:
        return self.to_dict()["video_id"]

    @property
    def snapshot_id(self) -> str:
        return self.to_dict()["snapshot_id"]

    @property
    def query_text(self) -> str:
        return self.to_dict()["query_text"]

    @property
    def run_id(self) -> str:
        return self.to_dict()["run_id"]

    @property
    def grounding_backend(self) -> str:
        return self.to_dict()["grounding_backend"]


def fixture_rgb_masks(image: np.ndarray) -> dict[str, np.ndarray]:
    """Shared explicit fixture policy; no annotation data or expected boxes."""
    if (not isinstance(image, np.ndarray) or image.dtype != np.uint8
            or image.ndim != 3 or image.shape[2] != 3):
        raise ValueError("fixture RGB policy requires canonical RGB8")
    red, green, blue = np.moveaxis(image.astype(np.int16), -1, 0)
    return {
        "red_square": (red > 120) & (red > 1.4 * green) & (red > 1.4 * blue),
        "blue_circle": (blue > 120) & (blue > 1.4 * red) & (blue > 1.15 * green),
        "green_object": (green > 120) & (green > 1.4 * red) & (green > 1.4 * blue),
    }


def fixture_rgb_features(image: np.ndarray) -> np.ndarray:
    masks = fixture_rgb_masks(image)
    return np.asarray([np.count_nonzero(masks[entity]) for entity in FIXTURE_ENTITIES],
                      dtype=np.float32)


class FixturePixelGrounder(ModelAdapter):
    """Synthetic contract adapter; RGB alone determines observed boxes."""

    def __init__(self) -> None:
        super().__init__(ModelCapabilities("fixture_pixel", "detection", True, ModelMode.MOCK))

    def _infer(self, inputs):
        if len(inputs) != 1:
            raise ValueError("fixture grounding requires batch one")
        item = inputs[0]
        ref = item.frame.frame_ref
        masks = fixture_rgb_masks(item.frame.image)
        detections = []
        for index, entity in enumerate(item.prompt.phrase_entity_ids):
            if entity not in masks:
                raise ValueError("fixture grounding does not support this entity")
            ys, xs = np.where(masks[entity])
            if not len(xs):
                continue
            phrase = item.prompt.positive_phrases[index]
            detection = DetectionObservation(
                f"fixture_pixel_{ref.stream_index}_{ref.pts}_{index}", ref, entity,
                (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1),
                0.8, "fixture_pixel", ModelMode.MOCK, "grounding.fixture_rgb_contract",
                phrase=phrase, provenance={
                    "algorithm": "rgb_mask_counts_v1",
                    "score_policy": "fixed_contract_score_not_calibrated",
                },
            )
            detections.append(GroundingDetection(
                detection, entity, phrase, index, item.prompt.prompt_version,
            ))
        return (GroundingBatch(ref, "fixture_pixel", ModelMode.MOCK,
                               "grounding.fixture_rgb_contract", item.prompt, tuple(detections)),)


BackendFactory = Callable[[ObservationJobRequest, VideoRecord], GroundingCandidate]


def _input_digest(video: VideoRecord, request: ObservationJobRequest) -> str:
    return hashlib.sha256(_canonical({
        "source_sha256": video.source_sha256, "snapshot_id": request.snapshot_id,
    }).encode("utf-8")).hexdigest()


def _fixture_admission(video: VideoRecord, request: ObservationJobRequest):
    if "grounding_backend_sha256" in request.to_dict():
        raise ValueError("fixture jobs cannot pin configured real backends")
    manifest = require_synthetic_contract(video.source_path)
    if video.source_sha256 not in manifest.source_sha256:
        raise ValueError("registered source hash differs from verified fixture sidecar")
    if _fingerprint(request.to_dict()["embedding_fingerprint"]).model_mode is not ModelMode.MOCK:
        raise ValueError("fixture observation retrieval must remain mock")
    if request.grounding_backend != "fixture_pixel":
        raise ValueError("fixture_observe requires the explicit fixture_pixel backend")
    query = parse_query(request.query_text, registry=_prompts(request.to_dict()["prompts"]))
    if any(entity not in FIXTURE_ENTITIES for entity in (query.entity, query.secondary_entity)
           if entity is not None):
        raise ValueError("fixture observation query uses unsupported entities")
    return manifest


def create_observation_job(
    queue: JobQueue, request: ObservationJobRequest, *, fixture: bool = False,
) -> Job:
    if not isinstance(queue, JobQueue) or not isinstance(request, ObservationJobRequest):
        raise TypeError("queue and request must be typed observation interfaces")
    if not isinstance(fixture, bool):
        raise ValueError("fixture must be an explicit boolean")
    video = queue.store.get_video(request.video_id)
    if video is None:
        raise ValueError("observation job requires a registered video")
    if fixture:
        if "grounding_backend_sha256" in request.to_dict():
            raise ValueError("fixture jobs cannot pin configured real backends")
        _fixture_admission(video, request)
    elif request.grounding_backend == "fixture_pixel":
        raise ValueError("fixture_pixel requires the explicit fixture_observe stage")
    return queue.create(
        stage="fixture_observe" if fixture else "observation_query",
        config_sha256=request.config_sha256, input_artifact_sha256=_input_digest(video, request),
        payload=request.to_dict(),
    )


def observation_job_handler(
    context: JobContext, *, snapshot_root: Path | None, evidence_root: Path | None,
    backend_factories: Mapping[str, BackendFactory] | None = None,
    sam2_backend=None,
    vlm_backend=None,
) -> CommittedJobOutput:
    """Run the public bridge, persist PNGs, then atomically admit the leased output."""
    context.ensure_lease()
    request = ObservationJobRequest.from_dict(context.job.payload)
    pin = request.to_dict().get("sam2_backend_sha256")
    if pin is not None and (sam2_backend is None or sam2_backend.identity_sha256 != pin):
        raise JobHandlerFailure("sam2_config_mismatch", "SAM2 backend is missing or changed")
    # An optional supplied SAM2 config never upgrades old unsegmented jobs silently.
    selected_sam2 = sam2_backend if pin is not None else None
    vlm_pin = request.to_dict().get("vlm_backend_sha256")
    if vlm_pin is not None and (vlm_backend is None or vlm_backend.identity_sha256 != vlm_pin):
        raise JobHandlerFailure("vlm_config_mismatch", "VLM backend is missing or changed")
    selected_vlm = vlm_backend if vlm_pin is not None else None
    if request.config_sha256 != context.job.config_sha256:
        raise JobHandlerFailure("request_hash_mismatch", "queued request is not canonical")
    store = context.queue.store
    video = store.get_video(request.video_id)
    if video is None or context.job.input_artifact_sha256 != _input_digest(video, request):
        raise JobHandlerFailure("source_binding", "registered observation input identity changed")
    if snapshot_root is None:
        raise JobHandlerFailure("snapshot_root_required", "snapshot root must be configured")
    snapshots = SnapshotStore(snapshot_root)
    loaded = snapshots.load(request.snapshot_id)
    pointer = store.get_index_snapshot(request.snapshot_id)
    descriptor = loaded.descriptor
    if pointer is None or any(pointer[name] != getattr(descriptor, name) for name in (
        "matrix_sha256", "metadata_sha256", "rows", "dimension", "size_bytes",
    )) or (
        pointer["relative_path"] != f"{snapshots.root.name}/{request.snapshot_id}"
        or pointer["kind"] != "index_snapshot"
        or pointer["content_sha256"] != request.snapshot_id
    ):
        raise JobHandlerFailure(
            "snapshot_binding", "snapshot is not registered with matching content",
        )
    fixture_manifest = None
    effective_backend = None
    if "retrieval_provenance" in request.to_dict():
        from video_intelligence.pipeline.retrieval_prepare import verify_snapshot_lineage

        verify_snapshot_lineage(request, loaded, video)
    if context.job.stage == "fixture_observe":
        fixture_manifest = _fixture_admission(video, request)
        candidate = GroundingCandidate(FixturePixelGrounder(), ModelSpec(
            "fixture_pixel", "detection", "synthetic_rgb_mask_contract", "rgb_mask_counts_v1",
            mode=ModelMode.MOCK, device="cpu", batch_size=1, allow_download=False,
        ))
    elif context.job.stage == "observation_query":
        if request.grounding_backend == "fixture_pixel":
            raise JobHandlerFailure(
                "fixture_stage_required", "generic observation has no fixture fallback",
            )
        factory = (backend_factories or {}).get(request.grounding_backend)
        if not callable(factory):
            raise JobHandlerFailure(
                "backend_not_registered", "grounding backend is not allowlisted",
            )
        pinned = request.to_dict().get("grounding_backend_sha256")
        configured = getattr(factory, "identity_sha256", None)
        if pinned is not None or configured is not None:
            if pinned != configured or not isinstance(pinned, str):
                raise JobHandlerFailure(
                    "backend_config_mismatch", "grounding backend config changed",
                )
            effective_backend = factory.to_dict()
        candidate = factory(request, video)
        if not isinstance(candidate, GroundingCandidate):
            raise JobHandlerFailure(
                "invalid_backend", "backend factory must return GroundingCandidate",
            )
    else:
        raise JobHandlerFailure("invalid_stage", "handler requires an observation job stage")
    payload = request.to_dict()
    registry = _prompts(payload["prompts"])
    query = parse_query(request.query_text,
                        query_namespace=f"observation:{request.video_id}:{request.run_id}",
                        registry=registry)
    fingerprint = _fingerprint(payload["embedding_fingerprint"])

    def checkpoint() -> None:
        context.check_cancelled()
        context.ensure_lease()

    def cancelled() -> bool:
        checkpoint()
        return False

    checkpoint()
    pipeline = ObservationPipeline(snapshots).run(
        video=video, query=query, fingerprint=fingerprint,
        query_embeddings={entity: EmbeddingQuery(replace(fingerprint, modality="text"), vector)
                          for entity, vector in payload["query_embeddings"].items()},
        shots=tuple(Shot(**row) for row in payload["shots"]), prompts=registry,
        grounding=candidate, run_id=request.run_id, config=_pipeline_config(payload["config"]),
        cancel_check=cancelled, snapshot_id=request.snapshot_id,
        sam2_backend=selected_sam2,
        segmentation_root=(evidence_root or ProjectPaths.from_environment().artifacts_root),
    )
    checkpoint()
    if effective_backend is not None:
        pipeline = replace(pipeline, query_result=pipeline.query_result.with_revision({
            **pipeline.query_result.query_config,
            "grounding_backend_sha256": request.to_dict()["grounding_backend_sha256"],
        }))
    if "retrieval_provenance" in request.to_dict():
        pipeline = replace(pipeline, query_result=pipeline.query_result.with_revision({
            **pipeline.query_result.query_config,
            "retrieval_provenance": request.to_dict()["retrieval_provenance"],
        }))
    if pipeline.query_result.processing_status is not ProcessingStatus.COMPLETE:
        raise JobHandlerFailure(pipeline.query_result.error_code or "observation_failed",
                                pipeline.error_message or "observation pipeline failed")
    artifact_root = (
        evidence_root if evidence_root is not None
        else ProjectPaths.from_environment().artifacts_root
    )
    frames = EvidenceFrameStore(artifact_root)
    evidence_config = EvidenceFrameConfig(**payload["evidence_config"])
    from video_intelligence.storage.evidence_masks import EvidenceMaskStore

    masked, mask_artifact = EvidenceMaskStore(artifact_root).persist(
        pipeline.query_result, video=video, config=evidence_config, checkpoint=checkpoint,
    )
    persisted = frames.persist(masked, video=video, checkpoint=checkpoint,
                               config=evidence_config)
    checked_artifact = frames.verify_result(persisted.query_result, checkpoint=checkpoint,
                                            config=evidence_config)
    if checked_artifact != persisted.artifact:
        raise JobHandlerFailure("artifact_binding", "persisted evidence readback differs")
    checkpoint()
    artifacts = tuple(artifact.to_dict() for artifact in (mask_artifact, persisted.artifact)
                      if artifact is not None)
    if sum(artifact["size_bytes"] for artifact in artifacts) > evidence_config.max_total_bytes:
        raise JobHandlerFailure("artifact_budget", "combined RGB/mask evidence byte cap exceeded")
    pipeline_payload = pipeline.to_dict()
    pipeline_payload["query_result"] = persisted.query_result.to_dict()
    output = {
        "schema_version": RESULT_SCHEMA_VERSION, "job_id": context.job.job_id,
        "request_sha256": request.config_sha256, "query_result": persisted.query_result.to_dict(),
        "pipeline": pipeline_payload, "artifacts": list(artifacts),
        "fixture_dataset": None if fixture_manifest is None else fixture_manifest.to_dict(),
    }
    if effective_backend is not None:
        output["grounding_backend_config"] = effective_backend
    if selected_vlm is not None:
        from video_intelligence.reasoning.vlm import VLMError

        try:
            output["structured_answer"] = selected_vlm.analyze(
                persisted.query_result, root=artifact_root, checkpoint=checkpoint,
                evidence_config=evidence_config,
            )
        except VLMError as error:
            raise JobHandlerFailure(error.code, str(error)) from error
        except (ValueError, KeyError, TypeError) as error:
            raise JobHandlerFailure("vlm_invalid_output", str(error)) from error
        checkpoint()
    if context.job.lease_token is None:
        raise JobLeaseLost(context.job.job_id)
    digest = store.commit_observation_result(
        context.job.job_id, owner=context.worker_id, lease_token=context.job.lease_token,
        request_sha256=request.config_sha256, result=persisted.query_result,
        output=output, artifacts=artifacts,
    )
    return CommittedJobOutput(context.job.job_id, digest)


def load_observation_result(
    store: SQLiteStore, job_id: str, evidence_store: EvidenceFrameStore,
) -> dict | None:
    output = store.get_observation_result(job_id)
    if output is None:
        return None
    job = store.get_job(job_id)
    request = ObservationJobRequest.from_dict(json.loads(job["payload_json"]))
    if "sam2_backend_sha256" in request.to_dict():
        from video_intelligence.detection.backends import backend_digest

        if backend_digest(output["query_result"]["query_config"].get("sam2_backend")) != (
            request.to_dict()["sam2_backend_sha256"]
        ):
            raise ValueError("saved SAM2 backend binding differs from queued request")
    if ("retrieval_provenance" in request.to_dict()
            and output["query_result"]["query_config"].get("retrieval_provenance")
            != request.to_dict()["retrieval_provenance"]):
        raise ValueError("saved learned retrieval binding differs from queued request")
    if "grounding_backend_sha256" in request.to_dict():
        from video_intelligence.detection.backends import backend_digest

        pinned = request.to_dict()["grounding_backend_sha256"]
        if (backend_digest(output.get("grounding_backend_config")) != pinned or
                output["query_result"]["query_config"].get("grounding_backend_sha256") != pinned):
            raise ValueError("saved grounding backend config binding differs from queued request")
    video = store.get_video(request.video_id)
    result = QueryResult.from_dict(output["query_result"], video=video)
    artifact = evidence_store.verify_result(
        result, config=EvidenceFrameConfig(**request.to_dict()["evidence_config"]),
    )
    from video_intelligence.storage.evidence_masks import EvidenceMaskStore

    mask_artifact = EvidenceMaskStore(evidence_store.root).verify_result(
        result, config=EvidenceFrameConfig(**request.to_dict()["evidence_config"]),
    )
    expected = [value.to_dict() for value in (mask_artifact, artifact) if value is not None]
    if output.get("artifacts") != expected:
        raise ValueError("saved observation artifacts differ from verified query evidence")
    for metadata in expected:
        registered = store.get_artifact(metadata["artifact_id"])
        if registered is None or any(registered[name] != value for name, value in metadata.items()):
            raise ValueError("registered evidence artifact metadata differs from verified files")
    return output
