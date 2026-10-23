"""Evaluation entry points that require explicit real-dataset provenance."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

from video_intelligence.core.types import FrameRef
from video_intelligence.data.dataset import (
    REAL_QUALITY_SPLITS,
    DatasetManifest,
    require_real_quality,
)
from video_intelligence.data.intake import (
    REAL_ANNOTATION_SCHEMA,
    REAL_DETECTION_ANNOTATION_SCHEMA,
    load_verified_real_annotations,
    validate_real_quality_split_policy,
)
from video_intelligence.detection.base import DetectionBatch
from video_intelligence.evaluation.detection_metrics import (
    DetectionAnnotation,
    DetectionMetrics,
    GroundTruthBox,
    evaluate_detection,
)
from video_intelligence.evaluation.retrieval_metrics import (
    RetrievalAnnotation,
    RetrievalMetrics,
    evaluate_retrieval,
)
from video_intelligence.models.contracts import ModelMode
from video_intelligence.retrieval.candidate_search import CandidateSet


def _context(manifest, annotation_path, split_manifests, schema):
    require_real_quality(manifest)
    if annotation_path is None or split_manifests is None:
        raise ValueError("real-quality evaluation requires annotation_path and split_manifests")
    validate_real_quality_split_policy(split_manifests)
    if sum(item == manifest for item in split_manifests) != 1:
        raise ValueError("split registry must contain the evaluated manifest exactly once")
    if {item.split for item in split_manifests} != REAL_QUALITY_SPLITS:
        raise ValueError("quality campaign requires train, validation and test split manifests")
    if any(
        (item.dataset_id, item.annotation_schema, item.source_provenance)
        != (manifest.dataset_id, schema, manifest.source_provenance)
        for item in split_manifests
    ):
        raise ValueError("split registry dataset, schema and provenance must match")
    return load_verified_real_annotations(manifest, annotation_path)


def _unique_map(values, expected_type, key, name):
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise ValueError(f"{name} must be a sequence")
    if any(not isinstance(item, expected_type) for item in values):
        raise ValueError(f"{name} contains invalid values")
    result = {key(item): item for item in values}
    if len(result) != len(values):
        raise ValueError(f"{name} contains duplicate identities")
    return result


def _real_members(manifest: DatasetManifest) -> frozenset[tuple[str, str]]:
    return frozenset(manifest.source_members)


def _require_member(frame_ref: FrameRef, members: frozenset[tuple[str, str]]) -> None:
    if frame_ref.source_sha256 is None:
        raise ValueError("real-quality frames require source_sha256")
    if (frame_ref.video_id, frame_ref.source_sha256) not in members:
        raise ValueError("frame source is not a member of the real-quality dataset")


def _validate_detection_inputs(
    manifest: DatasetManifest,
    predictions: Sequence[DetectionBatch],
    annotations: Sequence[DetectionAnnotation],
) -> None:
    members = _real_members(manifest)
    for batch in predictions:
        _require_member(batch.frame_ref, members)
        if batch.model_mode is not ModelMode.INFERENCE:
            raise ValueError("only inference detections can enter real-quality evaluation")
    for annotation in annotations:
        _require_member(annotation.frame_ref, members)


def _validate_retrieval_inputs(
    manifest: DatasetManifest,
    results: Sequence[CandidateSet],
) -> None:
    members = _real_members(manifest)
    for result in results:
        if result.model_id is None or result.model_mode is None:
            raise ValueError("real-quality retrieval requires model provenance")
        if result.model_mode is not ModelMode.INFERENCE:
            raise ValueError("only inference retrieval can enter real-quality evaluation")
        for candidate in result.candidates:
            video_id = candidate.metadata.get("video_id")
            source_sha256 = candidate.metadata.get("source_sha256")
            if (video_id, source_sha256) not in members:
                raise ValueError("retrieval candidate source is not a dataset member")
            if candidate.metadata.get("embedding_model_id") != result.model_id:
                raise ValueError("retrieval candidate model provenance does not match result")
            if candidate.metadata.get("embedding_model_mode") != result.model_mode.value:
                raise ValueError("retrieval candidate mode provenance does not match result")


def evaluate_real_detection(
    manifest: DatasetManifest,
    predictions: Sequence[DetectionBatch],
    annotations: Sequence[DetectionAnnotation],
    *,
    annotation_path: str | Path | None = None,
    split_manifests: Sequence[DatasetManifest] | None = None,
    iou_threshold: float = 0.5,
) -> DetectionMetrics:
    """Evaluate detector quality only when manifest proves a real quality dataset."""

    frozen = _context(manifest, annotation_path, split_manifests, REAL_DETECTION_ANNOTATION_SCHEMA)
    authoritative = tuple(
        DetectionAnnotation(
            item.frame_ref, tuple(GroundTruthBox(label, box) for label, box in item.boxes)
        )
        for item in frozen.frames
    )

    def key(item):
        return item.frame_ref.video_id, item.frame_ref.frame_index

    given = _unique_map(annotations, DetectionAnnotation, key, "annotations")
    expected = {key(item): item for item in authoritative}
    # Paths are operational locations, not annotation identity.
    normalized = {
        identity: replace(item, frame_ref=replace(item.frame_ref, source_path=None))
        for identity, item in given.items()
    }
    if normalized != expected:
        raise ValueError("detection annotations do not match frozen annotation artifact")
    observed = _unique_map(predictions, DetectionBatch, key, "predictions")
    if not set(observed) <= set(expected):
        raise ValueError("predictions contain frames outside the frozen denominator")
    _validate_detection_inputs(manifest, predictions, authoritative)
    return evaluate_detection(
        predictions,
        authoritative,
        iou_threshold=iou_threshold,
    )


def evaluate_real_retrieval(
    manifest: DatasetManifest,
    results: Sequence[CandidateSet],
    annotations: Sequence[RetrievalAnnotation],
    *,
    annotation_path: str | Path | None = None,
    split_manifests: Sequence[DatasetManifest] | None = None,
    k_values: Sequence[int] = (1, 5, 10),
) -> RetrievalMetrics:
    """Evaluate retrieval quality only when manifest proves a real quality dataset."""

    frozen = _context(manifest, annotation_path, split_manifests, REAL_ANNOTATION_SCHEMA)
    authoritative = tuple(
        RetrievalAnnotation(item.query_id, item.relevant_row_ids, item.forbidden_row_ids)
        for item in frozen.queries
    )
    given = _unique_map(annotations, RetrievalAnnotation, lambda item: item.query_id, "annotations")
    expected = {item.query_id: item for item in authoritative}
    if given != expected:
        raise ValueError("retrieval annotations do not match frozen annotation artifact")
    observed = _unique_map(results, CandidateSet, lambda item: item.query_id, "results")
    if not set(observed) <= set(expected):
        raise ValueError("results contain queries outside the frozen denominator")
    _validate_retrieval_inputs(manifest, results)
    videos = {item.query_id: item.video_id for item in frozen.queries}
    for result in results:
        if any(item.metadata["video_id"] != videos[result.query_id] for item in result.candidates):
            raise ValueError("retrieval candidate belongs to another query video")
        annotation = expected[result.query_id]
        if not annotation.is_positive and any(
            item.row_id not in annotation.forbidden_row_ids for item in result.candidates
        ):
            raise ValueError("negative-query results contain unjudged rows; extend annotations")
    return evaluate_retrieval(results, authoritative, k_values=k_values)
