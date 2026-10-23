"""Evidence construction and source/timestamp integrity checks."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from video_intelligence.core.types import FrameRef, TimestampQuality, VideoRecord
from video_intelligence.index.snapshot import SnapshotDescriptor
from video_intelligence.retrieval.schemas import EvidenceBundle, EvidenceItem


def evidence_from_metadata(
    metadata: Mapping[str, Any],
    *,
    video: VideoRecord,
    descriptor: SnapshotDescriptor,
    run_id: str,
    role: str = "entity",
) -> EvidenceItem:
    """Turn an index row into evidence tied to the actual source video."""

    if video.source_sha256 is None:
        raise ValueError("evidence requires a registered source hash")
    required = {
        "detection_id",
        "video_id",
        "frame_index",
        "timestamp_seconds",
        "label",
        "bbox_xyxy",
        "score",
    }
    missing = required - set(metadata)
    if missing:
        raise ValueError(f"index row is missing evidence fields: {sorted(missing)!r}")
    if metadata["video_id"] != video.video_id:
        raise ValueError("index row belongs to another video")
    if metadata.get("source_sha256") not in {None, video.source_sha256}:
        raise ValueError("index row source hash does not match video")
    width = video.metadata.width or int(metadata.get("width", 0))
    height = video.metadata.height or int(metadata.get("height", 0))
    if width <= 0 or height <= 0:
        raise ValueError("evidence requires known video dimensions")
    bbox = tuple(float(value) for value in metadata["bbox_xyxy"])
    if bbox[0] < 0 or bbox[1] < 0 or bbox[2] > width or bbox[3] > height:
        raise ValueError("evidence bbox is outside video dimensions")
    timestamp_quality = TimestampQuality(metadata.get("timestamp_quality", "cfr_estimate"))
    frame_ref = FrameRef(
        video_id=video.video_id,
        frame_index=metadata["frame_index"],
        timestamp_seconds=metadata["timestamp_seconds"],
        width=width,
        height=height,
        source_path=video.source_path,
        timestamp_quality=timestamp_quality,
        timestamp_origin=metadata.get("timestamp_origin", "fixture_decode"),
        source_sha256=video.source_sha256,
        decode_ordinal=metadata["frame_index"],
        backend="fixture",
    )
    return EvidenceItem(
        evidence_id=metadata["detection_id"],
        frame_ref=frame_ref,
        label=metadata["label"],
        bbox_xyxy=bbox,
        score=metadata["score"],
        source_sha256=video.source_sha256,
        snapshot_id=descriptor.snapshot_id,
        artifact_id=f"index_snapshot_{descriptor.snapshot_id}",
        source_stage="fixture.pixel_color_v1",
        role=role,
        provenance={
            "run_id": run_id,
            "source_sha256": video.source_sha256,
            "snapshot_id": descriptor.snapshot_id,
            "mode": "fixture",
        },
    )


def build_bundle(
    *,
    query_id: str,
    run_id: str,
    video: VideoRecord,
    descriptor: SnapshotDescriptor,
    items: Sequence[EvidenceItem],
    coverage_complete: bool,
    gaps: Sequence[str] = (),
    relation_policy: dict[str, Any] | None = None,
) -> EvidenceBundle:
    if not items:
        raise ValueError("cannot build an evidence bundle without items")
    timestamps = [item.frame_ref.timestamp_seconds for item in items]
    end = video.metadata.duration_seconds
    if end is None:
        end = max(timestamps)
    return EvidenceBundle(
        query_id=query_id,
        run_id=run_id,
        video_id=video.video_id,
        source_sha256=video.source_sha256 or "0" * 64,
        snapshot_id=descriptor.snapshot_id,
        items=tuple(items),
        coverage_start_seconds=0.0,
        coverage_end_seconds=end,
        coverage_complete=coverage_complete,
        gaps=tuple(gaps),
        relation_policy=relation_policy,
    )
