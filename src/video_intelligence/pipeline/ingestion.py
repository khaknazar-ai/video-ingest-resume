"""Registration and CPU fixture ingestion for the first end-to-end path."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from video_intelligence.core.types import Detection, FrameRef, VideoRecord
from video_intelligence.data.dataset import require_synthetic_contract
from video_intelligence.index.exact import ExactCosineIndex
from video_intelligence.pipeline.index_registry import IndexRegistry
from video_intelligence.storage.sqlite import SQLiteStore
from video_intelligence.video.decoder import VideoDecoder
from video_intelligence.video.manifest import build_video_manifest, sha256_file
from video_intelligence.video.sampler import uniform_budget_frames

LABELS = ("red_square", "blue_circle", "green_object")
MODEL_ID = "fixture.pixel_color_v1"


def _verify_registered_source(video: VideoRecord) -> None:
    if video.source_sha256 is None or sha256_file(video.source_path) != video.source_sha256:
        raise ValueError("source hash no longer matches the registered video")


@dataclass(frozen=True, slots=True)
class FixtureObservation:
    detection: Detection
    frame_ref: FrameRef


@dataclass(frozen=True, slots=True)
class IngestSummary:
    video_id: str
    mode: str
    processing_status: str
    sampled_frame_count: int
    detection_count: int
    coverage_complete: bool
    snapshot_id: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "video_id": self.video_id,
            "mode": self.mode,
            "processing_status": self.processing_status,
            "sampled_frame_count": self.sampled_frame_count,
            "detection_count": self.detection_count,
            "coverage_complete": self.coverage_complete,
            "snapshot_id": self.snapshot_id,
        }


class PixelFixtureAdapter:
    """Detect the fixture's colored regions from decoded RGB pixels only."""

    def __init__(self, *, frame_budget: int = 256) -> None:
        if isinstance(frame_budget, bool) or not isinstance(frame_budget, int) or frame_budget <= 0:
            raise ValueError("frame_budget must be a positive integer")
        self.frame_budget = frame_budget

    @staticmethod
    def _bbox_for_color(image: np.ndarray, label: str) -> tuple[tuple[float, ...], float] | None:
        red, green, blue = (image[..., index].astype(np.float32) for index in range(3))
        if label == "red_square":
            mask = (red > 120) & (red > green * 1.4) & (red > blue * 1.4)
        elif label == "blue_circle":
            mask = (blue > 120) & (blue > red * 1.4) & (blue > green * 1.15)
        elif label == "green_object":
            mask = (green > 100) & (green > red * 1.25) & (green > blue * 1.25)
        else:
            raise ValueError(f"unsupported fixture label: {label}")
        y_coordinates, x_coordinates = np.where(mask)
        if not len(x_coordinates):
            return None
        height, width = image.shape[:2]
        area = float(len(x_coordinates))
        score = min(1.0, area / (width * height * 0.01))
        bbox = (
            float(x_coordinates.min()),
            float(y_coordinates.min()),
            float(x_coordinates.max() + 1),
            float(y_coordinates.max() + 1),
        )
        return bbox, score

    def detect(
        self,
        video: VideoRecord,
        *,
        checkpoint: Callable[[], None] | None = None,
    ) -> tuple[list[FixtureObservation], int, bool]:
        require_synthetic_contract(video.source_path)
        if video.source_sha256 is None:
            raise ValueError("fixture ingestion requires a registered source hash")
        _verify_registered_source(video)
        if video.metadata.frame_count is None:
            raise ValueError("fixture ingestion requires a known frame_count")
        frame_budget = min(self.frame_budget, video.metadata.frame_count)
        frame_refs = uniform_budget_frames(video, frame_budget=frame_budget)
        observations: list[FixtureObservation] = []
        with VideoDecoder(video) as decoder:
            for decoded in decoder.iter_frames(frame_refs, strict=True):
                if checkpoint is not None:
                    checkpoint()
                for label in LABELS:
                    detected = self._bbox_for_color(decoded.image, label)
                    if detected is None:
                        continue
                    bbox, score = detected
                    detection = Detection(
                        detection_id=(
                            f"det_{video.video_id}_{decoded.frame_ref.frame_index:08d}_{label}"
                        ),
                        video_id=video.video_id,
                        frame_index=decoded.frame_ref.frame_index,
                        timestamp_seconds=decoded.actual_timestamp_seconds,
                        label=label,
                        score=score,
                        bbox_xyxy=bbox,
                        model_id=MODEL_ID,
                        track_id=f"{label}_fixture",
                    )
                    observations.append(FixtureObservation(detection, decoded.frame_ref))
                if checkpoint is not None:
                    checkpoint()
        coverage_complete = len(frame_refs) == video.metadata.frame_count
        _verify_registered_source(video)
        return observations, len(frame_refs), coverage_complete


class IngestionService:
    """Persist registration and build a fixture retrieval snapshot."""

    def __init__(self, store: SQLiteStore, registry: IndexRegistry) -> None:
        self.store = store
        self.registry = registry

    def register(self, source_path: str | Path, *, video_id: str | None = None) -> VideoRecord:
        manifest = build_video_manifest(source_path, video_id=video_id)
        record = VideoRecord.from_dict(manifest["video"])
        self.store.register_video(record)
        return record

    def ingest(
        self,
        video_id: str,
        *,
        adapter: PixelFixtureAdapter | None = None,
        checkpoint: Callable[[], None] | None = None,
        lease_fence: tuple[str, str, str] | None = None,
    ) -> IngestSummary:
        if lease_fence is not None and (
            len(lease_fence) != 3
            or any(not isinstance(value, str) or not value for value in lease_fence)
        ):
            raise ValueError("lease_fence must contain job_id, owner and lease_token")
        if checkpoint is not None:
            checkpoint()
        video = self.store.get_video(video_id)
        if video is None:
            raise ValueError(f"video is not registered: {video_id}")
        _verify_registered_source(video)
        adapter = adapter or PixelFixtureAdapter()
        observations, sampled_count, coverage_complete = adapter.detect(
            video,
            checkpoint=checkpoint,
        )
        _verify_registered_source(video)
        if not observations:
            if checkpoint is not None:
                checkpoint()
            return IngestSummary(
                video_id=video.video_id,
                mode="fixture",
                processing_status="partial",
                sampled_frame_count=sampled_count,
                detection_count=0,
                coverage_complete=coverage_complete,
                snapshot_id=None,
            )
        vectors = np.zeros((len(observations), len(LABELS)), dtype=np.float32)
        metadata: list[dict[str, Any]] = []
        row_ids: list[str] = []
        for row_index, observation in enumerate(observations):
            if checkpoint is not None:
                checkpoint()
            detection = observation.detection
            vectors[row_index, LABELS.index(detection.label)] = 1.0
            row_ids.append(detection.detection_id)
            metadata.append(
                {
                    "detection_id": detection.detection_id,
                    "video_id": detection.video_id,
                    "frame_index": detection.frame_index,
                    "timestamp_seconds": detection.timestamp_seconds,
                    "timestamp_quality": observation.frame_ref.timestamp_quality.value,
                    "timestamp_origin": observation.frame_ref.timestamp_origin,
                    "label": detection.label,
                    "score": detection.score,
                    "bbox_xyxy": list(detection.bbox_xyxy),
                    "width": observation.frame_ref.width,
                    "height": observation.frame_ref.height,
                    "source_sha256": video.source_sha256,
                    "track_id": detection.track_id,
                }
            )
        index = ExactCosineIndex(vectors, row_ids=row_ids, metadata=metadata)
        if checkpoint is not None:
            checkpoint()
        fence_kwargs = {}
        if lease_fence is not None:
            fence_kwargs = {
                "lease_job_id": lease_fence[0],
                "lease_owner": lease_fence[1],
                "lease_token": lease_fence[2],
            }
        def publication_checkpoint():
            if checkpoint is not None:
                checkpoint()
            _verify_registered_source(video)

        active = self.registry.publish_and_activate(
            f"fixture:{video.video_id}",
            index,
            attributes={
                "mode": "fixture",
                "adapter": MODEL_ID,
                "video_id": video.video_id,
                "source_sha256": video.source_sha256,
                "frame_budget": adapter.frame_budget,
                "sampled_frame_count": sampled_count,
                "coverage_complete": coverage_complete,
            },
            checkpoint=publication_checkpoint,
            **fence_kwargs,
        )
        return IngestSummary(
            video_id=video.video_id,
            mode="fixture",
            processing_status="complete" if coverage_complete else "partial",
            sampled_frame_count=sampled_count,
            detection_count=len(observations),
            coverage_complete=coverage_complete,
            snapshot_id=active.descriptor.snapshot_id,
        )
