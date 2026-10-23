"""Small deterministic video fixture for CPU-only CI and timestamp tests."""

from __future__ import annotations

import json
import math
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

from video_intelligence.core.paths import resolve_project_paths
from video_intelligence.data.dataset import (
    DatasetKind,
    DatasetManifest,
    DatasetPurpose,
    write_dataset_manifest,
)
from video_intelligence.data.synthetic_registry import (
    register_synthetic_hashes,
    synthetic_registry_path,
)
from video_intelligence.video.manifest import sha256_file


def _clipped_rect(
    bounds: tuple[int, int, int, int], width: int, height: int
) -> tuple[int, int, int, int] | None:
    x1, y1, x2, y2 = bounds
    clipped = (max(0, x1), max(0, y1), min(width, x2), min(height, y2))
    return clipped if clipped[0] < clipped[2] and clipped[1] < clipped[3] else None


def generate_synthetic_fixture(
    output_dir: str | Path,
    *,
    fps: float = 10.0,
    width: int = 320,
    height: int = 240,
    duration_seconds: float = 8.0,
    overwrite: bool = False,
) -> dict[str, Path]:
    """Generate the blueprint's red-square/blue-circle/green-object fixture."""

    import cv2
    import numpy as np

    if not isinstance(overwrite, bool):
        raise ValueError("overwrite must be a boolean")
    if (
        not all(math.isfinite(float(value)) for value in (fps, duration_seconds))
        or fps <= 0
        or width <= 0
        or height <= 0
        or duration_seconds <= 0
    ):
        raise ValueError("fps, dimensions and duration_seconds must be positive")
    directory = Path(output_dir).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    video_path = directory / "synthetic_fixture.mp4"
    ground_truth_path = directory / "synthetic_fixture.json"
    dataset_manifest_path = directory / "synthetic_fixture.dataset.json"
    output_paths = (video_path, ground_truth_path, dataset_manifest_path)
    if not overwrite:
        existing = tuple(path for path in output_paths if path.exists())
        if existing:
            raise FileExistsError(
                "fixture outputs already exist; choose a new output directory or "
                "pass overwrite=True: " + ", ".join(str(path) for path in existing)
            )

    staging_dir = Path(tempfile.mkdtemp(prefix=".synthetic-fixture-", dir=directory))
    staged_video_path = staging_dir / video_path.name
    staged_ground_truth_path = staging_dir / ground_truth_path.name
    staged_manifest_path = staging_dir / dataset_manifest_path.name

    writer = cv2.VideoWriter(
        str(staged_video_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    if not writer.isOpened():
        shutil.rmtree(staging_dir, ignore_errors=True)
        raise RuntimeError(f"Could not create video writer for {video_path}")

    total_frames = max(1, math.ceil(fps * duration_seconds - 1e-12))
    blue_start = duration_seconds * 0.25
    red_blue_end = duration_seconds * 0.75
    object_size = min(width, height)
    red_size = object_size * 0.15
    blue_radius = object_size / 12
    green_size = object_size * 0.13333333333333333
    red_track: list[dict[str, Any]] = []
    blue_track: list[dict[str, Any]] = []
    green_track: list[dict[str, Any]] = []
    try:
        for frame_index in range(total_frames):
            timestamp = frame_index / fps
            frame = np.full((height, width, 3), 245, dtype=np.uint8)
            cv2.rectangle(frame, (8, 8), (width - 8, height - 8), (210, 210, 210), 1)

            if timestamp < red_blue_end:
                red_cx = width * 0.140625 + width * 0.53125 * min(
                    timestamp, duration_seconds * 0.25
                ) / (duration_seconds * 0.25)
                if timestamp >= duration_seconds * 0.25:
                    red_cx = width * 0.671875 - width * 0.109375 * min(
                        timestamp - duration_seconds * 0.25, duration_seconds * 0.5
                    ) / (duration_seconds * 0.5)
                red_cy = height / 3
                raw_bounds = (
                    int(round(red_cx - red_size / 2)),
                    int(round(red_cy - red_size / 2)),
                    int(round(red_cx + red_size / 2)),
                    int(round(red_cy + red_size / 2)),
                )
                cv2.rectangle(
                    frame,
                    (raw_bounds[0], raw_bounds[1]),
                    (raw_bounds[2] - 1, raw_bounds[3] - 1),
                    (0, 0, 220),
                    -1,
                )
                red_bounds = _clipped_rect(raw_bounds, width, height)
                assert red_bounds is not None
                red_track.append(
                    {
                        "frame_index": frame_index,
                        "timestamp_seconds": timestamp,
                        "bbox_xyxy": [float(value) for value in red_bounds],
                    }
                )

            if timestamp >= blue_start and timestamp < red_blue_end:
                blue_cx = (
                    width * 0.734375 if timestamp < duration_seconds * 0.5 else width * 0.640625
                )
                blue_cy = height * 0.4375
                blue_center = (int(round(blue_cx)), int(round(blue_cy)))
                blue_radius_int = int(round(blue_radius))
                cv2.circle(frame, blue_center, blue_radius_int, (220, 80, 0), -1)
                blue_bounds = _clipped_rect(
                    (
                        blue_center[0] - blue_radius_int,
                        blue_center[1] - blue_radius_int,
                        blue_center[0] + blue_radius_int + 1,
                        blue_center[1] + blue_radius_int + 1,
                    ),
                    width,
                    height,
                )
                assert blue_bounds is not None
                blue_track.append(
                    {
                        "frame_index": frame_index,
                        "timestamp_seconds": timestamp,
                        "bbox_xyxy": [float(value) for value in blue_bounds],
                    }
                )

            if timestamp >= red_blue_end:
                progress = min((timestamp - red_blue_end) / (duration_seconds * 0.25), 1.0)
                green_cx = (
                    width + green_size / 2 + 4 - (width * 0.55 + green_size / 2 + 4) * progress
                )
                green_cy = height * 0.6875
                raw_bounds = (
                    int(round(green_cx - green_size / 2)),
                    int(round(green_cy - green_size / 2)),
                    int(round(green_cx + green_size / 2)),
                    int(round(green_cy + green_size / 2)),
                )
                cv2.rectangle(
                    frame,
                    (raw_bounds[0], raw_bounds[1]),
                    (raw_bounds[2] - 1, raw_bounds[3] - 1),
                    (50, 180, 50),
                    -1,
                )
                green_bounds = _clipped_rect(raw_bounds, width, height)
                if green_bounds is not None:
                    green_track.append(
                        {
                            "frame_index": frame_index,
                            "timestamp_seconds": timestamp,
                            "bbox_xyxy": [float(value) for value in green_bounds],
                        }
                    )
            writer.write(frame)
    finally:
        writer.release()

    events = {
        "blue_circle_first_appears": [
            blue_start,
            min(red_blue_end, blue_start + duration_seconds / 16),
        ],
        "red_object_near_blue_object": [duration_seconds * 0.5, red_blue_end],
        "green_object_enters": [red_blue_end, duration_seconds],
    }
    payload = {
        "video_id": "synthetic_fixture",
        "duration_seconds": duration_seconds,
        "fps": fps,
        "width": width,
        "height": height,
        "objects": {
            "red_square": {"track_id": "red_1", "observations": red_track},
            "blue_circle": {"track_id": "blue_1", "observations": blue_track},
            "green_object": {"track_id": "green_1", "observations": green_track},
        },
        "events": events,
        "queries": [
            {
                "query": "when does the blue circle first appear?",
                "ground_truth": list(events["blue_circle_first_appears"]),
            },
            {
                "query": "show red object near blue object",
                "ground_truth": list(events["red_object_near_blue_object"]),
            },
        ],
    }
    try:
        staged_ground_truth_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        # Build the sidecar only after the staged video is complete, so the hash and
        # the published bytes are created as one isolated fixture set.
        write_dataset_manifest(
            DatasetManifest(
                dataset_id="synthetic_fixture_contract",
                kind=DatasetKind.SYNTHETIC,
                purpose=DatasetPurpose.CONTRACT,
                split="contract",
                source_video_ids=("synthetic_fixture",),
                source_sha256=(sha256_file(staged_video_path),),
                annotation_schema="synthetic_fixture.v1",
                synthetic_recipe="video_intelligence.data.fixtures.generate_synthetic_fixture",
            ),
            staged_manifest_path,
        )
        if not overwrite and any(path.exists() for path in output_paths):
            raise FileExistsError("fixture output appeared during generation")
        for staged, destination in zip(
            (staged_video_path, staged_ground_truth_path, staged_manifest_path),
            output_paths,
            strict=True,
        ):
            os.replace(staged, destination)
        artifact_root = resolve_project_paths().artifacts_root.resolve()
        try:
            directory.relative_to(artifact_root)
        except ValueError:
            persist_path = None
        else:
            persist_path = synthetic_registry_path()
        register_synthetic_hashes((sha256_file(video_path),), persist_path=persist_path)
    except BaseException:
        shutil.rmtree(staging_dir, ignore_errors=True)
        raise
    shutil.rmtree(staging_dir, ignore_errors=True)
    return {
        "video": video_path,
        "ground_truth": ground_truth_path,
        "dataset_manifest": dataset_manifest_path,
    }
