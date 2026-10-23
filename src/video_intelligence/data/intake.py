"""Fail-closed intake for real quality datasets.

The intake contract binds source paths to content hashes and binds a strict,
versioned annotation file to the same source members. It never copies videos or
turns a synthetic fixture into a quality dataset.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields
from pathlib import Path

from video_intelligence.core.types import FrameRef
from video_intelligence.data.dataset import (
    REAL_QUALITY_SPLITS,
    DatasetKind,
    DatasetManifest,
    DatasetManifestError,
    DatasetPurpose,
    SyntheticDataUsageError,
    load_dataset_manifest,
    require_real_quality,
    write_dataset_manifest,
)
from video_intelligence.data.synthetic_registry import known_synthetic_hashes
from video_intelligence.detection.postprocess import BoundingBox
from video_intelligence.video.manifest import sha256_file

REAL_ANNOTATION_SCHEMA = "video_intelligence.real_quality.v1"
REAL_DETECTION_ANNOTATION_SCHEMA = "video_intelligence.real_detection.v1"


@dataclass(frozen=True, slots=True)
class QualityQuery:
    query_id: str
    video_id: str
    relevant_row_ids: frozenset[str]
    forbidden_row_ids: frozenset[str]


@dataclass(frozen=True, slots=True)
class QualityFrame:
    frame_ref: FrameRef
    boxes: tuple[tuple[str, tuple[float, float, float, float]], ...]


@dataclass(frozen=True, slots=True)
class VerifiedAnnotations:
    schema: str
    queries: tuple[QualityQuery, ...] = ()
    frames: tuple[QualityFrame, ...] = ()


class RealDatasetIntakeError(DatasetManifestError):
    """Invalid, incomplete or unsafe real-dataset intake input."""


def _identifier(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value.strip() != value
        or any(character.isspace() for character in value)
    ):
        raise RealDatasetIntakeError(f"{name} must be a non-empty identifier")
    return value


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise RealDatasetIntakeError(f"{name} must be a non-empty trimmed string")
    return value


def _sha256(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or value != value.lower()
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise RealDatasetIntakeError(f"{name} must be a lowercase SHA-256 hex digest")
    return value


def _reject_json_constant(value: str) -> None:
    raise RealDatasetIntakeError(f"non-finite JSON constant is not allowed: {value}")


def _load_annotation(path: Path) -> tuple[dict[str, object], str]:
    try:
        raw_bytes = path.read_bytes()
        payload = json.loads(raw_bytes.decode("utf-8"), parse_constant=_reject_json_constant)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise RealDatasetIntakeError(f"invalid annotation JSON: {path}") from error
    if not isinstance(payload, dict):
        raise RealDatasetIntakeError("annotation manifest must be a JSON object")
    return payload, hashlib.sha256(raw_bytes).hexdigest()


def _stable_source_hash(path: Path) -> str:
    before = path.stat()
    source_hash = sha256_file(path)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RealDatasetIntakeError(f"source changed while hashing: {path}")
    return source_hash


def _check_known_sidecars(source_path: Path, source_hash: str) -> None:
    if source_hash in known_synthetic_hashes(source_path=source_path):
        raise SyntheticDataUsageError(
            f"synthetic source cannot enter real-quality intake: {source_path}"
        )
    candidates = tuple(
        dict.fromkeys(
            (
                source_path.with_suffix(".dataset.json"),
                source_path.parent / "synthetic_fixture.dataset.json",
            )
        )
    )
    for candidate in candidates:
        if not candidate.is_file():
            continue
        try:
            manifest = load_dataset_manifest(candidate)
        except (DatasetManifestError, OSError, UnicodeError) as error:
            raise RealDatasetIntakeError(
                f"reserved dataset sidecar is invalid: {candidate}"
            ) from error
        if manifest.kind is DatasetKind.SYNTHETIC:
            raise SyntheticDataUsageError(
                f"synthetic source cannot enter real-quality intake: {source_path}"
            )
        if source_hash not in manifest.source_sha256:
            raise RealDatasetIntakeError(
                f"source sidecar hash does not match source file: {candidate}"
            )


def _validate_provenance(value: object) -> str:
    if not isinstance(value, dict):
        raise RealDatasetIntakeError("source_provenance must be an object")
    expected = {"provider", "dataset_revision", "license"}
    if set(value) != expected:
        raise RealDatasetIntakeError("source_provenance fields mismatch")
    provider = _text(value["provider"], "source_provenance.provider")
    revision = _identifier(value["dataset_revision"], "source_provenance.dataset_revision")
    license_name = _text(value["license"], "source_provenance.license")
    return json.dumps(
        {"dataset_revision": revision, "license": license_name, "provider": provider},
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _validate_row_ids(value: object, name: str) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise RealDatasetIntakeError(f"{name} must be a sequence")
    row_ids = tuple(_identifier(row_id, name) for row_id in value)
    if len(set(row_ids)) != len(row_ids):
        raise RealDatasetIntakeError(f"{name} must be unique")
    return row_ids


def _validate_annotation(
    payload: dict[str, object],
    *,
    dataset_id: str,
    split: str,
    source_members: Mapping[str, tuple[Path, str]],
) -> tuple[tuple[str, ...], str]:
    schema = payload.get("schema")
    if schema not in {REAL_ANNOTATION_SCHEMA, REAL_DETECTION_ANNOTATION_SCHEMA}:
        raise RealDatasetIntakeError(f"unsupported real annotation schema: {schema!r}")
    label_field = "queries" if schema == REAL_ANNOTATION_SCHEMA else "frames"
    expected = {"schema", "dataset_id", "split", "source_provenance", "videos", label_field}
    if set(payload) != expected:
        raise RealDatasetIntakeError("annotation fields mismatch")
    if payload["dataset_id"] != dataset_id:
        raise RealDatasetIntakeError("annotation dataset_id does not match intake")
    if payload["split"] != split:
        raise RealDatasetIntakeError("annotation split does not match intake")
    source_provenance = _validate_provenance(payload["source_provenance"])

    videos = payload["videos"]
    if isinstance(videos, (str, bytes)) or not isinstance(videos, Sequence) or not videos:
        raise RealDatasetIntakeError("annotation videos must be a non-empty sequence")
    annotated: dict[str, tuple[str, str]] = {}
    for row in videos:
        if not isinstance(row, dict) or set(row) != {
            "video_id",
            "source_sha256",
            "source_group_id",
        }:
            raise RealDatasetIntakeError("each annotation video row must have exactly three fields")
        video_id = _identifier(row["video_id"], "annotation.video_id")
        if video_id in annotated:
            raise RealDatasetIntakeError("annotation video_id values must be unique")
        annotated[video_id] = (
            _sha256(row["source_sha256"], "annotation.source_sha256"),
            _identifier(row["source_group_id"], "annotation.source_group_id"),
        )

    source_ids = set(source_members)
    if set(annotated) != source_ids:
        raise RealDatasetIntakeError("annotation videos must exactly match source registry")
    source_groups: list[str] = []
    for video_id in sorted(source_members):
        _, source_hash = source_members[video_id]
        annotation_hash, source_group_id = annotated[video_id]
        if annotation_hash != source_hash:
            raise RealDatasetIntakeError(
                f"annotation source hash does not match source file: {video_id}"
            )
        source_groups.append(source_group_id)

    if schema == REAL_DETECTION_ANNOTATION_SCHEMA:
        _quality_frames(payload["frames"], {key: value[1] for key, value in source_members.items()})
        return tuple(source_groups), source_provenance

    queries = payload["queries"]
    if isinstance(queries, (str, bytes)) or not isinstance(queries, Sequence) or not queries:
        raise RealDatasetIntakeError("annotation queries must be a non-empty sequence")
    query_ids: set[str] = set()
    positive_count = 0
    negative_count = 0
    for row in queries:
        expected_query_fields = {
            "query_id",
            "query",
            "video_id",
            "is_negative",
            "relevant_row_ids",
            "forbidden_row_ids",
        }
        if not isinstance(row, dict) or set(row) != expected_query_fields:
            raise RealDatasetIntakeError("each annotation query row has an invalid schema")
        query_id = _identifier(row["query_id"], "annotation.query_id")
        if query_id in query_ids:
            raise RealDatasetIntakeError("annotation query_id values must be unique")
        query_ids.add(query_id)
        _text(row["query"], "annotation.query")
        video_id = _identifier(row["video_id"], "annotation.query.video_id")
        if video_id not in source_members:
            raise RealDatasetIntakeError("annotation query references an unknown video_id")
        if not isinstance(row["is_negative"], bool):
            raise RealDatasetIntakeError("annotation query is_negative must be boolean")
        relevant = _validate_row_ids(row["relevant_row_ids"], "annotation.relevant_row_ids")
        forbidden = _validate_row_ids(row["forbidden_row_ids"], "annotation.forbidden_row_ids")
        if set(relevant) & set(forbidden):
            raise RealDatasetIntakeError("relevant and forbidden row ids must not overlap")
        if row["is_negative"]:
            negative_count += 1
            if relevant:
                raise RealDatasetIntakeError("negative queries must have no relevant rows")
            if not forbidden:
                raise RealDatasetIntakeError("negative queries require forbidden rows")
        else:
            positive_count += 1
            if not relevant:
                raise RealDatasetIntakeError("positive queries require relevant rows")
    if positive_count == 0 or negative_count == 0:
        raise RealDatasetIntakeError("real quality intake requires positive and negative queries")
    return tuple(source_groups), source_provenance


def _quality_frames(value: object, members: Mapping[str, str]) -> tuple[QualityFrame, ...]:
    if not isinstance(value, list) or not value:
        raise RealDatasetIntakeError("annotation frames must be a non-empty list")
    result: list[QualityFrame] = []
    keys: set[tuple[str, int]] = set()
    frame_fields = {field.name for field in fields(FrameRef)}
    for row in value:
        if not isinstance(row, dict) or set(row) != {"frame_ref", "boxes"}:
            raise RealDatasetIntakeError("annotation frame fields mismatch")
        ref = row["frame_ref"]
        if not isinstance(ref, dict) or set(ref) != frame_fields or ref["source_path"] is not None:
            raise RealDatasetIntakeError("frame_ref requires all fields and source_path=null")
        frame = FrameRef(**ref)
        if frame.video_id not in members or members[frame.video_id] != frame.source_sha256:
            raise RealDatasetIntakeError("annotation frame is not a source member")
        key = (frame.video_id, frame.frame_index)
        if key in keys:
            raise RealDatasetIntakeError("annotation frame keys must be unique")
        keys.add(key)
        if not isinstance(row["boxes"], list):
            raise RealDatasetIntakeError("annotation boxes must be a list")
        boxes = []
        for box in row["boxes"]:
            if not isinstance(box, dict) or set(box) != {"label", "bbox_xyxy"}:
                raise RealDatasetIntakeError("annotation box fields mismatch")
            label = _identifier(box["label"], "box.label")
            coordinates = BoundingBox(
                box["bbox_xyxy"], "source_pixels", frame.width, frame.height
            ).xyxy
            boxes.append((label, coordinates))
        if len(set(boxes)) != len(boxes):
            raise RealDatasetIntakeError("annotation boxes must be unique")
        result.append(QualityFrame(frame, tuple(boxes)))
    if not any(item.boxes for item in result) or not any(not item.boxes for item in result):
        raise RealDatasetIntakeError("real detection requires positive and negative frames")
    return tuple(result)


def load_verified_real_annotations(
    manifest: DatasetManifest, annotation_path: str | Path
) -> VerifiedAnnotations:
    """Hash, parse and validate the same bytes; never trust caller-supplied labels."""

    require_real_quality(manifest)
    payload, digest = _load_annotation(Path(annotation_path))
    if digest != manifest.annotation_sha256:
        raise RealDatasetIntakeError("annotation hash does not match manifest")
    if payload.get("schema") != manifest.annotation_schema:
        raise RealDatasetIntakeError("annotation schema does not match manifest")
    members = dict(manifest.source_members)
    groups, provenance = _validate_annotation(
        payload,
        dataset_id=manifest.dataset_id,
        split=manifest.split,
        source_members={key: (Path(), value) for key, value in members.items()},
    )
    expected_groups = dict((key, group) for key, _, group in manifest.source_group_members)
    if groups != tuple(expected_groups[key] for key in sorted(members)):
        raise RealDatasetIntakeError("annotation source groups do not match manifest")
    if provenance != manifest.source_provenance:
        raise RealDatasetIntakeError("annotation provenance does not match manifest")
    if manifest.annotation_schema == REAL_DETECTION_ANNOTATION_SCHEMA:
        return VerifiedAnnotations(
            manifest.annotation_schema, frames=_quality_frames(payload["frames"], members)
        )
    return VerifiedAnnotations(
        manifest.annotation_schema,
        queries=tuple(
            QualityQuery(
                row["query_id"],
                row["video_id"],
                frozenset(row["relevant_row_ids"]),
                frozenset(row["forbidden_row_ids"]),
            )
            for row in payload["queries"]
        ),
    )


def build_real_quality_manifest(
    dataset_id: str,
    source_videos: Mapping[str, str | Path],
    annotation_path: str | Path,
    *,
    split: str,
    manifest_path: str | Path | None = None,
    overwrite: bool = False,
) -> DatasetManifest:
    """Validate real inputs and build a content-addressed quality manifest.

    ``source_videos`` is an explicit ``video_id -> path`` registry. The function
    hashes current bytes and never accepts caller-provided source hashes.
    """

    dataset_id = _identifier(dataset_id, "dataset_id")
    split = _identifier(split, "split")
    if split not in REAL_QUALITY_SPLITS:
        raise RealDatasetIntakeError(f"split must be one of {sorted(REAL_QUALITY_SPLITS)!r}")
    if isinstance(source_videos, (str, bytes)) or not isinstance(source_videos, Mapping):
        raise RealDatasetIntakeError("source_videos must be a video_id -> path mapping")
    if not source_videos:
        raise RealDatasetIntakeError("source_videos must not be empty")

    normalized_sources: list[tuple[str, str | Path]] = []
    for raw_video_id, raw_path in source_videos.items():
        video_id = _identifier(raw_video_id, "source_video_id")
        if not isinstance(raw_path, (str, Path)):
            raise RealDatasetIntakeError(f"source path must be a path-like value: {video_id}")
        normalized_sources.append((video_id, raw_path))
    if len({video_id for video_id, _ in normalized_sources}) != len(normalized_sources):
        raise RealDatasetIntakeError("source_video_id values must be unique")

    resolved_sources: dict[str, tuple[Path, str]] = {}
    seen_source_hashes: set[str] = set()
    for video_id, raw_path in sorted(normalized_sources):
        source_path = Path(raw_path).expanduser().resolve()
        if not source_path.is_file():
            raise RealDatasetIntakeError(f"source video is not a file: {source_path}")
        if source_path.stat().st_size <= 0:
            raise RealDatasetIntakeError(f"source video is empty: {source_path}")
        source_hash = _stable_source_hash(source_path)
        _check_known_sidecars(source_path, source_hash)
        if source_hash in seen_source_hashes:
            raise RealDatasetIntakeError("source video hashes must be unique")
        seen_source_hashes.add(source_hash)
        resolved_sources[video_id] = (source_path, source_hash)

    annotation = Path(annotation_path).expanduser().resolve()
    if not annotation.is_file():
        raise RealDatasetIntakeError(f"annotation file is not a file: {annotation}")
    payload, annotation_hash = _load_annotation(annotation)
    source_groups, source_provenance = _validate_annotation(
        payload,
        dataset_id=dataset_id,
        split=split,
        source_members=resolved_sources,
    )
    source_ids = tuple(sorted(resolved_sources))
    manifest = DatasetManifest(
        dataset_id=dataset_id,
        kind=DatasetKind.REAL,
        purpose=DatasetPurpose.QUALITY,
        split=split,
        source_video_ids=source_ids,
        source_sha256=tuple(resolved_sources[video_id][1] for video_id in source_ids),
        annotation_schema=payload["schema"],
        annotation_sha256=annotation_hash,
        source_group_ids=source_groups,
        source_provenance=source_provenance,
    )
    require_real_quality(manifest)
    if manifest_path is not None:
        destination = Path(manifest_path).expanduser().resolve()
        protected_paths = {
            annotation,
            *(source_path for source_path, _ in resolved_sources.values()),
        }
        if destination in protected_paths:
            raise RealDatasetIntakeError("manifest output must not overwrite an intake input")
        write_dataset_manifest(manifest, destination, overwrite=overwrite)
    return manifest


def validate_real_quality_split_policy(
    manifests: Sequence[DatasetManifest],
) -> None:
    """Reject duplicate source ids, hashes or groups across evaluation splits."""

    if isinstance(manifests, (str, bytes)) or not isinstance(manifests, Sequence):
        raise TypeError("manifests must be a sequence")
    if not manifests:
        raise RealDatasetIntakeError("at least one quality manifest is required")
    seen_dataset_splits: set[tuple[str, str]] = set()
    seen_video_ids: set[str] = set()
    seen_hashes: set[str] = set()
    seen_groups: set[str] = set()
    for manifest in manifests:
        require_real_quality(manifest)
        dataset_split = (manifest.dataset_id, manifest.split)
        if dataset_split in seen_dataset_splits:
            raise RealDatasetIntakeError(
                f"duplicate dataset split manifest: {manifest.dataset_id}/{manifest.split}"
            )
        seen_dataset_splits.add(dataset_split)
        current_video_ids = {member[0] for member in manifest.source_group_members}
        current_hashes = {member[1] for member in manifest.source_group_members}
        current_groups = {member[2] for member in manifest.source_group_members}
        if seen_video_ids & current_video_ids:
            raise RealDatasetIntakeError("a source video_id appears in multiple splits")
        if seen_hashes & current_hashes:
            raise RealDatasetIntakeError("a source hash appears in multiple splits")
        if seen_groups & current_groups:
            raise RealDatasetIntakeError("a source group appears in multiple splits")
        seen_video_ids.update(current_video_ids)
        seen_hashes.update(current_hashes)
        seen_groups.update(current_groups)
