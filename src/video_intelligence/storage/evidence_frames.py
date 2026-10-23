"""Immutable, source-verified PNG evidence with portable query/observation binding."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import stat
import struct
import time
import uuid
from collections.abc import Callable
from copy import deepcopy
from dataclasses import asdict, dataclass, replace
from pathlib import Path, PurePosixPath
from typing import Any

import cv2
import numpy as np

from video_intelligence.core.types import (
    FrameRef,
    FrameRequest,
    TimestampQuality,
    VideoMetadata,
    VideoRecord,
)
from video_intelligence.retrieval.schemas import QueryResult, SupportStatus
from video_intelligence.video.decoder import PyAVVideoDecoder

SCHEMA_VERSION = "video_intelligence.evidence_frames.v1"
POLICY_VERSION = "lossless_full_frame_png.v1"


def _digest(value: object) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError("expected a lowercase SHA-256 digest")
    return value


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False,
                       separators=(",", ":")) + "\n").encode("utf-8")


def _portable(frame: FrameRef) -> dict[str, Any]:
    value = asdict(frame)
    value["source_path"] = None
    value["timestamp_quality"] = frame.timestamp_quality.value
    return value


def _semantics(frame: FrameRef) -> tuple[object, ...]:
    if frame.timestamp_quality is not TimestampQuality.EXACT_PTS or frame.backend == "legacy":
        raise ValueError("persisted evidence requires complete exact PTS identity")
    return (frame.identity_key, frame.frame_index, frame.timestamp_seconds,
            frame.width, frame.height, frame.timestamp_origin, frame.timestamp_quality)


def _safe(path: Path, *, directory: bool | None = None) -> None:
    """Reject lexical traversal and existing symlink/reparse components, without resolving them."""
    if ".." in path.parts:
        raise ValueError("artifact paths cannot contain parent traversal")
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current /= part
        try:
            info = current.lstat()
        except FileNotFoundError:
            if directory is not None and current == path:
                raise
            continue
        if stat.S_ISLNK(info.st_mode) or (
            getattr(info, "st_file_attributes", 0)
            & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
        ):
            raise ValueError("artifact paths cannot contain symlinks, junctions or reparse points")
        if current != path and not stat.S_ISDIR(info.st_mode):
            raise ValueError("artifact parent is not a directory")
        if current == path and directory is not None:
            expected = stat.S_ISDIR if directory else stat.S_ISREG
            if not expected(info.st_mode):
                raise ValueError("artifact member has an unsupported file type")


def _mkdir(path: Path) -> None:
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current /= part
        _safe(current)
        try:
            current.mkdir()
        except FileExistsError:
            pass
        _safe(current, directory=True)


@dataclass(frozen=True, slots=True)
class EvidenceFrameConfig:
    max_frames: int = 64
    max_frame_pixels: int = 2_073_600
    max_total_bytes: int = 64 * 1024**2
    max_runtime_seconds: float = 30.0

    def __post_init__(self) -> None:
        for name, maximum in (("max_frames", 256), ("max_frame_pixels", 8_294_400),
                              ("max_total_bytes", 512 * 1024**2)):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= maximum:
                raise ValueError(f"{name} must be an integer in [1,{maximum}]")
        value = self.max_runtime_seconds
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) or not 0 < value <= 60):
            raise ValueError("max_runtime_seconds must be finite in (0,60]")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class EvidenceFrameArtifact:
    artifact_id: str
    kind: str
    content_sha256: str
    relative_path: str
    size_bytes: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class PersistedEvidence:
    query_result: QueryResult
    artifact: EvidenceFrameArtifact | None


class _Budget:
    def __init__(self, config, checkpoint):
        if not isinstance(config, EvidenceFrameConfig):
            raise TypeError("config must be an EvidenceFrameConfig")
        if checkpoint is not None and not callable(checkpoint):
            raise TypeError("checkpoint must be callable or None")
        self.config, self.checkpoint = config, checkpoint
        self.started = time.perf_counter()
        self.deadline = self.started + config.max_runtime_seconds

    def check(self) -> None:
        if time.perf_counter() >= self.deadline:
            raise TimeoutError("evidence persistence exceeded its cooperative deadline")
        if self.checkpoint is not None:
            self.checkpoint()
        if time.perf_counter() >= self.deadline:
            raise TimeoutError("evidence persistence exceeded its cooperative deadline")


def _write(path: Path, payload: bytes, budget: _Budget) -> None:
    budget.check()
    _safe(path.parent, directory=True)
    _safe(path)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    flags |= getattr(os, "O_BINARY", 0)
    with os.fdopen(os.open(path, flags, 0o600), "wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    _safe(path, directory=False)
    budget.check()


def _read(path: Path, limit: int, budget: _Budget) -> bytes:
    budget.check()
    _safe(path, directory=False)
    before = path.stat()
    if before.st_size > limit:
        raise ValueError("artifact exceeds the configured byte cap")
    chunks, total = [], 0
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_BINARY", 0)
    with os.fdopen(os.open(path, flags), "rb") as stream:
        while chunk := stream.read(1024 * 1024):
            budget.check()
            total += len(chunk)
            if total > limit:
                raise ValueError("artifact exceeds the configured byte cap")
            chunks.append(chunk)
    _safe(path, directory=False)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns, before.st_ino) != (
        after.st_size, after.st_mtime_ns, after.st_ino
    ):
        raise ValueError("artifact changed during readback")
    return b"".join(chunks)


def _source_hash(path: Path, budget: _Budget) -> str:
    before = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            budget.check()
            digest.update(chunk)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns, before.st_ino) != (
        after.st_size, after.st_mtime_ns, after.st_ino
    ):
        raise ValueError("source changed while hashing")
    budget.check()
    return digest.hexdigest()


def _members(path: Path, maximum: int, budget: _Budget) -> set[str]:
    _safe(path, directory=True)
    names: set[str] = set()
    for member in path.iterdir():
        budget.check()
        if len(names) >= maximum:
            raise ValueError("unexpected artifact directory members exceed the cap")
        names.add(member.name)
    return names


def _png_rgb(payload: bytes, frame: FrameRef, config: EvidenceFrameConfig) -> np.ndarray:
    if (len(payload) < 33 or payload[:8] != b"\x89PNG\r\n\x1a\n"
            or payload[12:16] != b"IHDR"):
        raise ValueError("evidence image is not a PNG")
    width, height = struct.unpack(">II", payload[16:24])
    if ((width, height) != (frame.width, frame.height)
            or width * height > config.max_frame_pixels):
        raise ValueError("PNG dimensions disagree or exceed the pixel cap")
    image = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if image is None or image.dtype != np.uint8 or image.shape != (height, width, 3):
        raise ValueError("evidence PNG must decode as a complete RGB8 frame")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def _manifest_video(manifest: dict[str, Any]) -> VideoRecord:
    value = manifest["video"]
    if set(value) != {"video_id", "source_sha256", "metadata"}:
        raise ValueError("invalid portable manifest video")
    return VideoRecord(value["video_id"], Path("."), VideoMetadata.from_dict(value["metadata"]),
                       value["source_sha256"])


def _artifact(digest: str, size: int) -> EvidenceFrameArtifact:
    return EvidenceFrameArtifact(f"evidence_frames_{digest}", "evidence_frames", digest,
                                 f"evidence/{digest}/manifest.json", size)


def _updated_query(original, manifest, artifact) -> QueryResult:
    records = {identifier: row for row in manifest["frames"] for identifier in row["evidence_ids"]}
    items = []
    for item in original.evidence.items:
        row = records[item.evidence_id]
        provenance = deepcopy(item.provenance)
        provenance.update({
            "observation_artifact_id": item.artifact_id,
            "observation_artifact_kind": item.provenance.get("artifact_kind"),
            "artifact_kind": "persisted_evidence_frames", "artifact_persisted": True,
            "image_persisted": True,
            "image_artifact": {
                "artifact_id": artifact.artifact_id,
                "relative_path": f"evidence/{artifact.content_sha256}/{row['file']}",
                "png_sha256": row["png_sha256"], "rgb_sha256": row["rgb_sha256"],
                "frame_ref": _portable(item.frame_ref),
            },
        })
        items.append(replace(item, artifact_id=artifact.artifact_id, provenance=provenance))
    logical = original.logical_query_id or original.query.query_id
    unbound = replace(
        original, query=replace(original.query, query_id=logical), logical_query_id=None,
        evidence=replace(original.evidence, query_id=logical, items=tuple(items)),
    )
    policy = {**manifest["policy"], **artifact.to_dict(),
              "input_query_revision": original.query.query_id}
    return unbound.with_revision({**deepcopy(original.query_config), "evidence_frames": policy})


class EvidenceFrameStore:
    """Create-only PNG artifacts; READY is published last and failed staging is retained.

    Deadlines/checkpoints are cooperative around native codec calls. The caller
    still needs a process deadline for hung native code. The artifact root should
    be writable only by trusted workers; lstat checks do not lock concurrent writers.
    """

    def __init__(self, root: Path, *, decoder_factory=PyAVVideoDecoder):
        self.root = Path(root).expanduser().absolute()
        _safe(self.root)
        if not callable(decoder_factory):
            raise TypeError("decoder_factory must be callable")
        self.decoder_factory = decoder_factory

    def persist(self, result: QueryResult, *, video: VideoRecord,
                checkpoint: Callable[[], None] | None = None,
                config: EvidenceFrameConfig | None = None) -> PersistedEvidence:
        budget = _Budget(EvidenceFrameConfig() if config is None else config, checkpoint)
        if not isinstance(result, QueryResult) or not isinstance(video, VideoRecord):
            raise TypeError("result and video must use typed contracts")
        budget.check()
        result.validate_evidence()
        result.validate_video(video)
        budget.check()
        if result.support_status is not SupportStatus.SUPPORTED:
            if result.evidence and any(item.provenance.get("image_persisted")
                                       or item.provenance.get("artifact_persisted")
                                       or "image_artifact" in item.provenance
                                       for item in result.evidence.items):
                raise ValueError("unsupported results cannot carry persisted image claims")
            return PersistedEvidence(result, None)
        expected_source = _digest(video.source_sha256)
        if _source_hash(video.source_path, budget) != expected_source:
            raise ValueError("source SHA-256 differs from the registered video")
        if any(item.provenance.get("image_persisted") for item in result.evidence.items):
            artifact = self._verify_result(result, budget)
            if _source_hash(video.source_path, budget) != expected_source:
                raise ValueError("source changed during persistence")
            return PersistedEvidence(result, artifact)
        original = QueryResult.from_dict(result.to_dict(), video=video)
        if any("segmentation" in item.provenance and not item.provenance["segmentation"].get(
            "mask_persisted",
        ) for item in original.evidence.items):
            raise ValueError("segmentation masks must be persisted before RGB evidence")
        grouped: dict[tuple[object, ...], list] = {}
        for item in original.evidence.items:
            _semantics(item.frame_ref)
            _digest(item.provenance.get("frame_rgb_sha256"))
            if "image_artifact" in item.provenance or item.provenance.get("artifact_persisted"):
                raise ValueError("unpersisted evidence contains conflicting artifact claims")
            grouped.setdefault(item.frame_ref.identity_key, []).append(item)
        if len(grouped) > budget.config.max_frames:
            raise ValueError("selected evidence exceeds the frame cap")
        for items in grouped.values():
            ref = items[0].frame_ref
            if ref.width * ref.height > budget.config.max_frame_pixels:
                raise ValueError("selected evidence exceeds the pixel cap")
            if any(_semantics(item.frame_ref) != _semantics(ref)
                   or item.provenance["frame_rgb_sha256"] != items[0].provenance["frame_rgb_sha256"]
                   for item in items):
                raise ValueError("shared frame has contradictory identity or RGB hashes")
        budget.check()
        staging = self.root / "evidence" / f".staging-{uuid.uuid4().hex}"
        _mkdir(staging / "frames")
        rows, total = [], 0
        budget.check()
        with self.decoder_factory(video) as decoder:
            budget.check()
            for index, items in enumerate(grouped.values()):
                budget.check()
                ref = items[0].frame_ref
                decoded = decoder.read_request(
                    FrameRequest(video.video_id, ref.timestamp_seconds,
                                 source="evidence_persistence", tolerance_seconds=0.0),
                    selection_policy="nearest",
                )
                budget.check()
                if decoded is None or _semantics(decoded.frame_ref) != _semantics(ref):
                    raise ValueError("evidence re-decode did not match the exact source frame")
                if decoded.actual_timestamp_seconds != ref.timestamp_seconds or any(
                    not issue.recoverable for issue in getattr(decoder, "issues", ())
                ):
                    raise ValueError(
                        "evidence re-decode has conflicting actual time or fatal issues"
                    )
                rgb = decoded.image
                if (not isinstance(rgb, np.ndarray) or rgb.dtype != np.uint8
                        or rgb.shape != (ref.height, ref.width, 3)):
                    raise ValueError("evidence re-decode must return canonical RGB8")
                rgb_hash = hashlib.sha256(rgb.tobytes()).hexdigest()
                if rgb_hash != items[0].provenance["frame_rgb_sha256"]:
                    raise ValueError("re-decoded RGB differs from the grounding model input")
                ok, encoded = cv2.imencode(".png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                                           [cv2.IMWRITE_PNG_COMPRESSION, 3])
                if not ok:
                    raise ValueError("lossless PNG encoding failed")
                payload = encoded.tobytes()
                if hashlib.sha256(_png_rgb(payload, ref, budget.config).tobytes()).hexdigest() != (
                    rgb_hash
                ):
                    raise ValueError("lossless PNG readback changed RGB bytes")
                total += len(payload)
                if total > budget.config.max_total_bytes:
                    raise ValueError("selected PNG evidence exceeds the byte cap")
                name = f"frames/frame_{index:04d}.png"
                _write(staging / name, payload, budget)
                rows.append({"file": name, "frame_ref": _portable(ref),
                             "png_sha256": hashlib.sha256(payload).hexdigest(),
                             "rgb_sha256": rgb_hash, "png_size_bytes": len(payload),
                             "evidence_ids": [item.evidence_id for item in items]})
        if _source_hash(video.source_path, budget) != expected_source:
            raise ValueError("source changed during evidence re-decoding")
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "policy": {"version": POLICY_VERSION, "config": budget.config.to_dict()},
            "video": {"video_id": video.video_id, "source_sha256": expected_source,
                      "metadata": video.metadata.to_dict()},
            "input_query_result": original.to_dict(), "frames": rows,
        }
        payload = _json_bytes(manifest)
        total += len(payload) + 65
        if total > budget.config.max_total_bytes:
            raise ValueError("complete evidence artifact exceeds the byte cap")
        digest = hashlib.sha256(payload).hexdigest()
        _write(staging / "manifest.json", payload, budget)
        final = self.root / "evidence" / digest
        budget.check()
        _safe(final)
        try:
            final.mkdir()
        except FileExistsError:
            self._verify(f"evidence/{digest}/manifest.json", digest, budget)
        else:
            _mkdir(final / "frames")
            for name in [row["file"] for row in rows] + ["manifest.json"]:
                budget.check()
                _safe(staging / name, directory=False)
                _safe((final / name).parent, directory=True)
                os.link(staging / name, final / name, follow_symlinks=False)
            self._verify(f"evidence/{digest}/manifest.json", digest, budget, ready=False)
            _write(final / "READY", (digest + "\n").encode("ascii"), budget)
        artifact = _artifact(digest, total)
        updated = _updated_query(original, manifest, artifact)
        verified = self._verify_result(updated, budget)
        if verified != artifact:
            raise ValueError("published evidence artifact metadata changed")
        return PersistedEvidence(updated, artifact)

    def _verify(self, relative_path, expected_sha256, budget, *, ready=True):
        digest = _digest(expected_sha256)
        if not isinstance(relative_path, str) or relative_path != (
            f"evidence/{digest}/manifest.json"
        ) or "\\" in relative_path or PurePosixPath(relative_path).is_absolute():
            raise ValueError("invalid evidence manifest relative path")
        final = self.root / "evidence" / digest
        _safe(final, directory=True)
        expected_members = {"frames", "manifest.json"} | ({"READY"} if ready else set())
        if _members(final, len(expected_members), budget) != expected_members:
            raise ValueError("incomplete or unexpected evidence artifact members")
        if ready and _read(final / "READY", 65, budget) != (digest + "\n").encode("ascii"):
            raise ValueError("evidence READY digest differs")
        payload = _read(final / "manifest.json", budget.config.max_total_bytes, budget)
        if hashlib.sha256(payload).hexdigest() != digest:
            raise ValueError("evidence manifest SHA-256 mismatch")
        manifest = json.loads(payload)
        if (_json_bytes(manifest) != payload or set(manifest) != {
            "schema_version", "policy", "video", "input_query_result", "frames"
        } or manifest["schema_version"] != SCHEMA_VERSION):
            raise ValueError("invalid canonical evidence manifest")
        policy = manifest["policy"]
        if set(policy) != {"version", "config"} or policy["version"] != POLICY_VERSION:
            raise ValueError("unsupported evidence persistence policy")
        stored_config = EvidenceFrameConfig(**policy["config"])
        if stored_config.to_dict() != policy["config"]:
            raise ValueError("incomplete evidence persistence config")
        limits = EvidenceFrameConfig(
            min(budget.config.max_frames, stored_config.max_frames),
            min(budget.config.max_frame_pixels, stored_config.max_frame_pixels),
            min(budget.config.max_total_bytes, stored_config.max_total_bytes),
            min(budget.config.max_runtime_seconds, stored_config.max_runtime_seconds),
        )
        budget.deadline = min(budget.deadline, budget.started + limits.max_runtime_seconds)
        budget.check()
        video = _manifest_video(manifest)
        original = QueryResult.from_dict(manifest["input_query_result"], video=video)
        if original.support_status is not SupportStatus.SUPPORTED:
            raise ValueError("persisted artifact requires positive input evidence")
        rows = manifest["frames"]
        if not isinstance(rows, list) or not 1 <= len(rows) <= limits.max_frames:
            raise ValueError("evidence artifact exceeds the frame cap")
        input_items = {item.evidence_id: item for item in original.evidence.items}
        assigned, names, identities = set(), set(), set()
        total = len(payload) + 65
        if total > limits.max_total_bytes:
            raise ValueError("evidence artifact exceeds the stored byte cap")
        for row in rows:
            budget.check()
            if set(row) != {"file", "frame_ref", "png_sha256", "rgb_sha256",
                            "png_size_bytes", "evidence_ids"}:
                raise ValueError("invalid evidence frame manifest fields")
            name = row["file"]
            if (not isinstance(name, str) or re.fullmatch(r"frames/frame_[0-9]{4}\.png", name)
                    is None or name in names):
                raise ValueError("unsafe or duplicate PNG member")
            names.add(name)
            ref = FrameRef(**row["frame_ref"])
            if row["frame_ref"] != _portable(ref) or ref.identity_key in identities:
                raise ValueError("invalid portable or duplicate frame reference")
            identities.add(ref.identity_key)
            _semantics(ref)
            if not isinstance(row["evidence_ids"], list) or not row["evidence_ids"]:
                raise ValueError("PNG must bind at least one measured evidence item")
            for identifier in row["evidence_ids"]:
                item = input_items.get(identifier)
                if (item is None or identifier in assigned
                        or _semantics(item.frame_ref) != _semantics(ref)
                        or item.provenance.get("frame_rgb_sha256") != row["rgb_sha256"]
                        or item.provenance.get("image_persisted")
                        or "image_artifact" in item.provenance):
                    raise ValueError("PNG bindings contradict original observed evidence")
                assigned.add(identifier)
            size = row["png_size_bytes"]
            if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
                raise ValueError("invalid PNG byte count")
            total += size
            if total > limits.max_total_bytes:
                raise ValueError("evidence artifact exceeds the byte cap")
            image = _read(final / name, min(size, limits.max_total_bytes), budget)
            if (len(image) != size
                    or hashlib.sha256(image).hexdigest() != _digest(row["png_sha256"])):
                raise ValueError("evidence PNG SHA-256 mismatch")
            rgb = _png_rgb(image, ref, limits)
            if hashlib.sha256(rgb.tobytes()).hexdigest() != _digest(row["rgb_sha256"]):
                raise ValueError("evidence decoded RGB SHA-256 mismatch")
        if assigned != set(input_items):
            raise ValueError("artifact omitted original evidence items")
        _safe(final / "frames", directory=True)
        if {f"frames/{name}" for name in _members(final / "frames", len(names), budget)} != names:
            raise ValueError("unexpected PNG members in evidence artifact")
        budget.check()
        return manifest, _artifact(digest, total)

    def verify(self, relative_path: str, expected_sha256: str, *, checkpoint=None,
               config: EvidenceFrameConfig | None = None) -> dict[str, Any]:
        return self._verify(relative_path, expected_sha256, _Budget(
            EvidenceFrameConfig() if config is None else config, checkpoint,
        ))[0]

    load = verify

    def _verify_result(self, result, budget):
        if not isinstance(result, QueryResult):
            raise TypeError("result must be a QueryResult")
        budget.check()
        result.validate_evidence()
        from video_intelligence.storage.evidence_masks import EvidenceMaskStore

        EvidenceMaskStore(self.root).verify_result(
            result, config=budget.config, checkpoint=budget.check,
        )
        budget.check()
        if result.support_status is not SupportStatus.SUPPORTED:
            if result.evidence and any(item.provenance.get("image_persisted")
                                       or item.provenance.get("artifact_persisted")
                                       or "image_artifact" in item.provenance
                                       for item in result.evidence.items):
                raise ValueError("unsupported result contains persisted image claims")
            return None
        policy = result.query_config.get("evidence_frames")
        if not isinstance(policy, dict):
            raise ValueError("supported result has no persisted evidence policy")
        manifest, artifact = self._verify(policy.get("relative_path"),
                                          policy.get("content_sha256"), budget)
        original = QueryResult.from_dict(manifest["input_query_result"],
                                         video=_manifest_video(manifest))
        expected = _updated_query(original, manifest, artifact)
        if result.to_dict() != expected.to_dict():
            raise ValueError("persisted image pointers or provenance differ from original manifest")
        return artifact

    def verify_result(self, result: QueryResult, *, checkpoint=None,
                      config: EvidenceFrameConfig | None = None) -> EvidenceFrameArtifact | None:
        return self._verify_result(result, _Budget(
            EvidenceFrameConfig() if config is None else config, checkpoint,
        ))
