"""Reproducible video manifest creation and atomic persistence."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from video_intelligence.core.types import VideoRecord
from video_intelligence.video.probe import probe_video

SCHEMA_VERSION = "0.1"
MAX_HASH_CHUNK_SIZE = 16 * 1024 * 1024


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant is not allowed: {value}")


def _validated_sha256(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or value != value.lower()
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 hex digest")
    return value


def sha256_file(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    """Hash a source video incrementally without loading it into memory."""

    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int):
        raise ValueError("chunk_size must be an integer")
    if not 1 <= chunk_size <= MAX_HASH_CHUNK_SIZE:
        raise ValueError(f"chunk_size must be in [1, {MAX_HASH_CHUNK_SIZE}]")
    source = Path(path).expanduser().resolve()
    before = source.stat()
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    after = source.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError(f"source changed while hashing: {source}")
    return digest.hexdigest()


def build_video_manifest(path: str | Path, *, video_id: str | None = None) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    source_size = source.stat().st_size
    source_hash = sha256_file(source)
    record = VideoRecord(
        video_id=video_id or f"vid_{source_hash[:12]}",
        source_path=source,
        metadata=probe_video(source),
        source_sha256=source_hash,
    )
    if sha256_file(source) != source_hash:
        raise RuntimeError(f"source changed while building manifest: {source}")
    return {
        "schema_version": SCHEMA_VERSION,
        "video": record.to_dict(),
        "integrity": {"source_sha256": source_hash, "source_size_bytes": source_size},
    }


def write_manifest(manifest: dict[str, Any], path: str | Path, *, overwrite: bool = False) -> Path:
    """Publish JSON atomically, creating a new path unless overwrite is explicit."""

    destination = Path(path).expanduser().resolve()
    if destination.exists() and not overwrite:
        raise FileExistsError(f"manifest already exists; pass overwrite=True: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        if overwrite:
            os.replace(temporary_name, destination)
        else:
            # Same-filesystem hard-link publication is atomic create-if-absent.
            # An unsupported filesystem must fail, never degrade to exists()+replace.
            os.link(temporary_name, destination)
            os.unlink(temporary_name)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise
    return destination


def load_manifest(path: str | Path) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    payload = json.loads(
        source.read_text(encoding="utf-8"), parse_constant=_reject_json_constant
    )
    if not isinstance(payload, dict):
        raise ValueError("manifest must be a JSON object")
    expected_keys = {"schema_version", "video", "integrity"}
    unknown = set(payload) - expected_keys
    if unknown:
        raise ValueError(f"manifest contains unknown fields: {sorted(unknown)!r}")
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"Unsupported manifest schema: {payload.get('schema_version')!r}")
    if "video" not in payload or "integrity" not in payload:
        raise ValueError("manifest must contain video and integrity")
    record = VideoRecord.from_dict(payload["video"])
    integrity = payload["integrity"]
    if not isinstance(integrity, dict):
        raise ValueError("manifest integrity must be an object")
    integrity_unknown = set(integrity) - {"source_sha256", "source_size_bytes"}
    if integrity_unknown:
        raise ValueError(f"integrity contains unknown fields: {sorted(integrity_unknown)!r}")
    if "source_sha256" not in integrity:
        raise ValueError("integrity.source_sha256 is required")
    integrity_hash = _validated_sha256(integrity["source_sha256"], "integrity.source_sha256")
    if record.source_sha256 != integrity_hash:
        raise ValueError("manifest source hash copies do not match")
    size = integrity.get("source_size_bytes")
    if size is not None and (
        isinstance(size, bool) or not isinstance(size, int) or size < 0
    ):
        raise ValueError("integrity.source_size_bytes must be a non-negative integer")
    if record.source_path.exists() and not record.source_path.is_file():
        raise ValueError(f"manifest source_path is not a file: {record.source_path}")
    if record.source_path.is_file():
        actual_size = record.source_path.stat().st_size
        if size is not None and actual_size != size:
            raise ValueError("manifest source size does not match source file")
        if sha256_file(record.source_path) != integrity_hash:
            raise ValueError("manifest source hash does not match source file")
    return payload
