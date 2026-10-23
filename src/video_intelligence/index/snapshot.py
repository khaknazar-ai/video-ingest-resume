"""Content-addressed, immutable publication of exact index snapshots."""

from __future__ import annotations

import hashlib
import io
import json
import os
import shutil
import stat
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from video_intelligence.index.exact import ExactCosineIndex

FORMAT_VERSION = "0.1"
MAX_READY_BYTES = 4096
MAX_METADATA_BYTES = 16 * 1024 * 1024
MAX_MATRIX_BYTES = 256 * 1024 * 1024


class SnapshotIntegrityError(ValueError):
    """Raised when a snapshot is missing, incomplete, or tampered with."""


def _reject_links(path: Path) -> None:
    for part in reversed((path, *path.parents)):
        try:
            info = part.lstat()
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(info.st_mode) or (
            getattr(info, "st_file_attributes", 0)
            & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
        ):
            raise SnapshotIntegrityError("snapshot path contains a link or reparse point")


def _bounded_read(path: Path, limit: int) -> bytes:
    _reject_links(path)
    with path.open("rb") as handle:
        info = os.fstat(handle.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_size > limit:
            raise SnapshotIntegrityError("snapshot member exceeds its size limit")
        value = handle.read(limit + 1)
    if len(value) > limit:
        raise SnapshotIntegrityError("snapshot member exceeds its size limit")
    return value


def _validate_matrix_header(value: bytes, metadata: dict[str, Any]) -> None:
    """Validate dimensions and payload length before NumPy allocates an array."""
    stream = io.BytesIO(value)
    version = np.lib.format.read_magic(stream)
    if version == (1, 0):
        shape, _, dtype = np.lib.format.read_array_header_1_0(stream)
    elif version == (2, 0):
        shape, _, dtype = np.lib.format.read_array_header_2_0(stream)
    else:
        raise ValueError("unsupported snapshot matrix format")
    rows, dimension = metadata.get("rows"), metadata.get("dimension")
    if (
        type(rows) is not int or type(dimension) is not int
        or rows <= 0 or dimension <= 0 or shape != (rows, dimension)
        or dtype != np.dtype(np.float32)
        or rows * dimension * dtype.itemsize != len(value) - stream.tell()
        or not isinstance(metadata.get("row_ids"), list)
        or len(metadata["row_ids"]) != rows
        or not isinstance(metadata.get("metadata"), list)
        or len(metadata["metadata"]) != rows
        or not isinstance(metadata.get("attributes"), dict)
    ):
        raise ValueError("snapshot matrix header or payload length is invalid")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _json_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError("snapshot metadata must be finite JSON") from exc


def _digest_id(matrix_bytes: bytes, metadata_bytes: bytes) -> str:
    return _sha256_bytes(matrix_bytes + metadata_bytes)


def _validate_snapshot_id(snapshot_id: object) -> str:
    if (
        not isinstance(snapshot_id, str)
        or len(snapshot_id) != 64
        or snapshot_id != snapshot_id.lower()
        or any(character not in "0123456789abcdef" for character in snapshot_id)
    ):
        raise ValueError("snapshot_id must be a lowercase SHA-256 hex digest")
    return snapshot_id


def _write_bytes(path: Path, value: bytes) -> None:
    with path.open("wb") as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())


@dataclass(frozen=True, slots=True)
class SnapshotDescriptor:
    snapshot_id: str
    path: Path
    matrix_sha256: str
    metadata_sha256: str
    rows: int
    dimension: int
    size_bytes: int
    attributes: dict[str, Any]


@dataclass(frozen=True, slots=True)
class LoadedSnapshot:
    descriptor: SnapshotDescriptor
    index: ExactCosineIndex


class SnapshotStore:
    """Publish and load snapshots under a root directory.

    A snapshot is active only when its fully validated temporary directory has been
    atomically renamed into the content-addressed final directory. Incomplete writes
    remain unaddressable and are never returned by ``load``.
    """

    def __init__(self, root: str | Path) -> None:
        candidate = Path(root).expanduser().absolute()
        _reject_links(candidate)
        self.root = candidate.resolve()

    def publish(
        self,
        index: ExactCosineIndex,
        *,
        attributes: dict[str, Any] | None = None,
    ) -> SnapshotDescriptor:
        if not isinstance(index, ExactCosineIndex):
            raise TypeError("index must be an ExactCosineIndex")
        if attributes is not None and not isinstance(attributes, dict):
            raise ValueError("attributes must be a JSON object")
        snapshot_attributes = dict(attributes or {})
        matrix = index.matrix
        _reject_links(self.root)
        if matrix.nbytes + 1024 > MAX_MATRIX_BYTES:
            raise ValueError("snapshot matrix exceeds its size limit")
        self.root.mkdir(parents=True, exist_ok=True)
        temporary_path: Path | None = Path(tempfile.mkdtemp(prefix=".snapshot-", dir=self.root))
        try:
            assert temporary_path is not None
            matrix_path = temporary_path / "matrix.npy"
            with matrix_path.open("wb") as handle:
                np.save(handle, matrix, allow_pickle=False)
                handle.flush()
                os.fsync(handle.fileno())
            matrix_bytes = matrix_path.read_bytes()
            matrix_sha256 = _sha256_bytes(matrix_bytes)
            metadata_payload = {
                "format_version": FORMAT_VERSION,
                "matrix_sha256": matrix_sha256,
                "rows": index.rows,
                "dimension": index.dimension,
                "row_ids": list(index.row_ids),
                "metadata": index.metadata,
                "attributes": snapshot_attributes,
            }
            metadata_bytes = _json_bytes(metadata_payload)
            if len(metadata_bytes) > MAX_METADATA_BYTES:
                raise ValueError("snapshot metadata exceeds its size limit")
            metadata_sha256 = _sha256_bytes(metadata_bytes)
            _write_bytes(temporary_path / "metadata.json", metadata_bytes)
            snapshot_id = _digest_id(matrix_bytes, metadata_bytes)
            ready_bytes = _json_bytes(
                {
                    "format_version": FORMAT_VERSION,
                    "snapshot_id": snapshot_id,
                    "matrix_sha256": matrix_sha256,
                    "metadata_sha256": metadata_sha256,
                }
            )
            _write_bytes(temporary_path / "READY", ready_bytes)
            final_path = self.root / snapshot_id
            if final_path.exists():
                loaded = self.load(snapshot_id)
                if loaded.descriptor.attributes != snapshot_attributes:
                    raise SnapshotIntegrityError("existing snapshot has different attributes")
                return loaded.descriptor
            try:
                os.replace(str(temporary_path), str(final_path))
            except OSError as error:
                # On Windows, a concurrent publisher can win the directory
                # create race and make replace fail. Only convert that race
                # into success after validating the winner's complete content
                # and reserved attributes; unrelated I/O failures remain
                # failures.
                if not final_path.exists():
                    raise
                try:
                    loaded = self.load(snapshot_id)
                except SnapshotIntegrityError as integrity_error:
                    raise integrity_error from error
                if loaded.descriptor.attributes != snapshot_attributes:
                    raise SnapshotIntegrityError(
                        "existing snapshot has different attributes"
                    ) from error
                return loaded.descriptor
            temporary_path = None
            return self.load(snapshot_id).descriptor
        finally:
            if temporary_path is not None and temporary_path.exists():
                shutil.rmtree(temporary_path, ignore_errors=True)

    def load(self, snapshot_id: str) -> LoadedSnapshot:
        snapshot_id = _validate_snapshot_id(snapshot_id)
        snapshot_path = self.root / snapshot_id
        _reject_links(snapshot_path)
        if not snapshot_path.is_dir():
            raise SnapshotIntegrityError("snapshot directory does not exist")
        ready_path = snapshot_path / "READY"
        matrix_path = snapshot_path / "matrix.npy"
        metadata_path = snapshot_path / "metadata.json"
        if not ready_path.is_file() or not matrix_path.is_file() or not metadata_path.is_file():
            raise SnapshotIntegrityError("snapshot is incomplete and is not active")
        try:
            ready = json.loads(_bounded_read(ready_path, MAX_READY_BYTES).decode("utf-8"))
            metadata_bytes = _bounded_read(metadata_path, MAX_METADATA_BYTES)
            metadata = json.loads(metadata_bytes.decode("utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise SnapshotIntegrityError("snapshot JSON is unreadable") from exc
        try:
            matrix_bytes = _bounded_read(matrix_path, MAX_MATRIX_BYTES)
        except OSError as exc:
            raise SnapshotIntegrityError("snapshot matrix is unreadable") from exc
        matrix_sha256 = _sha256_bytes(matrix_bytes)
        metadata_sha256 = _sha256_bytes(metadata_bytes)
        if not isinstance(ready, dict) or not isinstance(metadata, dict):
            raise SnapshotIntegrityError("snapshot metadata must be JSON objects")
        if (
            ready.get("format_version") != FORMAT_VERSION
            or ready.get("snapshot_id") != snapshot_id
            or ready.get("matrix_sha256") != matrix_sha256
            or ready.get("metadata_sha256") != metadata_sha256
            or metadata.get("format_version") != FORMAT_VERSION
            or metadata.get("matrix_sha256") != matrix_sha256
        ):
            raise SnapshotIntegrityError("snapshot hash or format validation failed")
        if _digest_id(matrix_bytes, metadata_bytes) != snapshot_id:
            raise SnapshotIntegrityError("snapshot id does not match its content")
        try:
            # Decode the exact bytes whose digest was validated above. Reopening
            # matrix.npy would permit a replacement between verification and load.
            _validate_matrix_header(matrix_bytes, metadata)
            matrix = np.load(io.BytesIO(matrix_bytes), allow_pickle=False)
            index = ExactCosineIndex(
                matrix,
                row_ids=metadata["row_ids"],
                metadata=metadata["metadata"],
            )
            rows = metadata["rows"]
            dimension = metadata["dimension"]
            attributes = metadata["attributes"]
            if (
                not isinstance(rows, int)
                or not isinstance(dimension, int)
                or not isinstance(attributes, dict)
                or index.rows != rows
                or index.dimension != dimension
            ):
                raise ValueError("snapshot shape or attributes are invalid")
        except (OSError, ValueError, TypeError, KeyError, EOFError) as exc:
            raise SnapshotIntegrityError("snapshot matrix or metadata is invalid") from exc
        descriptor = SnapshotDescriptor(
            snapshot_id=snapshot_id,
            path=snapshot_path,
            matrix_sha256=matrix_sha256,
            metadata_sha256=metadata_sha256,
            rows=rows,
            dimension=dimension,
            size_bytes=sum(
                (snapshot_path / filename).stat().st_size
                for filename in ("matrix.npy", "metadata.json", "READY")
            ),
            attributes=dict(attributes),
        )
        return LoadedSnapshot(descriptor=descriptor, index=index)
