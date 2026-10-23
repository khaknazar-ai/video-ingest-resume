"""Orchestration between immutable index snapshots and SQLite active pointers."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any

from video_intelligence.index.exact import ExactCosineIndex
from video_intelligence.index.snapshot import (
    LoadedSnapshot,
    SnapshotDescriptor,
    SnapshotIntegrityError,
    SnapshotStore,
)
from video_intelligence.storage.sqlite import SQLiteStore


def _relative_namespace(value: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ValueError("artifact_relative_root must be a non-empty relative path")
    path = PurePosixPath(value.replace("\\", "/"))
    if path.is_absolute() or ".." in path.parts or "." in path.parts:
        raise ValueError("artifact_relative_root must stay inside the artifact root")
    return str(path)


@dataclass(frozen=True, slots=True)
class ActiveIndex:
    index_name: str
    descriptor: SnapshotDescriptor
    index: ExactCosineIndex


class IndexRegistry:
    """Publish immutable snapshots, then atomically update their SQLite pointer."""

    def __init__(
        self,
        store: SQLiteStore,
        snapshots: SnapshotStore,
        *,
        artifact_relative_root: str | None = None,
    ) -> None:
        self.store = store
        self.snapshots = snapshots
        # With no separately configured artifact root, its only verifiable
        # location is the parent of the snapshot store's root.
        actual_namespace = _relative_namespace(snapshots.root.name)
        declared = (
            actual_namespace
            if artifact_relative_root is None
            else _relative_namespace(artifact_relative_root)
        )
        if declared != actual_namespace:
            raise ValueError("artifact_relative_root does not match snapshot root")
        self.artifact_relative_root = declared

    def publish_and_activate(
        self,
        index_name: str,
        index: ExactCosineIndex,
        *,
        attributes: dict[str, Any] | None = None,
        checkpoint: Callable[[], None] | None = None,
        lease_job_id: str | None = None,
        lease_owner: str | None = None,
        lease_token: str | None = None,
    ) -> ActiveIndex:
        """Publish the blob first; SQLite activation is the second, recoverable step."""

        if checkpoint is not None:
            checkpoint()
        descriptor = self.snapshots.publish(index, attributes=attributes)
        if checkpoint is not None:
            checkpoint()
        return self._activate_descriptor(
            index_name,
            descriptor,
            lease_job_id=lease_job_id,
            lease_owner=lease_owner,
            lease_token=lease_token,
        )

    def activate_existing(self, index_name: str, snapshot_id: str) -> ActiveIndex:
        loaded = self.snapshots.load(snapshot_id)
        return self._activate_loaded(index_name, loaded)

    def get_active(self, index_name: str) -> ActiveIndex | None:
        row = self.store.get_active_index(index_name)
        if row is None:
            return None
        loaded = self.snapshots.load(row["snapshot_id"])
        self._verify_pointer(row, loaded)
        return ActiveIndex(index_name=index_name, descriptor=loaded.descriptor, index=loaded.index)

    def _activate_descriptor(
        self,
        index_name: str,
        descriptor: SnapshotDescriptor,
        *,
        lease_job_id: str | None = None,
        lease_owner: str | None = None,
        lease_token: str | None = None,
    ) -> ActiveIndex:
        loaded = self.snapshots.load(descriptor.snapshot_id)
        return self._activate_loaded(
            index_name,
            loaded,
            lease_job_id=lease_job_id,
            lease_owner=lease_owner,
            lease_token=lease_token,
        )

    def _activate_loaded(
        self,
        index_name: str,
        loaded: LoadedSnapshot,
        *,
        lease_job_id: str | None = None,
        lease_owner: str | None = None,
        lease_token: str | None = None,
    ) -> ActiveIndex:
        descriptor = loaded.descriptor
        artifact_id = f"index_snapshot_{descriptor.snapshot_id}"
        relative_path = f"{self.artifact_relative_root}/{descriptor.snapshot_id}"
        row = self.store.activate_index_snapshot(
            index_name=index_name,
            snapshot_id=descriptor.snapshot_id,
            artifact_id=artifact_id,
            relative_path=relative_path,
            size_bytes=descriptor.size_bytes,
            matrix_sha256=descriptor.matrix_sha256,
            metadata_sha256=descriptor.metadata_sha256,
            rows=descriptor.rows,
            dimension=descriptor.dimension,
            lease_job_id=lease_job_id,
            lease_owner=lease_owner,
            lease_token=lease_token,
        )
        self._verify_pointer(row, loaded)
        return ActiveIndex(index_name=index_name, descriptor=descriptor, index=loaded.index)

    def _verify_pointer(self, row, loaded: LoadedSnapshot) -> None:
        descriptor = loaded.descriptor
        expected_relative_path = f"{self.artifact_relative_root}/{descriptor.snapshot_id}"
        if (
            row["snapshot_id"] != descriptor.snapshot_id
            or row["relative_path"] != expected_relative_path
            or row["size_bytes"] != descriptor.size_bytes
            or row["matrix_sha256"] != descriptor.matrix_sha256
            or row["metadata_sha256"] != descriptor.metadata_sha256
            or row["rows"] != descriptor.rows
            or row["dimension"] != descriptor.dimension
        ):
            raise SnapshotIntegrityError("SQLite pointer does not match snapshot content")
