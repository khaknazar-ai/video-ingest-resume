"""Bounded embedding-index construction from typed model results."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from video_intelligence.embeddings.base import EmbeddingFingerprint, EmbeddingIndex, EmbeddingRow
from video_intelligence.index.snapshot import SnapshotDescriptor, SnapshotStore
from video_intelligence.models.contracts import ModelMode, ModelResult


def _positive_count(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


@dataclass(frozen=True, slots=True)
class EmbeddingBuildBatch:
    """One bounded model result paired with the rows for its inputs."""

    result: ModelResult[Any]
    rows: tuple[EmbeddingRow, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.result, ModelResult):
            raise TypeError("result must be a ModelResult")
        if isinstance(self.rows, (str, bytes)) or not isinstance(self.rows, Sequence):
            raise ValueError("rows must be a sequence of EmbeddingRow values")
        rows = tuple(self.rows)
        if not rows or any(not isinstance(row, EmbeddingRow) for row in rows):
            raise ValueError("rows must contain at least one EmbeddingRow")
        object.__setattr__(self, "rows", rows)
        if len(self.result.outputs) != len(rows):
            raise ValueError("model output count must match rows")
        if self.result.measurement.input_count != len(rows):
            raise ValueError("model measurement input_count must match rows")


@dataclass(frozen=True, slots=True)
class EmbeddingBuildManifest:
    """Reproducibility metadata for one in-memory build or published snapshot."""

    fingerprint: EmbeddingFingerprint
    row_count: int
    batch_count: int
    model_mode: ModelMode

    def __post_init__(self) -> None:
        if not isinstance(self.fingerprint, EmbeddingFingerprint):
            raise TypeError("fingerprint must be an EmbeddingFingerprint")
        object.__setattr__(self, "row_count", _positive_count(self.row_count, "row_count"))
        object.__setattr__(
            self,
            "batch_count",
            _positive_count(self.batch_count, "batch_count"),
        )
        try:
            model_mode = ModelMode(self.model_mode)
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"unsupported embedding build model mode: {self.model_mode!r}"
            ) from error
        if self.fingerprint.model_mode is None:
            raise ValueError("embedding fingerprint model_mode is required for a build manifest")
        if self.fingerprint.model_mode is not model_mode:
            raise ValueError("embedding build manifest model_mode does not match fingerprint")
        object.__setattr__(self, "model_mode", model_mode)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": "0.1",
            "fingerprint": self.fingerprint.to_dict(),
            "row_count": self.row_count,
            "batch_count": self.batch_count,
            "model_mode": self.model_mode.value,
        }


def _validate_result_provenance(
    result: ModelResult[Any],
    fingerprint: EmbeddingFingerprint,
) -> ModelMode:
    try:
        result_mode = ModelMode(result.mode)
    except (TypeError, ValueError) as error:
        raise ValueError(f"model result has an unsupported mode: {result.mode!r}") from error
    if result.model_id != fingerprint.model_id:
        raise ValueError("model result model_id does not match embedding fingerprint")
    provenance = result.provenance
    if not isinstance(provenance, dict):
        raise ValueError("model result provenance must be a JSON object")
    if "mode" not in provenance:
        raise ValueError("model result provenance mode is required")
    try:
        provenance_mode = ModelMode(provenance["mode"])
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"model result provenance has an unsupported mode: {provenance['mode']!r}"
        ) from error
    if provenance_mode is not result_mode:
        raise ValueError("model result mode does not match provenance mode")
    if provenance.get("model_id") != fingerprint.model_id:
        raise ValueError("model result provenance model_id does not match fingerprint")
    if provenance.get("revision") != fingerprint.revision:
        raise ValueError("model result revision does not match embedding fingerprint")
    if (
        fingerprint.weights_sha256 is not None
        and provenance.get("weights_sha256") != fingerprint.weights_sha256
    ):
        raise ValueError("model result weights hash does not match embedding fingerprint")
    if fingerprint.model_mode is None:
        raise ValueError("embedding fingerprint model_mode is required at build boundary")
    if fingerprint.model_mode is not result_mode:
        raise ValueError("model result mode does not match embedding fingerprint model_mode")
    return result_mode


def build_embedding_index(
    fingerprint: EmbeddingFingerprint,
    batches: Sequence[EmbeddingBuildBatch],
    *,
    block_size: int = 4096,
) -> tuple[EmbeddingIndex, EmbeddingBuildManifest]:
    """Build one index from bounded, provenance-checked model result batches."""

    if not isinstance(fingerprint, EmbeddingFingerprint):
        raise TypeError("fingerprint must be an EmbeddingFingerprint")
    if isinstance(batches, (str, bytes)) or not isinstance(batches, Sequence):
        raise ValueError("batches must be a sequence of EmbeddingBuildBatch values")
    batch_values = tuple(batches)
    if not batch_values or any(
        not isinstance(batch, EmbeddingBuildBatch) for batch in batch_values
    ):
        raise ValueError("batches must contain at least one EmbeddingBuildBatch")
    model_mode: ModelMode | None = None
    all_rows: list[EmbeddingRow] = []
    all_vectors: list[object] = []
    seen_row_ids: set[str] = set()
    for batch in batch_values:
        result_mode = _validate_result_provenance(batch.result, fingerprint)
        if model_mode is None:
            model_mode = result_mode
        elif result_mode is not model_mode:
            raise ValueError("all model result batches must use the same mode")
        for row, vector in zip(batch.rows, batch.result.outputs, strict=True):
            if row.row_id in seen_row_ids:
                raise ValueError("embedding row_ids must be unique across batches")
            seen_row_ids.add(row.row_id)
            all_rows.append(row)
            all_vectors.append(vector)
    assert model_mode is not None
    index = EmbeddingIndex(
        fingerprint,
        all_vectors,
        all_rows,
        block_size=block_size,
    )
    return index, EmbeddingBuildManifest(
        fingerprint=fingerprint,
        row_count=len(all_rows),
        batch_count=len(batch_values),
        model_mode=model_mode,
    )


def publish_embedding_snapshot(
    snapshots: SnapshotStore,
    fingerprint: EmbeddingFingerprint,
    batches: Sequence[EmbeddingBuildBatch],
    *,
    block_size: int = 4096,
    attributes: dict[str, object] | None = None,
) -> tuple[SnapshotDescriptor, EmbeddingIndex, EmbeddingBuildManifest]:
    """Build and atomically publish an embedding snapshot with its manifest."""

    if not isinstance(snapshots, SnapshotStore):
        raise TypeError("snapshots must be a SnapshotStore")
    index, manifest = build_embedding_index(
        fingerprint,
        batches,
        block_size=block_size,
    )
    snapshot_attributes = index.snapshot_attributes()
    snapshot_attributes["embedding_build"] = manifest.to_dict()
    if attributes is not None:
        if not isinstance(attributes, dict):
            raise ValueError("attributes must be a JSON object")
        collisions = set(snapshot_attributes).intersection(attributes)
        if collisions:
            raise ValueError(
                "attributes cannot override reserved snapshot attributes: "
                + ", ".join(sorted(collisions))
            )
        snapshot_attributes.update(attributes)
    descriptor = snapshots.publish(
        index.as_exact_index(),
        attributes=snapshot_attributes,
    )
    return descriptor, index, manifest
