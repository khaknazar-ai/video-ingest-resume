"""Embedding provenance, rows, queries, and exact-index integration."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

import numpy as np

from video_intelligence.embeddings.normalization import normalize_matrix, normalize_vector
from video_intelligence.index.exact import ExactCosineIndex, SearchHit
from video_intelligence.index.snapshot import LoadedSnapshot
from video_intelligence.models.contracts import ModelMode, ModelSpec

EMBEDDING_FINGERPRINT_METADATA_KEY = "embedding_fingerprint_id"
EMBEDDING_SCHEMA_VERSION = "0.1"


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _identifier(value: object, name: str) -> str:
    result = _text(value, name)
    if any(character.isspace() for character in result):
        raise ValueError(f"{name} must not contain whitespace")
    return result


def _sha256(value: object, name: str) -> str | None:
    if value is None:
        return None
    if (
        not isinstance(value, str)
        or len(value) != 64
        or value != value.lower()
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 hex digest or null")
    return value


def _dimension(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 < value <= 1_000_000:
        raise ValueError("dimension must be an integer in [1, 1000000]")
    return value


def _canonical_json(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError("embedding provenance must be finite JSON") from exc


def _optional_nonnegative_float(value: object, name: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite non-negative number or null")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be a finite non-negative number or null")
    return result


class EmbeddingModality(StrEnum):
    """The two modalities produced by a joint image/text embedding model."""

    IMAGE = "image"
    TEXT = "text"


@dataclass(frozen=True, slots=True)
class EmbeddingFingerprint:
    """Immutable model and processor identity for one embedding space."""

    model_id: str
    revision: str
    processor_revision: str
    modality: EmbeddingModality | str
    dimension: int
    weights_sha256: str | None = None
    model_mode: ModelMode | str | None = None

    def __post_init__(self) -> None:
        for name in ("model_id", "processor_revision"):
            object.__setattr__(self, name, _identifier(getattr(self, name), name))
        object.__setattr__(self, "revision", _text(self.revision, "revision"))
        object.__setattr__(
            self,
            "weights_sha256",
            _sha256(self.weights_sha256, "weights_sha256"),
        )
        object.__setattr__(self, "dimension", _dimension(self.dimension))
        if self.model_mode is not None:
            object.__setattr__(self, "model_mode", ModelMode(self.model_mode))
        try:
            object.__setattr__(self, "modality", EmbeddingModality(self.modality))
        except (TypeError, ValueError) as error:
            raise ValueError(f"unsupported embedding modality: {self.modality!r}") from error

    @classmethod
    def from_spec(
        cls,
        spec: ModelSpec,
        *,
        modality: EmbeddingModality | str,
        dimension: int,
        processor_revision: str | None = None,
    ) -> EmbeddingFingerprint:
        if not isinstance(spec, ModelSpec):
            raise TypeError("spec must be a ModelSpec")
        return cls(
            model_id=spec.model_id,
            revision=spec.revision,
            processor_revision=spec.revision if processor_revision is None else processor_revision,
            modality=modality,
            dimension=dimension,
            weights_sha256=spec.weights_sha256,
            model_mode=spec.mode,
        )

    def _identity_dict(self) -> dict[str, object]:
        return {
            "model_id": self.model_id,
            "revision": self.revision,
            "processor_revision": self.processor_revision,
            "modality": self.modality.value,
            "dimension": self.dimension,
            "weights_sha256": self.weights_sha256,
            "model_mode": None if self.model_mode is None else self.model_mode.value,
        }

    @property
    def fingerprint_id(self) -> str:
        return hashlib.sha256(_canonical_json(self._identity_dict())).hexdigest()

    def is_compatible_query(self, other: object) -> bool:
        """Return whether two modalities share one joint embedding space."""

        if not isinstance(other, EmbeddingFingerprint):
            return False
        return (
            self.model_id == other.model_id
            and self.revision == other.revision
            and self.processor_revision == other.processor_revision
            and self.dimension == other.dimension
            and self.weights_sha256 == other.weights_sha256
            and self.model_mode == other.model_mode
            and {self.modality, other.modality}
            <= {EmbeddingModality.IMAGE, EmbeddingModality.TEXT}
        )

    def to_dict(self) -> dict[str, object]:
        return {
            **self._identity_dict(),
            "fingerprint_id": self.fingerprint_id,
        }


@dataclass(frozen=True, slots=True)
class EmbeddingRow:
    """One row identity and filterable metadata for an embedding matrix."""

    row_id: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "row_id", _identifier(self.row_id, "row_id"))
        if not isinstance(self.metadata, Mapping):
            raise ValueError("metadata must be a mapping")
        copied = dict(self.metadata)
        _canonical_json(copied)
        object.__setattr__(self, "metadata", copied)


@dataclass(frozen=True, slots=True)
class EmbeddingQuery:
    """A normalized query vector that cannot silently cross embedding spaces."""

    fingerprint: EmbeddingFingerprint
    vector: np.ndarray

    def __post_init__(self) -> None:
        if not isinstance(self.fingerprint, EmbeddingFingerprint):
            raise TypeError("fingerprint must be an EmbeddingFingerprint")
        normalized = normalize_vector(
            self.vector,
            expected_dimension=self.fingerprint.dimension,
        )
        object.__setattr__(self, "vector", normalized)


def _metadata_with_provenance(
    row: EmbeddingRow,
    fingerprint: EmbeddingFingerprint,
) -> dict[str, Any]:
    metadata = dict(row.metadata)
    expected = {
        EMBEDDING_FINGERPRINT_METADATA_KEY: fingerprint.fingerprint_id,
        "embedding_model_id": fingerprint.model_id,
        "embedding_revision": fingerprint.revision,
        "embedding_processor_revision": fingerprint.processor_revision,
        "embedding_modality": fingerprint.modality.value,
        "embedding_dimension": fingerprint.dimension,
        "embedding_model_mode": (
            None if fingerprint.model_mode is None else fingerprint.model_mode.value
        ),
    }
    for key, value in expected.items():
        if key in metadata and metadata[key] != value:
            raise ValueError(f"metadata field {key!r} conflicts with embedding fingerprint")
        metadata[key] = value
    _canonical_json(metadata)
    return metadata


class EmbeddingIndex:
    """Exact cosine search over rows from one verified embedding space."""

    def __init__(
        self,
        fingerprint: EmbeddingFingerprint,
        vectors: object,
        rows: Sequence[EmbeddingRow],
        *,
        block_size: int = 4096,
    ) -> None:
        if not isinstance(fingerprint, EmbeddingFingerprint):
            raise TypeError("fingerprint must be an EmbeddingFingerprint")
        if isinstance(rows, (str, bytes)) or not isinstance(rows, Sequence):
            raise ValueError("rows must be a sequence of EmbeddingRow values")
        records = tuple(rows)
        if any(not isinstance(row, EmbeddingRow) for row in records):
            raise ValueError("rows must contain only EmbeddingRow values")
        normalized = normalize_matrix(vectors, expected_dimension=fingerprint.dimension)
        if len(records) != normalized.shape[0]:
            raise ValueError("rows length must match vectors rows")
        metadata = tuple(_metadata_with_provenance(row, fingerprint) for row in records)
        self._fingerprint = fingerprint
        self._rows = tuple(
            EmbeddingRow(row_id=row.row_id, metadata=metadata[index])
            for index, row in enumerate(records)
        )
        self._exact = ExactCosineIndex(
            normalized,
            row_ids=tuple(row.row_id for row in self._rows),
            metadata=metadata,
            block_size=block_size,
        )

    @property
    def fingerprint(self) -> EmbeddingFingerprint:
        return self._fingerprint

    @property
    def rows(self) -> tuple[EmbeddingRow, ...]:
        return self._rows

    @property
    def dimension(self) -> int:
        return self._exact.dimension

    @property
    def matrix(self) -> np.ndarray:
        return self._exact.matrix

    def as_exact_index(self) -> ExactCosineIndex:
        """Expose the validated index to the existing snapshot store."""

        return self._exact

    def snapshot_attributes(self) -> dict[str, object]:
        return {
            "embedding_schema_version": EMBEDDING_SCHEMA_VERSION,
            "embedding_fingerprint": self.fingerprint.to_dict(),
            "storage_dtype": "float32",
            "normalized": True,
        }

    @classmethod
    def from_loaded_snapshot(
        cls,
        loaded: LoadedSnapshot,
        fingerprint: EmbeddingFingerprint,
    ) -> EmbeddingIndex:
        if not isinstance(loaded, LoadedSnapshot):
            raise TypeError("loaded must be a LoadedSnapshot")
        if not isinstance(fingerprint, EmbeddingFingerprint):
            raise TypeError("fingerprint must be an EmbeddingFingerprint")
        attributes = loaded.descriptor.attributes
        if (
            attributes.get("embedding_schema_version") != EMBEDDING_SCHEMA_VERSION
            or attributes.get("embedding_fingerprint") != fingerprint.to_dict()
        ):
            raise ValueError("snapshot embedding fingerprint does not match requested space")
        return cls.from_exact_index(fingerprint, loaded.index)

    @classmethod
    def from_exact_index(
        cls,
        fingerprint: EmbeddingFingerprint,
        index: ExactCosineIndex,
    ) -> EmbeddingIndex:
        if not isinstance(index, ExactCosineIndex):
            raise TypeError("index must be an ExactCosineIndex")
        if index.dimension != fingerprint.dimension:
            raise ValueError("index dimension does not match embedding fingerprint")
        rows = []
        for row_id, metadata in zip(index.row_ids, index.metadata, strict=True):
            if metadata.get(EMBEDDING_FINGERPRINT_METADATA_KEY) != fingerprint.fingerprint_id:
                raise ValueError("index row fingerprint does not match requested space")
            rows.append(EmbeddingRow(row_id=row_id, metadata=metadata))
        return cls(fingerprint, index.matrix, rows)

    def search(
        self,
        query: EmbeddingQuery | object,
        *,
        top_k: int,
        video_id: str | None = None,
        start_seconds: float | None = None,
        end_seconds: float | None = None,
        shot_id: str | None = None,
        metadata_filter: Callable[[Mapping[str, Any]], bool] | None = None,
    ) -> list[SearchHit]:
        if isinstance(query, EmbeddingQuery):
            if (
                query.fingerprint != self.fingerprint
                and not self.fingerprint.is_compatible_query(query.fingerprint)
            ):
                raise ValueError("query fingerprint does not match index fingerprint")
            vector: object = query.vector
        else:
            vector = normalize_vector(query, expected_dimension=self.dimension)
        video_id = None if video_id is None else _text(video_id, "video_id")
        shot_id = None if shot_id is None else _text(shot_id, "shot_id")
        start = _optional_nonnegative_float(start_seconds, "start_seconds")
        end = _optional_nonnegative_float(end_seconds, "end_seconds")
        if start is not None and end is not None and end < start:
            raise ValueError("end_seconds must be greater than or equal to start_seconds")
        if metadata_filter is not None and not callable(metadata_filter):
            raise ValueError("metadata_filter must be callable or null")

        def eligible(metadata: Mapping[str, Any]) -> bool:
            if video_id is not None and metadata.get("video_id") != video_id:
                return False
            if shot_id is not None and metadata.get("shot_id") != shot_id:
                return False
            if start is not None or end is not None:
                timestamp = metadata.get("timestamp_seconds")
                if (
                    isinstance(timestamp, bool)
                    or not isinstance(timestamp, (int, float))
                    or not math.isfinite(float(timestamp))
                ):
                    return False
                timestamp_value = float(timestamp)
                if start is not None and timestamp_value < start:
                    return False
                if end is not None and timestamp_value >= end:
                    return False
            return metadata_filter is None or metadata_filter(metadata)

        return self._exact.search(vector, top_k=top_k, metadata_filter=eligible)
