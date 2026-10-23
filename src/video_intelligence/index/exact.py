"""Small, deterministic exact cosine index backed by normalized NumPy arrays."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from math import isfinite
from typing import Any

import numpy as np

Metadata = Mapping[str, Any]
MetadataFilter = Callable[[Metadata], bool]


def _normalise_matrix(matrix: object) -> np.ndarray:
    try:
        raw = np.asarray(matrix)
    except (TypeError, ValueError) as exc:
        raise ValueError("matrix must be a numeric two-dimensional array") from exc
    if raw.ndim != 2:
        raise ValueError("matrix must be two-dimensional")
    if raw.shape[0] == 0 or raw.shape[1] == 0:
        raise ValueError("matrix must contain at least one row and one dimension")
    if not np.issubdtype(raw.dtype, np.number) or np.issubdtype(raw.dtype, np.complexfloating):
        raise ValueError("matrix must contain real numeric values")
    try:
        values = raw.astype(np.float64, copy=False)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("matrix must contain real numeric values") from exc
    if not np.isfinite(values).all():
        raise ValueError("matrix must contain only finite values")
    norms = np.linalg.norm(values, axis=1)
    if not np.isfinite(norms).all() or np.any(norms <= 0.0):
        raise ValueError("matrix rows must have finite non-zero norms")
    normalized = (values / norms[:, None]).astype(np.float32)
    if not np.isfinite(normalized).all():
        raise ValueError("normalized matrix must contain only finite values")
    normalized.setflags(write=False)
    return normalized


def _normalise_query(query: object, dimension: int) -> np.ndarray:
    try:
        raw = np.asarray(query)
    except (TypeError, ValueError) as exc:
        raise ValueError("query must be a real one-dimensional array") from exc
    if raw.ndim != 1 or raw.shape[0] != dimension:
        raise ValueError(f"query must have dimension {dimension}")
    if not np.issubdtype(raw.dtype, np.number) or np.issubdtype(raw.dtype, np.complexfloating):
        raise ValueError("query must contain real numeric values")
    try:
        values = raw.astype(np.float64, copy=False)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("query must contain real numeric values") from exc
    if not np.isfinite(values).all():
        raise ValueError("query must contain only finite values")
    norm = float(np.linalg.norm(values))
    if not isfinite(norm) or norm <= 0.0:
        raise ValueError("query must have a finite non-zero norm")
    normalized = (values / norm).astype(np.float32)
    if not np.isfinite(normalized).all():
        raise ValueError("normalized query must contain only finite values")
    return normalized


def _validate_row_ids(row_ids: Sequence[str] | None, rows: int) -> tuple[str, ...]:
    values = (
        tuple(f"row_{index:08d}" for index in range(rows))
        if row_ids is None
        else tuple(row_ids)
    )
    if len(values) != rows:
        raise ValueError("row_ids length must match matrix rows")
    if any(not isinstance(value, str) or not value or value.strip() != value for value in values):
        raise ValueError("row_ids must be non-empty strings without surrounding whitespace")
    if len(set(values)) != len(values):
        raise ValueError("row_ids must be unique")
    return values


def _validate_metadata(
    metadata: Sequence[Metadata] | None, rows: int
) -> tuple[dict[str, Any], ...]:
    values = [{} for _ in range(rows)] if metadata is None else list(metadata)
    if len(values) != rows:
        raise ValueError("metadata length must match matrix rows")
    result: list[dict[str, Any]] = []
    for value in values:
        if not isinstance(value, Mapping):
            raise ValueError("each metadata row must be a mapping")
        try:
            result.append(deepcopy(dict(value)))
        except (TypeError, ValueError) as exc:
            raise ValueError("metadata must be deepcopyable") from exc
    return tuple(result)


@dataclass(frozen=True, slots=True)
class SearchHit:
    """One exact search result, ordered by score then stable row identity."""

    row_id: str
    row_index: int
    score: float
    metadata: Mapping[str, Any]


class ExactCosineIndex:
    """An in-memory exact cosine index with bounded query scoring.

    Rows are normalized once at construction. Search computes every eligible score
    exactly (there is no approximate nearest-neighbor layer) and applies stable
    ``(-score, row_id, row_index)`` ordering before truncating to ``top_k``.
    """

    def __init__(
        self,
        matrix: object,
        *,
        row_ids: Sequence[str] | None = None,
        metadata: Sequence[Metadata] | None = None,
        block_size: int = 4096,
    ) -> None:
        if isinstance(block_size, bool) or not isinstance(block_size, int) or block_size <= 0:
            raise ValueError("block_size must be a positive integer")
        normalized = _normalise_matrix(matrix)
        self._matrix = normalized
        self._row_ids = _validate_row_ids(row_ids, normalized.shape[0])
        self._metadata = _validate_metadata(metadata, normalized.shape[0])
        self._block_size = block_size

    @property
    def rows(self) -> int:
        return int(self._matrix.shape[0])

    @property
    def dimension(self) -> int:
        return int(self._matrix.shape[1])

    @property
    def row_ids(self) -> tuple[str, ...]:
        return self._row_ids

    @property
    def metadata(self) -> tuple[dict[str, Any], ...]:
        return tuple(deepcopy(value) for value in self._metadata)

    @property
    def matrix(self) -> np.ndarray:
        """Return a read-only copy suitable for persistence or diagnostics."""

        return self._matrix.copy()

    def search(
        self,
        query: object,
        *,
        top_k: int,
        metadata_filter: MetadataFilter | None = None,
    ) -> list[SearchHit]:
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
            raise ValueError("top_k must be a positive integer")
        normalized_query = _normalise_query(query, self.dimension)
        best: list[tuple[float, str, int]] = []
        for start in range(0, self.rows, self._block_size):
            block_indices = [
                index
                for index in range(start, min(start + self._block_size, self.rows))
                if metadata_filter is None or metadata_filter(deepcopy(self._metadata[index]))
            ]
            if not block_indices:
                continue
            scores = self._matrix[block_indices] @ normalized_query
            best.extend(
                (float(score), self._row_ids[index], index)
                for score, index in zip(scores, block_indices, strict=True)
            )
            best.sort(key=lambda item: (-item[0], item[1], item[2]))
            del best[top_k:]
        best.sort(key=lambda item: (-item[0], item[1], item[2]))
        return [
            SearchHit(
                row_id=row_id,
                row_index=index,
                score=score,
                metadata=deepcopy(self._metadata[index]),
            )
            for score, row_id, index in best
        ]
