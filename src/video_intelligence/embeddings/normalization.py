"""Strict float32 normalization shared by image and text embeddings."""

from __future__ import annotations

from math import isfinite

import numpy as np


def _expected_dimension(value: int | None) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError("expected_dimension must be a positive integer or null")
    return value


def _real_array(value: object, *, ndim: int, name: str) -> np.ndarray:
    try:
        raw = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a real numeric array") from exc
    if raw.ndim != ndim:
        raise ValueError(f"{name} must be {ndim}-dimensional")
    if not np.issubdtype(raw.dtype, np.number) or np.issubdtype(
        raw.dtype, np.complexfloating
    ):
        raise ValueError(f"{name} must contain real numeric values")
    try:
        values = raw.astype(np.float64, copy=False)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must contain real numeric values") from exc
    if not np.isfinite(values).all():
        raise ValueError(f"{name} must contain only finite values")
    return values


def normalize_matrix(
    vectors: object,
    *,
    expected_dimension: int | None = None,
) -> np.ndarray:
    """Return a read-only, finite, non-zero-row-norm float32 matrix."""

    expected = _expected_dimension(expected_dimension)
    values = _real_array(vectors, ndim=2, name="vectors")
    if values.shape[0] == 0 or values.shape[1] == 0:
        raise ValueError("vectors must contain at least one row and one dimension")
    if expected is not None and values.shape[1] != expected:
        raise ValueError(f"vectors must have dimension {expected}")
    norms = np.linalg.norm(values, axis=1)
    if not np.isfinite(norms).all() or np.any(norms <= 0.0):
        raise ValueError("vectors rows must have finite non-zero norms")
    normalized = (values / norms[:, None]).astype(np.float32)
    if not np.isfinite(normalized).all():
        raise ValueError("normalized vectors must contain only finite values")
    normalized.setflags(write=False)
    return normalized


def normalize_vector(
    vector: object,
    *,
    expected_dimension: int | None = None,
) -> np.ndarray:
    """Return a read-only, finite, non-zero-norm float32 vector."""

    expected = _expected_dimension(expected_dimension)
    values = _real_array(vector, ndim=1, name="vector")
    if values.shape[0] == 0:
        raise ValueError("vector must contain at least one dimension")
    if expected is not None and values.shape[0] != expected:
        raise ValueError(f"vector must have dimension {expected}")
    norm = float(np.linalg.norm(values))
    if not isfinite(norm) or norm <= 0.0:
        raise ValueError("vector must have a finite non-zero norm")
    normalized = (values / norm).astype(np.float32)
    if not np.isfinite(normalized).all():
        raise ValueError("normalized vector must contain only finite values")
    normalized.setflags(write=False)
    return normalized
