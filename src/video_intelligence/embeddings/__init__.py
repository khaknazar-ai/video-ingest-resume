"""Typed, CPU-first embedding contracts and exact-index integration."""

from video_intelligence.embeddings.base import (
    EMBEDDING_FINGERPRINT_METADATA_KEY,
    EmbeddingFingerprint,
    EmbeddingIndex,
    EmbeddingModality,
    EmbeddingQuery,
    EmbeddingRow,
)
from video_intelligence.embeddings.normalization import normalize_matrix, normalize_vector
from video_intelligence.embeddings.siglip2 import (
    SIGLIP2_MODEL_ID,
    SIGLIP2_ROLE,
    SigLIP2Adapter,
    SigLIP2Input,
)

__all__ = [
    "EMBEDDING_FINGERPRINT_METADATA_KEY",
    "EmbeddingFingerprint",
    "EmbeddingIndex",
    "EmbeddingModality",
    "EmbeddingQuery",
    "EmbeddingRow",
    "normalize_matrix",
    "normalize_vector",
    "SIGLIP2_MODEL_ID",
    "SIGLIP2_ROLE",
    "SigLIP2Adapter",
    "SigLIP2Input",
]
