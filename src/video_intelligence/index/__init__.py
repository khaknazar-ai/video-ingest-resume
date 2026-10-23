"""CPU-first exact vector indexing and immutable snapshot primitives."""

from video_intelligence.index.exact import ExactCosineIndex, SearchHit
from video_intelligence.index.snapshot import (
    LoadedSnapshot,
    SnapshotDescriptor,
    SnapshotIntegrityError,
    SnapshotStore,
)

__all__ = [
    "ExactCosineIndex",
    "EmbeddingBuildBatch",
    "EmbeddingBuildManifest",
    "LoadedSnapshot",
    "SearchHit",
    "SnapshotDescriptor",
    "SnapshotIntegrityError",
    "SnapshotStore",
    "build_embedding_index",
    "publish_embedding_snapshot",
]


def __getattr__(name: str) -> object:
    if name in {
        "EmbeddingBuildBatch",
        "EmbeddingBuildManifest",
        "build_embedding_index",
        "publish_embedding_snapshot",
    }:
        from video_intelligence.index import build

        return getattr(build, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
