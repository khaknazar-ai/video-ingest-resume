"""Model-agnostic candidate retrieval over a verified embedding index."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from video_intelligence.embeddings.base import EmbeddingIndex, EmbeddingQuery
from video_intelligence.models.contracts import ModelMode
from video_intelligence.retrieval.schemas import ParseStatus, QuerySpec


def _identifier(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value.strip() != value
        or any(character.isspace() for character in value)
    ):
        raise ValueError(f"{name} must be a non-empty identifier")
    return value


@dataclass(frozen=True, slots=True)
class Candidate:
    """One ranked retrieval result; it is not yet grounded evidence."""

    row_id: str
    rank: int
    score: float
    metadata: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "row_id", _identifier(self.row_id, "row_id"))
        if isinstance(self.rank, bool) or not isinstance(self.rank, int) or self.rank <= 0:
            raise ValueError("rank must be a positive integer")
        if isinstance(self.score, bool) or not isinstance(self.score, (int, float)):
            raise ValueError("score must be finite")
        score = float(self.score)
        if not math.isfinite(score):
            raise ValueError("score must be finite")
        object.__setattr__(self, "score", score)
        if not isinstance(self.metadata, Mapping):
            raise ValueError("metadata must be a mapping")
        object.__setattr__(self, "metadata", dict(self.metadata))


@dataclass(frozen=True, slots=True)
class CandidateSet:
    """Ranked candidates plus the provenance needed for later grounding."""

    query_id: str
    fingerprint_id: str
    top_k: int
    candidates: tuple[Candidate, ...]
    model_id: str | None = None
    model_mode: ModelMode | str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "query_id", _identifier(self.query_id, "query_id"))
        object.__setattr__(
            self,
            "fingerprint_id",
            _identifier(self.fingerprint_id, "fingerprint_id"),
        )
        if isinstance(self.top_k, bool) or not isinstance(self.top_k, int) or self.top_k <= 0:
            raise ValueError("top_k must be a positive integer")
        object.__setattr__(self, "candidates", tuple(self.candidates))
        if self.model_id is not None:
            object.__setattr__(self, "model_id", _identifier(self.model_id, "model_id"))
        if self.model_mode is not None:
            object.__setattr__(self, "model_mode", ModelMode(self.model_mode))
        if (self.model_id is None) != (self.model_mode is None):
            raise ValueError("model_id and model_mode must be provided together")
        if any(not isinstance(candidate, Candidate) for candidate in self.candidates):
            raise ValueError("candidates must contain only Candidate values")
        expected_ranks = tuple(range(1, len(self.candidates) + 1))
        if tuple(candidate.rank for candidate in self.candidates) != expected_ranks:
            raise ValueError("candidate ranks must be contiguous and start at one")
        if len({candidate.row_id for candidate in self.candidates}) != len(self.candidates):
            raise ValueError("candidate row_ids must be unique")
        if len(self.candidates) > self.top_k:
            raise ValueError("candidate count must not exceed top_k")

    @property
    def row_ids(self) -> tuple[str, ...]:
        return tuple(candidate.row_id for candidate in self.candidates)

    def to_dict(self) -> dict[str, object]:
        return {
            "query_id": self.query_id,
            "fingerprint_id": self.fingerprint_id,
            "top_k": self.top_k,
            "model_id": self.model_id,
            "model_mode": None if self.model_mode is None else self.model_mode.value,
            "candidates": [
                {
                    "row_id": candidate.row_id,
                    "rank": candidate.rank,
                    "score": candidate.score,
                    "metadata": dict(candidate.metadata),
                }
                for candidate in self.candidates
            ],
        }


def search_candidates(
    index: EmbeddingIndex,
    query: QuerySpec,
    query_embedding: EmbeddingQuery,
    *,
    top_k: int,
    video_id: str | None = None,
    shot_id: str | None = None,
    metadata_filter: Callable[[Mapping[str, Any]], bool] | None = None,
) -> CandidateSet:
    """Retrieve candidates without turning similarity into a supported answer."""

    if not isinstance(index, EmbeddingIndex):
        raise TypeError("index must be an EmbeddingIndex")
    if not isinstance(query, QuerySpec):
        raise TypeError("query must be a QuerySpec")
    if query.parse_status is not ParseStatus.SUPPORTED:
        raise ValueError("candidate search requires a supported query")
    if not isinstance(query_embedding, EmbeddingQuery):
        raise TypeError("query_embedding must be an EmbeddingQuery")
    hits = index.search(
        query_embedding,
        top_k=top_k,
        video_id=video_id,
        shot_id=shot_id,
        start_seconds=query.time_start_seconds,
        end_seconds=query.time_end_seconds,
        metadata_filter=metadata_filter,
    )
    candidates = tuple(
        Candidate(
            row_id=hit.row_id,
            rank=rank,
            score=hit.score,
            metadata=hit.metadata,
        )
        for rank, hit in enumerate(hits, start=1)
    )
    return CandidateSet(
        query_id=query.query_id,
        fingerprint_id=index.fingerprint.fingerprint_id,
        top_k=top_k,
        candidates=candidates,
        model_id=index.fingerprint.model_id,
        model_mode=index.fingerprint.model_mode,
    )
