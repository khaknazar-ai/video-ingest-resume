"""Transparent Recall@K/MRR evaluation with explicit negative annotations."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from video_intelligence.retrieval.candidate_search import CandidateSet


def _query_id(value: object) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value.strip() != value
        or any(character.isspace() for character in value)
    ):
        raise ValueError("query_id must be a non-empty identifier")
    return value


def _row_ids(value: object, name: str) -> frozenset[str]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Iterable):
        raise ValueError(f"{name} must be an iterable of row ids")
    result = frozenset(value)
    if any(
        not isinstance(row_id, str)
        or not row_id
        or row_id.strip() != row_id
        or any(character.isspace() for character in row_id)
        for row_id in result
    ):
        raise ValueError(f"{name} must contain non-empty identifiers")
    return result


@dataclass(frozen=True, slots=True)
class RetrievalAnnotation:
    """Annotation-only relevance contract; never read by production retrieval."""

    query_id: str
    relevant_row_ids: frozenset[str] = field(default_factory=frozenset)
    forbidden_row_ids: frozenset[str] = field(default_factory=frozenset)

    def __post_init__(self) -> None:
        object.__setattr__(self, "query_id", _query_id(self.query_id))
        object.__setattr__(
            self,
            "relevant_row_ids",
            _row_ids(self.relevant_row_ids, "relevant_row_ids"),
        )
        object.__setattr__(
            self,
            "forbidden_row_ids",
            _row_ids(self.forbidden_row_ids, "forbidden_row_ids"),
        )
        if self.relevant_row_ids & self.forbidden_row_ids:
            raise ValueError("relevant and forbidden row ids must be disjoint")

    @property
    def is_positive(self) -> bool:
        return bool(self.relevant_row_ids)


@dataclass(frozen=True, slots=True)
class RetrievalMetrics:
    """Measured metrics with nulls for undefined denominators."""

    query_count: int
    positive_query_count: int
    negative_query_count: int
    missing_result_count: int
    recall_at_k: Mapping[int, float | None]
    mrr: float | None
    false_negative_count: int
    false_positive_count: int
    negatives_without_forbidden_labels: int

    def __post_init__(self) -> None:
        for name in (
            "query_count",
            "positive_query_count",
            "negative_query_count",
            "missing_result_count",
            "false_negative_count",
            "false_positive_count",
            "negatives_without_forbidden_labels",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if self.positive_query_count + self.negative_query_count != self.query_count:
            raise ValueError("positive and negative counts must sum to query_count")
        if not isinstance(self.recall_at_k, Mapping) or not self.recall_at_k:
            raise ValueError("recall_at_k must be a non-empty mapping")
        normalized: dict[int, float | None] = {}
        for key, value in self.recall_at_k.items():
            if isinstance(key, bool) or not isinstance(key, int) or key <= 0:
                raise ValueError("recall_at_k keys must be positive integers")
            if value is not None:
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    raise ValueError("recall_at_k values must be finite numbers or null")
                value = float(value)
                if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                    raise ValueError("recall_at_k values must be finite numbers in [0, 1]")
            normalized[key] = value
        object.__setattr__(self, "recall_at_k", normalized)
        if self.mrr is not None:
            if (
                isinstance(self.mrr, bool)
                or not isinstance(self.mrr, (int, float))
                or not math.isfinite(float(self.mrr))
                or not 0.0 <= float(self.mrr) <= 1.0
            ):
                raise ValueError("mrr must be a finite number in [0, 1] or null")
            object.__setattr__(self, "mrr", float(self.mrr))

    def to_dict(self) -> dict[str, Any]:
        return {
            "query_count": self.query_count,
            "positive_query_count": self.positive_query_count,
            "negative_query_count": self.negative_query_count,
            "missing_result_count": self.missing_result_count,
            "recall_at_k": {str(key): value for key, value in self.recall_at_k.items()},
            "mrr": self.mrr,
            "false_negative_count": self.false_negative_count,
            "false_positive_count": self.false_positive_count,
            "negatives_without_forbidden_labels": self.negatives_without_forbidden_labels,
        }


def evaluate_retrieval(
    results: Sequence[CandidateSet],
    annotations: Sequence[RetrievalAnnotation],
    *,
    k_values: Sequence[int] = (1, 5, 10),
) -> RetrievalMetrics:
    """Compute retrieval metrics from ranked results and separate annotations."""

    if isinstance(results, (str, bytes)) or not isinstance(results, Sequence):
        raise ValueError("results must be a sequence of CandidateSet values")
    if isinstance(annotations, (str, bytes)) or not isinstance(annotations, Sequence):
        raise ValueError("annotations must be a sequence of RetrievalAnnotation values")
    if isinstance(k_values, (str, bytes)) or not isinstance(k_values, Sequence):
        raise ValueError("k_values must be a sequence of positive integers")
    normalized_k = tuple(k_values)
    if not normalized_k or any(
        isinstance(value, bool) or not isinstance(value, int) or value <= 0
        for value in normalized_k
    ):
        raise ValueError("k_values must contain positive integers")
    if len(set(normalized_k)) != len(normalized_k):
        raise ValueError("k_values must be unique")
    annotation_values = tuple(annotations)
    if any(not isinstance(value, RetrievalAnnotation) for value in annotation_values):
        raise ValueError("annotations must contain only RetrievalAnnotation values")
    if len({value.query_id for value in annotation_values}) != len(annotation_values):
        raise ValueError("annotation query_ids must be unique")
    result_values = tuple(results)
    if any(not isinstance(value, CandidateSet) for value in result_values):
        raise ValueError("results must contain only CandidateSet values")
    if len({value.query_id for value in result_values}) != len(result_values):
        raise ValueError("result query_ids must be unique")
    result_by_query = {value.query_id: value for value in result_values}
    max_k = max(normalized_k)
    positive = tuple(value for value in annotation_values if value.is_positive)
    negative = tuple(value for value in annotation_values if not value.is_positive)
    missing = sum(value.query_id not in result_by_query for value in annotation_values)
    recall_at_k: dict[int, float | None] = {}
    for k in normalized_k:
        hits = 0
        for annotation in positive:
            result = result_by_query.get(annotation.query_id)
            if result is not None and annotation.relevant_row_ids & set(result.row_ids[:k]):
                hits += 1
        recall_at_k[k] = hits / len(positive) if positive else None
    reciprocal_ranks: list[float] = []
    for annotation in positive:
        result = result_by_query.get(annotation.query_id)
        if result is None:
            continue
        first = next(
            (
                candidate.rank
                for candidate in result.candidates
                if candidate.row_id in annotation.relevant_row_ids
            ),
            None,
        )
        if first is not None:
            reciprocal_ranks.append(1.0 / first)
    false_negative_count = sum(
        annotation.query_id not in result_by_query
        or not annotation.relevant_row_ids
        & set(result_by_query[annotation.query_id].row_ids[:max_k])
        for annotation in positive
    )
    false_positive_count = 0
    for annotation in negative:
        result = result_by_query.get(annotation.query_id)
        returned_row_ids = set(result.row_ids[:max_k]) if result is not None else set()
        if annotation.forbidden_row_ids & returned_row_ids:
            false_positive_count += 1
    return RetrievalMetrics(
        query_count=len(annotation_values),
        positive_query_count=len(positive),
        negative_query_count=len(negative),
        missing_result_count=missing,
        recall_at_k=recall_at_k,
        mrr=sum(reciprocal_ranks) / len(positive) if positive else None,
        false_negative_count=false_negative_count,
        false_positive_count=false_positive_count,
        negatives_without_forbidden_labels=sum(
            not annotation.forbidden_row_ids for annotation in negative
        ),
    )
