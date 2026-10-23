"""Application query service for the deterministic fixture profile."""

from __future__ import annotations

import math
import sqlite3
from collections.abc import Mapping
from dataclasses import replace
from typing import Any

import numpy as np

from video_intelligence.pipeline.index_registry import ActiveIndex, IndexRegistry
from video_intelligence.retrieval.evidence import build_bundle, evidence_from_metadata
from video_intelligence.retrieval.predicates import spatial_relation_holds, within_time
from video_intelligence.retrieval.query_parser import parse_query
from video_intelligence.retrieval.schemas import (
    RELATION_POLICY_VERSION,
    EvidenceItem,
    ProcessingStatus,
    QueryResult,
    QuerySpec,
    SupportStatus,
)
from video_intelligence.storage.sqlite import SQLiteStore
from video_intelligence.video.manifest import sha256_file

LABEL_VECTORS = {
    "red_square": np.array([1.0, 0.0, 0.0], dtype=np.float32),
    "blue_circle": np.array([0.0, 1.0, 0.0], dtype=np.float32),
    "green_object": np.array([0.0, 0.0, 1.0], dtype=np.float32),
}


def _within_time(metadata: Mapping[str, Any], query: QuerySpec) -> bool:
    timestamp = float(metadata["timestamp_seconds"])
    return within_time(timestamp, query.time_start_seconds, query.time_end_seconds)


class QueryService:
    """Run supported queries against a verified fixture snapshot."""

    def __init__(
        self,
        store: SQLiteStore,
        registry: IndexRegistry,
        *,
        max_evidence_items: int = 8,
        near_distance_fraction: float = 0.15,
    ) -> None:
        if isinstance(max_evidence_items, bool) or not isinstance(max_evidence_items, int):
            raise ValueError("max_evidence_items must be a positive integer")
        if max_evidence_items <= 0:
            raise ValueError("max_evidence_items must be a positive integer")
        if (
            isinstance(near_distance_fraction, bool)
            or not isinstance(near_distance_fraction, (int, float))
            or not math.isfinite(near_distance_fraction) or near_distance_fraction <= 0
        ):
            raise ValueError("near_distance_fraction must be finite and positive")
        self.store = store
        self.registry = registry
        self.max_evidence_items = max_evidence_items
        self.near_distance_fraction = float(near_distance_fraction)

    def query(self, video_id: str, raw_query: str, *, run_id: str = "run_local") -> QueryResult:
        result = self._query(video_id, raw_query, run_id=run_id)
        if self.store.get_video(video_id) is None:
            return result
        try:
            self.store.save_query_result(result)
        except (OSError, RuntimeError, ValueError, TypeError, sqlite3.DatabaseError):
            spec = replace(result.query, query_id=result.logical_query_id or result.query.query_id)
            return self._failed(
                spec, video_id, run_id, "query_persistence_failed"
            ).with_revision(result.query_config)
        return result

    def load_persisted(self, query_id: str) -> dict | None:
        """Load the exact JSON result saved by a previous process."""

        return self.store.get_query_payload(query_id)

    def _query(self, video_id: str, raw_query: str, *, run_id: str = "run_local") -> QueryResult:
        context = {
            "version": "fixture_query.v2",
            "max_evidence_items": self.max_evidence_items,
            "near_distance_fraction": self.near_distance_fraction,
            "source_sha256": None,
            "snapshot_id": None,
        }
        spec = parse_query(raw_query, query_namespace=f"{video_id}|{run_id}")
        try:
            result = self._compute(video_id, raw_query, run_id=run_id, context=context)
        except (TypeError, ValueError, KeyError, IndexError):
            result = self._failed(spec, video_id, run_id, "invalid_evidence")
        return result.with_revision(context)

    def _compute(
        self, video_id: str, raw_query: str, *, run_id: str, context: dict[str, Any]
    ) -> QueryResult:
        spec = parse_query(raw_query, query_namespace=f"{video_id}|{run_id}")
        if spec.parse_status.value != "supported":
            return QueryResult(
                query=spec,
                video_id=video_id,
                run_id=run_id,
                processing_status=ProcessingStatus.COMPLETE,
                support_status=SupportStatus.UNSUPPORTED_QUERY,
                mode="fixture",
            )
        video = self.store.get_video(video_id)
        if video is None:
            return self._failed(spec, video_id, run_id, "unknown_video")
        context["source_sha256"] = video.source_sha256
        if video.source_sha256 is None or not video.source_path.is_file():
            return self._failed(spec, video_id, run_id, "source_unavailable")
        try:
            current_source_sha256 = sha256_file(video.source_path)
        except (OSError, RuntimeError):
            return self._failed(spec, video_id, run_id, "source_unavailable")
        if current_source_sha256 != video.source_sha256:
            return self._failed(spec, video_id, run_id, "source_hash_mismatch")
        try:
            active = self.registry.get_active(f"fixture:{video_id}")
        except Exception:
            return self._failed(spec, video_id, run_id, "corrupt_index")
        if active is None:
            return QueryResult(
                query=spec,
                video_id=video_id,
                run_id=run_id,
                processing_status=ProcessingStatus.PARTIAL,
                support_status=SupportStatus.INSUFFICIENT_EVIDENCE,
                mode="fixture",
                error_code="index_unavailable",
            )
        context["snapshot_id"] = active.descriptor.snapshot_id
        if active.descriptor.attributes.get("source_sha256") != video.source_sha256:
            return self._failed(spec, video_id, run_id, "source_hash_mismatch")
        if spec.relation is not None and self.max_evidence_items < 2:
            return QueryResult(
                spec,
                video_id,
                run_id,
                ProcessingStatus.PARTIAL,
                SupportStatus.INSUFFICIENT_EVIDENCE,
                "fixture",
                error_code="evidence_budget_exhausted",
            )
        try:
            items = self._retrieve(spec, video, active, run_id)
        except (KeyError, TypeError, ValueError, IndexError):
            return self._failed(spec, video_id, run_id, "invalid_evidence")
        coverage_complete = bool(active.descriptor.attributes.get("coverage_complete", False))
        if not items:
            return QueryResult(
                query=spec,
                video_id=video_id,
                run_id=run_id,
                processing_status=(
                    ProcessingStatus.COMPLETE if coverage_complete else ProcessingStatus.PARTIAL
                ),
                support_status=(
                    SupportStatus.NOT_FOUND
                    if coverage_complete
                    else SupportStatus.INSUFFICIENT_EVIDENCE
                ),
                mode="fixture",
                error_code=None if coverage_complete else "incomplete_coverage",
            )
        if spec.temporal is not None and not coverage_complete:
            return QueryResult(
                query=spec,
                video_id=video_id,
                run_id=run_id,
                processing_status=ProcessingStatus.PARTIAL,
                support_status=SupportStatus.INSUFFICIENT_EVIDENCE,
                mode="fixture",
                error_code="incomplete_coverage",
            )
        descriptor = active.descriptor
        try:
            bundle = build_bundle(
                query_id=spec.query_id,
                run_id=run_id,
                video=video,
                descriptor=descriptor,
                items=items,
                coverage_complete=coverage_complete,
                gaps=() if coverage_complete else ("sampling_budget",),
                relation_policy={
                    "version": RELATION_POLICY_VERSION,
                    "near_distance_fraction": self.near_distance_fraction,
                } if spec.relation is not None else None,
            )
        except (TypeError, ValueError):
            return self._failed(spec, video_id, run_id, "invalid_evidence")
        answer: dict[str, Any]
        if spec.answer_type == "time":
            answer = {"timestamp_seconds": items[0].frame_ref.timestamp_seconds}
        else:
            answer = {"evidence_count": len(items)}
            if spec.relation is not None:
                answer["relation"] = spec.relation
        return QueryResult(
            query=spec,
            video_id=video_id,
            run_id=run_id,
            processing_status=ProcessingStatus.COMPLETE,
            support_status=SupportStatus.SUPPORTED,
            mode="fixture",
            answer=answer,
            evidence=bundle,
        )

    def _retrieve(
        self,
        spec: QuerySpec,
        video,
        active: ActiveIndex,
        run_id: str,
    ) -> list[EvidenceItem]:
        assert spec.entity is not None
        query_vector = LABEL_VECTORS.get(spec.entity)
        if query_vector is None:
            raise ValueError("fixture query entity has no deterministic vector")
        primary_hits = active.index.search(
            query_vector,
            top_k=active.index.rows,
            metadata_filter=lambda row: (
                row.get("video_id") == video.video_id
                and row.get("label") == spec.entity
                and _within_time(row, spec)
            ),
        )
        ordered_hits = sorted(
            primary_hits,
            key=lambda hit: (float(hit.metadata["timestamp_seconds"]), hit.row_id),
        )
        if spec.relation is None:
            if spec.temporal == "last":
                ordered_hits = list(reversed(ordered_hits))
            selected = (
                ordered_hits[:1]
                if spec.temporal is not None
                else ordered_hits[: self.max_evidence_items]
            )
            return [
                evidence_from_metadata(
                    hit.metadata,
                    video=video,
                    descriptor=active.descriptor,
                    run_id=run_id,
                )
                for hit in selected
            ]
        if spec.secondary_entity is None or spec.relation not in {
            "near",
            "left",
            "right",
            "overlap",
        }:
            raise ValueError("unsupported relation")
        secondary_rows = [
            row
            for row in active.index.metadata
            if row.get("video_id") == video.video_id
            and row.get("label") == spec.secondary_entity
            and _within_time(row, spec)
        ]
        result: list[EvidenceItem] = []
        for hit in ordered_hits:
            if len(result) + 2 > self.max_evidence_items:
                break
            primary = hit.metadata
            candidates = [
                row
                for row in secondary_rows
                if row["frame_index"] == primary["frame_index"]
                and row["detection_id"] != primary["detection_id"]
                and self._relation_holds(spec.relation, primary, row, video)
            ]
            if not candidates:
                continue
            secondary = sorted(candidates, key=lambda row: row["detection_id"])[0]
            result.extend(
                (
                    evidence_from_metadata(
                        primary,
                        video=video,
                        descriptor=active.descriptor,
                        run_id=run_id,
                        role="primary",
                    ),
                    evidence_from_metadata(
                        secondary,
                        video=video,
                        descriptor=active.descriptor,
                        run_id=run_id,
                        role="secondary",
                    ),
                )
            )
            if len(result) >= self.max_evidence_items:
                break
        return result

    def _is_near(self, first: Mapping[str, Any], second: Mapping[str, Any], video) -> bool:
        return self._relation_holds("near", first, second, video)

    def _relation_holds(
        self,
        relation: str,
        first: Mapping[str, Any],
        second: Mapping[str, Any],
        video,
    ) -> bool:
        return spatial_relation_holds(
            relation, first["bbox_xyxy"], second["bbox_xyxy"],
            width=video.metadata.width or int(first.get("width", 0)),
            height=video.metadata.height or int(first.get("height", 0)),
            near_distance_fraction=self.near_distance_fraction,
        )

    @staticmethod
    def _failed(spec: QuerySpec, video_id: str, run_id: str, error_code: str) -> QueryResult:
        return QueryResult(
            query=spec,
            video_id=video_id,
            run_id=run_id,
            processing_status=ProcessingStatus.FAILED,
            support_status=SupportStatus.INSUFFICIENT_EVIDENCE,
            mode="fixture",
            error_code=error_code,
        )
