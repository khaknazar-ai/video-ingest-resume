"""Typed contracts for the CPU query/evidence vertical slice."""

from __future__ import annotations

import hashlib
import json
import math
from copy import deepcopy
from dataclasses import asdict, dataclass, field, fields, replace
from enum import StrEnum
from fractions import Fraction
from typing import Any

from video_intelligence.core.types import FrameRef, TimestampQuality, VideoRecord
from video_intelligence.retrieval.predicates import spatial_relation_holds, within_time

RESULT_SCHEMA_VERSION = "0.2"
RELATION_POLICY_VERSION = "source_box_geometry.v1"
OBSERVATION_SUPPORT_POLICY = "unverified_learned_observations_abstain.v1"


def requires_semantic_verification(mode, query_config, items=()) -> bool:
    """Model execution and phrase-conditioned boxes do not establish query truth.

    No semantic verifier is qualified in this release. Preserve explicit mock-only
    contracts, but do not let a mixed mock/learned path bypass the answer boundary.
    Provenance is an admission check, not a caller-controlled verification receipt.
    """
    if mode not in {"fixture", "mock"}:
        return True
    for name, mode_key in (("embedding_fingerprint", "model_mode"), ("grounding_model", "mode")):
        if name in query_config:
            stage = query_config[name]
            if not isinstance(stage, dict) or stage.get(mode_key) != "mock":
                return True
    for item in items:
        provenance = item.provenance
        if provenance.get("mode") not in {"fixture", "mock"}:
            return True
        if "grounding" in provenance:
            grounding = provenance["grounding"]
            if not isinstance(grounding, dict) or grounding.get("model_mode") != "mock":
                return True
            spec = grounding.get("model_spec")
            if not isinstance(spec, dict) or spec.get("mode") != "mock":
                return True
    return False


def _portable_frame(frame: FrameRef) -> dict[str, Any]:
    payload = asdict(frame)
    payload["source_path"] = None
    payload["timestamp_quality"] = frame.timestamp_quality.value
    return payload


def _validate_exact_frame(frame: FrameRef) -> None:
    if frame.timestamp_quality is TimestampQuality.EXACT_PTS:
        if frame.pts is None or frame.time_base is None or frame.source_sha256 is None:
            raise ValueError("exact frame requires PTS, time_base and source identity")
        try:
            valid = Fraction(frame.time_base) > 0
        except (ValueError, ZeroDivisionError) as error:
            raise ValueError("exact frame time_base must be a positive rational") from error
        if not valid or frame.backend == "legacy":
            raise ValueError("exact frame identity is incomplete")


def _revision_id(payload: dict[str, Any]) -> str:
    body = deepcopy(payload)
    logical_id = body["logical_query_id"]
    body["query"]["query_id"] = logical_id
    if body["evidence"] is not None:
        body["evidence"]["query_id"] = logical_id
    material = json.dumps(body, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return "q_rev_" + hashlib.sha256(material.encode("utf-8")).hexdigest()


def _validate_relation_policy(policy: dict[str, Any] | None) -> None:
    if policy is None:
        return
    if not isinstance(policy, dict) or set(policy) != {"version", "near_distance_fraction"}:
        raise ValueError("invalid relation policy schema")
    if policy["version"] != RELATION_POLICY_VERSION:
        raise ValueError("unsupported relation policy version")
    fraction = policy["near_distance_fraction"]
    if (
        isinstance(fraction, bool) or not isinstance(fraction, (int, float))
        or not math.isfinite(fraction) or fraction <= 0
    ):
        raise ValueError("relation policy distance must be finite and positive")


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ValueError(f"{name} must be a non-empty trimmed string")
    return value


def _identifier(value: object, name: str) -> str:
    value = _text(value, name)
    if any(character.isspace() for character in value):
        raise ValueError(f"{name} must not contain whitespace")
    return value


def _sha256(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or value != value.lower()
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 hex digest")
    return value


def _bbox(value: object) -> tuple[float, float, float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise ValueError("bbox_xyxy must contain four coordinates")
    coordinates = tuple(float(item) for item in value)
    if not all(math.isfinite(item) for item in coordinates):
        raise ValueError("bbox_xyxy must contain finite coordinates")
    x1, y1, x2, y2 = coordinates
    if x1 < 0 or y1 < 0 or x2 <= x1 or y2 <= y1:
        raise ValueError("bbox_xyxy must have positive in-frame coordinates")
    return coordinates


class ParseStatus(StrEnum):
    SUPPORTED = "supported"
    UNSUPPORTED_QUERY = "unsupported_query"


class ProcessingStatus(StrEnum):
    COMPLETE = "complete"
    PARTIAL = "partial"
    FAILED = "failed"


class SupportStatus(StrEnum):
    SUPPORTED = "supported"
    NOT_FOUND = "not_found"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    UNSUPPORTED_QUERY = "unsupported_query"


@dataclass(frozen=True, slots=True)
class QuerySpec:
    query_id: str
    raw_text: str
    language: str
    parse_status: ParseStatus
    entity: str | None = None
    secondary_entity: str | None = None
    relation: str | None = None
    temporal: str | None = None
    answer_type: str = "evidence"
    time_start_seconds: float | None = None
    time_end_seconds: float | None = None
    reason: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "query_id", _identifier(self.query_id, "query_id"))
        object.__setattr__(self, "raw_text", _text(self.raw_text, "raw_text"))
        object.__setattr__(self, "language", _identifier(self.language, "language"))
        object.__setattr__(self, "parse_status", ParseStatus(self.parse_status))
        for name in ("entity", "secondary_entity", "relation", "temporal"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _identifier(value, name))
        if self.reason is not None:
            object.__setattr__(self, "reason", _text(self.reason, "reason"))
        if self.temporal not in {None, "first", "last"}:
            raise ValueError("temporal must be first, last or null")
        if self.answer_type not in {"evidence", "time"}:
            raise ValueError("answer_type must be evidence or time")
        for name in ("time_start_seconds", "time_end_seconds"):
            value = getattr(self, name)
            if value is not None:
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    raise ValueError(f"{name} must be finite and non-negative")
                value = float(value)
                if not math.isfinite(value) or value < 0:
                    raise ValueError(f"{name} must be finite and non-negative")
                object.__setattr__(self, name, value)
        if (
            self.time_start_seconds is not None
            and self.time_end_seconds is not None
            and self.time_end_seconds < self.time_start_seconds
        ):
            raise ValueError("time_end_seconds must not precede time_start_seconds")
        if self.parse_status is ParseStatus.SUPPORTED and self.entity is None:
            raise ValueError("supported query requires an entity")
        if self.relation not in {None, "near", "left", "right", "overlap"}:
            raise ValueError("unsupported relation")
        if (self.relation is None) != (self.secondary_entity is None):
            raise ValueError("relation and secondary_entity must be provided together")
        if self.relation is not None and (
            self.secondary_entity == self.entity or self.temporal is not None
        ):
            raise ValueError("relation requires distinct entities and no temporal selector")
        if (self.answer_type == "time") != (self.temporal is not None):
            raise ValueError("time answer requires an explicit temporal selector")

    def to_dict(self) -> dict[str, Any]:
        return {
            "query_id": self.query_id,
            "raw_text": self.raw_text,
            "language": self.language,
            "parse_status": self.parse_status.value,
            "entity": self.entity,
            "secondary_entity": self.secondary_entity,
            "relation": self.relation,
            "temporal": self.temporal,
            "answer_type": self.answer_type,
            "time_start_seconds": self.time_start_seconds,
            "time_end_seconds": self.time_end_seconds,
            "reason": self.reason,
        }


@dataclass(frozen=True, slots=True)
class EvidenceItem:
    evidence_id: str
    frame_ref: FrameRef
    label: str
    bbox_xyxy: tuple[float, float, float, float]
    score: float
    source_sha256: str
    snapshot_id: str
    artifact_id: str
    source_stage: str
    role: str = "entity"
    provenance: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "evidence_id", _identifier(self.evidence_id, "evidence_id"))
        if not isinstance(self.frame_ref, FrameRef):
            raise ValueError("frame_ref must be a FrameRef")
        _validate_exact_frame(self.frame_ref)
        object.__setattr__(self, "label", _identifier(self.label, "label"))
        object.__setattr__(self, "bbox_xyxy", _bbox(self.bbox_xyxy))
        if isinstance(self.score, bool) or not isinstance(self.score, (int, float)):
            raise ValueError("score must be a finite number between 0 and 1")
        score = float(self.score)
        if not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError("score must be a finite number between 0 and 1")
        object.__setattr__(self, "score", score)
        object.__setattr__(self, "source_sha256", _sha256(self.source_sha256, "source_sha256"))
        object.__setattr__(self, "snapshot_id", _sha256(self.snapshot_id, "snapshot_id"))
        object.__setattr__(self, "artifact_id", _identifier(self.artifact_id, "artifact_id"))
        object.__setattr__(self, "source_stage", _identifier(self.source_stage, "source_stage"))
        object.__setattr__(self, "role", _identifier(self.role, "role"))
        if self.frame_ref.source_sha256 != self.source_sha256:
            raise ValueError("frame_ref source hash does not match evidence source hash")
        if not isinstance(self.provenance, dict):
            raise ValueError("provenance must be a JSON object")
        object.__setattr__(self, "provenance", deepcopy(self.provenance))
        if self.bbox_xyxy[2] > self.frame_ref.width or self.bbox_xyxy[3] > self.frame_ref.height:
            raise ValueError("evidence bbox is outside frame")


@dataclass(frozen=True, slots=True)
class EvidenceBundle:
    query_id: str
    run_id: str
    video_id: str
    source_sha256: str
    snapshot_id: str
    items: tuple[EvidenceItem, ...]
    coverage_start_seconds: float
    coverage_end_seconds: float
    coverage_complete: bool
    gaps: tuple[str, ...] = ()
    relation_policy: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "query_id", _identifier(self.query_id, "query_id"))
        object.__setattr__(self, "run_id", _identifier(self.run_id, "run_id"))
        object.__setattr__(self, "video_id", _identifier(self.video_id, "video_id"))
        object.__setattr__(self, "source_sha256", _sha256(self.source_sha256, "source_sha256"))
        object.__setattr__(self, "snapshot_id", _sha256(self.snapshot_id, "snapshot_id"))
        if not isinstance(self.items, (tuple, list)) or any(
            not isinstance(item, EvidenceItem) for item in self.items
        ):
            raise ValueError("evidence items must be EvidenceItem values")
        object.__setattr__(self, "items", tuple(self.items))
        if not self.items:
            raise ValueError("evidence bundle requires at least one item")
        if len({item.evidence_id for item in self.items}) != len(self.items):
            raise ValueError("evidence ids must be unique")
        for item in self.items:
            if item.frame_ref.video_id != self.video_id or item.source_sha256 != self.source_sha256:
                raise ValueError("evidence item belongs to another source")
            if item.snapshot_id != self.snapshot_id:
                raise ValueError("evidence item belongs to another snapshot")
        for name in ("coverage_start_seconds", "coverage_end_seconds"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{name} must be finite and non-negative")
            value = float(value)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
            object.__setattr__(self, name, value)
        if self.coverage_end_seconds < self.coverage_start_seconds:
            raise ValueError("coverage end must not precede coverage start")
        if not isinstance(self.coverage_complete, bool):
            raise ValueError("coverage_complete must be a boolean")
        object.__setattr__(self, "gaps", tuple(_identifier(gap, "gap") for gap in self.gaps))
        if self.relation_policy is not None:
            _validate_relation_policy(self.relation_policy)
            object.__setattr__(self, "relation_policy", deepcopy(self.relation_policy))


@dataclass(frozen=True, slots=True)
class QueryResult:
    query: QuerySpec
    video_id: str
    run_id: str
    processing_status: ProcessingStatus
    support_status: SupportStatus
    mode: str
    answer: dict[str, Any] | None = None
    evidence: EvidenceBundle | None = None
    error_code: str | None = None
    logical_query_id: str | None = None
    query_config: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: dict[str, Any], *, video: VideoRecord) -> QueryResult:
        """Revalidate stored wire data, including legacy payloads, before serving it."""
        try:
            version = payload.get("schema_version", "0.1")
            if version not in {"0.1", RESULT_SCHEMA_VERSION}:
                raise ValueError("unsupported persisted query schema version")
            query = QuerySpec(**payload["query"])
            evidence = payload["evidence"]
            if evidence is not None:
                items = []
                for item in evidence["items"]:
                    if version == RESULT_SCHEMA_VERSION:
                        wire_ref = item["frame_ref"]
                        if (
                            not isinstance(wire_ref, dict)
                            or set(wire_ref) != {value.name for value in fields(FrameRef)}
                            or wire_ref["source_path"] is not None
                        ):
                            raise ValueError("frame_ref must contain the full portable schema")
                        ref = FrameRef(**wire_ref)
                        for name in (
                            "video_id", "frame_index", "timestamp_seconds",
                            "timestamp_quality", "timestamp_origin", "source_sha256",
                        ):
                            if getattr(ref, name) != item[name]:
                                raise ValueError("flattened evidence fields differ from frame_ref")
                    else:
                        if item["timestamp_quality"] == TimestampQuality.EXACT_PTS.value:
                            raise ValueError("legacy exact frame identity is missing")
                        ref = FrameRef(
                            video_id=item["video_id"], frame_index=item["frame_index"],
                            timestamp_seconds=item["timestamp_seconds"],
                            timestamp_quality=item["timestamp_quality"],
                            timestamp_origin=item["timestamp_origin"],
                            width=video.metadata.width, height=video.metadata.height,
                            source_sha256=item["source_sha256"], backend="legacy",
                        )
                    items.append(
                        EvidenceItem(
                            frame_ref=ref,
                            **{
                                key: item[key]
                                for key in (
                                    "evidence_id",
                                    "label",
                                    "bbox_xyxy",
                                    "score",
                                    "source_sha256",
                                    "snapshot_id",
                                    "artifact_id",
                                    "source_stage",
                                    "role",
                                    "provenance",
                                )
                            },
                        )
                    )
                evidence = EvidenceBundle(**{**evidence, "items": tuple(items)})
            values = {key: value for key, value in payload.items() if key != "schema_version"}
            result = cls(**{**values, "query": query, "evidence": evidence})
            result.validate_video(video)
            canonical = result.to_dict()
            if version == "0.1":
                for name in ("schema_version", "logical_query_id", "query_config"):
                    canonical.pop(name)
                if canonical["evidence"] is not None:
                    canonical["evidence"].pop("relation_policy")
                    for item in canonical["evidence"]["items"]:
                        item.pop("frame_ref")
                if "schema_version" in payload:
                    canonical["schema_version"] = "0.1"
            if canonical != payload:
                raise ValueError("persisted query contains unknown or inconsistent fields")
            return result
        except (KeyError, TypeError, AttributeError) as error:
            raise ValueError("invalid persisted query schema") from error

    def __post_init__(self) -> None:
        if not isinstance(self.query, QuerySpec):
            raise ValueError("query must be a QuerySpec")
        object.__setattr__(self, "video_id", _identifier(self.video_id, "video_id"))
        object.__setattr__(self, "run_id", _identifier(self.run_id, "run_id"))
        object.__setattr__(self, "processing_status", ProcessingStatus(self.processing_status))
        object.__setattr__(self, "support_status", SupportStatus(self.support_status))
        object.__setattr__(self, "mode", _identifier(self.mode, "mode"))
        if self.answer is not None and not isinstance(self.answer, dict):
            raise ValueError("answer must be a JSON object or null")
        if self.error_code is not None:
            object.__setattr__(self, "error_code", _identifier(self.error_code, "error_code"))
        if self.logical_query_id is not None:
            object.__setattr__(
                self, "logical_query_id", _identifier(self.logical_query_id, "logical_query_id")
            )
        if not isinstance(self.query_config, dict):
            raise ValueError("query_config must be a JSON object")
        object.__setattr__(self, "query_config", deepcopy(self.query_config))
        self.validate_evidence()

    def with_revision(self, query_config: dict[str, Any]) -> QueryResult:
        """Bind one immutable revision to logical request, effective config and result."""
        logical = self.logical_query_id or self.query.query_id
        payload = self._wire_payload()
        payload["logical_query_id"] = logical
        payload["query_config"] = deepcopy(query_config)
        revision = _revision_id(payload)
        return replace(
            self, logical_query_id=logical, query_config=query_config,
            query=replace(self.query, query_id=revision),
            evidence=(
                replace(self.evidence, query_id=revision) if self.evidence is not None else None
            ),
        )

    def validate_video(self, video: VideoRecord) -> None:
        """Validate identity and known source bounds at storage/replay boundaries."""
        if self.video_id != video.video_id:
            raise ValueError("persisted query video identity mismatch")
        if self.evidence is None:
            return
        if self.evidence.source_sha256 != video.source_sha256:
            raise ValueError("persisted evidence source hash mismatch")
        for item in self.evidence.items:
            frame = item.frame_ref
            if (
                video.metadata.duration_seconds is not None
                and frame.timestamp_seconds >= video.metadata.duration_seconds
            ):
                raise ValueError("evidence timestamp is outside video duration")
            if (
                video.metadata.width is not None and frame.width != video.metadata.width
            ) or (
                video.metadata.height is not None and frame.height != video.metadata.height
            ):
                raise ValueError("evidence dimensions differ from video")

    def validate_evidence(self) -> None:
        """Recheck mutable nested provenance at construction and persistence boundaries."""
        json.dumps(self.query_config, allow_nan=False)
        if self.support_status is SupportStatus.SUPPORTED:
            if requires_semantic_verification(
                self.mode, self.query_config, self.evidence.items if self.evidence else (),
            ):
                raise ValueError("learned observations require independent semantic verification")
            if self.evidence is None:
                raise ValueError("supported answer requires evidence")
            if (
                self.query.parse_status is not ParseStatus.SUPPORTED
                or self.processing_status is not ProcessingStatus.COMPLETE
                or self.error_code is not None
            ):
                raise ValueError("supported answer requires successful supported processing")
        if self.evidence is None:
            if self.answer is not None:
                raise ValueError("answer requires supporting evidence")
            self._validate_revision()
            return
        if not isinstance(self.evidence, EvidenceBundle):
            raise ValueError("evidence must be an EvidenceBundle")
        bundle = self.evidence
        _validate_relation_policy(bundle.relation_policy)
        for name, expected in (
            ("source_sha256", bundle.source_sha256), ("snapshot_id", bundle.snapshot_id)
        ):
            if name in self.query_config and self.query_config[name] != expected:
                raise ValueError("query config does not match supporting evidence")
        if "max_evidence_items" in self.query_config:
            budget = self.query_config["max_evidence_items"]
            if isinstance(budget, bool) or not isinstance(budget, int) or budget <= 0:
                raise ValueError("invalid query evidence budget")
            if len(bundle.items) > budget:
                raise ValueError("cited evidence exceeds the query budget")
        if bundle.relation_policy is not None and "near_distance_fraction" in self.query_config:
            if (
                self.query_config["near_distance_fraction"]
                != bundle.relation_policy["near_distance_fraction"]
            ):
                raise ValueError("query config does not match relation policy")
        if (bundle.video_id, bundle.query_id, bundle.run_id) != (
            self.video_id,
            self.query.query_id,
            self.run_id,
        ):
            raise ValueError("evidence belongs to another video, query or run")
        for item in bundle.items:
            _validate_exact_frame(item.frame_ref)
            expected = {
                "mode": self.mode,
                "run_id": self.run_id,
                "source_sha256": bundle.source_sha256,
                "snapshot_id": bundle.snapshot_id,
            }
            if any(item.provenance.get(key) != value for key, value in expected.items()):
                raise ValueError("evidence provenance does not match result")
            if not within_time(
                item.frame_ref.timestamp_seconds,
                self.query.time_start_seconds, self.query.time_end_seconds,
            ):
                raise ValueError("evidence timestamp is outside query time constraints")
        if self.support_status is not SupportStatus.SUPPORTED:
            if self.answer is not None:
                raise ValueError("unsupported result must not contain a positive answer")
            self._validate_revision()
            return
        if self.query.relation is None and any(
            item.label != self.query.entity or item.role != "entity" for item in bundle.items
        ):
            raise ValueError("evidence entity binding does not match query")
        if self.query.temporal is not None and (
            not bundle.coverage_complete or bundle.gaps or len(bundle.items) != 1
        ):
            raise ValueError("temporal answer requires complete coverage and one cited frame")
        if self.query.relation is not None and len(bundle.items) % 2:
            raise ValueError("relation evidence requires complete pairs")
        if self.support_status is SupportStatus.SUPPORTED and self.query.answer_type == "time":
            timestamp = self.answer.get("timestamp_seconds") if self.answer is not None else None
            if (
                isinstance(timestamp, bool)
                or not isinstance(timestamp, (int, float))
                or not math.isfinite(timestamp)
                or timestamp != bundle.items[0].frame_ref.timestamp_seconds
                or set(self.answer) != {"timestamp_seconds"}
            ):
                raise ValueError("answer timestamp must match supporting evidence")
        elif (
            self.answer is None
            or set(self.answer) != (
                {"evidence_count", "relation"} if self.query.relation is not None
                else {"evidence_count"}
            )
            or isinstance(self.answer.get("evidence_count"), bool)
            or not isinstance(self.answer.get("evidence_count"), int)
            or self.answer["evidence_count"] != len(bundle.items)
        ):
            raise ValueError("answer evidence_count must match the cited evidence")
        if self.support_status is SupportStatus.SUPPORTED and self.query.relation is not None:
            if self.answer["relation"] != self.query.relation:
                raise ValueError("answer relation does not match query")
            for primary, secondary in zip(bundle.items[::2], bundle.items[1::2], strict=True):
                if (
                    primary.role != "primary"
                    or secondary.role != "secondary"
                    or primary.frame_ref.identity_key != secondary.frame_ref.identity_key
                    or primary.frame_ref.timestamp_seconds != secondary.frame_ref.timestamp_seconds
                    or (primary.frame_ref.width, primary.frame_ref.height)
                    != (secondary.frame_ref.width, secondary.frame_ref.height)
                    or primary.label != self.query.entity
                    or secondary.label != self.query.secondary_entity
                ):
                    raise ValueError("relation evidence must pair both entities in the same frame")
                policy = bundle.relation_policy
                if not spatial_relation_holds(
                    self.query.relation, primary.bbox_xyxy, secondary.bbox_xyxy,
                    width=primary.frame_ref.width, height=primary.frame_ref.height,
                    near_distance_fraction=(policy["near_distance_fraction"] if policy else None),
                ):
                    raise ValueError("relation is not supported by the cited boxes")
        self._validate_revision()

    def _validate_revision(self) -> None:
        payload = self._wire_payload()
        json.dumps(payload, allow_nan=False)
        if self.query.query_id.startswith("q_rev_") and self.logical_query_id is None:
            raise ValueError("immutable query revision requires logical_query_id")
        if self.logical_query_id is not None and self.query.query_id != _revision_id(payload):
            raise ValueError("query revision does not match immutable result content")

    def to_dict(self) -> dict[str, Any]:
        self.validate_evidence()
        return self._wire_payload()

    def _wire_payload(self) -> dict[str, Any]:
        evidence = None
        if self.evidence is not None:
            evidence = {
                "query_id": self.evidence.query_id,
                "run_id": self.evidence.run_id,
                "video_id": self.evidence.video_id,
                "source_sha256": self.evidence.source_sha256,
                "snapshot_id": self.evidence.snapshot_id,
                "coverage_start_seconds": self.evidence.coverage_start_seconds,
                "coverage_end_seconds": self.evidence.coverage_end_seconds,
                "coverage_complete": self.evidence.coverage_complete,
                "gaps": list(self.evidence.gaps),
                "relation_policy": deepcopy(self.evidence.relation_policy),
                "items": [
                    {
                        "evidence_id": item.evidence_id,
                        "video_id": item.frame_ref.video_id,
                        "frame_index": item.frame_ref.frame_index,
                        "timestamp_seconds": item.frame_ref.timestamp_seconds,
                        "timestamp_quality": item.frame_ref.timestamp_quality.value,
                        "timestamp_origin": item.frame_ref.timestamp_origin,
                        "frame_ref": _portable_frame(item.frame_ref),
                        "label": item.label,
                        "bbox_xyxy": list(item.bbox_xyxy),
                        "score": item.score,
                        "source_sha256": item.source_sha256,
                        "snapshot_id": item.snapshot_id,
                        "artifact_id": item.artifact_id,
                        "source_stage": item.source_stage,
                        "role": item.role,
                        "provenance": deepcopy(item.provenance),
                    }
                    for item in self.evidence.items
                ],
            }
        return {
            "schema_version": RESULT_SCHEMA_VERSION,
            "logical_query_id": self.logical_query_id,
            "query_config": deepcopy(self.query_config),
            "query": self.query.to_dict(),
            "video_id": self.video_id,
            "run_id": self.run_id,
            "processing_status": self.processing_status.value,
            "support_status": self.support_status.value,
            "mode": self.mode,
            "answer": deepcopy(self.answer),
            "evidence": evidence,
            "error_code": self.error_code,
        }
