"""Bounded semantic-verifier transport contract, without answer promotion.

Matching a label, a box or its propagated mask is not independent verification.
This module binds future assessments to source pixels and immutable observations.
It supplies no qualified backend and cannot change a QueryResult's support status.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

import numpy as np

from video_intelligence.core.types import TimestampQuality
from video_intelligence.retrieval.schemas import ParseStatus, ProcessingStatus, QueryResult
from video_intelligence.video.decoder import DecodedFrame

SCHEMA_VERSION = "video_intelligence.semantic_assessment.v1"
MAX_FRAMES = 8
MAX_ITEMS = 16
MAX_PIXELS = 8_388_608
MAX_REQUEST_BYTES = 262_144
MAX_RESPONSE_BYTES = 32_768


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _digest(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sha(value):
    if (not isinstance(value, str) or len(value) != 64
            or any(c not in "0123456789abcdef" for c in value)):
        raise ValueError("semantic identity must be a lowercase SHA-256 digest")
    return value


def _rgb_hash(frame):
    ref, image = frame.frame_ref, frame.image
    if (ref.timestamp_quality is not TimestampQuality.EXACT_PTS
            or ref.pts is None or ref.time_base is None or ref.source_sha256 is None):
        raise ValueError("semantic input requires exact source-bound PTS")
    if (not isinstance(image, np.ndarray) or image.dtype != np.uint8
            or image.shape != (ref.height, ref.width, 3)):
        raise ValueError("semantic input must be source-sized uint8 RGB")
    return hashlib.sha256(image.tobytes(order="C")).hexdigest()


@dataclass(frozen=True, slots=True, init=False)
class SemanticRequest:
    """Detached canonical transport body; not a verification or authorization receipt."""

    _body: str

    def __init__(self, *args, **kwargs):
        raise TypeError("use SemanticRequest.from_result with bound result and decoded frames")

    @classmethod
    def from_result(cls, result: QueryResult, frames: tuple[DecodedFrame, ...], *,
                    backend_sha256: str) -> SemanticRequest:
        """Include every retained candidate, or reject the request on budget overflow.

        No truncation hides missing candidates or breaks spatial pairs. Frames must
        be the exact decoded RGB supplied to the future backend, in evidence order.
        Hash binding detects substitution; it does not authenticate a caller's pixels.
        """
        payload = result.to_dict()
        if (result.processing_status is not ProcessingStatus.COMPLETE
                or result.query.parse_status is not ParseStatus.SUPPORTED
                or not result.query.query_id.startswith("q_rev_")
                or result.logical_query_id is None or result.evidence is None):
            raise ValueError("semantic request requires complete revision-bound candidates")
        items = result.evidence.items
        if not 1 <= len(items) <= MAX_ITEMS or not 1 <= len(frames) <= MAX_FRAMES:
            raise ValueError("semantic frame/item budget exceeded")
        if sum(f.frame_ref.width * f.frame_ref.height for f in frames) > MAX_PIXELS:
            raise ValueError("semantic pixel budget exceeded")
        expected = []
        seen_ids = set()
        for item in items:
            if item.evidence_id in seen_ids:
                raise ValueError("semantic candidate IDs must be unique")
            seen_ids.add(item.evidence_id)
            if item.frame_ref.identity_key not in expected:
                expected.append(item.frame_ref.identity_key)
        if [f.frame_ref.identity_key for f in frames] != expected:
            raise ValueError("semantic frames must match all candidates in source order")
        frame_rows = []
        for frame in frames:
            matching = [i for i in items
                        if i.frame_ref.identity_key == frame.frame_ref.identity_key]
            if any((i.frame_ref.width, i.frame_ref.height, i.frame_ref.timestamp_seconds,
                    i.frame_ref.timestamp_origin, i.frame_ref.time_base)
                   != (frame.frame_ref.width, frame.frame_ref.height,
                       frame.frame_ref.timestamp_seconds, frame.frame_ref.timestamp_origin,
                       frame.frame_ref.time_base)
                   for i in matching):
                raise ValueError("semantic frame geometry/timestamp differs")
            row = next(i for i in payload["evidence"]["items"]
                       if i["evidence_id"] == matching[0].evidence_id)["frame_ref"]
            rgb_sha256 = _rgb_hash(frame)
            if any(i.provenance.get("frame_rgb_sha256") != rgb_sha256
                   or i.provenance.get("frame_rgb_encoding") != "rgb8_hwc" for i in matching):
                raise ValueError("semantic pixels differ from grounded candidate RGB")
            frame_rows.append({"frame_ref": row, "rgb_sha256": rgb_sha256})
        groups = []
        step = 2 if result.query.relation else 1
        if len(items) % step:
            raise ValueError("semantic relation requires complete pairs")
        for offset in range(0, len(items), step):
            group = items[offset:offset + step]
            if step == 2 and (
                group[0].role != "primary" or group[1].role != "secondary"
                or group[0].frame_ref.identity_key != group[1].frame_ref.identity_key
                or group[0].label != result.query.entity
                or group[1].label != result.query.secondary_entity
            ):
                raise ValueError("semantic relation candidates must bind both query entities")
            if step == 1 and (group[0].role != "entity" or group[0].label != result.query.entity):
                raise ValueError("semantic candidate does not bind the query entity")
            ids = [i.evidence_id for i in group]
            groups.append({"unit_id": _digest(_canonical(ids)), "evidence_ids": ids})
        body = _canonical({
            "schema_version": SCHEMA_VERSION, "backend_sha256": _sha(backend_sha256),
            "result_sha256": _digest(_canonical(payload)),
            "query_revision": result.query.query_id,
            "query": result.query.to_dict(),
            "source_sha256": result.evidence.source_sha256,
            "snapshot_id": result.evidence.snapshot_id,
            "coverage_complete": result.evidence.coverage_complete,
            "gaps": list(result.evidence.gaps),
            "frames": frame_rows, "candidates": payload["evidence"]["items"],
            "units": groups, "omitted_evidence_ids": [],
        })
        if len(body.encode("utf-8")) > MAX_REQUEST_BYTES:
            raise ValueError("semantic request byte budget exceeded")
        request = object.__new__(cls)
        object.__setattr__(request, "_body", body)
        return request

    @property
    def sha256(self):
        return _digest(self._body)

    def to_dict(self):
        return json.loads(self._body)

    def validate_frames(self, frames: tuple[DecodedFrame, ...]):
        """Recheck immediately before inference; decoded arrays remain mutable."""
        rows = self.to_dict()["frames"]
        if len(rows) != len(frames):
            raise ValueError("semantic frame count changed")
        for row, frame in zip(rows, frames, strict=True):
            ref = frame.frame_ref
            expected = row["frame_ref"]
            if (any(getattr(ref, key) != expected[key] for key in (
                    "video_id", "source_sha256", "stream_index", "pts", "time_base",
                    "width", "height", "timestamp_seconds", "timestamp_origin"))
                    or _rgb_hash(frame) != row["rgb_sha256"]):
                raise ValueError("semantic frame identity/pixels changed")


class SemanticVerdict(StrEnum):
    SUPPORTS = "supports_query"
    CONTRADICTS = "contradicts_query"
    UNCERTAIN = "uncertain"


@dataclass(frozen=True, slots=True)
class UnitAssessment:
    unit_id: str
    verdict: SemanticVerdict


@dataclass(frozen=True, slots=True)
class SemanticReport:
    """Untrusted model assessments, even when the wire contract is valid."""

    request_sha256: str
    backend_sha256: str
    assessments: tuple[UnitAssessment, ...]

    @classmethod
    def from_json(cls, request: SemanticRequest, raw: str) -> SemanticReport:
        if not isinstance(raw, str) or len(raw.encode("utf-8")) > MAX_RESPONSE_BYTES:
            raise ValueError("semantic response byte budget exceeded")

        def unique(pairs):
            result = {}
            for key, value in pairs:
                if key in result:
                    raise ValueError("semantic response has duplicate keys")
                result[key] = value
            return result

        def nonfinite(value):
            raise ValueError("semantic response contains a nonfinite number")

        try:
            body = json.loads(raw, object_pairs_hook=unique, parse_constant=nonfinite)
            if not isinstance(body, dict) or set(body) != {
                "schema_version", "request_sha256", "backend_sha256", "assessments",
            }:
                raise ValueError("semantic response fields differ")
            source = request.to_dict()
            if (body["schema_version"] != SCHEMA_VERSION
                    or body["request_sha256"] != request.sha256
                    or body["backend_sha256"] != source["backend_sha256"]):
                raise ValueError("semantic response request/backend binding differs")
            rows = body["assessments"]
            if not isinstance(rows, list) or len(rows) != len(source["units"]):
                raise ValueError("semantic response must assess every unit exactly once")
            assessments = []
            for row, unit in zip(rows, source["units"], strict=True):
                if (not isinstance(row, dict) or set(row) != {"unit_id", "verdict"}
                        or row["unit_id"] != unit["unit_id"]):
                    raise ValueError("semantic response unit identity/order differs")
                assessments.append(UnitAssessment(row["unit_id"], SemanticVerdict(row["verdict"])))
            return cls(request.sha256, body["backend_sha256"], tuple(assessments))
        except (KeyError, TypeError, RecursionError, json.JSONDecodeError) as error:
            raise ValueError("invalid semantic response") from error


class SemanticVerifier(Protocol):
    """Future local backend. Runtime and quality qualification are separate gates."""

    def assess(self, request: SemanticRequest,
               frames: tuple[DecodedFrame, ...]) -> SemanticReport: ...
