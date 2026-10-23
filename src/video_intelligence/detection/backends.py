"""Explicit local backend configs, pinned into canonical observation requests."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

from video_intelligence.core.paths import ProjectPaths
from video_intelligence.detection.grounding import GroundingCandidate
from video_intelligence.detection.grounding_dino import (
    POLICY_VERSION,
    TRANSFORMERS_VERSION,
    GroundingDinoAdapter,
    GroundingDinoConfig,
    validate_spec,
)
from video_intelligence.models.contracts import ModelSpec
from video_intelligence.retrieval.query_parser import parse_query


def backend_digest(payload):
    return hashlib.sha256(json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False,
    ).encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class GroundingDinoFactory:
    spec: ModelSpec
    config: GroundingDinoConfig

    def __post_init__(self):
        validate_spec(self.spec)
        if not isinstance(self.config, GroundingDinoConfig):
            raise TypeError("config must be a GroundingDinoConfig")

    def to_dict(self):
        return {"adapter": "grounding_dino", "policy_version": POLICY_VERSION,
                "transformers_version": TRANSFORMERS_VERSION,
                "model_spec": self.spec.to_dict(), "config": self.config.to_dict()}

    @property
    def identity_sha256(self):
        return backend_digest(self.to_dict())

    def __call__(self, request, video):
        payload = request.to_dict()
        if payload.get("grounding_backend_sha256") != self.identity_sha256:
            raise ValueError("request does not pin the configured grounding backend")
        self.validate_plan(payload)
        return GroundingCandidate(GroundingDinoAdapter(self.config), self.spec)

    def validate_plan(self, payload):
        from video_intelligence.pipeline.observation_jobs import _pipeline_config, _prompts

        if self.config.max_detections > _pipeline_config(
            payload.get("config", {}),
        ).max_detections_per_frame:
            raise ValueError("grounder detection budget exceeds pipeline admission budget")
        registry = _prompts(payload["prompts"])
        query = parse_query(payload["query_text"], registry=registry)
        if query.parse_status.value == "supported":
            entities = (query.entity,) if query.secondary_entity is None else (
                query.entity, query.secondary_entity,
            )
            plan = registry.plan(payload["query_text"], entities)
            # Validate policy before constructing/loading an optional runtime.
            if plan.negative_phrases or len(plan.positive_phrases) > self.config.max_phrases:
                raise ValueError("prompt exceeds Grounding DINO positive-only phrase policy")
            if any(len(phrase) > 256 or "." in phrase for phrase in plan.positive_phrases):
                raise ValueError("Grounding DINO phrase text exceeds policy")


def _unique(pairs):
    values = {}
    for key, value in pairs:
        if key in values:
            raise ValueError(f"duplicate backend config field: {key}")
        values[key] = value
    return values


def _nonfinite(value):
    raise ValueError(f"nonfinite backend config value: {value}")


def load_grounding_backends(path: Path, *, paths: ProjectPaths | None = None):
    """Parse configuration without importing ML, reading weights or accepting code paths."""
    with Path(path).open("rb") as stream:
        raw = stream.read(128 * 1024 + 1)
    if len(raw) > 128 * 1024:
        raise ValueError("backend config exceeds 128 KiB")
    try:
        payload = json.loads(raw.decode("utf-8"), object_pairs_hook=_unique,
                             parse_constant=_nonfinite)
        if not isinstance(payload, dict) or set(payload) != {"schema_version", "backends"}:
            raise ValueError("invalid backend config fields")
        if payload["schema_version"] != "video_intelligence.grounding_backends.v1":
            raise ValueError("unsupported backend config schema")
        entries = payload["backends"]
        if not isinstance(entries, dict) or not 1 <= len(entries) <= 8:
            raise ValueError("backend config requires 1..8 explicit entries")
        paths = paths or ProjectPaths.from_environment()
        result = {}
        for name, entry in entries.items():
            if (not isinstance(name, str) or re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]{0,63}", name)
                    is None or name == "fixture_pixel"):
                raise ValueError("invalid or reserved backend name")
            if not isinstance(entry, dict) or set(entry) != {"adapter", "model_spec", "config"}:
                raise ValueError("backend entry requires adapter, model_spec and config")
            if entry["adapter"] != "grounding_dino":
                raise ValueError("backend adapter is not allowlisted")
            fields = dict(entry["model_spec"])
            location = fields.get("weights_location")
            if not isinstance(location, str) or not location.strip():
                raise ValueError("backend requires an explicit local weights_location")
            local = Path(location).expanduser()
            if not local.is_absolute():
                if local.drive or any(
                    part in {".", ".."} for part in location.replace("\\", "/").split("/")
                ):
                    raise ValueError("relative weights_location must remain under VI_MODELS_ROOT")
                local = paths.models_root / local
            fields["weights_location"] = str(local.resolve())
            result[name] = GroundingDinoFactory(
                ModelSpec(**fields), GroundingDinoConfig(**entry["config"]),
            )
        return result
    except (TypeError, KeyError, AttributeError, OverflowError) as error:
        raise ValueError("malformed backend configuration") from error
