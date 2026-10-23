"""Closed-set detector adapter boundary without requiring model weights."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from video_intelligence.detection.base import DetectionBatch
from video_intelligence.models.contracts import ModelAdapter, ModelSpec, UnavailableModelAdapter


@dataclass(frozen=True, slots=True)
class ClosedSetCandidate:
    """An adapter and matching detection ModelSpec for comparison runs."""

    adapter: ModelAdapter[Any, DetectionBatch]
    spec: ModelSpec

    def __post_init__(self) -> None:
        if not isinstance(self.adapter, ModelAdapter):
            raise TypeError("adapter must be a ModelAdapter")
        if not isinstance(self.spec, ModelSpec):
            raise TypeError("spec must be a ModelSpec")
        if self.spec.role != "detection":
            raise ValueError("closed-set spec role must be detection")
        capabilities = self.adapter.capabilities()
        if self.spec.model_id != capabilities.model_id:
            raise ValueError("spec model_id does not match adapter capabilities")
        if self.spec.mode is not capabilities.mode:
            raise ValueError("spec mode does not match adapter capabilities")


class UnavailableClosedSetAdapter(UnavailableModelAdapter):
    """Explicit closed-set fallback that never fabricates detector outputs."""

    def __init__(
        self,
        *,
        model_id: str = "closed_set_unavailable",
        reason: str = "closed-set detector backend or weights are unavailable",
    ) -> None:
        super().__init__(
            model_id=model_id,
            role="detection",
            reason=reason,
        )
