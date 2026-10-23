"""Explicit model registry and readiness checks for local or cloud profiles."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from video_intelligence.models.contracts import (
    AdapterState,
    ModelAdapter,
    ModelAdapterError,
    ModelCapabilities,
    ModelErrorCode,
    ModelMode,
    ModelResult,
    ModelSpec,
    OutputT,
)


@dataclass(frozen=True, slots=True)
class ModelRegistration:
    """Inspectable model entry; weights and backend are represented by their spec."""

    spec: ModelSpec
    capabilities: ModelCapabilities
    state: AdapterState

    def to_dict(self) -> dict[str, object]:
        return {
            "spec": self.spec.to_dict(),
            "capabilities": self.capabilities.to_dict(),
            "state": self.state.value,
        }


@dataclass(frozen=True, slots=True)
class ReadinessReport:
    """Truthful readiness for one role/model selection."""

    role: str
    model_id: str | None
    supported: bool
    available: bool
    loaded: bool
    ready: bool
    mode: ModelMode | None
    reason: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "role": self.role,
            "model_id": self.model_id,
            "supported": self.supported,
            "available": self.available,
            "loaded": self.loaded,
            "ready": self.ready,
            "mode": None if self.mode is None else self.mode.value,
            "reason": self.reason,
        }


class ModelRegistry:
    """Register adapters once and keep model selection explicit and inspectable."""

    def __init__(self) -> None:
        self._entries: dict[str, tuple[ModelSpec, ModelAdapter[Any, Any]]] = {}

    def register(self, adapter: ModelAdapter[Any, Any], spec: ModelSpec) -> ModelRegistration:
        if spec.model_id in self._entries:
            raise ValueError(f"model is already registered: {spec.model_id}")
        capabilities = adapter.capabilities()
        if capabilities.model_id != spec.model_id or capabilities.role != spec.role:
            raise ValueError("model spec does not match adapter capabilities")
        if capabilities.mode is not spec.mode:
            raise ValueError("model spec mode does not match adapter capabilities")
        self._entries[spec.model_id] = (spec, adapter)
        return self.get(spec.model_id)

    def get(self, model_id: str) -> ModelRegistration:
        spec, adapter = self._entry(model_id)
        return ModelRegistration(spec, adapter.capabilities(), adapter.state)

    def list_models(self) -> tuple[ModelRegistration, ...]:
        return tuple(self.get(model_id) for model_id in sorted(self._entries))

    def readiness(self, role: str, *, model_id: str | None = None) -> ReadinessReport:
        candidates = [
            (registered_id, entry)
            for registered_id, entry in self._entries.items()
            if entry[0].role == role and (model_id is None or registered_id == model_id)
        ]
        if not candidates:
            return ReadinessReport(
                role=role,
                model_id=model_id,
                supported=False,
                available=False,
                loaded=False,
                ready=False,
                mode=None,
                reason="no adapter registered for role/model",
            )
        if len(candidates) > 1:
            return ReadinessReport(
                role=role,
                model_id=None,
                supported=True,
                available=False,
                loaded=False,
                ready=False,
                mode=None,
                reason="multiple adapters require an explicit model_id",
            )
        selected_id, (spec, adapter) = candidates[0]
        capabilities = adapter.capabilities()
        if adapter.state not in {AdapterState.UNLOADED, AdapterState.LOADED}:
            return ReadinessReport(
                role=role,
                model_id=selected_id,
                supported=True,
                available=False,
                loaded=False,
                ready=False,
                mode=spec.mode,
                reason=f"adapter is {adapter.state.value}; create a new adapter or await cleanup",
            )
        available, availability_reason = adapter.availability(spec)
        if not available:
            return ReadinessReport(
                role=role,
                model_id=selected_id,
                supported=True,
                available=False,
                loaded=adapter.state is AdapterState.LOADED,
                ready=False,
                mode=spec.mode,
                reason=availability_reason or capabilities.reason or "model backend is unavailable",
            )
        return ReadinessReport(
            role=role,
            model_id=selected_id,
            supported=True,
            available=True,
            loaded=adapter.state is AdapterState.LOADED,
            ready=True,
            mode=spec.mode,
            reason=None if adapter.state is AdapterState.LOADED else "available; lazy load pending",
        )

    def load(self, model_id: str) -> ModelRegistration:
        spec, adapter = self._entry(model_id)
        adapter.load(spec)
        return self.get(model_id)

    def infer(
        self,
        model_id: str,
        inputs: Sequence[Any],
        *,
        cancel_check: Callable[[], bool] | None = None,
    ) -> ModelResult[OutputT]:
        spec, adapter = self._entry(model_id)
        if adapter.state is AdapterState.UNLOADED:
            adapter.load(spec)
        if not adapter.capabilities().available:
            raise ModelAdapterError(
                ModelErrorCode.MODEL_UNAVAILABLE,
                adapter.capabilities().reason or "model backend is unavailable",
                model_id=model_id,
            )
        return adapter.infer(inputs, cancel_check=cancel_check)

    def close(self, model_id: str) -> ModelRegistration:
        _spec, adapter = self._entry(model_id)
        adapter.close()
        return self.get(model_id)

    def close_all(self) -> None:
        for model_id in self._entries:
            self.close(model_id)

    def _entry(self, model_id: str) -> tuple[ModelSpec, ModelAdapter[Any, Any]]:
        try:
            return self._entries[model_id]
        except KeyError as error:
            raise ModelAdapterError(
                ModelErrorCode.MODEL_UNAVAILABLE,
                "model is not registered",
                model_id=model_id,
            ) from error
