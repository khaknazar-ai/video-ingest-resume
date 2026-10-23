"""Dependency-light contracts and explicit fallback adapters for model stages."""

from video_intelligence.models.contracts import (
    AdapterState,
    MockAdapter,
    ModelAdapter,
    ModelAdapterError,
    ModelCapabilities,
    ModelErrorCode,
    ModelMode,
    ModelResult,
    ModelSpec,
    ResourceMeasurement,
    UnavailableModelAdapter,
    hash_weights_location,
)
from video_intelligence.models.registry import ModelRegistration, ModelRegistry, ReadinessReport

__all__ = [
    "AdapterState",
    "ModelAdapter",
    "ModelAdapterError",
    "ModelCapabilities",
    "ModelErrorCode",
    "ModelMode",
    "hash_weights_location",
    "ModelResult",
    "ModelRegistration",
    "ModelRegistry",
    "ModelSpec",
    "MockAdapter",
    "ResourceMeasurement",
    "UnavailableModelAdapter",
    "ReadinessReport",
]
