"""Lazy optional SigLIP2 adapter with truthful unavailable behavior."""

from __future__ import annotations

import importlib.util
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from video_intelligence.embeddings.base import EmbeddingModality
from video_intelligence.embeddings.normalization import normalize_matrix
from video_intelligence.models.contracts import (
    ModelAdapter,
    ModelAdapterError,
    ModelCapabilities,
    ModelErrorCode,
    ModelMode,
    ModelSpec,
)

SIGLIP2_MODEL_ID = "google/siglip2-base-patch16-224"
SIGLIP2_ROLE = "embedding"
_SUPPORTED_DTYPES = frozenset({"float32", "float16", "bfloat16"})


def _missing_dependencies() -> tuple[str, ...]:
    return tuple(
        name
        for name in ("torch", "transformers", "PIL")
        if importlib.util.find_spec(name) is None
    )


@dataclass(frozen=True, slots=True)
class SigLIP2Input:
    """One image or text value accepted by the optional SigLIP2 backend."""

    modality: EmbeddingModality | str
    value: object

    def __post_init__(self) -> None:
        try:
            object.__setattr__(self, "modality", EmbeddingModality(self.modality))
        except (TypeError, ValueError) as error:
            raise ValueError(f"unsupported SigLIP2 modality: {self.modality!r}") from error
        if self.value is None:
            raise ValueError("value must not be null")
        if self.modality is EmbeddingModality.TEXT and (
            not isinstance(self.value, str) or not self.value.strip()
        ):
            raise ValueError("text input value must be a non-empty string")


class SigLIP2Adapter(ModelAdapter[SigLIP2Input, tuple[float, ...]]):
    """Legacy contracts only; model loading requires LocalSigLIP2Adapter's strict policy."""

    def __init__(self) -> None:
        missing = _missing_dependencies()
        reason = (
            f"missing optional dependencies: {', '.join(missing)}"
            if missing
            else None
        )
        if type(self) is SigLIP2Adapter:
            reason = "legacy SigLIP2 model loading is disabled; use LocalSigLIP2Adapter"
        super().__init__(
            ModelCapabilities(
                model_id=SIGLIP2_MODEL_ID,
                role=SIGLIP2_ROLE,
                available=reason is None,
                mode=ModelMode.INFERENCE,
                supports_cpu=True,
                supports_cuda=True,
                max_batch_size=1,
                reason=reason,
            )
        )
        self._processor: Any = None
        self._model: Any = None
        self._dtype: Any = None

    @property
    def actual_dtype(self) -> str | None:
        """Backend precision after a successful load, rather than declared intent."""

        if self._model is None:
            return None
        return str(self._model.dtype).removeprefix("torch.")

    def _load(self, spec: ModelSpec) -> None:
        raise ModelAdapterError(
            ModelErrorCode.MODEL_UNAVAILABLE,
            "legacy SigLIP2 model loading is disabled; use LocalSigLIP2Adapter",
            model_id=spec.model_id,
        )

    def availability(self, spec: ModelSpec) -> tuple[bool, str | None]:
        if spec.dtype not in _SUPPORTED_DTYPES:
            return False, "SigLIP2 dtype must be float32, float16 or bfloat16"
        available, reason = super().availability(spec)
        if not available:
            return available, reason
        try:
            self._resolve_source(spec)
        except ModelAdapterError as error:
            return False, error.message
        return True, None

    @staticmethod
    def _resolve_source(spec: ModelSpec) -> str:
        if spec.weights_location is None:
            if not spec.allow_download:
                raise ModelAdapterError(
                    ModelErrorCode.MODEL_UNAVAILABLE,
                    "weights_location is required when downloads are disabled",
                    model_id=spec.model_id,
                )
            return spec.source
        location = Path(spec.weights_location).expanduser().resolve()
        if not location.is_dir():
            raise ModelAdapterError(
                ModelErrorCode.MODEL_UNAVAILABLE,
                "weights_location must be an existing local directory",
                model_id=spec.model_id,
            )
        return str(location)

    def _infer(self, inputs: tuple[SigLIP2Input, ...]) -> Sequence[tuple[float, ...]]:
        if self._processor is None or self._model is None:
            raise ModelAdapterError(
                ModelErrorCode.MODEL_UNAVAILABLE,
                "SigLIP2 backend is not loaded",
                model_id=SIGLIP2_MODEL_ID,
            )
        modalities = {item.modality for item in inputs}
        if len(modalities) != 1:
            raise ValueError("one inference batch must contain one modality")
        modality = next(iter(modalities))
        values = [item.value for item in inputs]
        try:
            if modality is EmbeddingModality.IMAGE:
                encoded = self._processor(images=values, return_tensors="pt")
                extractor = self._model.get_image_features
            else:
                encoded = self._processor(
                    text=values, return_tensors="pt", padding="max_length", max_length=64
                )
                extractor = self._model.get_text_features
            device = self.spec.device if self.spec is not None else "cpu"
            dtype = self._dtype or self._torch.float32
            encoded = {
                key: (
                    value.to(device, dtype=dtype)
                    if self._torch.is_tensor(value) and value.is_floating_point()
                    else value.to(device) if hasattr(value, "to") else value
                )
                for key, value in encoded.items()
            }
            with self._torch.inference_mode():
                features = extractor(**encoded)
        except self._torch.cuda.OutOfMemoryError as error:
            raise ModelAdapterError(
                ModelErrorCode.OOM,
                "SigLIP2 inference ran out of CUDA memory",
                model_id=SIGLIP2_MODEL_ID,
            ) from error
        if not hasattr(features, "detach"):
            raise ValueError("SigLIP2 feature output is not a tensor")
        # NumPy cannot represent torch.bfloat16; normalize in CPU float32.
        normalized = normalize_matrix(features.detach().float().cpu().numpy())
        if normalized.shape[0] != len(inputs):
            raise ValueError("SigLIP2 returned a row count different from the input batch")
        return tuple(tuple(float(value) for value in row) for row in normalized)

    def _close(self) -> None:
        self._processor = None
        self._model = None
        self._torch = None
        self._dtype = None
