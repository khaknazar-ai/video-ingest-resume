"""Strict local FixRes retrieval policy; lazy ML imports and no download fallback."""

from __future__ import annotations

import importlib.metadata
import re
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from video_intelligence.core.paths import ProjectPaths
from video_intelligence.detection.backends import _nonfinite, _unique, backend_digest
from video_intelligence.embeddings.base import EmbeddingFingerprint
from video_intelligence.embeddings.normalization import normalize_matrix
from video_intelligence.embeddings.siglip2 import SIGLIP2_MODEL_ID, SigLIP2Adapter, SigLIP2Input
from video_intelligence.models.contracts import (
    ModelAdapterError,
    ModelErrorCode,
    ModelMode,
    ModelSpec,
)
from video_intelligence.models.local_bundle import validate_local_bundle

TRANSFORMERS_VERSION = "5.10.2"
POLICY_VERSION = "siglip2_fixres224_first_phrase_lower64_hf5102.v2"


def validate_spec(spec):
    if (spec.model_id != SIGLIP2_MODEL_ID or spec.source != SIGLIP2_MODEL_ID
            or spec.role != "embedding" or spec.mode is not ModelMode.INFERENCE
            or spec.batch_size != 1 or spec.allow_download
            or not spec.weights_location or not spec.weights_sha256
            or re.fullmatch(r"[0-9a-f]{40}", spec.revision) is None):
        raise ValueError("local SigLIP2 requires exact model, inference/batch1 and local assets")
    if (spec.device != "cpu" and re.fullmatch(r"cuda(?::[0-9]+)?", spec.device) is None
            or spec.dtype not in {"float32", "float16", "bfloat16"}
            or spec.device == "cpu" and spec.dtype != "float32"
            or spec.max_image_side != 224 or spec.timeout_seconds > 60):
        raise ValueError("local SigLIP2 requires FixRes 224 and bounded supported precision/device")


@dataclass(frozen=True, slots=True)
class RetrievalConfig:
    frame_budget: int = 16
    max_runtime_seconds: float = 60.0
    seek_tolerance_seconds: float = 0.2
    max_input_pixels: int = 2_073_600
    max_source_bytes: int = 512 * 1024 * 1024

    def __post_init__(self):
        for name, cap in (("frame_budget", 256), ("max_input_pixels", 8_294_400),
                          ("max_source_bytes", 2 * 1024**3)):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= cap:
                raise ValueError(f"{name} must be an integer in [1,{cap}]")
        for name in ("max_runtime_seconds", "seek_tolerance_seconds"):
            value = getattr(self, name)
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not 0 < value <= 60):
                raise ValueError(f"{name} must be finite in (0,60]")

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True, slots=True)
class LocalRetrievalBackend:
    spec: ModelSpec
    config: RetrievalConfig

    def __post_init__(self):
        validate_spec(self.spec)
        if not isinstance(self.config, RetrievalConfig):
            raise TypeError("config must be a RetrievalConfig")

    def to_dict(self):
        return {"model_spec": self.spec.to_dict(), "config": self.config.to_dict(),
                "policy_version": POLICY_VERSION, "transformers_version": TRANSFORMERS_VERSION}

    @property
    def identity_sha256(self):
        return backend_digest(self.to_dict())

    @property
    def fingerprint(self):
        # Includes dtype/device, processor and inference policy, not only the model revision.
        return EmbeddingFingerprint.from_spec(self.spec, modality="image", dimension=768,
                                             processor_revision=f"siglip2:{self.identity_sha256}")

    def adapter(self):
        return LocalSigLIP2Adapter()


def load_retrieval_backend(path: Path, *, paths: ProjectPaths | None = None):
    import json

    with Path(path).open("rb") as stream:
        raw = stream.read(128 * 1024 + 1)
    if len(raw) > 128 * 1024:
        raise ValueError("retrieval config exceeds 128 KiB")
    try:
        payload = json.loads(raw.decode("utf-8"), object_pairs_hook=_unique,
                             parse_constant=_nonfinite)
        if (not isinstance(payload, dict)
                or set(payload) != {"schema_version", "model_spec", "config"}
                or payload["schema_version"] != "video_intelligence.local_retrieval.v1"):
            raise ValueError("invalid local retrieval config schema/fields")
        fields = dict(payload["model_spec"])
        location = fields.get("weights_location")
        if not isinstance(location, str) or not location.strip():
            raise ValueError("retrieval requires explicit local weights_location")
        local = Path(location).expanduser()
        if not local.is_absolute():
            if local.drive or any(x in {".", ".."} for x in location.replace("\\", "/").split("/")):
                raise ValueError("relative retrieval weights must remain under VI_MODELS_ROOT")
            local = (paths or ProjectPaths.from_environment()).models_root / local
        fields["weights_location"] = str(local.resolve())
        return LocalRetrievalBackend(ModelSpec(**fields), RetrievalConfig(**payload["config"]))
    except (TypeError, KeyError, AttributeError, OverflowError) as error:
        raise ValueError("malformed retrieval backend configuration") from error


def _load_runtime():
    if importlib.metadata.version("transformers") != TRANSFORMERS_VERSION:
        raise RuntimeError(f"local SigLIP2 requires transformers=={TRANSFORMERS_VERSION}")
    import torch
    from transformers import SiglipModel, SiglipProcessor

    return torch, SiglipProcessor, SiglipModel


class LocalSigLIP2Adapter(SigLIP2Adapter):
    """Strict local loader; the legacy base adapter no longer loads models."""

    def availability(self, spec):
        try:
            validate_spec(spec)
            available, reason = super().availability(spec)
            if not available:
                return available, reason
            if importlib.metadata.version("transformers") != TRANSFORMERS_VERSION:
                return False, f"local SigLIP2 requires transformers=={TRANSFORMERS_VERSION}"
        except (ValueError, importlib.metadata.PackageNotFoundError) as error:
            return False, str(error)
        return True, None

    def _load(self, spec):
        validate_spec(spec)
        try:
            validate_local_bundle(spec.weights_location, model_type="siglip")
            self._torch, processor, model = _load_runtime()
            self._dtype = getattr(self._torch, spec.dtype)
            source = self._resolve_source(spec)
            self._processor = processor.from_pretrained(
                source, local_files_only=True, trust_remote_code=False, backend="pil",
            )
            self._model = model.from_pretrained(
                source, local_files_only=True, trust_remote_code=False, use_safetensors=True,
                dtype=self._dtype, attn_implementation="eager", use_kernels=False,
                allow_all_kernels=False,
            )
            self._model.to(spec.device, dtype=self._dtype)
            self._model.eval()
            config = self._model.config
            actual = str(self._model.device)
            if (config.model_type != "siglip" or config.vision_config.image_size != 224
                    or config.vision_config.hidden_size != 768
                    or config.text_config.max_position_embeddings != 64
                    or self._model.dtype != self._dtype
                    or not (actual == spec.device or spec.device == "cuda"
                            and re.fullmatch(r"cuda:[0-9]+", actual))):
                raise ValueError("loaded SigLIP2 architecture/device/dtype differs from policy")
        except Exception as error:
            code = (ModelErrorCode.OOM if isinstance(error, MemoryError)
                    or getattr(self, "_torch", None) is not None
                    and isinstance(error, self._torch.cuda.OutOfMemoryError)
                    else ModelErrorCode.MODEL_UNAVAILABLE)
            raise ModelAdapterError(code, f"local SigLIP2 load failed: {error}",
                                    model_id=SIGLIP2_MODEL_ID) from error

    def _infer(self, inputs):
        if len(inputs) != 1 or not isinstance(inputs[0], SigLIP2Input):
            raise ValueError("local SigLIP2 requires one typed input")
        item = inputs[0]
        try:
            if item.modality.value == "image":
                rgb = item.value
                if (not isinstance(rgb, np.ndarray) or rgb.dtype != np.uint8
                        or rgb.ndim != 3 or rgb.shape[2] != 3 or min(rgb.shape[:2]) <= 0
                        or rgb.shape[0] * rgb.shape[1] > 8_294_400):
                    raise ValueError("local SigLIP2 requires canonical RGB8")
                encoded = self._processor(images=[rgb], return_tensors="pt")
                extractor = self._model.get_image_features
                expected_key, expected_shape = "pixel_values", (1, 3, 224, 224)
            else:
                if len(item.value) > 1024 or item.value != item.value.lower():
                    raise ValueError("SigLIP2 rendered text must be lowercase and bounded")
                encoded = self._processor(text=[item.value], return_tensors="pt",
                                          padding="max_length", max_length=64, truncation=False)
                extractor = self._model.get_text_features
                expected_key, expected_shape = "input_ids", (1, 64)
            if (not isinstance(encoded, Mapping)
                    or tuple(encoded[expected_key].shape) != expected_shape):
                raise ValueError("SigLIP2 processed input exceeds fixed image/token budget")
            if encoded[expected_key].is_floating_point() != (item.modality.value == "image"):
                raise ValueError("SigLIP2 processed pixel/token dtype differs from modality")
            for value in encoded.values():
                array = np.asarray(value.detach().float().cpu().numpy())
                if not np.isfinite(array).all():
                    raise ValueError("SigLIP2 processed tensor is not finite")
            encoded = {key: value.to(self.spec.device, **(
                {"dtype": self._dtype} if value.is_floating_point() else {}
            )) for key, value in encoded.items()}
            with self._torch.inference_mode():
                features = extractor(**encoded, return_dict=True).pooler_output
        except self._torch.cuda.OutOfMemoryError as error:
            raise ModelAdapterError(ModelErrorCode.OOM, "local SigLIP2 inference CUDA OOM",
                                    model_id=SIGLIP2_MODEL_ID) from error
        array = np.asarray(features.detach().float().cpu().numpy())
        if array.shape != (1, 768):
            raise ValueError("SigLIP2 feature tensor must have shape (1,768)")
        normalized = normalize_matrix(array, expected_dimension=768)
        return (tuple(float(value) for value in normalized[0]),)
