"""Optional local-only Grounding DINO adapter; no import-time ML dependency."""

from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import re
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

from video_intelligence.detection.base import DetectionObservation
from video_intelligence.detection.grounding import (
    GroundingBatch,
    GroundingDetection,
    GroundingInput,
)
from video_intelligence.detection.prompts import PromptPlan
from video_intelligence.models.contracts import (
    ModelAdapter,
    ModelAdapterError,
    ModelCapabilities,
    ModelErrorCode,
    ModelMode,
    ModelSpec,
)
from video_intelligence.models.local_bundle import validate_local_bundle

MODEL_ID = "IDEA-Research/grounding-dino-tiny"
TRANSFORMERS_VERSION = "5.10.2"
POLICY_VERSION = "grounding_dino_single_phrase_hf5102.v3"


@dataclass(frozen=True, slots=True)
class GroundingDinoConfig:
    box_threshold: float = 0.35
    text_threshold: float = 0.25
    max_phrases: int = 8
    max_detections: int = 32

    def __post_init__(self):
        for name in ("box_threshold", "text_threshold"):
            value = getattr(self, name)
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not 0 < value < 1):
                raise ValueError(f"{name} must be finite in (0,1)")
        for name, cap in (("max_phrases", 8), ("max_detections", 256)):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= cap:
                raise ValueError(f"{name} must be an integer in [1,{cap}]")

    def to_dict(self):
        return asdict(self)


def validate_spec(spec: ModelSpec) -> None:
    if not isinstance(spec, ModelSpec):
        raise TypeError("spec must be a ModelSpec")
    if (spec.model_id != MODEL_ID or spec.source != MODEL_ID or spec.role != "detection"
            or spec.mode is not ModelMode.INFERENCE or spec.batch_size != 1):
        raise ValueError("Grounding DINO requires exact model/source, detection/inference, batch 1")
    if spec.allow_download or not spec.weights_location or not spec.weights_sha256:
        raise ValueError("Grounding DINO requires local hash-bound weights and disabled downloads")
    if re.fullmatch(r"[0-9a-f]{40}", spec.revision) is None:
        raise ValueError("Grounding DINO revision must be an immutable 40-character commit")
    if spec.device != "cpu" and re.fullmatch(r"cuda(?::[0-9]+)?", spec.device) is None:
        raise ValueError("Grounding DINO supports explicit CPU or CUDA devices only")
    if spec.dtype not in {"float32", "float16", "bfloat16"}:
        raise ValueError("unsupported Grounding DINO precision")
    if spec.device == "cpu" and spec.dtype != "float32":
        raise ValueError("CPU Grounding DINO requires float32")
    if spec.max_image_side > 1024 or spec.timeout_seconds > 60:
        raise ValueError("Grounding DINO image/time budgets exceed this adapter's limits")


def _missing_dependencies():
    return tuple(name for name in ("torch", "transformers")
                 if importlib.util.find_spec(name) is None)


def _load_runtime():
    if importlib.metadata.version("transformers") != TRANSFORMERS_VERSION:
        raise RuntimeError(f"Grounding DINO requires transformers=={TRANSFORMERS_VERSION}")
    import torch
    from transformers import GroundingDinoForObjectDetection, GroundingDinoProcessor

    return torch, GroundingDinoProcessor, GroundingDinoForObjectDetection


def _array(value):
    if not hasattr(value, "detach"):
        raise ValueError("Grounding DINO must return tensors")
    return np.asarray(value.detach().float().cpu().numpy())


class GroundingDinoAdapter(ModelAdapter):
    """One forward per registered phrase; outputs retain the full original prompt plan."""

    def __init__(self, config: GroundingDinoConfig | None = None):
        self.config = GroundingDinoConfig() if config is None else config
        if not isinstance(self.config, GroundingDinoConfig):
            raise TypeError("config must be a GroundingDinoConfig")
        missing = _missing_dependencies()
        super().__init__(ModelCapabilities(
            MODEL_ID, "detection", not missing, ModelMode.INFERENCE,
            supports_cpu=True, supports_cuda=True, max_batch_size=1,
            reason=f"missing optional dependencies: {', '.join(missing)}" if missing else None,
        ))
        self._processor = self._model = self._torch = None
        self._dtype = None

    def validate_prompt(self, prompt: PromptPlan):
        if not isinstance(prompt, PromptPlan):
            raise TypeError("prompt must be a PromptPlan")
        if prompt.negative_phrases:
            raise ValueError("Grounding DINO negative phrases are not supported by this policy")
        if len(prompt.positive_phrases) > self.config.max_phrases:
            raise ValueError("Grounding DINO phrase budget exceeded")
        if any(len(phrase) > 256 or "." in phrase for phrase in prompt.positive_phrases):
            raise ValueError("Grounding DINO phrases must be <=256 characters without periods")

    def availability(self, spec):
        try:
            validate_spec(spec)
        except (ValueError, OverflowError) as error:
            return False, str(error)
        available, reason = super().availability(spec)
        if not available:
            return available, reason
        try:
            if importlib.metadata.version("transformers") != TRANSFORMERS_VERSION:
                return False, f"Grounding DINO requires transformers=={TRANSFORMERS_VERSION}"
        except importlib.metadata.PackageNotFoundError:
            return False, "transformers package metadata is unavailable"
        if not Path(spec.weights_location).expanduser().is_dir():
            return False, "Grounding DINO weights_location must be an existing directory"
        return True, None

    def _load(self, spec):
        validate_spec(spec)
        source = str(Path(spec.weights_location).expanduser().resolve())
        try:
            validate_local_bundle(spec.weights_location, model_type="grounding-dino")
            self._torch, processor_loader, model_loader = _load_runtime()
            self._dtype = getattr(self._torch, spec.dtype)
            self._processor = processor_loader.from_pretrained(
                source, local_files_only=True, trust_remote_code=False, backend="pil",
            )
            # Keep partial state for ModelAdapter.close if a later check/load fails.
            self._model = model_loader.from_pretrained(
                source, local_files_only=True, trust_remote_code=False,
                use_safetensors=True, dtype=self._dtype, attn_implementation="eager",
                use_kernels=False, allow_all_kernels=False, disable_custom_kernels=True,
            )
            self._model.to(spec.device, dtype=self._dtype)
            self._model.eval()
            actual_device = str(self._model.device)
            device_matches = actual_device == spec.device or (
                spec.device == "cuda" and re.fullmatch(r"cuda:[0-9]+", actual_device) is not None
            )
            if (self._model.dtype != self._dtype or not device_matches
                    or self._model.config.model_type != "grounding-dino"
                    or self._model.config.backbone_config.model_type != "swin"
                    or self._model.config.text_config.model_type != "bert"
                    or not self._model.config.disable_custom_kernels):
                raise ValueError("loaded model architecture, dtype or device differs from spec")
        except MemoryError:
            raise
        except Exception as error:
            code = (ModelErrorCode.OOM if self._torch is not None
                    and isinstance(error, self._torch.cuda.OutOfMemoryError)
                    else ModelErrorCode.MODEL_UNAVAILABLE)
            raise ModelAdapterError(code, f"Grounding DINO local load failed: {error}",
                                    model_id=MODEL_ID) from error

    def _infer(self, inputs):
        if len(inputs) != 1 or not isinstance(inputs[0], GroundingInput):
            raise ValueError("Grounding DINO requires one typed GroundingInput")
        item = inputs[0]
        self.validate_prompt(item.prompt)
        ref, rgb = item.frame.frame_ref, item.frame.image
        if (not isinstance(rgb, np.ndarray) or rgb.dtype != np.uint8
                or rgb.shape != (ref.height, ref.width, 3)):
            raise ValueError("Grounding DINO expects canonical source RGB8")
        scale = min(1.0, self.spec.max_image_side / max(ref.width, ref.height))
        dimensions = (max(1, round(ref.width * scale)), max(1, round(ref.height * scale)))
        resized = rgb if dimensions == (ref.width, ref.height) else cv2.resize(
            rgb, dimensions, interpolation=cv2.INTER_LINEAR,
        )
        resized_hash = hashlib.sha256(resized.tobytes()).hexdigest()
        detections = []
        for index, phrase in enumerate(item.prompt.positive_phrases):
            text = phrase.lower() + "."
            try:
                encoded = self._processor(images=resized, text=text, return_tensors="pt",
                                          do_resize=False, truncation=False)
                if not isinstance(encoded, Mapping):
                    raise ValueError("processor must return a tensor mapping")
                pixels, tokens = encoded["pixel_values"], encoded["input_ids"]
                if (len(pixels.shape) != 4 or tuple(pixels.shape[:2]) != (1, 3)
                        or min(pixels.shape[-2:]) <= 0
                        or max(pixels.shape[-2:]) > self.spec.max_image_side
                        or len(tokens.shape) != 2 or tokens.shape[0] != 1
                        or not 1 <= tokens.shape[1] <= min(256, self._model.config.max_text_len)):
                    raise ValueError("processor image/token shape exceeds the declared budget")
                if not np.isfinite(_array(pixels)).all():
                    raise ValueError("processor pixels are not finite")
                encoded = {key: value.to(
                    self.spec.device,
                    **({"dtype": self._dtype} if value.is_floating_point() else {}),
                ) for key, value in encoded.items()}
                with self._torch.inference_mode():
                    outputs = self._model(**encoded)
                results = self._processor.post_process_grounded_object_detection(
                    outputs, encoded["input_ids"], threshold=self.config.box_threshold,
                    text_threshold=self.config.text_threshold,
                    target_sizes=[(ref.height, ref.width)],
                )
            except self._torch.cuda.OutOfMemoryError as error:
                raise ModelAdapterError(ModelErrorCode.OOM, "Grounding DINO inference CUDA OOM",
                                        model_id=MODEL_ID) from error
            if not isinstance(results, list) or len(results) != 1:
                raise ValueError("Grounding DINO postprocessor must return one result")
            boxes, scores = _array(results[0]["boxes"]), _array(results[0]["scores"])
            labels = results[0].get("text_labels")
            # HF 5.10.2's empty batch_decode emits [""] for zero detections.
            # Accept only this exact sentinel with correctly shaped empty tensors.
            if (boxes.shape == (0, 4) and scores.shape == (0,)
                    and isinstance(labels, list) and labels == [""]):
                labels = []
            if (boxes.ndim != 2 or boxes.shape[1] != 4 or scores.shape != (len(boxes),)
                    or len(boxes) > 900 or not np.isfinite(boxes).all()
                    or not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any()
                    or not isinstance(labels, (list, tuple)) or len(labels) != len(boxes)):
                raise ValueError("invalid Grounding DINO boxes/scores")
            for query_index, (box, score) in enumerate(zip(boxes, scores, strict=True)):
                label = labels[query_index]
                if (not isinstance(label, str) or not label.strip(" .")
                        or label.lower().strip(" .") not in phrase.lower()):
                    raise ValueError("decoded token label does not bind to the submitted phrase")
                if score <= self.config.box_threshold:
                    raise ValueError("postprocessor score contradicts configured threshold")
                x1, y1, x2, y2 = (float(value) for value in box)
                if x2 <= x1 or y2 <= y1:
                    raise ValueError("Grounding DINO returned a reversed/empty raw box")
                clipped = (max(0., x1), max(0., y1), min(float(ref.width), x2),
                           min(float(ref.height), y2))
                if clipped[2] <= clipped[0] or clipped[3] <= clipped[1]:
                    raise ValueError("Grounding DINO box lies entirely outside the source")
                entity = item.prompt.phrase_entity_ids[index]
                observation = DetectionObservation(
                    f"gdino_{ref.stream_index}_{ref.pts}_{index}_{query_index}", ref, entity,
                    clipped, float(score), MODEL_ID, ModelMode.INFERENCE, "grounding.dino_hf",
                    phrase=phrase, provenance={
                        "adapter_policy": POLICY_VERSION,
                        "transformers_version": TRANSFORMERS_VERSION,
                        "config": self.config.to_dict(), "conditioned_text": text,
                        "phrase_call_index": index, "query_index": query_index,
                        "decoded_label": label,
                        "raw_source_xyxy": [x1, y1, x2, y2], "clipping": "source_bounds",
                        "resized_rgb_sha256": resized_hash, "resized_width": dimensions[0],
                        "resized_height": dimensions[1], "resize": "opencv_inter_linear_down_only",
                        "actual_device": str(self._model.device),
                        "actual_dtype": str(self._model.dtype),
                    },
                )
                detections.append(GroundingDetection(
                    observation, entity, phrase, index, item.prompt.prompt_version,
                ))
                if len(detections) > self.config.max_detections:
                    raise ValueError("Grounding DINO detection budget exceeded; no truncation")
        return (GroundingBatch(ref, MODEL_ID, ModelMode.INFERENCE, "grounding.dino_hf",
                               item.prompt, tuple(detections)),)

    def _close(self):
        self._model = self._processor = self._torch = self._dtype = None
