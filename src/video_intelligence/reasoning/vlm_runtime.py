"""Deferred Qwen3-VL entry point; model loading awaits patched SDK qualification."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path

import cv2
import numpy as np

from video_intelligence.core.types import FrameRef
from video_intelligence.detection.backends import _nonfinite, _unique
from video_intelligence.models import hash_weights_location
from video_intelligence.storage.evidence_frames import EvidenceFrameConfig, _json_bytes, _png_rgb
from video_intelligence.storage.evidence_frames import _safe as safe_path
from video_intelligence.storage.evidence_masks import _png_mask


def _load_runtime():
    version = importlib.metadata.version("transformers")
    raise RuntimeError(
        f"VLM model runtime disabled pending patched SDK qualification; installed={version}; "
        "legacy transformers==4.57.1 is not permitted"
    )


def predict(request, directory):
    from video_intelligence.reasoning.contracts import SYSTEM_PROMPT
    from video_intelligence.reasoning.vlm import backend_from_effective, validate_grids

    backend = backend_from_effective(request["backend"])
    spec, config = backend.spec, backend.config
    safe_path(Path(spec.weights_location), directory=True)
    source_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    if (request["schema_version"] != "video_intelligence.vlm_input.v1"
            or request["backend_sha256"] != backend.identity_sha256
            or request["system_prompt"] != SYSTEM_PROMPT
            or request["runtime_source_sha256"] != source_sha
            or hash_weights_location(spec.weights_location) != spec.weights_sha256):
        raise ValueError("VLM child backend/source/weights binding differs")
    rows = request["media"]
    if not isinstance(rows, list) or not 1 <= len(rows) <= config.max_images:
        raise ValueError("VLM input image budget exceeded")
    images = []
    for i, row in enumerate(rows):
        ref = FrameRef(**row["frame_ref"])
        if (row["file"] != f"image_{i:02d}.png" or row["kind"] not in {"rgb", "mask"}
                or ref.width * ref.height > config.max_source_pixels):
            raise ValueError("VLM input image path/geometry differs")
        path = directory / row["file"]
        safe_path(path, directory=False)
        with path.open("rb") as stream:
            payload = stream.read(config.max_source_pixels * 4 + 4097)
        if (len(payload) > config.max_source_pixels * 4 + 4096
                or hashlib.sha256(payload).hexdigest() != row["reference"]["png_sha256"]):
            raise ValueError("VLM input PNG byte/hash bound differs")
        limits = EvidenceFrameConfig(max_frame_pixels=config.max_source_pixels)
        if row["kind"] == "rgb":
            image = _png_rgb(payload, ref, limits)
            digest = hashlib.sha256(image.tobytes()).hexdigest()
            expected = row["reference"]["rgb_sha256"]
        else:
            mask = _png_mask(payload, ref, limits)
            digest = hashlib.sha256(mask.astype(np.uint8).tobytes()).hexdigest()
            expected = row["reference"]["binary_sha256"]
            image = np.repeat((mask.astype(np.uint8) * 255)[..., None], 3, axis=2)
        if digest != expected:
            raise ValueError("VLM decoded RGB/mask differs")
        # Bound geometry BEFORE processor allocations; processor may only round to patch grid.
        scale = min(1., 480 / max(ref.width, ref.height),
                    (config.max_image_pixels / (ref.width * ref.height)) ** .5 * .9)
        size = (max(32, int(ref.width * scale)), max(32, int(ref.height * scale)))
        image = cv2.resize(image, size, interpolation=(
            cv2.INTER_NEAREST if row["kind"] == "mask" else cv2.INTER_AREA
        ))
        images.append(image)
    torch, model_class, processor_class = _load_runtime()
    dtype = getattr(torch, spec.dtype)
    processor = processor_class.from_pretrained(spec.weights_location, local_files_only=True,
                                                 trust_remote_code=False)
    content = [{"type": "image"} for _ in images] + [{
        "type": "text", "text": json.dumps(request["prompt_data"], ensure_ascii=False,
                                             sort_keys=True, allow_nan=False),
    }]
    messages = [{"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
                {"role": "user", "content": content}]
    rendered = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    if not isinstance(rendered, str) or len(rendered.encode("utf-8")) > 128 * 1024:
        raise ValueError("VLM rendered prompt byte budget exceeded")
    inputs = processor(text=[rendered], images=images, return_tensors="pt", padding=True,
                       min_pixels=1024, max_pixels=config.max_image_pixels)
    ids, pixels, grid = inputs["input_ids"], inputs["pixel_values"], inputs["image_grid_thw"]
    if (ids.ndim != 2 or ids.shape[0] != 1 or not 1 <= ids.shape[1] <= config.max_input_tokens
            or ids.dtype != torch.int64 or tuple(grid.shape) != (len(images), 3)
            or grid.dtype != torch.int64 or not torch.isfinite(pixels).all().item()):
        raise ValueError("VLM processor token/grid/pixel values exceed policy")
    grids = grid.tolist()
    validate_grids(grids, len(images), config)
    patches = sum(t * h * w for t, h, w in grids)
    if tuple(pixels.shape) != (patches, 3 * 2 * 16 * 16):
        raise ValueError("VLM processed pixel tensor differs from Qwen3 patch policy")
    model = model_class.from_pretrained(
        spec.weights_location, local_files_only=True, trust_remote_code=False,
        use_safetensors=True, torch_dtype=dtype, attn_implementation="eager",
    )
    try:
        model.to(spec.device, dtype=dtype)
        model.eval()
        vision = model.config.vision_config
        if (model.config.model_type != "qwen3_vl" or vision.patch_size != 16
                or vision.temporal_patch_size != 2 or vision.spatial_merge_size != 2
                or model.dtype != dtype or not (str(model.device) == spec.device
                    or spec.device == "cuda" and str(model.device).startswith("cuda:"))):
            raise ValueError("VLM actual architecture/device/dtype differs")
        moved = {key: value.to(model.device) for key, value in inputs.items()}
        with torch.inference_mode():
            generated = model.generate(**moved, do_sample=False, num_beams=1,
                                       max_new_tokens=config.max_new_tokens)
        if (generated.ndim != 2 or generated.shape[0] != 1 or generated.dtype != torch.int64
                or not 1 <= generated.shape[1] - ids.shape[1] <= config.max_new_tokens
                or not torch.equal(generated[:, :ids.shape[1]].cpu(), ids.cpu())):
            raise ValueError("VLM generation length/prefix differs")
        output = generated[:, ids.shape[1]:]
        raw = processor.batch_decode(output, skip_special_tokens=True,
                                     clean_up_tokenization_spaces=False)[0]
        if not isinstance(raw, str) or len(raw.encode("utf-8")) > 16 * 1024:
            raise ValueError("VLM generated text exceeds byte budget")
        runtime = {"request_sha256": hashlib.sha256(_json_bytes(request)).hexdigest(),
                   "runtime_source_sha256": request["runtime_source_sha256"],
                   "transformers_version": "4.57.1", "actual_device": str(model.device),
                   "actual_dtype": str(model.dtype).removeprefix("torch."),
                   "input_tokens": ids.shape[1], "output_tokens": output.shape[1],
                   "image_grid_thw": grids,
                   "rendered_prompt_sha256": hashlib.sha256(rendered.encode("utf-8")).hexdigest()}
        return {"status": "succeeded", "raw_output": raw, "runtime": runtime}
    finally:
        model = processor = inputs = None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", required=True, type=Path)
    args = parser.parse_args(argv)
    safe_path(args.request.absolute(), directory=False)
    with args.request.open("rb") as stream:
        payload = stream.read(1024 * 1024 + 1)
    if len(payload) > 1024 * 1024:
        raise ValueError("VLM request exceeds 1 MiB")
    request = json.loads(payload, object_pairs_hook=_unique, parse_constant=_nonfinite)
    try:
        if payload != _json_bytes(request):
            raise ValueError("VLM request is not canonical")
        result = predict(request, args.request.absolute().parent)
        code = 0
    except Exception as error:
        error_code = ("vlm_invalid_output" if isinstance(error, ValueError)
                      else "vlm_model_unavailable")
        if isinstance(error, MemoryError) or type(error).__name__ == "OutOfMemoryError":
            error_code = "vlm_oom"
        result = {"status": "failed", "error_code": error_code,
                  "error": f"{type(error).__name__}: {error}"[:1000]}
        code = 2
    with (args.request.parent / "result.json").open("x", encoding="utf-8") as stream:
        json.dump({**result, "request_sha256": hashlib.sha256(payload).hexdigest()},
                  stream, sort_keys=True, allow_nan=False)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
