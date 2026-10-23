"""Deferred SAM2 Video entry point; model loading awaits patched SDK qualification."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path

import numpy as np

from video_intelligence.core.types import FrameRef
from video_intelligence.models import ModelSpec, hash_weights_location
from video_intelligence.segmentation.masks import pack_mask


def _load_runtime():
    version = importlib.metadata.version("transformers")
    raise RuntimeError(
        f"SAM2 model runtime disabled pending patched SDK qualification; installed={version}; "
        "legacy transformers==4.57.1 is not permitted"
    )


def predict(request, directory):
    from video_intelligence.segmentation.sam2 import SAM2Backend, SAM2Config

    backend = request["backend"]
    configured = SAM2Backend(ModelSpec(**backend["model_spec"]), backend["python_executable"],
                             backend["python_sha256"], SAM2Config(**backend["config"]))
    if (backend != configured.to_dict() or request["backend_sha256"] != configured.identity_sha256
            or request["schema_version"] != "video_intelligence.sam2_window.v1"):
        raise ValueError("SAM2 child backend identity differs")
    torch, model_class, processor_class = _load_runtime()
    spec, config = configured.spec, configured.config
    if hash_weights_location(spec.weights_location) != spec.weights_sha256:
        raise ValueError("SAM2 child weights hash differs")
    rows, seeds = request["frames"], request["seeds"]
    if (not isinstance(rows, list) or not 1 <= len(rows) <= config.max_frames
            or not isinstance(seeds, list) or not 1 <= len(seeds) <= config.max_objects):
        raise ValueError("SAM2 child window budget exceeded")
    images = []
    width = height = None
    for index, row in enumerate(rows):
        ref = FrameRef(**row["frame_ref"])
        if (row["file"] != f"frame_{index:04d}.png"
                or ref.width * ref.height > config.max_input_pixels
                or width is not None and (ref.width, ref.height) != (width, height)):
            raise ValueError("SAM2 child frame geometry/path differs")
        width, height = ref.width, ref.height
        path = directory / row["file"]
        from video_intelligence.storage.evidence_frames import EvidenceFrameConfig, _png_rgb, _safe

        _safe(path, directory=False)
        if path.stat().st_size > config.max_input_pixels * 4 + 4096:
            raise ValueError("SAM2 input PNG byte budget exceeded")
        payload = path.read_bytes()
        if hashlib.sha256(payload).hexdigest() != row["png_sha256"]:
            raise ValueError("SAM2 child input PNG hash differs")
        rgb = _png_rgb(payload, ref, EvidenceFrameConfig(max_frame_pixels=config.max_input_pixels))
        if hashlib.sha256(rgb.tobytes()).hexdigest() != row["rgb_sha256"]:
            raise ValueError("SAM2 child RGB hash differs")
        images.append(rgb)
    dtype = getattr(torch, spec.dtype)
    model = processor = session = None
    try:
        processor = processor_class.from_pretrained(spec.weights_location, local_files_only=True,
                                                     trust_remote_code=False)
        model = model_class.from_pretrained(
            spec.weights_location, local_files_only=True, trust_remote_code=False,
            use_safetensors=True, torch_dtype=dtype, attn_implementation="eager",
        )
        model.to(spec.device, dtype=dtype)
        model.eval()
        if (model.config.model_type != "sam2_video" or model.config.image_size != 1024
                or model.dtype != dtype or not (str(model.device) == spec.device
                    or spec.device == "cuda" and str(model.device).startswith("cuda:"))):
            raise ValueError("actual SAM2 architecture/device/dtype differs")
        with torch.inference_mode():
            session = processor.init_video_session(
                video=images, inference_device=spec.device, dtype=dtype,
                inference_state_device="cpu", processing_device="cpu", video_storage_device="cpu",
                max_vision_features_cache_size=1,
            )
            processed_shape = (len(images), 3, 1024, 1024)
            if (tuple(session.video.shape) != processed_shape
                    or session.video.dtype != dtype or str(session.video.device) != "cpu"
                    or (session.video_height, session.video_width) != (height, width)):
                raise ValueError("SAM2 processed video exceeds fixed shape/device/dtype policy")
            seed_ids = []
            for seed in seeds:
                object_id, index = seed["object_id"], seed["frame_idx"]
                box = seed["bbox_xyxy"]
                if (isinstance(object_id, bool) or not isinstance(object_id, int) or object_id <= 0
                        or object_id in seed_ids or isinstance(index, bool)
                        or not isinstance(index, int) or not 0 <= index < len(images)
                        or len(box) != 4 or not np.isfinite(box).all()
                        or not 0 <= box[0] < box[2] <= width
                        or not 0 <= box[1] < box[3] <= height):
                    raise ValueError("SAM2 seed object/frame/box differs")
                seed_ids.append(object_id)
                processor.add_inputs_to_inference_session(
                    inference_session=session, frame_idx=index, obj_ids=object_id,
                    input_boxes=[[box]],
                )
                model(inference_session=session, frame_idx=index)
            if list(session.obj_ids) != seed_ids:
                raise ValueError("SAM2 changed seed object ordering")
            start = min(seed["frame_idx"] for seed in seeds)
            results, total = [], 0
            for output in model.propagate_in_video_iterator(
                inference_session=session, start_frame_idx=start,
                max_frame_num_to_track=len(images) - start, reverse=False,
            ):
                index = output.frame_idx
                if index != start + len(results) or index >= len(images):
                    raise ValueError("SAM2 emitted duplicate/out-of-order frame")
                masks = processor.post_process_masks(
                    [output.pred_masks], original_sizes=[[height, width]], binarize=False,
                )[0].detach().float().cpu().numpy()
                masks = np.asarray(masks)
                if masks.shape != (len(seed_ids), 1, height, width) or not np.isfinite(masks).all():
                    raise ValueError("SAM2 mask logits have invalid shape or nonfinite values")
                values = {str(key): pack_mask(masks[i, 0] > config.mask_threshold)
                          for i, key in enumerate(seed_ids)}
                total += sum(len(value["data"]) for value in values.values())
                if total > config.max_mask_bytes:
                    raise ValueError("SAM2 total mask transport budget exceeded")
                results.append({"frame_idx": index, "masks": values})
            if len(results) != len(images) - start:
                raise ValueError("SAM2 omitted requested propagation frames")
        return {"status": "succeeded", "frames": results, "actual_device": str(model.device),
                "actual_dtype": str(model.dtype), "transformers_version": "4.57.1",
                "processed_video_shape": list(processed_shape)}
    finally:
        if session is not None:
            session.reset_inference_session()
        model = processor = session = None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", required=True, type=Path)
    args = parser.parse_args(argv)
    directory = args.request.absolute().parent
    payload = args.request.read_bytes()
    if len(payload) > 1024 * 1024:
        raise ValueError("SAM2 request exceeds 1 MiB")
    request = json.loads(payload)
    source_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    bindings = {"request_sha256": hashlib.sha256(payload).hexdigest(),
                "backend_sha256": request["backend_sha256"], "runtime_source_sha256": source_sha}
    try:
        if source_sha != request["runtime_source_sha256"]:
            raise ValueError("SAM2 installed runtime source differs from parent")
        result = predict(request, directory)
        code = 0
    except Exception as error:
        result = {"status": "failed", "error_code": "sam2_model_unavailable",
                  "error": f"{type(error).__name__}: {error}"}
        if isinstance(error, ValueError):
            result["error_code"] = "sam2_invalid_output"
        # No global torch import merely for error handling.
        if isinstance(error, MemoryError) or type(error).__name__ == "OutOfMemoryError":
            result["error_code"] = "sam2_oom"
        code = 2
    with (directory / "result.json").open("x", encoding="utf-8") as stream:
        json.dump({**result, **bindings}, stream, sort_keys=True, allow_nan=False)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
