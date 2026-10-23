"""Explicit isolated SAM2 backend and verified bounded window IPC."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import subprocess
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2

from video_intelligence.core.paths import ProjectPaths
from video_intelligence.core.types import FrameRequest
from video_intelligence.detection.backends import _nonfinite, _unique, backend_digest
from video_intelligence.models import ModelMode, ModelSpec, hash_weights_location
from video_intelligence.segmentation.masks import unpack_mask
from video_intelligence.storage.evidence_frames import _mkdir, _portable, _safe, _semantics
from video_intelligence.video.decoder import PyAVVideoDecoder
from video_intelligence.video.manifest import sha256_file

MODEL_ID = "facebook/sam2.1-hiera-tiny"
TRANSFORMERS_VERSION = "4.57.1"
POLICY_VERSION = "sam2_video_iou_seed_masklets.v1"


class SAM2Error(RuntimeError):
    def __init__(self, code, message):
        self.code = code
        super().__init__(message)


@dataclass(frozen=True, slots=True)
class SAM2Config:
    max_frames: int = 16
    max_objects: int = 8
    max_input_pixels: int = 2_073_600
    max_window_seconds: float = 8.0
    max_mask_bytes: int = 8 * 1024 * 1024
    mask_threshold: float = 0.0
    min_box_overlap: float = 0.1

    def __post_init__(self):
        for name, cap in (("max_frames", 32), ("max_objects", 16),
                          ("max_input_pixels", 8_294_400), ("max_mask_bytes", 32 * 1024 * 1024)):
            x = getattr(self, name)
            if isinstance(x, bool) or not isinstance(x, int) or not 1 <= x <= cap:
                raise ValueError(f"{name} must be an integer in [1,{cap}]")
        for name, low, high in (("max_window_seconds", 0, 16),
                                ("min_box_overlap", 0, 1), ("mask_threshold", -10, 10)):
            x = getattr(self, name)
            if (isinstance(x, bool) or not isinstance(x, (int, float))
                    or not low < x <= high):
                raise ValueError(f"{name} must be finite in ({low},{high}]")


@dataclass(frozen=True, slots=True)
class SAM2Backend:
    spec: ModelSpec
    python_executable: str
    python_sha256: str
    config: SAM2Config = SAM2Config()

    def __post_init__(self):
        s = self.spec
        if (s.model_id != MODEL_ID or s.source != MODEL_ID or s.role != "segmentation"
                or s.mode is not ModelMode.INFERENCE or s.batch_size != 1 or s.allow_download
                or not s.weights_location or not s.weights_sha256
                or re.fullmatch(r"[0-9a-f]{40}", s.revision) is None
                or s.max_image_side != 1024 or s.timeout_seconds > 60
                or s.dtype not in {"float32", "float16", "bfloat16"}
                or s.device == "cpu" and s.dtype != "float32"
                or s.device != "cpu" and re.fullmatch(r"cuda(?::[0-9]+)?", s.device) is None):
            raise ValueError("SAM2 requires local hash-bound tiny/inference/batch1")
        if (not isinstance(self.python_executable, str)
                or not Path(self.python_executable).is_absolute()
                or re.fullmatch(r"[0-9a-f]{64}", self.python_sha256) is None
                or not isinstance(self.config, SAM2Config)):
            raise ValueError("SAM2 requires an absolute hash-bound Python and typed config")

    def to_dict(self):
        return {"model_spec": self.spec.to_dict(), "python_executable": self.python_executable,
                "python_sha256": self.python_sha256, "config": asdict(self.config),
                "policy_version": POLICY_VERSION, "transformers_version": TRANSFORMERS_VERSION}

    @property
    def identity_sha256(self):
        return backend_digest(self.to_dict())

    def _invoke(self, directory, *, checkpoint, timeout):
        environment = dict(os.environ)
        for key in ("PYTHONPATH", "PYTHONHOME"):
            environment.pop(key, None)
        environment.update({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                            "PYTHONDONTWRITEBYTECODE": "1"})
        argv = [self.python_executable, "-I", "-m", "video_intelligence.segmentation.sam2_runtime",
                "--request", str(directory / "request.json")]
        started = time.monotonic()
        with ((directory / "stdout.txt").open("xb") as out,
              (directory / "stderr.txt").open("xb") as err):
            process = subprocess.Popen(argv, cwd=directory, env=environment, stdout=out, stderr=err)
            try:
                while process.poll() is None:
                    checkpoint()
                    if time.monotonic() - started >= timeout:
                        raise SAM2Error("sam2_timeout", "SAM2 process exceeded its window deadline")
                    time.sleep(0.05)
                checkpoint()
            finally:
                if process.poll() is None:
                    process.kill()
                process.wait(timeout=10)
        return process.returncode

    def segment_window(self, *, video, batches, tracking, rgb_hashes, root, checkpoint, timeout):
        from video_intelligence.segmentation import sam2_runtime

        checkpoint()
        refs = [batch.frame_ref for batch in batches]
        if (not refs or len(refs) > self.config.max_frames
                or refs[-1].timestamp_seconds - refs[0].timestamp_seconds
                > self.config.max_window_seconds
                or any(ref.width * ref.height > self.config.max_input_pixels for ref in refs)
                or len(tracking.tracks) > self.config.max_objects):
            raise SAM2Error("sam2_budget", "SAM2 window/frame/object/pixel budget exceeded")
        if not tracking.tracks:
            return {"backend": self.to_dict(), "frames": [], "seeds": [], "status": "no_seeds"}, {}
        python = Path(self.python_executable)
        try:
            _safe(python, directory=False)
            if sha256_file(python) != self.python_sha256:
                raise SAM2Error("sam2_runtime_binding", "SAM2 Python binary hash differs")
            if not Path(self.spec.weights_location).is_dir():
                raise SAM2Error("sam2_model_unavailable", "SAM2 asset directory is unavailable")
            if hash_weights_location(self.spec.weights_location) != self.spec.weights_sha256:
                raise SAM2Error("sam2_weights_binding", "SAM2 local asset hash differs")
        except OSError as error:
            raise SAM2Error("sam2_model_unavailable",
                            f"SAM2 runtime/assets unavailable: {error}") from error
        checkpoint()
        root = Path(root).absolute()
        _safe(root)
        directory = root / "sam2-inputs" / uuid.uuid4().hex
        _mkdir(directory)
        _safe(directory, directory=True)
        rows = []
        with PyAVVideoDecoder(video) as decoder:
            for index, ref in enumerate(refs):
                checkpoint()
                frame = decoder.read_request(FrameRequest(video.video_id, ref.timestamp_seconds),
                                             selection_policy="nearest")
                if (frame is None or _semantics(frame.frame_ref) != _semantics(ref)
                        or frame.actual_timestamp_seconds != ref.timestamp_seconds
                        or hashlib.sha256(frame.image.tobytes()).hexdigest()
                        != rgb_hashes[ref.identity_key]):
                    raise SAM2Error("sam2_rgb_binding", "SAM2 re-decode differs from grounding RGB")
                ok, png = cv2.imencode(".png", cv2.cvtColor(frame.image, cv2.COLOR_RGB2BGR))
                if not ok:
                    raise ValueError("SAM2 input PNG encoding failed")
                name = f"frame_{index:04d}.png"
                payload = png.tobytes()
                with (directory / name).open("xb") as stream:
                    stream.write(payload)
                rows.append({"file": name, "frame_ref": _portable(ref),
                             "png_sha256": hashlib.sha256(payload).hexdigest(),
                             "rgb_sha256": rgb_hashes[ref.identity_key]})
        positions = {ref.identity_key: i for i, ref in enumerate(refs)}
        seeds = []
        bindings = {}
        for object_id, track in enumerate(tracking.tracks, 1):
            first = track.observations[0].detection
            seeds.append({"object_id": object_id,
                          "frame_idx": positions[first.frame_ref.identity_key],
                          "bbox_xyxy": list(first.bbox_xyxy), "detection_id": first.detection_id,
                          "seed_track_id": track.track_id, "label": track.label})
            for observation in track.observations:
                bindings[observation.detection.detection_id] = object_id
        request = {"schema_version": "video_intelligence.sam2_window.v1",
                   "backend": self.to_dict(), "backend_sha256": self.identity_sha256,
                   "runtime_source_sha256": sha256_file(Path(sam2_runtime.__file__)),
                   "frames": rows, "seeds": seeds}
        payload = json.dumps(request, sort_keys=True, allow_nan=False).encode("utf-8")
        (directory / "request.json").write_bytes(payload)
        request_sha = hashlib.sha256(payload).hexdigest()
        code = self._invoke(directory, checkpoint=checkpoint,
                            timeout=min(timeout, self.spec.timeout_seconds))
        checkpoint()
        path = directory / "result.json"
        _safe(path, directory=False)
        if path.stat().st_size > self.config.max_mask_bytes * 2 + 128 * 1024:
            raise SAM2Error("sam2_budget", "SAM2 output exceeds mask transport cap")
        result = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_unique,
                            parse_constant=_nonfinite)
        if (result.get("request_sha256") != request_sha
                or result.get("backend_sha256") != self.identity_sha256
                or result.get("runtime_source_sha256") != request["runtime_source_sha256"]):
            raise SAM2Error("sam2_output_binding", "SAM2 process result identity differs")
        if code or result.get("status") != "succeeded":
            raise SAM2Error(result.get("error_code", "sam2_failed"),
                            result.get("error", "SAM2 failed"))
        actual_device = result.get("actual_device")
        if (result.get("transformers_version") != TRANSFORMERS_VERSION
                or result.get("processed_video_shape") != [len(refs), 3, 1024, 1024]
                or str(result.get("actual_dtype")).removeprefix("torch.") != self.spec.dtype
                or not (actual_device == self.spec.device or self.spec.device == "cuda"
                        and isinstance(actual_device, str)
                        and re.fullmatch(r"cuda:[0-9]+", actual_device))):
            raise SAM2Error("sam2_output_binding", "SAM2 actual runtime/device/dtype differs")
        runtime = {key: result[key] for key in (
            "request_sha256", "runtime_source_sha256", "actual_device", "actual_dtype",
            "transformers_version",
            "processed_video_shape",
        )}
        outputs = result["frames"]
        start = min(seed["frame_idx"] for seed in seeds)
        if not isinstance(outputs, list) or len(outputs) != len(refs) - start:
            raise SAM2Error("sam2_output_binding", "SAM2 omitted propagated frames")
        mask_values, total = {}, 0
        for index, output in enumerate(outputs, start):
            if output.get("frame_idx") != index or set(output.get("masks", {})) != {
                str(seed["object_id"]) for seed in seeds
            }:
                raise SAM2Error("sam2_output_binding", "SAM2 frame/object coverage differs")
            for key, mask in output["masks"].items():
                total += len(mask.get("data", ""))
                if total > self.config.max_mask_bytes:
                    raise SAM2Error("sam2_budget", "SAM2 packed mask budget exceeded")
                unpack_mask(mask, width=refs[index].width, height=refs[index].height)
                mask_values[index, int(key)] = mask
        attached, rejected = {}, []
        seed_lookup = {seed["object_id"]: seed for seed in seeds}
        for index, batch in enumerate(batches):
            for item in batch.detections:
                detection = item.observation
                object_id = bindings[detection.detection_id]
                value = mask_values.get((index, object_id))
                reason = "empty_or_unpropagated_mask"
                if value is not None and index >= seed_lookup[object_id]["frame_idx"]:
                    mask = unpack_mask(value, width=refs[index].width, height=refs[index].height)
                    x1, y1, x2, y2 = detection.bbox_xyxy
                    overlap = int(mask[math.floor(y1):math.ceil(y2),
                                       math.floor(x1):math.ceil(x2)].sum()) / max(1, value["area"])
                    if value["area"] and overlap >= self.config.min_box_overlap:
                        attached[detection.detection_id] = {
                            "backend_sha256": self.identity_sha256,
                            "model_spec": self.spec.to_dict(), "policy_version": POLICY_VERSION,
                            "transformers_version": TRANSFORMERS_VERSION,
                            "window_id": tracking.window_id, "object_id": object_id,
                            "sam2_track_id": f"sam2_{tracking.window_id}_{object_id}",
                            "seed": seed_lookup[object_id], "mask": value,
                            "tracking_algorithm_id": "sam2_video_propagation.v1",
                            "mask_box_overlap": overlap,
                            "frame_rgb_sha256": rows[index]["rgb_sha256"],
                            "identity_scope": "within_window_seed_conditioned_masklet",
                            "quality_status": "unmeasured", "mask_persisted": False,
                            "runtime": runtime,
                        }
                        continue
                    reason = "empty_mask" if not value["area"] else "mask_box_disagreement"
                rejected.append({"detection_id": detection.detection_id, "reason": reason})
        return {"backend": self.to_dict(), "seeds": seeds, "frames": outputs,
                "rejected_observations": rejected, "status": "succeeded",
                "algorithm_id": "sam2_video_propagation.v1", "runtime": runtime}, attached


def load_sam2_backend(path, *, paths=None):
    with Path(path).open("rb") as stream:
        raw = stream.read(128 * 1024 + 1)
    if len(raw) > 128 * 1024:
        raise ValueError("SAM2 config exceeds 128 KiB")
    try:
        value = json.loads(raw, object_pairs_hook=_unique, parse_constant=_nonfinite)
        if (not isinstance(value, dict) or set(value) != {
            "schema_version", "model_spec", "python_executable", "python_sha256", "config",
        }
                or value["schema_version"] != "video_intelligence.sam2_backend.v1"):
            raise ValueError("invalid SAM2 backend config fields/schema")
        fields = dict(value["model_spec"])
        location = Path(fields["weights_location"])
        if not location.is_absolute():
            if location.drive or ".." in location.parts:
                raise ValueError("SAM2 weights must remain under VI_MODELS_ROOT")
            location = (paths or ProjectPaths.from_environment()).models_root / location
        fields["weights_location"] = str(location.resolve())
        return SAM2Backend(ModelSpec(**fields), value["python_executable"], value["python_sha256"],
                           SAM2Config(**value["config"]))
    except (TypeError, KeyError, AttributeError, OverflowError) as error:
        raise ValueError("malformed SAM2 config") from error
