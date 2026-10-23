"""Local, isolated Qwen3-VL inference over verified persisted evidence only."""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath

from video_intelligence.core.paths import ProjectPaths
from video_intelligence.detection.backends import _nonfinite, _unique, backend_digest
from video_intelligence.models import ModelMode, ModelSpec, hash_weights_location
from video_intelligence.reasoning.contracts import (
    PROMPT_VERSION,
    SYSTEM_PROMPT,
    assemble_answer,
    prompt_data,
    select_evidence,
)
from video_intelligence.storage.evidence_frames import (
    EvidenceFrameConfig,
    EvidenceFrameStore,
    _Budget,
    _digest,
    _json_bytes,
    _mkdir,
    _read,
    _safe,
)
from video_intelligence.video.manifest import sha256_file

MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"
TRANSFORMERS_VERSION = "4.57.1"


class VLMError(RuntimeError):
    def __init__(self, code, message):
        self.code = code
        super().__init__(message)


@dataclass(frozen=True, slots=True)
class VLMConfig:
    max_frames: int = 2
    max_images: int = 6
    max_items: int = 16
    max_source_pixels: int = 2_073_600
    max_image_pixels: int = 65_536
    max_input_tokens: int = 4096
    max_new_tokens: int = 512

    def __post_init__(self):
        for name, low, high in (("max_frames", 1, 8), ("max_images", 1, 16),
                                ("max_items", 1, 32), ("max_source_pixels", 1, 8_294_400),
                                ("max_image_pixels", 1024, 262_144),
                                ("max_input_tokens", 128, 8192), ("max_new_tokens", 1, 1024)):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
                raise ValueError(f"{name} must be an integer in [{low},{high}]")


@dataclass(frozen=True, slots=True)
class VLMBackend:
    spec: ModelSpec
    python_executable: str
    python_sha256: str
    config: VLMConfig = VLMConfig()

    def __post_init__(self):
        s = self.spec
        if (s.model_id != MODEL_ID or s.source != MODEL_ID or s.role != "vlm"
                or s.mode is not ModelMode.INFERENCE or s.batch_size != 1 or s.allow_download
                or not s.weights_location or not s.weights_sha256
                or not Path(s.weights_location).is_absolute()
                or re.fullmatch(r"[0-9a-f]{40}", s.revision) is None
                or s.max_image_side != 512 or s.timeout_seconds > 60
                or s.dtype not in {"float32", "float16", "bfloat16"}
                or s.device == "cpu" and s.dtype != "float32"
                or s.device != "cpu" and re.fullmatch(r"cuda(?::[0-9]+)?", s.device) is None):
            raise ValueError("VLM requires local hash-bound Qwen3-VL-2B/inference/batch1")
        if (not isinstance(self.python_executable, str)
                or not Path(self.python_executable).is_absolute()
                or re.fullmatch(r"[0-9a-f]{64}", self.python_sha256) is None
                or not isinstance(self.config, VLMConfig)):
            raise ValueError("VLM requires absolute hash-bound Python and typed config")

    def to_dict(self):
        return {"model_spec": self.spec.to_dict(), "python_executable": self.python_executable,
                "python_sha256": self.python_sha256, "config": asdict(self.config),
                "prompt_version": PROMPT_VERSION, "transformers_version": TRANSFORMERS_VERSION}

    @property
    def identity_sha256(self):
        return backend_digest(self.to_dict())

    def _invoke(self, directory, *, checkpoint, timeout):
        environment = dict(os.environ)
        for key in ("PYTHONPATH", "PYTHONHOME"):
            environment.pop(key, None)
        environment.update({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                            "PYTHONDONTWRITEBYTECODE": "1"})
        argv = [self.python_executable, "-I", "-m", "video_intelligence.reasoning.vlm_runtime",
                "--request", str(directory / "request.json")]
        started = time.monotonic()
        with ((directory / "stdout.txt").open("xb") as out,
              (directory / "stderr.txt").open("xb") as err):
            process = subprocess.Popen(argv, cwd=directory, env=environment, stdout=out, stderr=err)
            try:
                while process.poll() is None:
                    checkpoint()
                    if time.monotonic() - started >= timeout:
                        raise VLMError("vlm_timeout", "VLM process exceeded its deadline")
                    time.sleep(0.05)
                checkpoint()
            finally:
                if process.poll() is None:
                    process.kill()
                process.wait(timeout=10)
        return process.returncode

    def analyze(self, result, *, root, checkpoint, evidence_config=None):
        from video_intelligence.reasoning import vlm_runtime

        checkpoint()
        evidence_config = evidence_config or EvidenceFrameConfig()
        frames = EvidenceFrameStore(root)
        frames.verify_result(result, checkpoint=checkpoint, config=evidence_config)
        selection = select_evidence(result, self.config)
        if not selection["catalog"]:
            return assemble_answer(result, self)  # No model load for insufficient evidence.
        python = Path(self.python_executable)
        _safe(python, directory=False)
        if sha256_file(python) != self.python_sha256:
            raise VLMError("vlm_runtime_binding", "VLM Python binary hash differs")
        if (not Path(self.spec.weights_location).is_dir()
                or hash_weights_location(self.spec.weights_location) != self.spec.weights_sha256):
            raise VLMError("vlm_model_unavailable", "VLM local asset directory/hash is unavailable")
        _safe(Path(self.spec.weights_location), directory=True)
        manifest = input_manifest(result, self, sha256_file(Path(vlm_runtime.__file__)))
        directory = Path(root).absolute() / "vlm-inputs" / uuid.uuid4().hex
        _mkdir(directory)
        budget = _Budget(evidence_config, checkpoint)
        for row in manifest["media"]:
            ref = row["reference"]
            relative = PurePosixPath(ref["relative_path"])
            if (relative.is_absolute() or Path(str(relative)).drive
                    or ".." in relative.parts or "\\" in str(relative)):
                raise ValueError("VLM evidence path is not portable relative")
            payload = _read(frames.root.joinpath(*relative.parts),
                            evidence_config.max_total_bytes, budget)
            if hashlib.sha256(payload).hexdigest() != ref["png_sha256"]:
                raise ValueError("VLM evidence PNG changed after verification")
            with (directory / row["file"]).open("xb") as stream:
                stream.write(payload)
        payload = _json_bytes(manifest)
        with (directory / "request.json").open("xb") as stream:
            stream.write(payload)
        try:
            code = self._invoke(directory, checkpoint=checkpoint, timeout=self.spec.timeout_seconds)
            raw = _read(directory / "result.json", 128 * 1024, _Budget(evidence_config, checkpoint))
        except OSError as error:
            raise VLMError("vlm_process_failed",
                           f"VLM process/output unavailable: {error}") from error
        reply = json.loads(raw, object_pairs_hook=_unique, parse_constant=_nonfinite)
        if not isinstance(reply, dict) or reply.get("request_sha256") != hashlib.sha256(
            payload
        ).hexdigest():
            raise VLMError("vlm_output_binding", "VLM result/request binding differs")
        if code != 0 or reply.get("status") != "succeeded":
            error_code = reply.get("error_code")
            if error_code not in {"vlm_model_unavailable", "vlm_invalid_output", "vlm_oom"}:
                error_code = "vlm_inference_failed"
            raise VLMError(error_code,
                           str(reply.get("error", "VLM process failed"))[:1000])
        if set(reply) != {"status", "request_sha256", "runtime", "raw_output"}:
            raise VLMError("vlm_output_binding", "VLM result contains unexpected fields")
        validate_runtime(reply.get("runtime"), manifest, result, self)
        answer = assemble_answer(result, self, raw_output=reply.get("raw_output"),
                                 runtime=reply["runtime"], input_manifest=manifest)
        checkpoint()
        frames.verify_result(result, checkpoint=checkpoint, config=evidence_config)
        return answer


def input_manifest(result, backend, runtime_source_sha256):
    selection = select_evidence(result, backend.config)
    media = [{**row, "file": f"image_{i:02d}.png"} for i, row in enumerate(selection["media"])]
    if any(row["frame_ref"]["width"] * row["frame_ref"]["height"]
           > backend.config.max_source_pixels for row in media):
        raise ValueError("VLM source image pixel budget exceeded")
    return {"schema_version": "video_intelligence.vlm_input.v1", "backend": backend.to_dict(),
            "backend_sha256": backend.identity_sha256,
            "runtime_source_sha256": _digest(runtime_source_sha256),
            "upstream_result_sha256": backend_digest(result.to_dict()),
            "system_prompt": SYSTEM_PROMPT, "prompt_data": prompt_data(result, selection),
            "media": media}


def validate_runtime(runtime, manifest, result, backend):
    """Pure bounded replay validation; file verification is separately mandatory."""
    try:
        if manifest != input_manifest(result, backend, manifest["runtime_source_sha256"]):
            raise ValueError("VLM input manifest differs from upstream evidence")
        expected = {"request_sha256", "runtime_source_sha256", "transformers_version",
                    "actual_device", "actual_dtype", "input_tokens", "output_tokens",
                    "image_grid_thw", "rendered_prompt_sha256"}
        if not isinstance(runtime, dict) or set(runtime) != expected:
            raise ValueError("VLM runtime metadata missing or unknown")
        if (runtime["request_sha256"] != hashlib.sha256(_json_bytes(manifest)).hexdigest()
                or runtime["runtime_source_sha256"] != manifest["runtime_source_sha256"]
                or runtime["transformers_version"] != TRANSFORMERS_VERSION
                or runtime["actual_dtype"] != backend.spec.dtype
                or not (runtime["actual_device"] == backend.spec.device or backend.spec.device
                        == "cuda" and re.fullmatch(r"cuda:[0-9]+", runtime["actual_device"]))):
            raise ValueError("VLM actual runtime/binding differs")
        _digest(runtime["rendered_prompt_sha256"])
        for name, cap in (("input_tokens", backend.config.max_input_tokens),
                          ("output_tokens", backend.config.max_new_tokens)):
            x = runtime[name]
            if isinstance(x, bool) or not isinstance(x, int) or not 1 <= x <= cap:
                raise ValueError("VLM token budget exceeded")
        validate_grids(runtime["image_grid_thw"], len(manifest["media"]), backend.config)
    except (KeyError, TypeError, AttributeError) as error:
        raise ValueError("malformed VLM runtime/input manifest") from error


def validate_grids(grids, count, config):
    if not isinstance(grids, list) or len(grids) != count:
        raise ValueError("VLM processor omitted images")
    for grid in grids:
        if (not isinstance(grid, list) or len(grid) != 3
                or any(isinstance(x, bool) or not isinstance(x, int) or x <= 0 for x in grid)
                or grid[0] != 1 or grid[1] % 2 or grid[2] % 2
                or max(grid[1:]) * 16 > 512
                or grid[1] * grid[2] * 256 > config.max_image_pixels):
            raise ValueError("VLM processed image grid/pixel budget exceeded")


def backend_from_effective(value):
    if not isinstance(value, dict) or set(value) != {"model_spec", "python_executable",
            "python_sha256", "config", "prompt_version", "transformers_version"}:
        raise ValueError("invalid effective VLM config")
    backend = VLMBackend(ModelSpec(**value["model_spec"]), value["python_executable"],
                         value["python_sha256"], VLMConfig(**value["config"]))
    if backend.to_dict() != value:
        raise ValueError("VLM config version/normalization differs")
    return backend


def load_vlm_backend(path, *, paths=None):
    with Path(path).open("rb") as stream:
        raw = stream.read(128 * 1024 + 1)
    if len(raw) > 128 * 1024:
        raise ValueError("VLM config exceeds 128 KiB")
    try:
        value = json.loads(raw, object_pairs_hook=_unique, parse_constant=_nonfinite)
        if (not isinstance(value, dict) or set(value) != {
            "schema_version", "model_spec", "python_executable", "python_sha256", "config",
        } or value["schema_version"] != "video_intelligence.vlm_backend.v1"):
            raise ValueError("invalid VLM backend config fields/schema")
        fields = dict(value["model_spec"])
        location = Path(fields["weights_location"])
        if not location.is_absolute():
            if location.drive or ".." in location.parts:
                raise ValueError("VLM weights must remain under VI_MODELS_ROOT")
            location = (paths or ProjectPaths.from_environment()).models_root / location
        fields["weights_location"] = str(location.absolute())
        return VLMBackend(ModelSpec(**fields), value["python_executable"], value["python_sha256"],
                          VLMConfig(**value["config"]))
    except (TypeError, KeyError, AttributeError, OverflowError) as error:
        raise ValueError("malformed VLM config") from error
