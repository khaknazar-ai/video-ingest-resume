"""Explicit local native semantic transport, isolated per request and never promoted."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from video_intelligence.core.types import VideoRecord
from video_intelligence.reasoning.isolated_worker import WorkerCommand, file_hash
from video_intelligence.reasoning.semantic import SemanticRequest
from video_intelligence.reasoning.semantic_backend import SYSTEM_PROMPT, BoundSemanticVerifier
from video_intelligence.reasoning.semantic_generation import ReportChoices
from video_intelligence.reasoning.semantic_input import (
    FULL_VIEW,
    model_input_view,
    validate_input_view,
)
from video_intelligence.reasoning.semantic_native_worker import checked_source_rows
from video_intelligence.retrieval.schemas import QueryResult
from video_intelligence.video.decoder import DecodedFrame


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def source_digest():
    root = Path(__file__).absolute().parents[2]
    rows = checked_source_rows(root)
    return hashlib.sha256(canonical(rows).encode()).hexdigest()


@dataclass(frozen=True, slots=True)
class NativeProfile:
    python: str
    python_sha256: str
    model_root: str
    bundle_sha256: str
    source_sha256: str
    input_view: str = FULL_VIEW

    def __post_init__(self):
        validate_input_view(self.input_view)

    def to_dict(self):
        profile = {"python": self.python, "python_sha256": self.python_sha256,
            "model_root": self.model_root, "bundle_sha256": self.bundle_sha256,
            "source_sha256": self.source_sha256,
            "model_id": "Qwen/Qwen3-VL-2B-Instruct",
            "revision": "89644892e4d85e24eaac8bacfd4f463576704203",
            "torch": "2.13.0+cu130", "transformers": "5.10.2", "device": "cuda:0",
            "dtype": "bfloat16", "max_frames": 1, "max_units": 1,
            "max_input_tokens": 4096, "max_new_tokens": 256,
            "max_source_pixels": 2_073_600, "max_processor_pixels": 65_536,
            "timeout_seconds": 60, "policy": "isolated_native_semantic.v1"}
        if self.input_view != FULL_VIEW:
            profile["input_view"] = self.input_view
        return profile

    @property
    def identity_sha256(self):
        return hashlib.sha256(canonical(self.to_dict()).encode()).hexdigest()


def result_and_frames(data):
    import numpy as np
    from PIL import Image

    video = VideoRecord.from_dict(data["video"])
    if file_hash(video.source_path) != video.source_sha256:
        raise ValueError("native source video changed")
    result = QueryResult.from_dict(data["result"], video=video)
    step = 2 if result.query.relation else 1
    if len(result.evidence.items) != step:
        raise ValueError("native profile requires one complete unit, without truncation")
    row = data["frame"]
    if file_hash(row["png_path"]) != row["png_sha256"]:
        raise ValueError("native PNG changed")
    ref = result.evidence.items[0].frame_ref
    if ref.width * ref.height > 2_073_600:
        raise ValueError("native source pixel budget exceeded")
    with Image.open(row["png_path"]) as image:
        if image.size != (ref.width, ref.height) or image.width * image.height > 2_073_600:
            raise ValueError("native PNG geometry differs or exceeds pixel budget")
        rgb = np.asarray(image.convert("RGB")).copy()
    if hashlib.sha256(rgb.tobytes()).hexdigest() != row["rgb_sha256"]:
        raise ValueError("native RGB changed")
    frame = DecodedFrame(ref, rgb, ref.timestamp_seconds, ref.timestamp_seconds)
    request = SemanticRequest.from_result(result, (frame,), backend_sha256="0" * 64)
    if len(request.to_dict()["units"]) != 1:
        raise ValueError("native profile requires one complete evidence unit")
    return result, (frame,)


def prepare_inputs(processor, request, frames, *, input_view=FULL_VIEW):
    from PIL import Image

    request.validate_frames(frames)
    body = request.to_dict()
    view = model_input_view(request, input_view)
    schema = {"schema_version": "video_intelligence.semantic_assessment.v1",
        "request_sha256": request.sha256, "backend_sha256": body["backend_sha256"],
        "assessments": [{"unit_id": u["unit_id"],
                         "verdict_choices": ["supports_query", "contradicts_query", "uncertain"]}
                        for u in body["units"]]}
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": [
        {"type": "image"}, {"type": "text", "text": canonical({"request": view,
                                                                  "response_schema": schema})}]}]
    rendered = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[rendered], images=[Image.fromarray(frames[0].image)],
                       return_tensors="pt")
    ids = inputs["input_ids"][0].tolist()
    grids = inputs["image_grid_thw"].tolist()
    if len(ids) > 4096 or len(grids) != 1 or any(t*h*w*256 > 65_536 for t, h, w in grids):
        raise ValueError("native processor budget exceeded")
    constraint = ReportChoices.from_request(request).constrain(
        processor.tokenizer, ids, eos_token_id=151645, max_new_tokens=256)
    binding = {"prompt_sha256": hashlib.sha256(rendered.encode()).hexdigest(),
        "input_ids_sha256": hashlib.sha256(inputs["input_ids"].numpy().tobytes()).hexdigest(),
        "pixels_sha256": hashlib.sha256(inputs["pixel_values"].numpy().tobytes()).hexdigest(),
        "input_tokens": len(ids), "image_grid_thw": grids,
        "max_complete_report_tokens": constraint.max_sequence_tokens}
    if input_view != FULL_VIEW:
        binding.update(input_view=input_view,
                       model_view_sha256=hashlib.sha256(canonical(view).encode()).hexdigest())
    return inputs, constraint, binding


class IsolatedSemanticTransport:
    def __init__(self, profile, data, directory, *, checkpoint=lambda: None):
        self.profile, self.data, self.directory = profile, data, Path(directory).absolute()
        self.checkpoint = checkpoint
        self.last_lifecycle = self.last_reply = None

    @property
    def identity_sha256(self):
        return self.profile.identity_sha256

    def generate(self, *, request: SemanticRequest, frames, system_prompt, choices):
        request.validate_frames(frames)
        if (system_prompt != SYSTEM_PROMPT or choices.request_sha256 != request.sha256
                or request.sha256 != self.data["request_sha256"]
                or self.data["input_binding"].get("input_view", FULL_VIEW)
                   != self.profile.input_view
                or source_digest() != self.profile.source_sha256):
            raise ValueError("native transport binding differs")
        self.directory.mkdir(exist_ok=False)
        job = {"profile": self.profile.to_dict(), "data": self.data}
        payload = canonical(job).encode()
        job_path = self.directory / "job.json"
        with job_path.open("xb") as stream:
            stream.write(payload)
        script = Path(__file__).with_name("semantic_native_worker.py")
        command = WorkerCommand(Path(self.profile.python), self.profile.python_sha256,
                                script, file_hash(script))
        self.last_lifecycle = command.run(job_path, self.directory, checkpoint=self.checkpoint)
        reply_path = self.directory / "reply.json"
        if reply_path.is_file() and reply_path.stat().st_size <= 131_072:
            self.last_reply = json.loads(reply_path.read_bytes())
        if (self.last_lifecycle["status"] != "passed" or self.last_reply is None
                or self.last_reply.get("status") != "passed"):
            raise RuntimeError("isolated native worker failed; retained lifecycle/logs/reply")
        reply = self.last_reply
        if (reply["job_sha256"] != hashlib.sha256(payload).hexdigest()
                or reply["request_sha256"] != request.sha256
                or reply["runtime"]["binding"] != self.data["input_binding"]):
            raise ValueError("native worker output binding differs")
        return reply["raw_response"]


def assess_materialized(profile, data, directory, *, checkpoint=lambda: None):
    """Public Python assessment path; preserves the original abstaining QueryResult."""
    result, frames = result_and_frames(data)
    transport = IsolatedSemanticTransport(profile, data, directory, checkpoint=checkpoint)
    verifier = BoundSemanticVerifier(transport)
    request = verifier.request(result, frames)
    if request.sha256 != data["request_sha256"]:
        raise ValueError("frozen native request differs")
    report = verifier.assess(request, frames)
    return report, transport, result
