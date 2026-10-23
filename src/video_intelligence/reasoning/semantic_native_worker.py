"""Isolated source entry point: verify the source tree before importing it."""

from __future__ import annotations

import argparse
import ctypes
import gc
import hashlib
import json
import os
import sys
import time
from pathlib import Path


def checked_source_rows(root):
    """Bind operator-owned Python source; this is not a sandbox or dependency digest."""
    root = Path(root).absolute()

    def regular_path(path):
        if path.is_symlink() or getattr(path.lstat(), "st_file_attributes", 0) & 0x400:
            raise ValueError("source tree has a reparse point")

    for ancestor in (root, *root.parents):
        regular_path(ancestor)
    rows = []
    for directory, dirs, files in os.walk(root, topdown=True, followlinks=False):
        for name in sorted(dirs + files):
            regular_path(Path(directory) / name)
        for name in sorted(files):
            path = Path(directory) / name
            if path.suffix.lower() in {".pyc", ".pyo", ".pyd", ".dll", ".so"}:
                raise ValueError("source tree has an unexpected import artifact")
            if path.suffix == ".py":
                rows.append({"path": path.relative_to(root).as_posix(),
                             "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    return sorted(rows, key=lambda row: row["path"])


def checked_request(verifier, result, frames, expected):
    request = verifier.request(result, frames)
    if request.sha256 != expected:
        raise ValueError("child request differs before inference")
    return request


def retain_generation(reply, token_ids, decode, binding, elapsed, reserved):
    """Keep bounded primary diagnostics before checking a failed generation."""
    reply["runtime"] = {"binding": binding, "generated_token_ids": token_ids[:257],
        "generated_tokens": len(token_ids), "generation_seconds": elapsed,
        "actual_dtype": "bfloat16", "actual_device": "cuda:0",
        "diagnostic_tokens_truncated": len(token_ids) > 257}
    text = decode(token_ids[:257])
    encoded = text.encode("utf-8")
    reply["raw_response"] = encoded[:32768].decode("utf-8", errors="ignore")
    reply["raw_response_truncated"] = len(encoded) > 32768
    reservation = reserved() if callable(reserved) else reserved
    if reservation > 8_589_934_592:
        raise MemoryError("native between-call reservation stop reached")
    if not token_ids or len(token_ids) > 256 or token_ids[-1] != 151645:
        raise ValueError("native generation is truncated or lacks terminal EOS")
    if reply["raw_response_truncated"]:
        raise ValueError("native raw diagnostic byte budget exceeded")
    return text


def own_memory():
    class Counters(ctypes.Structure):
        _fields_ = [("cb", ctypes.c_ulong), ("faults", ctypes.c_ulong)] + [(n, ctypes.c_size_t)
            for n in ("peak", "current", "qpp", "qpu", "qnp", "qnu", "page", "peakpage")]
    value = Counters()
    value.cb = ctypes.sizeof(value)
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.GetCurrentProcess.restype = ctypes.c_void_p
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    psapi.GetProcessMemoryInfo.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_ulong]
    if not psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(value), value.cb):
        raise OSError("own process memory query failed")
    return {"working_set_bytes": value.current, "peak_working_set_bytes": value.peak}


def main():
    if sys.flags.optimize or not sys.flags.isolated:
        raise RuntimeError("worker requires isolated, non-optimized Python")
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", type=Path, required=True)
    args = parser.parse_args()
    if args.request.stat().st_size > 1_048_576:
        raise ValueError("worker request byte cap exceeded")
    raw = args.request.read_bytes()
    job = json.loads(raw)
    root = Path(__file__).absolute().parents[2]
    rows = checked_source_rows(root)
    digest = hashlib.sha256(json.dumps(rows, sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    if digest != job["profile"]["source_sha256"]:
        raise ValueError("source inventory differs before import")
    # Operator-owned source check, not a dependency digest or an execution sandbox.
    sys.path.insert(0, str(root))
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                      HF_HUB_DISABLE_PROGRESS_BARS="1", PYTHONDONTWRITEBYTECODE="1")
    from video_intelligence.models import hash_weights_location
    from video_intelligence.reasoning.semantic_backend import BoundSemanticVerifier
    from video_intelligence.reasoning.semantic_native import (
        NativeProfile,
        prepare_inputs,
        result_and_frames,
    )

    started = time.monotonic()
    torch = model = processor = inputs = generated = output = None
    reply = {"status": "failed", "job_sha256": hashlib.sha256(raw).hexdigest()}
    try:
        import importlib.metadata

        import torch
        from transformers import AutoProcessor, GenerationConfig, Qwen3VLForConditionalGeneration

        profile_fields = {k: job["profile"][k] for k in (
            "python", "python_sha256", "model_root", "bundle_sha256", "source_sha256")}
        if "input_view" in job["profile"]:
            profile_fields["input_view"] = job["profile"]["input_view"]
        profile = NativeProfile(**profile_fields)
        if profile.to_dict() != job["profile"]:
            raise ValueError("native profile fields differ")
        for name in ("torch", "transformers"):
            if importlib.metadata.version(name) != job["profile"][name]:
                raise ValueError("native package version differs")
        if hash_weights_location(profile.model_root) != profile.bundle_sha256:
            raise ValueError("native model bundle differs")
        result, frames = result_and_frames(job["data"])

        class LocalTransport:
            identity_sha256 = profile.identity_sha256

            def generate(self, *, request, frames, system_prompt, choices):
                nonlocal processor, model, inputs, generated, output
                processor = AutoProcessor.from_pretrained(profile.model_root,
                    local_files_only=True, trust_remote_code=False,
                    size={"longest_edge": 65_536, "shortest_edge": 1024})
                inputs, constraint, binding = prepare_inputs(processor, request, frames,
                                                             input_view=profile.input_view)
                if binding != job["data"]["input_binding"]:
                    raise ValueError("actual native prompt/pixels/tokens differ")
                model = Qwen3VLForConditionalGeneration.from_pretrained(profile.model_root,
                    local_files_only=True, trust_remote_code=False, use_safetensors=True,
                    dtype=torch.bfloat16, attn_implementation="eager").eval().to("cuda:0")
                if (str(next(model.parameters()).dtype) != "torch.bfloat16"
                        or str(next(model.parameters()).device) != "cuda:0"):
                    raise ValueError("actual native device/dtype differs")
                if torch.cuda.memory_reserved() > 8_589_934_592:
                    raise MemoryError("native between-call reservation stop reached")
                inputs = inputs.to("cuda:0")
                settings = constraint.generation_kwargs()
                hooks = {k: settings.pop(k)
                         for k in ("prefix_allowed_tokens_fn", "logits_processor")}
                settings.update(pad_token_id=151643, bos_token_id=151643, use_cache=True)
                config = GenerationConfig(**settings)  # No inherited model generation defaults.
                began = time.monotonic()
                with torch.inference_mode():
                    generated = model.generate(**inputs, generation_config=config, **hooks)
                torch.cuda.synchronize()
                output = generated[0, inputs["input_ids"].shape[1]:].detach().cpu()
                text = retain_generation(reply, output.tolist(), lambda ids: processor.decode(
                    ids, skip_special_tokens=True, clean_up_tokenization_spaces=False),
                    binding, time.monotonic()-began, torch.cuda.memory_reserved)
                return text

        verifier = BoundSemanticVerifier(LocalTransport())
        request = checked_request(verifier, result, frames, job["data"]["request_sha256"])
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable")
        torch.cuda.reset_peak_memory_stats()
        torch.manual_seed(0)
        reply["device"] = {"index": 0, "name": torch.cuda.get_device_name(0),
                           "capability": list(torch.cuda.get_device_capability(0)),
                           "cuda_build": torch.version.cuda}
        report = verifier.assess(request, frames)
        reply["request_sha256"] = report.request_sha256
        reply["verdicts"] = [a.verdict.value for a in report.assessments]
        reply["status"] = "passed"
    except Exception as error:
        reply["error_type"] = type(error).__name__
        reply["error_message"] = str(error)[:2048]
    finally:
        errors = []
        try:
            if torch is not None and torch.cuda.is_initialized():
                reply["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
                reply["peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
        except Exception as error:
            errors.append({"phase": "peak", "type": type(error).__name__})
        model = processor = inputs = generated = output = None
        try:
            gc.collect()
            if torch is not None and torch.cuda.is_initialized():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                reply["after_cleanup_allocated_bytes"] = torch.cuda.memory_allocated()
                reply["after_cleanup_reserved_bytes"] = torch.cuda.memory_reserved()
            reply["process_memory"] = own_memory()
        except Exception as error:
            errors.append({"phase": "cleanup_or_memory", "type": type(error).__name__})
        reply["cleanup_errors"] = errors
        reply["inference_status"] = reply["status"]
        if errors:
            reply["status"] = "cleanup_failed"
        reply["wall_seconds"] = time.monotonic()-started
        with (args.request.parent / "reply.json").open("x", encoding="utf-8") as stream:
            json.dump(reply, stream, indent=2)
    return int(reply["status"] != "passed")


if __name__ == "__main__":
    raise SystemExit(main())
