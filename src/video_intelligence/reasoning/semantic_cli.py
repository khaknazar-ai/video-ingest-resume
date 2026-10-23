"""Opt-in learned CLI assessments; reports never promote a query's support status."""

from __future__ import annotations

import hashlib
import math
import sys
import time
import uuid
from copy import deepcopy
from dataclasses import asdict, dataclass, replace
from pathlib import Path

from video_intelligence.core.types import FrameRequest
from video_intelligence.reasoning.semantic import SemanticReport, SemanticRequest, _canonical, _sha
from video_intelligence.reasoning.semantic_backend import BoundSemanticVerifier
from video_intelligence.reasoning.semantic_native import (
    IsolatedSemanticTransport,
    NativeProfile,
    file_hash,
    prepare_inputs,
    source_digest,
)
from video_intelligence.retrieval.schemas import (
    ProcessingStatus,
    QueryResult,
    requires_semantic_verification,
)
from video_intelligence.storage.evidence_frames import _safe, _source_hash
from video_intelligence.video.decoder import PyAVVideoDecoder

MAX_SEQUENTIAL_UNITS = 4
SEQUENTIAL_POLICY = "sequential_complete_units.v1"


def _unit_result(result, parent, index):
    """Explicit, ephemeral projection; parent result and provenance remain intact."""
    body = parent.to_dict()
    unit = body["units"][index]
    items = tuple(i for i in result.evidence.items if i.evidence_id in unit["evidence_ids"])
    query = replace(result.query, query_id=result.logical_query_id)
    times = [i.frame_ref.timestamp_seconds for i in items]
    bundle = replace(result.evidence, query_id=query.query_id, items=items,
        coverage_start_seconds=min(times), coverage_end_seconds=max(times), coverage_complete=False,
        gaps=(*result.evidence.gaps, "sequential_unit_projection"))
    config = {**result.query_config, "semantic_partition": {
        "policy": SEQUENTIAL_POLICY, "parent_query_revision": result.query.query_id,
        "parent_result_sha256": body["result_sha256"], "parent_request_sha256": parent.sha256,
        "parent_evidence_ids": [i.evidence_id for i in result.evidence.items],
        "parent_unit_ids": [u["unit_id"] for u in body["units"]],
        "unit_id": unit["unit_id"], "index": index,
    }}
    return QueryResult(query, result.video_id, result.run_id, result.processing_status,
        result.support_status, result.mode, evidence=bundle,
        error_code=result.error_code).with_revision(config)


class SemanticCancelled(RuntimeError):
    """A caller checkpoint or cancellation marker stopped the assessment."""


def _local(path):
    path = Path(path).expanduser()
    if not path.is_absolute() or path.anchor.startswith("\\\\"):
        raise ValueError("semantic paths must be absolute local paths")
    _safe(path)
    return path


def load_source_frames(video, refs, checkpoint):
    """Decode all selected units, without selecting or truncating candidates."""
    frames = []
    checkpoint()
    with PyAVVideoDecoder(video) as decoder:
        for ref in refs:
            checkpoint()
            frame = decoder.read_request(FrameRequest(video.video_id, ref.timestamp_seconds,
                source="semantic_cli"), selection_policy="nearest")
            if frame is None:
                raise ValueError("semantic source frame unavailable")
            frames.append(frame)
    checkpoint()
    return tuple(frames)


def require_native_checkout():
    root = Path(__file__).absolute().parents[2]
    if (not sys.dont_write_bytecode or root.name != "src"
            or not (root.parent / "pyproject.toml").is_file()):
        raise ValueError("native semantic CLI requires a verified source checkout and Python -B")


def native_transport(*, profile, result, video, frames, directory, checkpoint):
    """Prepare a source-bound input, then reuse the supervised native transport."""
    require_native_checkout()
    checkpoint()
    if (file_hash(profile.python) != profile.python_sha256
            or source_digest() != profile.source_sha256):
        raise ValueError("semantic interpreter/source binding differs")
    from PIL import Image
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(profile.model_root, local_files_only=True,
        trust_remote_code=False, size={"longest_edge": 65_536, "shortest_edge": 1024})
    request = BoundSemanticVerifier(profile).request(result, frames)
    _, _, binding = prepare_inputs(processor, request, frames, input_view=profile.input_view)
    checkpoint()
    # This is private worker input, not an immutable published evidence artifact.
    png = directory / "source-frame.png"
    with png.open("xb") as stream:
        Image.fromarray(frames[0].image).save(stream, format="PNG")
    data = {"video": video.to_dict(), "result": result.to_dict(),
        "frame": {"png_path": str(png), "png_sha256": file_hash(png),
                  "rgb_sha256": hashlib.sha256(frames[0].image.tobytes()).hexdigest()},
        "request_sha256": request.sha256, "input_binding": binding}
    checkpoint()
    return IsolatedSemanticTransport(profile, data, directory / "worker", checkpoint=checkpoint)


@dataclass(frozen=True, slots=True)
class SemanticServices:
    """Trusted Python injection only; there is no dynamic import or mock CLI option."""

    transport_factory: object = None
    frame_loader: object = load_source_frames
    checkpoint: object = None
    clock: object = time.monotonic


@dataclass
class SemanticCLI:
    profile: NativeProfile
    work_root: Path
    timeout_seconds: float
    cancel_file: Path | None
    services: SemanticServices

    @classmethod
    def from_args(cls, args, payload, services=None):
        if (not isinstance(payload, dict) or set(payload) != {"schema_version", "profile"}
                or payload["schema_version"] != "video_intelligence.semantic_cli_config.v1"
                or not isinstance(payload["profile"], dict)):
            raise ValueError("invalid semantic CLI config")
        fields = ("python", "python_sha256", "model_root", "bundle_sha256", "source_sha256")
        raw = payload["profile"]
        if (set(raw) not in (set(fields), {*fields, "input_view"})
                or any(not isinstance(raw[name], str) for name in raw)):
            raise ValueError("semantic profile fields differ")
        profile = NativeProfile(**raw)
        _local(profile.python)
        _local(profile.model_root)
        for name in ("python_sha256", "bundle_sha256", "source_sha256"):
            _sha(getattr(profile, name))
        root = _local(args.semantic_work_root)
        if not root.is_dir():
            raise ValueError("semantic-work-root must be an existing local directory")
        timeout = args.semantic_timeout_seconds
        if isinstance(timeout, bool) or not math.isfinite(timeout) or not 0 < timeout <= 240:
            raise ValueError("semantic timeout must be finite and in (0,240]")
        cancel = None if args.semantic_cancel_file is None else _local(args.semantic_cancel_file)
        if cancel is not None:
            _safe(cancel.parent, directory=True)
            if cancel.exists():
                _safe(cancel, directory=False)
        services = services or SemanticServices()
        if (not isinstance(services, SemanticServices) or not callable(services.frame_loader)
                or not callable(services.clock)
                or services.transport_factory is not None
                   and not callable(services.transport_factory)
                or services.checkpoint is not None and not callable(services.checkpoint)):
            raise ValueError("semantic services must be explicit callable dependencies")
        return cls(profile, root, timeout, cancel, services)

    def attach(self, payload, video):
        """Return a detached result plus untrusted assessment; never persist promotion."""
        original = deepcopy(payload)
        value = payload.get("query_result", payload)
        assessment = {"schema_version": "video_intelligence.semantic_cli.v1", "status": "not_run",
            "input_view": self.profile.input_view,
            "analysis_semantics": "unverified_model_judgment", "quality_status": "unqualified",
            "inference_mode": ("native" if self.services.transport_factory is None
                               else "controlled_transport"),
            "controlled_frames": self.services.frame_loader is not load_source_frames,
            "answer_promotion": False, "answer": None, "report": None}
        transport = None
        started = self.services.clock()
        deadline = started + self.timeout_seconds

        def check():
            if self.cancel_file is not None and self.cancel_file.exists():
                raise SemanticCancelled("semantic cancellation marker")
            if self.services.checkpoint is not None:
                self.services.checkpoint()
            if self.services.clock() >= deadline:
                raise TimeoutError("semantic cooperative deadline")

        class Budget:
            def check(self):
                check()

        try:
            check()
            if video is None:
                raise ValueError("semantic assessment requires the registered source video")
            result = QueryResult.from_dict(value, video=video)
            if (result.processing_status is not ProcessingStatus.COMPLETE
                    or result.evidence is None or not result.evidence.items):
                assessment["reason"] = "no_complete_candidates"
            elif not requires_semantic_verification(result.mode, result.query_config,
                                                    result.evidence.items):
                assessment["reason"] = "explicit_mock_contract_not_learned_inference"
            else:
                step = 2 if result.query.relation else 1
                if (len(result.evidence.items) % step
                        or not 1 <= len(result.evidence.items) // step <= MAX_SEQUENTIAL_UNITS):
                    raise ValueError("semantic sequential unit budget exceeded")
                refs = []
                for item in result.evidence.items:
                    if item.frame_ref.identity_key not in [r.identity_key for r in refs]:
                        refs.append(item.frame_ref)
                if any(ref.width * ref.height > 2_073_600 for ref in refs):
                    raise ValueError("semantic source pixel budget exceeded")
                _local(video.source_path)
                if _source_hash(video.source_path, Budget()) != video.source_sha256:
                    raise ValueError("semantic source hash differs")
                frames = self.services.frame_loader(video, tuple(refs), check)
                check()
                # Validate the complete set, including later units, before any factory call.
                unit_backend = BoundSemanticVerifier(self.profile).identity_sha256
                multiple = len(result.evidence.items) > step
                backend = hashlib.sha256(_canonical({"unit_backend_sha256": unit_backend,
                    "policy": SEQUENTIAL_POLICY, "max_units": MAX_SEQUENTIAL_UNITS,
                }).encode()).hexdigest() if multiple else unit_backend
                parent = SemanticRequest.from_result(result, frames, backend_sha256=backend)
                units = parent.to_dict()["units"]
                assessment.update(request_sha256=parent.sha256, backend_sha256=backend)
                if multiple:
                    assessment.update(aggregation_policy=SEQUENTIAL_POLICY,
                                      unit_backend_sha256=unit_backend)
                receipts = [{**unit, "status": "not_run", "report": None} for unit in units]
                assessment["unit_assessments"] = receipts
                directory = self.work_root / ("semantic-" + uuid.uuid4().hex)
                _safe(directory)
                directory.mkdir()
                assessment["work_directory"] = str(directory)
                factory = self.services.transport_factory or native_transport
                reports = []
                for index, receipt in enumerate(receipts):
                    transport = None
                    receipt["status"] = "failed"
                    try:
                        check()
                        parent.validate_frames(frames)
                        projected = _unit_result(result, parent, index) if multiple else result
                        keys = {i.frame_ref.identity_key for i in projected.evidence.items}
                        selected_frames = tuple(
                            f for f in frames if f.frame_ref.identity_key in keys)
                        unit_dir = directory / f"unit-{index:03d}" if multiple else directory
                        if multiple:
                            unit_dir.mkdir()
                        receipt["work_directory"] = str(unit_dir)
                        transport = factory(profile=self.profile, result=projected, video=video,
                            frames=selected_frames, directory=unit_dir, checkpoint=check)
                        verifier = BoundSemanticVerifier(transport)
                        if verifier.identity_sha256 != unit_backend:
                            raise ValueError("sequential unit backend differs")
                        request = verifier.request(projected, selected_frames)
                        if request.to_dict()["units"] != [units[index]]:
                            raise ValueError("sequential projection changed evidence membership")
                        receipt.update(request_sha256=request.sha256,
                            query_revision=projected.query.query_id,
                            result_sha256=request.to_dict()["result_sha256"])
                        report = verifier.assess(request, selected_frames)
                        check()
                        parent.validate_frames(frames)
                        if _source_hash(video.source_path, Budget()) != video.source_sha256:
                            raise ValueError("semantic source changed during assessment")
                        receipt.update(status="assessed", report=asdict(report))
                        reports.extend(report.assessments)
                    finally:
                        lifecycle = getattr(transport, "last_lifecycle", None)
                        if lifecycle is not None:
                            receipt["worker_lifecycle"] = deepcopy(lifecycle)
                if result.to_dict() != value:
                    raise ValueError("semantic assessment changed the upstream result")
                # Host assembly is explicit and validates exact complete parent membership.
                report = SemanticReport.from_json(parent, _canonical({
                    "schema_version": parent.to_dict()["schema_version"],
                    "request_sha256": parent.sha256, "backend_sha256": backend,
                    "assessments": [asdict(row) for row in reports],
                }))
                assessment.update(status="assessed", report=asdict(report),
                                  reason="unqualified_assessment_cannot_support_answer")
        except (Exception, KeyboardInterrupt) as error:
            if isinstance(error, (SemanticCancelled, KeyboardInterrupt)):
                status, code = "cancelled", "semantic_cancelled"
            elif isinstance(error, TimeoutError):
                status, code = "deadline_exceeded", "semantic_deadline"
            else:
                status, code = "failed", "semantic_validation_or_transport_failed"
                # The supervised worker wraps errors; check the outer deadline/cancel state too.
                try:
                    check()
                except (Exception, KeyboardInterrupt) as state:
                    if isinstance(state, (SemanticCancelled, KeyboardInterrupt)):
                        status, code = "cancelled", "semantic_cancelled"
                    elif isinstance(state, TimeoutError):
                        status, code = "deadline_exceeded", "semantic_deadline"
            assessment.update(status=status, error_code=code, error_type=type(error).__name__)
        finally:
            assessment["wall_seconds"] = max(0, self.services.clock()-started)
            if transport is not None and "aggregation_policy" not in assessment:
                lifecycle = getattr(transport, "last_lifecycle", None)
                if lifecycle is not None:
                    assessment["worker_lifecycle"] = deepcopy(lifecycle)
        envelope = {"schema_version": "video_intelligence.semantic_cli_output.v1",
            "query_result": deepcopy(value), "semantic_assessment": assessment,
            "upstream_output": original if "query_result" in original else None}
        return envelope, 0 if assessment["status"] in {"assessed", "not_run"} else 1
