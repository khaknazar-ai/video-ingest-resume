"""CPU preparation/scoring of paired diagnostics, never inference or quality admission.

Worker files contain proposed ROIs, not detector outputs or oracle verdicts. The
closed evaluator retains provisional labels and raw source evidence separately.
All writes are exclusive in a NEW diagnostic directory. This is not production
snapshot publication and does not replace its immutable/hardlink semantics.
"""

from __future__ import annotations

import hashlib
import io
import json
import re
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import numpy as np
from PIL import Image

from video_intelligence.core.types import FrameRef
from video_intelligence.data.dataset import DatasetManifest, require_synthetic_contract
from video_intelligence.data.synthetic_registry import known_synthetic_hashes
from video_intelligence.reasoning.semantic import (
    MAX_RESPONSE_BYTES,
    SemanticReport,
    SemanticRequest,
)
from video_intelligence.reasoning.semantic_backend import SYSTEM_PROMPT
from video_intelligence.reasoning.semantic_generation import (
    GENERATION_POLICY,
    MAX_INPUT_TOKENS,
    MAX_NEW_TOKENS,
    ReportChoices,
)
from video_intelligence.reasoning.semantic_input import EVIDENCE_VIEW, FULL_VIEW, model_input_view
from video_intelligence.retrieval.schemas import (
    EvidenceBundle,
    EvidenceItem,
    QueryResult,
    QuerySpec,
)
from video_intelligence.video.decoder import DecodedFrame

SCHEMA = "video_intelligence.semantic_diagnostic.v1"
MAX_FILE_BYTES = 16_777_216
LABELS = {"supports_query", "contradicts_query", "insufficient_evidence"}
STATES = {"completed", "timeout", "error", "not_run"}


def canonical(value):
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode("utf-8")


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def _sha(value):
    if not isinstance(value, str) or re.fullmatch("[0-9a-f]{64}", value) is None:
        raise ValueError("expected SHA-256")
    return value


def _keys(value, keys):
    if not isinstance(value, dict) or set(value) != set(keys):
        raise ValueError("diagnostic fields differ")


def _unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON field")
        result[key] = value
    return result


def _nonfinite(value):
    raise ValueError("nonfinite JSON")


def read_json(path):
    return _parse(_bytes(Path(path)))


def _parse(data):
    return json.loads(data, object_pairs_hook=_unique, parse_constant=_nonfinite)


def _bytes(path):
    if path.is_symlink() or not path.is_file() or path.stat().st_size > MAX_FILE_BYTES:
        raise ValueError("missing, linked or oversized diagnostic file")
    data = path.read_bytes()
    if len(data) > MAX_FILE_BYTES:
        raise ValueError("diagnostic file grew beyond cap")
    return data


def _file_sha(path):
    before = path.stat()
    h = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(262144):
            h.update(chunk)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError("source changed while hashing")
    return h.hexdigest()


def _rooted(root, relative):
    if not isinstance(relative, str):
        raise ValueError("relative path required")
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("diagnostic path escapes root")
    current = root
    for part in path.parts:
        current = current / part
        if current.is_symlink() or getattr(current, "is_junction", lambda: False)():
            raise ValueError("linked diagnostic path")
    resolved = current.resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError("diagnostic path escapes root")
    return resolved


def _write(root, relative, data):
    path = _rooted(root, relative)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(data)
    return {"path": relative, "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}


def _code_identity():
    # Pin exactly the package used to construct/validate requests, without imports
    # of model SDKs or inspection of weights/environment/services.
    root = Path(__file__).resolve().parents[1]
    rows = [
        {"path": p.relative_to(root).as_posix(), "sha256": _file_sha(p)}
        for p in sorted(root.rglob("*.py"))
    ]
    return digest(rows)


def validate_profile(profile):
    _keys(profile, {"model_id", "revision", "bundle_sha256", "runtime_profile_sha256", "caps"})
    for key in ("model_id", "revision"):
        if not isinstance(profile[key], str) or not 1 <= len(profile[key]) <= 256:
            raise ValueError("model identity required")
    for key in ("bundle_sha256", "runtime_profile_sha256"):
        _sha(profile[key])
    limits = {
        "max_input_tokens": MAX_INPUT_TOKENS,
        "max_new_tokens": MAX_NEW_TOKENS,
        "max_source_pixels": 2_073_600,
        "max_processor_pixels": 65_536,
        "timeout_seconds": 60,
    }
    _keys(profile["caps"], limits)
    for key, maximum in limits.items():
        if type(profile["caps"][key]) is not int or not 1 <= profile["caps"][key] <= maximum:
            raise ValueError("unsupported diagnostic cap")


def _load_rgb(path, case, caps):
    raw = _bytes(path)
    if hashlib.sha256(raw).hexdigest() != case["png_sha256"]:
        raise ValueError("diagnostic PNG changed")
    ref = FrameRef(**case["frame_ref"])
    if ref.width * ref.height > caps["max_source_pixels"]:
        raise ValueError("diagnostic source pixel cap")
    with Image.open(io.BytesIO(raw)) as image:
        if image.mode != "RGB" or image.size != (ref.width, ref.height):
            raise ValueError("diagnostic RGB geometry differs")
        rgb = np.asarray(image).copy()
    if hashlib.sha256(rgb.tobytes()).hexdigest() != case["rgb_sha256"]:
        raise ValueError("diagnostic RGB changed")
    return raw, DecodedFrame(ref, rgb, ref.timestamp_seconds)


def _request(case, frame, backend):
    """Use the existing strict request factory; never instantiate unchecked wire."""
    identity = digest(case)
    query = QuerySpec(
        "q_" + identity[:24], case["query"], "en", "supported", entity=case["category"]
    )
    ref = frame.frame_ref
    provenance = {
        "mode": "diagnostic",
        "run_id": "diagnostic_" + identity,
        "source_sha256": ref.source_sha256,
        "snapshot_id": identity,
        "frame_rgb_sha256": case["rgb_sha256"],
        "frame_rgb_encoding": "rgb8_hwc",
        "candidate_origin": "proposed_roi_not_detector_output",
        "score_semantics": "structural_zero_not_measured_confidence",
    }
    item = EvidenceItem(
        "proposal_" + identity,
        ref,
        case["category"],
        tuple(case["bbox_xyxy"]),
        0.0,
        ref.source_sha256,
        identity,
        "diagnostic_" + identity,
        "diagnostic_roi_proposal",
        provenance=provenance,
    )
    bundle = EvidenceBundle(
        query.query_id,
        provenance["run_id"],
        ref.video_id,
        ref.source_sha256,
        identity,
        (item,),
        ref.timestamp_seconds,
        ref.timestamp_seconds,
        False,
        ("diagnostic_roi_only", "not_retrieval_snapshot", "identity_unverified"),
    )
    result = QueryResult(
        query,
        ref.video_id,
        provenance["run_id"],
        "complete",
        "insufficient_evidence",
        "diagnostic",
        evidence=bundle,
    ).with_revision({"policy": SCHEMA, "input_sha256": identity})
    return SemanticRequest.from_result(result, (frame,), backend_sha256=backend)


def _jobs(inputs, profile, code_sha, worker):
    jobs = []
    for case in inputs:
        _, frame = _load_rgb(_rooted(worker, case["image"]), case, profile["caps"])
        for arm, view in (("baseline", FULL_VIEW), ("candidate", EVIDENCE_VIEW)):
            policy = {
                "input_view": view,
                "generation_policy": GENERATION_POLICY,
                "seed": 0,
                "system_prompt": SYSTEM_PROMPT,
                "model": profile,
                "code_sha256": code_sha,
            }
            request = _request(case, frame, digest(policy))
            choices = ReportChoices.from_request(request)
            job = {
                "schema": SCHEMA,
                "case_id": case["case_id"],
                "arm": arm,
                "paired_input_sha256": digest(case),
                "image": case["image"],
                "png_sha256": case["png_sha256"],
                "rgb_sha256": case["rgb_sha256"],
                "policy": policy,
                "request": request.to_dict(),
                "request_sha256": request.sha256,
                "model_input": model_input_view(request, view),
                "report_choices": list(choices.wires),
                "assets_and_processor_qualified": False,
                "execution_authorized": False,
            }
            jobs.append((job, request))
    return jobs


def prepare(
    packet_path,
    review_path,
    provenance_path,
    profile_path,
    output,
    *,
    packet_sha256,
    contract_manifest: DatasetManifest | None = None,
):
    """Freeze label-free worker files and private provisional evaluator evidence."""
    packet_path, review_path = Path(packet_path), Path(review_path)
    provenance_path, profile_path = Path(provenance_path), Path(profile_path)
    raw_inputs = {
        name: _bytes(path)
        for name, path in (
            ("packet.json", packet_path),
            ("review.json", review_path),
            ("provenance.json", provenance_path),
            ("profile.json", profile_path),
        )
    }
    packet, reviews = _parse(raw_inputs["packet.json"]), _parse(raw_inputs["review.json"])
    provenance, profile = _parse(raw_inputs["provenance.json"]), _parse(raw_inputs["profile.json"])
    validate_profile(profile)
    if digest({k: v for k, v in packet.items() if k != "reviews"}) != _sha(packet_sha256):
        raise ValueError("frozen annotation packet binding differs")
    if reviews["content_sha256"] != packet_sha256:
        raise ValueError("review belongs to a different packet")
    if (
        provenance.get("quality_gold_approved") is not False
        or provenance.get("quality_metrics") is not None
    ):
        raise ValueError("N0 accepts provisional non-quality labels only")
    if provenance.get("reviewer_kind") != "model_assisted_blind_review":
        raise ValueError("explicit model-review provenance required")
    if (
        provenance.get("packet_content_sha256") != packet_sha256
        or provenance.get("review_file_sha256")
        != hashlib.sha256(raw_inputs["review.json"]).hexdigest()
    ):
        raise ValueError("label provenance does not bind this packet/review")
    _keys(reviews, {"content_sha256", "reviews"})
    sources = {s["video_id"]: s for s in packet["sources"]}
    if len(sources) != len(packet["sources"]) or not 1 <= len(packet["cases"]) <= 32:
        raise ValueError("duplicate sources or case cap exceeded")
    contract = packet["kind"] == "synthetic"
    if contract:
        if (
            not isinstance(contract_manifest, DatasetManifest)
            or contract_manifest.kind.value != "synthetic"
            or contract_manifest.purpose.value != "contract"
        ):
            raise ValueError("synthetic input requires explicit contract manifest")
        if set(contract_manifest.source_members) != {
            (s["video_id"], s["source_sha256"]) for s in sources.values()
        }:
            raise ValueError("synthetic contract membership differs")
    elif (packet["kind"], packet["purpose"], packet["split"], packet["usage"]) != (
        "real",
        "quality",
        "validation",
        "development_only_not_final_test",
    ):
        raise ValueError("development-only packet required")
    seen, inputs, images = set(), [], {}
    for case in packet["cases"]:
        identifier = case["case_id"]
        if (
            not isinstance(identifier, str)
            or re.fullmatch(r"[a-zA-Z0-9_-]{1,80}", identifier) is None
        ):
            raise ValueError("portable case ID required")
        if identifier in seen:
            raise ValueError("duplicate case")
        seen.add(identifier)
        source = sources[case["video_id"]]
        ref = FrameRef(**case["frame_ref"])
        if ref.video_id != source["video_id"] or ref.source_sha256 != source["source_sha256"]:
            raise ValueError("case source differs")
        source_path = Path(source["source_path"])
        if contract and require_synthetic_contract(source_path) != contract_manifest:
            raise ValueError("synthetic sidecar differs from explicit contract manifest")
        if _file_sha(source_path) != ref.source_sha256:
            raise ValueError("source bytes differ")
        if not contract and ref.source_sha256 in known_synthetic_hashes(source_path=source_path):
            raise ValueError("known synthetic input requires contract mode")
        # Allowlist only. Packet hash, labels, source titles and review rationale
        # must not influence any worker path, request ID, input hash or payload.
        row = {
            key: case[key]
            for key in ("case_id", "query", "category", "bbox_xyxy", "png_sha256", "rgb_sha256")
        }
        row.update(frame_ref={**asdict(ref), "source_path": None}, image=f"images/{identifier}.png")
        raw, frame = _load_rgb(
            _rooted(packet_path.parent, case["relative_png"]), row, profile["caps"]
        )
        _request(row, frame, "a" * 64)  # Validate all identities/units before writing.
        images[row["image"]] = raw
        inputs.append(row)
    inputs.sort(key=lambda row: row["case_id"])
    label_rows = {}
    for row in reviews["reviews"]:
        _keys(
            row,
            {
                "case_id",
                "reviewer_id",
                "content_sha256",
                "blinded_to_model_outputs",
                "label",
                "rationale",
            },
        )
        if (
            row["case_id"] in label_rows
            or row["case_id"] not in seen
            or row["content_sha256"] != packet_sha256
            or row["label"] not in LABELS
            or row["blinded_to_model_outputs"] is not True
            or not isinstance(row["reviewer_id"], str)
            or not row["reviewer_id"].strip()
            or row["reviewer_id"] == packet.get("proposer_id")
            or not isinstance(row["rationale"], str)
            or not row["rationale"]
        ):
            raise ValueError("invalid or duplicate provisional label")
        label_rows[row["case_id"]] = row["label"]
    if set(label_rows) != seen:
        raise ValueError("every diagnostic case needs a provisional label")
    output = Path(output).absolute()
    output.mkdir(parents=False, exist_ok=False)
    worker = output / "worker"
    worker.mkdir()
    code_sha = _code_identity()
    files = []
    for relative, raw in images.items():
        files.append(_write(output, "worker/" + relative, raw))
    files.append(_write(output, "worker/inputs.json", canonical(inputs)))
    files.append(_write(output, "worker/profile.json", canonical(profile)))
    jobs = []
    for job, _ in _jobs(inputs, profile, code_sha, worker):
        row = _write(output, "worker/jobs/" + digest(job) + ".json", canonical(job))
        files.append(row)
        jobs.append(
            {
                "job_sha256": row["sha256"],
                "path": row["path"],
                "case_id": job["case_id"],
                "arm": job["arm"],
            }
        )
    evaluator = {
        "labels_kind": "provisional_model_review",
        "labels": label_rows,
        "quality_gold_approved": False,
        "primary_accuracy": None,
        "primary_false_support": None,
        "data_kind": "synthetic" if contract else "real",
    }
    files.append(_write(output, "evaluator/labels.json", canonical(evaluator)))
    for name in ("packet.json", "review.json", "provenance.json"):
        files.append(_write(output, "evaluator/" + name, raw_inputs[name]))
    if contract:
        files.append(
            _write(
                output, "evaluator/contract-manifest.json", canonical(contract_manifest.to_dict())
            )
        )
    manifest = {
        "schema": SCHEMA,
        "code_sha256": code_sha,
        "inputs_sha256": digest(inputs),
        "model_profile_sha256": digest(profile),
        "files": files,
        "jobs": jobs,
        "case_count": len(inputs),
        "data_kind": evaluator["data_kind"],
        "quality_gold_approved": False,
        "primary_accuracy": None,
        "execution_authorized": False,
        "native_model_calls": 0,
        "proposal_kind": "roi_diagnostic_not_grounded",
        "arms": ["baseline", "candidate"],
    }
    envelope = {"manifest_sha256": digest(manifest), "manifest": manifest}
    _write(output, "manifest.json", canonical(envelope))  # Written last; no overwrite/retry.
    return envelope


def _verified(root, manifest_sha256):
    root = Path(root).absolute()
    envelope = read_json(root / "manifest.json")
    manifest = envelope["manifest"]
    if digest(manifest) != _sha(manifest_sha256) or envelope["manifest_sha256"] != manifest_sha256:
        raise ValueError("experiment manifest binding differs")
    if manifest["code_sha256"] != _code_identity():
        raise ValueError("experiment source code changed")
    for row in manifest["files"]:
        raw = _bytes(_rooted(root, row["path"]))
        if len(raw) != row["bytes"] or hashlib.sha256(raw).hexdigest() != row["sha256"]:
            raise ValueError("experiment file binding differs")
    inputs, profile = (
        read_json(root / "worker/inputs.json"),
        read_json(root / "worker/profile.json"),
    )
    validate_profile(profile)
    if (
        digest(inputs) != manifest["inputs_sha256"]
        or digest(profile) != manifest["model_profile_sha256"]
    ):
        raise ValueError("experiment input/profile differs")
    built = _jobs(inputs, profile, manifest["code_sha256"], root / "worker")
    expected = {digest(job): (job, request) for job, request in built}
    if set(expected) != {row["job_sha256"] for row in manifest["jobs"]}:
        raise ValueError("experiment job membership differs")
    for row in manifest["jobs"]:
        if read_json(_rooted(root, row["path"])) != expected[row["job_sha256"]][0]:
            raise ValueError("experiment payload differs from inputs/policy")
    return root, manifest, expected


def score(experiment, receipts_path, output, *, manifest_sha256):
    """Score exact bound responses; missing/timeout/malformed stay in all denominators."""
    root, manifest, jobs = _verified(experiment, manifest_sha256)
    raw = _bytes(Path(receipts_path))
    output = Path(output).absolute()
    output.mkdir(parents=False, exist_ok=False)
    _write(output, "raw-receipts.json", raw)  # Preserve exact bytes before receipt parsing.
    try:
        rows = json.loads(raw, object_pairs_hook=_unique, parse_constant=_nonfinite)
        if not isinstance(rows, list) or len(rows) > 2 * len(jobs):
            raise ValueError("receipt list cap/type differs")
    except (ValueError, RecursionError):
        rows = []
        parse_error = True
    else:
        parse_error = False
    indexed, invalid, duplicates = {}, [], set()
    for index, row in enumerate(rows):
        try:
            _keys(row, {"job_sha256", "status", "raw_response"})
            key = row["job_sha256"]
            if key in indexed:
                duplicates.add(key)
            if key not in jobs or key in indexed or row["status"] not in STATES:
                raise ValueError("unknown/duplicate job or status")
            if row["status"] == "completed":
                if not isinstance(row["raw_response"], str):
                    raise ValueError("completed receipt needs raw response")
            elif row["raw_response"] is not None:
                raise ValueError("non-completed receipt cannot carry a response")
            indexed[key] = row
        except (ValueError, TypeError):
            invalid.append(index)
    labels = read_json(root / "evaluator/labels.json")
    if labels["quality_gold_approved"] is not False or labels["primary_accuracy"] is not None:
        raise ValueError("diagnostic labels cannot acquire gold status")
    outcomes, arms = [], {}
    for key, (job, request) in jobs.items():
        receipt = indexed.get(key)
        state = receipt["status"] if receipt else "not_run"
        if key in duplicates:
            state = "invalid_receipt"
        verdict = None
        if state == "completed":
            try:
                if len(receipt["raw_response"].encode("utf-8")) > MAX_RESPONSE_BYTES:
                    raise ValueError("raw response cap")
                report = SemanticReport.from_json(request, receipt["raw_response"])
                verdict = report.assessments[0].verdict.value
            except ValueError:
                state = "malformed"
        provisional = labels["labels"][job["case_id"]]
        normalized = "insufficient_evidence" if verdict == "uncertain" else verdict
        outcome = {
            "case_id": job["case_id"],
            "arm": job["arm"],
            "job_sha256": key,
            "status": state,
            "verdict": verdict,
            "provisional_label": provisional,
            "diagnostic_agreement": normalized == provisional,
        }
        outcomes.append(outcome)
        arms.setdefault(job["arm"], []).append(outcome)
    summaries = {}
    for arm, values in arms.items():

        def ratio(label, predicate, values=values):
            subset = [v for v in values if v["provisional_label"] == label]
            return {"count": sum(predicate(v) for v in subset), "denominator": len(subset)}

        summaries[arm] = {
            "scheduled": len(values),
            "status_counts": dict(Counter(v["status"] for v in values)),
            "provisional_agreement": {
                "count": sum(v["diagnostic_agreement"] for v in values),
                "denominator": len(values),
            },
            "support_against_provisional_contradiction": ratio(
                "contradicts_query", lambda v: v["verdict"] == "supports_query"
            ),
            "support_on_provisional_uncertainty": ratio(
                "insufficient_evidence", lambda v: v["verdict"] == "supports_query"
            ),
            "positive_provisional_agreement": ratio(
                "supports_query", lambda v: v["diagnostic_agreement"]
            ),
            "abstention": {
                "count": sum(v["verdict"] == "uncertain" for v in values),
                "denominator": len(values),
            },
        }
    complete = not (parse_error or invalid) and all(v["status"] == "completed" for v in outcomes)
    result = {
        "schema": SCHEMA,
        "experiment_manifest_sha256": manifest_sha256,
        "raw_receipts_sha256": hashlib.sha256(raw).hexdigest(),
        "status": "diagnostic_complete" if complete else "diagnostic_incomplete",
        "invalid_receipt_rows": invalid,
        "receipt_parse_error": parse_error,
        "arms": summaries,
        "outcomes": outcomes,
        "labels_kind": labels["labels_kind"],
        "data_kind": manifest["data_kind"],
        "primary_accuracy": None,
        "primary_false_support": None,
        "quality_gold_approved": False,
        "quality_metrics": None,
        "native_model_calls": 0,
        "answer_promotion": False,
        "result_identity_sha256": None,
    }
    result["result_identity_sha256"] = digest(result)
    _write(output, "score.json", canonical(result))
    return result
