"""Create-only binary mask PNG artifacts, independently verified on replay."""

from __future__ import annotations

import hashlib
import json
import os
import re
import struct
import uuid
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np

from video_intelligence.retrieval.schemas import QueryResult, SupportStatus
from video_intelligence.segmentation.masks import unpack_mask
from video_intelligence.storage.evidence_frames import (
    EvidenceFrameArtifact,
    EvidenceFrameConfig,
    _Budget,
    _digest,
    _json_bytes,
    _manifest_video,
    _members,
    _mkdir,
    _portable,
    _read,
    _safe,
    _write,
)

SCHEMA = "video_intelligence.evidence_masks.v1"
MAX_MASKS = 1024


def _artifact(digest, size):
    return EvidenceFrameArtifact(f"evidence_masks_{digest}", "evidence_masks", digest,
                                 f"masks/{digest}/manifest.json", size)


def _png_mask(data, ref, config):
    if (len(data) < 33 or data[:8] != b"\x89PNG\r\n\x1a\n" or data[12:16] != b"IHDR"
            or struct.unpack(">II", data[16:24]) != (ref.width, ref.height)
            or ref.width * ref.height > config.max_frame_pixels):
        raise ValueError("mask PNG header/geometry exceeds limits")
    mask = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if (mask is None or mask.dtype != np.uint8 or mask.shape != (ref.height, ref.width)
            or not np.isin(mask, [0, 255]).all()):
        raise ValueError("mask PNG must contain only grayscale 0/255")
    return mask == 255


def _updated(original, manifest, artifact):
    records = {row["evidence_id"]: row for row in manifest["masks"]}
    items = []
    for item in original.evidence.items:
        if item.evidence_id not in records:
            items.append(item)
            continue
        row = records[item.evidence_id]
        provenance = deepcopy(item.provenance)
        segmentation = provenance["segmentation"]
        segmentation["mask"] = {key: value for key, value in segmentation["mask"].items()
                                if key != "data"}
        segmentation["mask_persisted"] = True
        segmentation["mask_artifact"] = {
            "artifact_id": artifact.artifact_id, "relative_path":
                f"masks/{artifact.content_sha256}/{row['file']}",
            "png_sha256": row["png_sha256"], "binary_sha256": row["binary_sha256"],
            "frame_ref": _portable(item.frame_ref),
        }
        items.append(replace(item, provenance=provenance))
    logical = original.logical_query_id or original.query.query_id
    unbound = replace(original, query=replace(original.query, query_id=logical),
                      logical_query_id=None,
                      evidence=replace(original.evidence, query_id=logical, items=tuple(items)))
    return unbound.with_revision({**original.query_config, "mask_evidence": artifact.to_dict()})


def _item_binding(item):
    return {"frame_ref": _portable(item.frame_ref), "label": item.label,
            "bbox_xyxy": item.bbox_xyxy, "score": item.score,
            "source_sha256": item.source_sha256, "snapshot_id": item.snapshot_id,
            "source_stage": item.source_stage, "role": item.role,
            "lineage": {key: item.provenance.get(key) for key in (
                "grounding", "tracking", "segmentation", "frame_rgb_sha256",
            )}}


class EvidenceMaskStore:
    def __init__(self, root):
        self.root = Path(root).expanduser().absolute()
        _safe(self.root)

    def persist(self, result, *, video, config=None, checkpoint=None):
        budget = _Budget(config or EvidenceFrameConfig(), checkpoint)
        budget.check()
        result.validate_video(video)
        if result.support_status is not SupportStatus.SUPPORTED:
            return result, None
        selected = [item for item in result.evidence.items if "segmentation" in item.provenance]
        if not selected:
            return result, None
        if "mask_evidence" in result.query_config:
            return result, self.verify_result(result, config=config, checkpoint=checkpoint)
        if len(selected) > MAX_MASKS or len({x.frame_ref.identity_key for x in selected}) > (
            budget.config.max_frames
        ):
            raise ValueError("selected mask count/frame cap exceeded")
        original = QueryResult.from_dict(result.to_dict(), video=video)
        staging = self.root / "masks" / f".staging-{uuid.uuid4().hex}"
        _mkdir(staging / "images")
        rows, total = [], 0
        for index, item in enumerate(selected):
            budget.check()
            value = item.provenance["segmentation"]
            if value.get("mask_persisted") or "mask_artifact" in value:
                raise ValueError("unpersisted mask has conflicting artifact claims")
            mask = unpack_mask(value["mask"], width=item.frame_ref.width,
                               height=item.frame_ref.height)
            if not mask.any() or mask.size > budget.config.max_frame_pixels:
                raise ValueError("evidence mask must be nonempty and within pixel budget")
            if value.get("frame_rgb_sha256") != item.provenance.get("frame_rgb_sha256"):
                raise ValueError("mask source RGB differs from grounded evidence")
            ok, png = cv2.imencode(".png", mask.astype(np.uint8) * 255)
            if not ok:
                raise ValueError("mask PNG encoding failed")
            data = png.tobytes()
            if not np.array_equal(_png_mask(data, item.frame_ref, budget.config), mask):
                raise ValueError("mask PNG roundtrip differs")
            total += len(data)
            if total > budget.config.max_total_bytes:
                raise ValueError("mask artifact byte budget exceeded")
            name = f"images/mask_{index:04d}.png"
            _write(staging / name, data, budget)
            rows.append({"file": name, "evidence_id": item.evidence_id,
                         "frame_ref": _portable(item.frame_ref), "png_size_bytes": len(data),
                         "png_sha256": hashlib.sha256(data).hexdigest(),
                         "binary_sha256": value["mask"]["sha256"], "area": value["mask"]["area"]})
        manifest = {"schema_version": SCHEMA, "config": budget.config.to_dict(), "masks": rows,
                    "video": {"video_id": video.video_id, "source_sha256": video.source_sha256,
                              "metadata": video.metadata.to_dict()},
                    "input_query_result": original.to_dict()}
        data = _json_bytes(manifest)
        total += len(data) + 65
        if total > budget.config.max_total_bytes:
            raise ValueError("complete mask artifact byte budget exceeded")
        digest = hashlib.sha256(data).hexdigest()
        _write(staging / "manifest.json", data, budget)
        final = self.root / "masks" / digest
        _safe(final)
        try:
            final.mkdir()
        except FileExistsError:
            self._verify(digest, budget)
        else:
            _mkdir(final / "images")
            for name in [row["file"] for row in rows] + ["manifest.json"]:
                budget.check()
                _safe(staging / name, directory=False)
                _safe((final / name).parent, directory=True)
                os.link(staging / name, final / name, follow_symlinks=False)
            self._verify(digest, budget, ready=False)
            _write(final / "READY", (digest + "\n").encode("ascii"), budget)
        artifact = _artifact(digest, total)
        updated = _updated(original, manifest, artifact)
        if self.verify_result(updated, config=config, checkpoint=checkpoint) != artifact:
            raise ValueError("mask publication readback differs")
        return updated, artifact

    def _verify(self, digest, budget, *, ready=True):
        digest = _digest(digest)
        final = self.root / "masks" / digest
        expected = {"images", "manifest.json"} | ({"READY"} if ready else set())
        if _members(final, len(expected), budget) != expected:
            raise ValueError("incomplete or unexpected mask artifact members")
        if ready and _read(final / "READY", 65, budget) != (digest + "\n").encode("ascii"):
            raise ValueError("mask READY differs")
        data = _read(final / "manifest.json", budget.config.max_total_bytes, budget)
        if hashlib.sha256(data).hexdigest() != digest:
            raise ValueError("mask manifest hash differs")
        manifest = json.loads(data)
        if (_json_bytes(manifest) != data or set(manifest) != {
            "schema_version", "config", "masks", "video", "input_query_result",
        } or manifest["schema_version"] != SCHEMA):
            raise ValueError("invalid mask manifest schema")
        stored = EvidenceFrameConfig(**manifest["config"])
        if stored.to_dict() != manifest["config"]:
            raise ValueError("incomplete mask persistence config")
        limits = EvidenceFrameConfig(*(min(getattr(budget.config, name), getattr(stored, name))
                                      for name in ("max_frames", "max_frame_pixels",
                                                   "max_total_bytes", "max_runtime_seconds")))
        budget.deadline = min(budget.deadline, budget.started + limits.max_runtime_seconds)
        video = _manifest_video(manifest)
        original = QueryResult.from_dict(manifest["input_query_result"], video=video)
        if original.support_status is not SupportStatus.SUPPORTED:
            raise ValueError("mask artifact requires supported observed evidence")
        selected = {item.evidence_id: item for item in original.evidence.items
                    if "segmentation" in item.provenance}
        rows = manifest["masks"]
        if not isinstance(rows, list) or not 1 <= len(rows) <= MAX_MASKS:
            raise ValueError("mask artifact count exceeded")
        seen, names, frame_ids = set(), set(), set()
        total = len(data) + 65
        for row in rows:
            budget.check()
            if set(row) != {"file", "evidence_id", "frame_ref", "png_size_bytes", "png_sha256",
                            "binary_sha256", "area"}:
                raise ValueError("invalid mask member fields")
            name = row["file"]
            if (not isinstance(name, str)
                    or re.fullmatch(r"images/mask_[0-9]{4}\.png", name) is None):
                raise ValueError("unsafe mask image path")
            item = selected.get(row["evidence_id"])
            if (item is None or item.evidence_id in seen or name in names
                    or row["frame_ref"] != _portable(item.frame_ref)):
                raise ValueError("mask item identity differs")
            seen.add(item.evidence_id)
            names.add(name)
            frame_ids.add(item.frame_ref.identity_key)
            if len(frame_ids) > limits.max_frames:
                raise ValueError("mask frame count cap exceeded")
            expected = unpack_mask(item.provenance["segmentation"]["mask"],
                                   width=item.frame_ref.width, height=item.frame_ref.height)
            size = row["png_size_bytes"]
            if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
                raise ValueError("invalid mask byte count")
            total += size
            if total > limits.max_total_bytes:
                raise ValueError("mask artifact exceeds byte cap")
            png = _read(final / name, min(size, limits.max_total_bytes), budget)
            actual = _png_mask(png, item.frame_ref, limits)
            if (len(png) != size or hashlib.sha256(png).hexdigest() != row["png_sha256"]
                    or not np.array_equal(actual, expected) or not actual.any()
                    or int(actual.sum()) != row["area"]
                    or hashlib.sha256(actual.astype(np.uint8).tobytes()).hexdigest()
                    != row["binary_sha256"]):
                raise ValueError("mask PNG/area/hash differs from segmentation")
        if seen != set(selected) or _members(final / "images", len(names), budget) != {
            name.split("/")[1] for name in names
        }:
            raise ValueError("mask artifact omitted or added mask members")
        return manifest, _artifact(digest, total)

    def verify_result(self, result, *, config=None, checkpoint=None):
        budget = _Budget(config or EvidenceFrameConfig(), checkpoint)
        policy = result.query_config.get("mask_evidence")
        segmented = [] if result.evidence is None else [item for item in result.evidence.items
                    if item.provenance.get("segmentation", {}).get("mask_persisted")]
        if policy is None:
            if segmented:
                raise ValueError("persisted masks require a bound mask artifact")
            return None
        if result.support_status is not SupportStatus.SUPPORTED:
            raise ValueError("unsupported result cannot claim persisted masks")
        manifest, artifact = self._verify(policy.get("content_sha256"), budget)
        if artifact.to_dict() != policy:
            raise ValueError("mask artifact metadata differs")
        original = QueryResult.from_dict(manifest["input_query_result"],
                                         video=_manifest_video(manifest))
        expected = _updated(original, manifest, artifact)
        if (any(result.query_config.get(key) != value
                for key, value in original.query_config.items())
                or result.video_id != original.video_id or result.run_id != original.run_id
                or result.mode != original.mode
                or {k: v for k, v in result.query.to_dict().items() if k != "query_id"}
                != {k: v for k, v in original.query.to_dict().items() if k != "query_id"}):
            raise ValueError("saved mask query identity differs")
        expected_masks = {item.evidence_id: _item_binding(item) for item in expected.evidence.items}
        actual_masks = {item.evidence_id: _item_binding(item) for item in result.evidence.items}
        if expected_masks != actual_masks:
            raise ValueError("mask evidence pointers or provenance differ")
        return artifact
