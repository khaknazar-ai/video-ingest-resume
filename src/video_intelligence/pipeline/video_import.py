"""Local reference/managed import with fenced, durable copy/probe checkpoints.

No transcoding, inference, cleanup, uploads, or downstream stage admission. A
filesystem object without a validated SQLite checkpoint is never a ready import.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import subprocess
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from video_intelligence.core.paths import resolve_project_paths
from video_intelligence.core.types import VideoMetadata, VideoRecord
from video_intelligence.pipeline.jobs import JobQueue, JobStatus
from video_intelligence.storage.evidence_frames import _mkdir, _safe
from video_intelligence.storage.sqlite import LeaseFenceError, SQLiteStore
from video_intelligence.video import probe as probe_module

SCHEMA = "video_intelligence.local_import.v1"
_TOOL_CACHE = {}


def _json(value):
    return json.dumps(value, sort_keys=True, allow_nan=False, separators=(",", ":"))


def _digest(value):
    return hashlib.sha256(_json(value).encode()).hexdigest()


def _stat(path):
    stat = path.stat()
    return [stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns]


def _probe_code():
    return hashlib.sha256(Path(probe_module.__file__).read_bytes()).hexdigest()


class ImportFailure(RuntimeError):
    def __init__(self, code, message, *, job_id=None, target=None):
        super().__init__(message)
        self.code, self.job_id, self.target = code, job_id, target


@dataclass(frozen=True)
class ImportConfig:
    max_bytes: int = 512 * 1024**2
    timeout_seconds: float = 60.
    min_free_bytes: int = 64 * 1024**2
    chunk_bytes: int = 1024**2
    max_pixels: int = 2_073_600
    max_duration_seconds: float = 3600.
    pipeline_version: str = "local-import.v1"
    probe_profile: str = "ffprobe-strict.v1"

    def __post_init__(self):
        for name, maximum in (("max_bytes", 512 * 1024**2), ("chunk_bytes", 16 * 1024**2),
                              ("max_pixels", 8_294_400), ("min_free_bytes", 1024**4)):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= maximum:
                raise ValueError(f"{name} must be an integer in [1,{maximum}]")
        for name, maximum in (("timeout_seconds", 60.), ("max_duration_seconds", 86_400.)):
            value = getattr(self, name)
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(value) or not 0 < value <= maximum):
                raise ValueError(f"{name} must be finite in (0,{maximum}]")
        for name in ("pipeline_version", "probe_profile"):
            value = getattr(self, name)
            if (not isinstance(value, str) or not value or len(value) > 128
                    or any(c.isspace() for c in value)):
                raise ValueError(f"{name} must be a bounded identifier")


class LocalVideoImport:
    def __init__(self, store: SQLiteStore, config: ImportConfig | None = None):
        self.store, self.queue = store, JobQueue(store)
        self.config = config or ImportConfig()
        self.job = None
        self.started = time.monotonic()
        self.cancel_file = None
        self.progress = None
        self.next_heartbeat = 0.

    def _runtime(self, job=None, *, cancel_file=None, progress=None):
        self.job, self.started, self.next_heartbeat = job, time.monotonic(), 0.
        self.cancel_file = Path(cancel_file).expanduser().absolute() if cancel_file else None
        if self.cancel_file is not None:
            _safe(self.cancel_file)
        self.progress = progress

    def _check(self, *, renew=True):
        remaining = self.config.timeout_seconds-(time.monotonic()-self.started)
        if remaining <= 0:
            raise ImportFailure("import_timeout", "local import deadline exceeded")
        if self.cancel_file is not None and self.cancel_file.exists():
            raise ImportFailure("import_cancelled", "local cancellation marker exists")
        if self.job is not None:
            row = self.queue.get(self.job.job_id)
            if row is not None and row.status == JobStatus.CANCELLED:
                raise ImportFailure("import_cancelled", "import job cancelled")
            if (row is None or row.status != JobStatus.RUNNING
                    or row.owner != self.job.owner or row.lease_token != self.job.lease_token
                    or row.lease_until is None or row.lease_until <= time.time()):
                raise LeaseFenceError("local import lost its lease")
            if renew and time.monotonic() >= self.next_heartbeat:
                if not self.queue.heartbeat(self.job.job_id, owner=self.job.owner,
                        lease_token=self.job.lease_token, lease_seconds=max(.1, remaining)):
                    raise LeaseFenceError("local import heartbeat was fenced")
                self.next_heartbeat = time.monotonic()+.5

    def _event(self, name):
        if self.progress is not None:
            self.progress(name)
        self._check()

    def _probe_tool(self):
        self._check()
        executable = shutil.which("ffprobe")
        if executable is None:
            raise ImportFailure("probe_unavailable", "strict import requires existing ffprobe")
        path = Path(executable).absolute()
        _safe(path, directory=False)
        identity = _stat(path)
        if not 0 < identity[2] <= 512*1024**2:
            raise ImportFailure("probe_tool_limit", "ffprobe identity exceeds fingerprint budget")
        key = (str(path), *identity)
        if key not in _TOOL_CACHE:
            digest = hashlib.sha256()
            with path.open("rb") as handle:
                while chunk := handle.read(1024**2):
                    self._check()
                    digest.update(chunk)
            if _stat(path) != identity:
                raise ImportFailure("pipeline_changed", "ffprobe changed during fingerprinting")
            _TOOL_CACHE[key] = digest.hexdigest()
        return dict(path=str(path), sha256=_TOOL_CACHE[key])

    def _hash(self, path, *, renew=True):
        _safe(path, directory=False)
        before = _stat(path)
        digest = hashlib.sha256()
        count = 0
        with path.open("rb") as handle:
            while True:
                self._check(renew=renew)
                chunk = handle.read(self.config.chunk_bytes)
                if not chunk:
                    break
                count += len(chunk)
                if count > self.config.max_bytes:
                    raise ImportFailure("source_size_limit", "hash input exceeds byte budget")
                digest.update(chunk)
        if _stat(path) != before:
            raise ImportFailure("source_changed", f"file changed during hashing: {path}")
        self._check(renew=renew)
        return digest.hexdigest()

    def create(self, source, *, mode="reference", video_id=None, managed_root=None):
        self._runtime()
        if mode not in {"reference", "managed"}:
            raise ValueError("import mode must be reference or managed")
        source = Path(source).expanduser().absolute()
        _safe(source, directory=False)
        identity = _stat(source)
        if not 0 < identity[2] <= self.config.max_bytes:
            raise ImportFailure("source_size_limit", "source size exceeds import budget")
        digest = self._hash(source)
        if _stat(source) != identity:
            raise ImportFailure("source_changed", "source changed during import preparation")
        root = None
        if mode == "managed":
            root = Path(managed_root or (
                resolve_project_paths().datasets_root/"video_intelligence"/"managed"
            )).expanduser().absolute()
            _safe(root)
        elif managed_root is not None:
            raise ValueError("managed_root is only valid in managed mode")
        video_id = video_id or f"vid_{digest[:12]}"
        # Reuse the public record identifier validation without changing its schema.
        VideoRecord(video_id, source, VideoMetadata(None, None, None, None, None), digest)
        payload = dict(schema=SCHEMA, mode=mode, source=str(source), source_identity=identity,
            source_sha256=digest, source_size=identity[2], video_id=video_id,
            managed_root=str(root) if root else None, config=asdict(self.config),
            probe_code_sha256=_probe_code(), probe_tool=self._probe_tool())
        return self.queue.create(stage="video_import", config_sha256=_digest(payload["config"]),
                                 input_artifact_sha256=digest, payload=payload)

    def _source(self, *, renew=True):
        source = Path(self.job.payload["source"])
        if (_stat(source) != self.job.payload["source_identity"]
                or self._hash(source, renew=renew) != self.job.payload["source_sha256"]):
            raise ImportFailure("source_changed", "original input differs from its pinned identity",
                                target=str(source))
        return source

    def _space(self, root, remaining):
        self._check()
        _safe(root, directory=True)
        if shutil.disk_usage(root).free < remaining+self.config.min_free_bytes:
            raise ImportFailure("insufficient_space", "managed copy exceeds free-space reserve")

    def _copy(self):
        source = self._source()
        root = Path(self.job.payload["managed_root"])
        _mkdir(root)
        objects, staging = root/"objects", root/"staging"
        _mkdir(objects)
        _mkdir(staging)
        digest, size = self.job.payload["source_sha256"], self.job.payload["source_size"]
        target = objects/digest
        key = _digest(dict(stage="copy.v1", root=str(root), source_sha256=digest, size=size))
        expected = dict(stage="copy", source_sha256=digest, bytes=size, path=str(target))
        cached = self.store.get_import_checkpoint(key)
        _safe(target)
        if target.exists():
            if target.stat().st_size != size or self._hash(target) != digest:
                raise ImportFailure("destination_conflict", "managed object has different bytes",
                                    target=str(target))
            self._source()
            if cached == expected:
                return target, key, True
            self.store.save_import_checkpoint(self.job.job_id, self.job.owner, self.job.lease_token,
                stage="copy", cache_key=key, output=expected,
                publish=lambda: self._source(renew=False))
            return target, key, True
        self._space(root, size)
        descriptor, name = tempfile.mkstemp(
            prefix=f"{self.job.job_id}-", suffix=".partial", dir=staging
        )
        partial = Path(name)
        copied, copied_hash = 0, hashlib.sha256()
        # Failed/interrupted partials are retained, never resumed or marked ready.
        with os.fdopen(descriptor, "wb") as output, source.open("rb") as input_handle:
            while chunk := input_handle.read(self.config.chunk_bytes):
                self._space(root, max(0, size-copied))
                copied += len(chunk)
                if copied > size:
                    raise ImportFailure("source_changed", "source grew while copying")
                output.write(chunk)
                copied_hash.update(chunk)
                self._event("copy_chunk")
            output.flush()
            os.fsync(output.fileno())
        if copied != size or copied_hash.hexdigest() != digest:
            raise ImportFailure("source_changed", "copied bytes do not match the pinned source")
        self._source()

        def publish():
            self._source(renew=False)
            if self._hash(partial, renew=False) != digest:
                raise ImportFailure("copy_changed", "staged copy changed before publication")
            _safe(target)
            try:
                # Same-filesystem immutable create-if-absent, never exists()+replace.
                os.link(partial, target)
            except PermissionError as error:
                raise ImportFailure("publication_denied", str(error), target=str(target)) from error
            except FileExistsError as error:
                raise ImportFailure("destination_conflict", "target appeared during publication",
                                    target=str(target)) from error
            if self._hash(target, renew=False) != digest:
                raise ImportFailure("copy_changed", "published bytes differ from the pinned source")
            self._source(renew=False)

        self.store.save_import_checkpoint(self.job.job_id, self.job.owner, self.job.lease_token,
            stage="copy", cache_key=key, output=expected, publish=publish)
        # Only our newly created, successfully linked temporary alias is removed.
        partial.unlink()
        return target, key, False

    def _metadata(self, path):
        probe_identity = _stat(path)
        key = _digest(dict(stage="probe.v1", source_sha256=self.job.payload["source_sha256"],
            pipeline_version=self.config.pipeline_version, profile=self.config.probe_profile,
            code=self.job.payload["probe_code_sha256"], tool=self.job.payload["probe_tool"],
            max_pixels=self.config.max_pixels,
            max_duration_seconds=self.config.max_duration_seconds))
        cached = self.store.get_import_checkpoint(key)
        metadata = None
        if cached is not None and set(cached) == {"source_sha256", "probe_key", "metadata"}:
            if (cached["source_sha256"] == self.job.payload["source_sha256"]
                    and cached["probe_key"] == key):
                try:
                    metadata = VideoMetadata.from_dict(cached["metadata"])
                except (ValueError, TypeError):
                    pass
        reused = metadata is not None
        if metadata is None:
            self._check()
            remaining = self.config.timeout_seconds-(time.monotonic()-self.started)
            try:
                metadata = probe_module.probe_video(path, allow_fallback=False,
                                                    timeout_seconds=max(.000001, remaining))
            except subprocess.TimeoutExpired as error:
                raise ImportFailure("probe_timeout", str(error)) from error
            self._check()
        if (metadata.width is None or metadata.height is None
                or metadata.width*metadata.height > self.config.max_pixels
                or metadata.duration_seconds is None
                or metadata.duration_seconds > self.config.max_duration_seconds):
            raise ImportFailure("media_limit", "probe geometry/duration is missing or over budget")
        if self._hash(path) != self.job.payload["source_sha256"]:
            raise ImportFailure("copy_changed", "probe input changed")
        if _stat(path) != probe_identity:
            raise ImportFailure("copy_changed", "probe input identity changed during probe")
        self._source()
        if not reused:
            value = dict(source_sha256=self.job.payload["source_sha256"], probe_key=key,
                         metadata=metadata.to_dict())
            self.store.save_import_checkpoint(self.job.job_id, self.job.owner, self.job.lease_token,
                stage="probe", cache_key=key, output=value,
                publish=lambda: self._source(renew=False))
        return metadata, key, reused

    def _validate_job(self, job):
        payload = job.payload
        keys = {"schema", "mode", "source", "source_identity", "source_sha256", "source_size",
                "video_id", "managed_root", "config", "probe_code_sha256", "probe_tool"}
        if (job.stage != "video_import" or set(payload) != keys or payload["schema"] != SCHEMA):
            raise ValueError("job is not a local video import")
        self.config = ImportConfig(**payload["config"])
        if (job.config_sha256 != _digest(payload["config"])
                or job.input_artifact_sha256 != payload["source_sha256"]
                or payload["mode"] not in {"reference", "managed"}):
            raise ValueError("import request identity mismatch")
        identity = payload["source_identity"]
        if (not isinstance(identity, list) or len(identity) != 4
                or any(type(v) is not int for v in identity)
                or type(payload["source_size"]) is not int
                or payload["source_size"] != identity[2]
                or not 0 < identity[2] <= self.config.max_bytes):
            raise ValueError("invalid pinned source size/identity")
        source = Path(payload["source"])
        if not source.is_absolute():
            raise ValueError("import source must be absolute")
        _safe(source, directory=False)
        VideoRecord(payload["video_id"], source, VideoMetadata(None, None, None, None, None),
                    payload["source_sha256"])
        if payload["mode"] == "managed":
            root = Path(payload["managed_root"])
            if not root.is_absolute():
                raise ValueError("managed root must be absolute")
            _safe(root)
        elif payload["managed_root"] is not None:
            raise ValueError("reference imports cannot declare a managed root")
        if payload["probe_code_sha256"] != _probe_code():
            raise ImportFailure("pipeline_changed", "probe code changed; create a new import")
        if payload["probe_tool"] != self._probe_tool():
            raise ImportFailure("pipeline_changed", "ffprobe binary changed; create a new import")

    def execute_claimed(self, job, *, cancel_file=None, progress=None):
        self._runtime(job, cancel_file=cancel_file, progress=progress)
        self._validate_job(job)
        source = self._source()
        target, copy_key, reused_copy = source, None, False
        if job.payload["mode"] == "managed":
            target, copy_key, reused_copy = self._copy()
            self._event("copy_checkpoint")
        metadata, probe_key, reused_probe = self._metadata(target)
        self._event("probe_checkpoint")
        record = VideoRecord(
            job.payload["video_id"], target, metadata, job.payload["source_sha256"]
        )
        receipt = dict(schema=SCHEMA, job_id=job.job_id, mode=job.payload["mode"],
            original_source=str(source), video=record.to_dict(),
            source_size=job.payload["source_size"],
            copy_key=copy_key, probe_key=probe_key, config_sha256=job.config_sha256,
            pipeline_version=self.config.pipeline_version, probe_code_sha256=_probe_code(),
            probe_tool=job.payload["probe_tool"],
            reused_copy=reused_copy, reused_probe=reused_probe, quality_status="unmeasured")
        self._event("before_activate")

        def activation_guard():
            self._source(renew=False)
            if self._hash(target, renew=False) != record.source_sha256:
                raise ImportFailure("copy_changed", "source changed before activation")

        output_sha = self.store.commit_video_import(job.job_id, job.owner, job.lease_token,
            record=record, receipt=receipt, checkpoint=activation_guard)
        return receipt, output_sha

    def run(self, job_id, *, cancel_file=None, progress=None):
        existing = self.queue.get(job_id)
        if existing is None or existing.stage != "video_import":
            raise ValueError("unknown local import job")
        if existing.status == JobStatus.SUCCEEDED:
            receipt = self.store.get_video_import_result(job_id)
            if receipt is None:
                raise ImportFailure("receipt_missing", "completed import receipt is missing")
            self._runtime(cancel_file=cancel_file)
            self._validate_job(existing)
            if _stat(Path(existing.payload["source"])) != existing.payload["source_identity"]:
                raise ImportFailure("source_changed", "original source identity changed")
            for path in {Path(receipt["original_source"]), Path(receipt["video"]["source_path"])}:
                if self._hash(path) != existing.payload["source_sha256"]:
                    raise ImportFailure("source_changed", "completed import source bytes changed")
            return receipt
        self.store.retry_video_import(job_id)
        job = self.queue.claim(job_id, owner=f"import_{os.getpid()}", lease_seconds=60.)
        if job is None:
            raise ImportFailure("import_busy", "another live lease owns this import", job_id=job_id)
        try:
            return self.execute_claimed(job, cancel_file=cancel_file, progress=progress)[0]
        except Exception as error:
            code = getattr(error, "code", "lost_lease" if isinstance(error, LeaseFenceError)
                           else "import_failed")
            status = JobStatus.CANCELLED if code == "import_cancelled" else JobStatus.FAILED
            self.queue.complete(job_id, owner=job.owner, lease_token=job.lease_token,
                                status=status, error_code=code)
            if isinstance(error, ImportFailure):
                error.job_id = job_id
                raise
            raise ImportFailure(code, str(error), job_id=job_id) from error

    def status(self, job_id):
        job = self.queue.get(job_id)
        if job is None or job.stage != "video_import":
            raise ValueError("unknown local import job")
        return dict(job=job.to_dict(), receipt=self.store.get_video_import_result(job_id),
                    validation="historical checkpoint status; resume revalidates current bytes")
