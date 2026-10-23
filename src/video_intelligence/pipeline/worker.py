"""Bounded local job worker with cooperative cancellation and leases."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from video_intelligence.index.snapshot import SnapshotStore
from video_intelligence.pipeline.index_registry import IndexRegistry
from video_intelligence.pipeline.ingestion import IngestionService, PixelFixtureAdapter
from video_intelligence.pipeline.jobs import Job, JobQueue, JobStatus
from video_intelligence.storage.sqlite import LeaseFenceError

JobHandler = Callable[["JobContext"], object]


class JobCancelled(RuntimeError):
    """A handler can raise this after observing a cancellation request."""


class JobLeaseLost(RuntimeError):
    """A handler attempted to continue after its per-attempt lease was fenced."""


class JobHandlerFailure(RuntimeError):
    """A domain failure whose stable code must survive worker dispatch."""

    def __init__(self, code: str, message: str) -> None:
        if (
            not isinstance(code, str) or not code or len(code) > 256
            or any(character.isspace() for character in code)
        ):
            raise ValueError("handler failure code must be a bounded identifier")
        self.code = code
        super().__init__(message)


@dataclass(frozen=True, slots=True)
class CommittedJobOutput:
    """A receipt; the worker reloads data from the authoritative committed row."""

    job_id: str
    output_sha256: str
    kind: str = "observation"


@dataclass(frozen=True, slots=True)
class JobContext:
    queue: JobQueue
    job: Job
    worker_id: str
    lease_seconds: float

    def heartbeat(self) -> bool:
        """Extend the lease while a handler is still making progress."""

        return self.queue.heartbeat(
            self.job.job_id,
            owner=self.worker_id,
            lease_token=self.job.lease_token,
            lease_seconds=self.lease_seconds,
        )

    def cancellation_requested(self) -> bool:
        return self.queue.cancel_requested(self.job.job_id)

    def check_cancelled(self) -> None:
        if self.cancellation_requested():
            raise JobCancelled(self.job.job_id)

    def ensure_lease(self) -> None:
        """Renew and validate the attempt before a bounded unit of work or side effect."""

        if not self.heartbeat():
            raise JobLeaseLost(self.job.job_id)
        self.check_cancelled()


@dataclass(frozen=True, slots=True)
class JobOutcome:
    job_id: str
    stage: str
    status: JobStatus
    output: object | None = None
    error_code: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "stage": self.stage,
            "status": self.status.value,
            "output": self.output,
            "error_code": self.error_code,
        }


class JobWorker:
    """Run registered local handlers one job at a time with a bounded loop."""

    def __init__(
        self,
        queue: JobQueue,
        *,
        handlers: Mapping[str, JobHandler] | None = None,
        worker_id: str = "local_worker",
        lease_seconds: float = 300.0,
    ) -> None:
        if (
            not isinstance(worker_id, str)
            or not worker_id
            or worker_id.strip() != worker_id
            or any(character.isspace() for character in worker_id)
            or len(worker_id) > 256
        ):
            raise ValueError("worker_id must be a non-empty identifier")
        if lease_seconds <= 0:
            raise ValueError("lease_seconds must be positive")
        self.queue = queue
        self.handlers = dict(handlers or {})
        self.worker_id = worker_id
        self.lease_seconds = lease_seconds

    def run_once(self, *, job_id: str | None = None) -> JobOutcome | None:
        job = (
            self.queue.claim_next(owner=self.worker_id, lease_seconds=self.lease_seconds)
            if job_id is None else self.queue.claim(
                job_id, owner=self.worker_id, lease_seconds=self.lease_seconds,
            )
        )
        if job is None:
            return None
        context = JobContext(
            queue=self.queue,
            job=job,
            worker_id=self.worker_id,
            lease_seconds=self.lease_seconds,
        )
        if context.cancellation_requested():
            return self._current_outcome(job, fallback_status=JobStatus.CANCELLED)

        handler = self.handlers.get(job.stage)
        if handler is None:
            return self._finish_failed(job, "handler_not_registered")
        try:
            output = handler(context)
        except JobCancelled:
            self.queue.cancel(
                job.job_id,
                owner=self.worker_id,
                lease_token=job.lease_token,
            )
            return self._current_outcome(job, fallback_status=JobStatus.CANCELLED)
        except (JobLeaseLost, LeaseFenceError):
            status = (JobStatus.CANCELLED if context.cancellation_requested()
                      else JobStatus.FAILED)
            return JobOutcome(job.job_id, job.stage, status, error_code="lost_lease")
        except JobHandlerFailure as error:
            if context.cancellation_requested():
                return self._current_outcome(job, fallback_status=JobStatus.CANCELLED)
            return self._finish_failed(job, error.code)
        except Exception:
            if context.cancellation_requested():
                return self._current_outcome(job, fallback_status=JobStatus.CANCELLED)
            return self._finish_failed(job, "handler_failed")
        if isinstance(output, CommittedJobOutput):
            try:
                if output.kind == "video_import" and output.job_id == job.job_id:
                    payload = self.queue.store.get_video_import_result(
                        job.job_id, owner=self.worker_id, lease_token=job.lease_token,
                        output_sha256=output.output_sha256,
                    )
                else:
                    payload = (
                        self.queue.store.get_observation_result(
                            job.job_id, owner=self.worker_id, lease_token=job.lease_token,
                            output_sha256=output.output_sha256,
                        ) if output.job_id == job.job_id and output.kind == "observation" else None
                    )
            except (TypeError, ValueError):
                return JobOutcome(job.job_id, job.stage, JobStatus.FAILED,
                                  error_code="invalid_committed_output")
            if payload is None:
                status = (JobStatus.CANCELLED if context.cancellation_requested()
                          else JobStatus.FAILED)
                return JobOutcome(job.job_id, job.stage, status, error_code="lost_lease")
            return JobOutcome(job.job_id, job.stage, JobStatus.SUCCEEDED, output=payload)
        if context.cancellation_requested():
            return self._current_outcome(job, fallback_status=JobStatus.CANCELLED)
        try:
            self._validate_output(output)
        except ValueError:
            return self._finish_failed(job, "invalid_handler_output")

        completed = self.queue.complete(
            job.job_id,
            owner=self.worker_id,
            lease_token=job.lease_token,
            status=JobStatus.SUCCEEDED,
        )
        if not completed:
            return self._current_outcome(
                job, fallback_status=JobStatus.FAILED, error_code="lost_lease"
            )
        return JobOutcome(job.job_id, job.stage, JobStatus.SUCCEEDED, output=output)

    def run_until_idle(self, *, max_jobs: int = 100) -> tuple[JobOutcome, ...]:
        if isinstance(max_jobs, bool) or not isinstance(max_jobs, int) or max_jobs <= 0:
            raise ValueError("max_jobs must be a positive integer")
        outcomes: list[JobOutcome] = []
        for _ in range(max_jobs):
            outcome = self.run_once()
            if outcome is None:
                break
            outcomes.append(outcome)
        return tuple(outcomes)

    def _finish_failed(self, job: Job, error_code: str) -> JobOutcome:
        completed = self.queue.complete(
            job.job_id,
            owner=self.worker_id,
            lease_token=job.lease_token,
            status=JobStatus.FAILED,
            error_code=error_code,
        )
        if not completed:
            return self._current_outcome(
                job, fallback_status=JobStatus.FAILED, error_code="lost_lease"
            )
        return JobOutcome(job.job_id, job.stage, JobStatus.FAILED, error_code=error_code)

    def _current_outcome(
        self,
        job: Job,
        *,
        fallback_status: JobStatus,
        error_code: str | None = None,
    ) -> JobOutcome:
        current = self.queue.get(job.job_id)
        if current is None:
            return JobOutcome(job.job_id, job.stage, fallback_status, error_code=error_code)
        return JobOutcome(
            current.job_id,
            current.stage,
            current.status,
            error_code=current.error_code or error_code,
        )

    @staticmethod
    def _validate_output(output: object) -> None:
        try:
            json.dumps(output, ensure_ascii=False, allow_nan=False)
        except (TypeError, ValueError) as error:
            raise ValueError("handler output must be JSON-serializable and finite") from error


def fixture_ingest_handler(
    context: JobContext,
    *,
    snapshot_root: Path | None = None,
    artifact_relative_root: str | None = None,
) -> dict[str, Any]:
    """Run the existing CPU fixture ingestion using only job payload inputs."""

    payload = context.job.payload
    video_id = payload.get("video_id")
    if not isinstance(video_id, str) or not video_id:
        raise ValueError("fixture_ingest payload requires video_id")
    payload_root = payload.get("snapshot_root", snapshot_root)
    if not isinstance(payload_root, (str, Path)) or not str(payload_root):
        raise ValueError("fixture_ingest payload requires snapshot_root")
    frame_budget = payload.get("frame_budget", 256)
    if isinstance(frame_budget, bool) or not isinstance(frame_budget, int) or frame_budget <= 0:
        raise ValueError("fixture_ingest frame_budget must be a positive integer")
    payload_namespace = payload.get("artifact_relative_root", artifact_relative_root)
    if payload_namespace is not None and not isinstance(payload_namespace, str):
        raise ValueError("fixture_ingest artifact_relative_root must be a string or null")

    context.ensure_lease()
    if context.job.lease_token is None:
        raise JobLeaseLost(context.job.job_id)
    registry = IndexRegistry(
        context.queue.store,
        SnapshotStore(Path(payload_root)),
        artifact_relative_root=payload_namespace,
    )
    summary = IngestionService(context.queue.store, registry).ingest(
        video_id,
        adapter=PixelFixtureAdapter(frame_budget=frame_budget),
        checkpoint=context.ensure_lease,
        lease_fence=(context.job.job_id, context.worker_id, context.job.lease_token),
    )
    context.ensure_lease()
    return summary.to_dict()


def default_handlers(
    *,
    snapshot_root: Path | None = None,
    artifact_relative_root: str | None = None,
    evidence_root: Path | None = None,
    backend_factories: Mapping[str, Callable] | None = None,
    sam2_backend=None,
    vlm_backend=None,
) -> dict[str, JobHandler]:
    """Return handlers that are safe to run without external models or services."""

    def handle_fixture_ingest(context: JobContext) -> object:
        return fixture_ingest_handler(
            context,
            snapshot_root=snapshot_root,
            artifact_relative_root=artifact_relative_root,
        )

    def handle_observation(context: JobContext) -> object:
        from video_intelligence.pipeline.observation_jobs import observation_job_handler

        return observation_job_handler(
            context, snapshot_root=snapshot_root, evidence_root=evidence_root,
            backend_factories=backend_factories,
            sam2_backend=sam2_backend,
            vlm_backend=vlm_backend,
        )

    def handle_video_import(context: JobContext) -> object:
        from video_intelligence.pipeline.video_import import ImportFailure, LocalVideoImport

        try:
            _, digest = LocalVideoImport(context.queue.store).execute_claimed(context.job)
        except ImportFailure as error:
            raise JobHandlerFailure(error.code, str(error)) from error
        return CommittedJobOutput(context.job.job_id, digest, kind="video_import")

    return {
        "video_import": handle_video_import,
        "fixture_ingest": handle_fixture_ingest,
        "observation_query": handle_observation,
        "fixture_observe": handle_observation,
    }
