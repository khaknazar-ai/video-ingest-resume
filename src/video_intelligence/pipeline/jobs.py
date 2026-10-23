"""Typed job queue facade over the SQLite metadata store."""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import StrEnum

from video_intelligence.storage.sqlite import SQLiteStore


class JobStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass(frozen=True, slots=True)
class Job:
    job_id: str
    stage: str
    status: JobStatus
    config_sha256: str
    input_artifact_sha256: str | None
    payload: dict[str, object]
    owner: str | None
    lease_until: float | None
    lease_token: str | None
    attempt: int
    error_code: str | None

    def to_dict(self) -> dict[str, object]:
        return {
            "job_id": self.job_id,
            "stage": self.stage,
            "status": self.status.value,
            "config_sha256": self.config_sha256,
            "input_artifact_sha256": self.input_artifact_sha256,
            "payload": self.payload,
            "owner": self.owner,
            "lease_until": self.lease_until,
            "lease_token": self.lease_token,
            "attempt": self.attempt,
            "error_code": self.error_code,
        }


def _job(row) -> Job:
    payload = json.loads(row["payload_json"])
    if not isinstance(payload, dict):
        raise ValueError("job payload must be a JSON object")
    return Job(
        job_id=row["job_id"],
        stage=row["stage"],
        status=JobStatus(row["status"]),
        config_sha256=row["config_sha256"],
        input_artifact_sha256=row["input_artifact_sha256"],
        payload=payload,
        owner=row["owner"],
        lease_until=row["lease_until"],
        lease_token=row["lease_token"],
        attempt=row["attempt"],
        error_code=row["error_code"],
    )


class JobQueue:
    def __init__(self, store: SQLiteStore) -> None:
        self.store = store

    def create(
        self,
        *,
        stage: str,
        config_sha256: str,
        input_artifact_sha256: str | None = None,
        payload: dict[str, object] | None = None,
    ) -> Job:
        return _job(
            self.store.create_job(
                stage=stage,
                config_sha256=config_sha256,
                input_artifact_sha256=input_artifact_sha256,
                payload=payload,
            )
        )

    def get(self, job_id: str) -> Job | None:
        row = self.store.get_job(job_id)
        return None if row is None else _job(row)

    def claim(self, job_id: str, *, owner: str, lease_seconds: float = 300.0) -> Job | None:
        row = self.store.claim_job(job_id, owner=owner, lease_seconds=lease_seconds)
        return None if row is None or row["status"] != JobStatus.RUNNING else _job(row)

    def claim_next(self, *, owner: str, lease_seconds: float = 300.0) -> Job | None:
        row = self.store.claim_next_job(owner=owner, lease_seconds=lease_seconds)
        return None if row is None or row["status"] != JobStatus.RUNNING else _job(row)

    def heartbeat(
        self,
        job_id: str,
        *,
        owner: str,
        lease_token: str | None = None,
        lease_seconds: float = 300.0,
    ) -> bool:
        return self.store.heartbeat(
            job_id,
            owner=owner,
            lease_token=lease_token,
            lease_seconds=lease_seconds,
        )

    def cancel(
        self,
        job_id: str,
        *,
        owner: str | None = None,
        lease_token: str | None = None,
    ) -> bool:
        return self.store.cancel_job(
            job_id,
            owner=owner,
            lease_token=lease_token,
        )

    def cancel_requested(self, job_id: str) -> bool:
        job = self.get(job_id)
        return job is None or job.status is JobStatus.CANCELLED

    def complete(
        self,
        job_id: str,
        *,
        owner: str,
        lease_token: str | None = None,
        status: JobStatus,
        error_code: str | None = None,
    ) -> bool:
        return self.store.finish_job(
            job_id,
            owner=owner,
            lease_token=lease_token,
            status=status.value,
            error_code=error_code,
        )

    def recover_stale(self, *, now: float | None = None) -> int:
        return self.store.recover_stale_jobs(now=now)
