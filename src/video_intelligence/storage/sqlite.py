"""Small SQLite metadata store with explicit transactions and foreign keys."""

from __future__ import annotations

import hashlib
import json
import math
import sqlite3
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path, PurePosixPath, PureWindowsPath

from video_intelligence.core.types import VideoMetadata, VideoRecord


def _validate_structured_answer(output, result, request):
    pin = request.get("vlm_backend_sha256")
    if pin is None:
        if "structured_answer" in output:
            raise ValueError("unrequested structured answer cannot upgrade an observation job")
        return
    from video_intelligence.reasoning.contracts import validate_answer

    validate_answer(output.get("structured_answer"), result, expected_backend_sha256=pin)


SCHEMA = """
CREATE TABLE IF NOT EXISTS videos (
    video_id TEXT PRIMARY KEY,
    source_path TEXT NOT NULL,
    source_sha256 TEXT NOT NULL,
    metadata_json TEXT NOT NULL,
    created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS artifacts (
    artifact_id TEXT PRIMARY KEY,
    kind TEXT NOT NULL,
    content_sha256 TEXT NOT NULL,
    relative_path TEXT NOT NULL,
    size_bytes INTEGER NOT NULL CHECK(size_bytes >= 0),
    created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS jobs (
    job_id TEXT PRIMARY KEY,
    dedupe_key TEXT NOT NULL UNIQUE,
    stage TEXT NOT NULL,
    status TEXT NOT NULL CHECK(
        status IN ('pending', 'running', 'succeeded', 'failed', 'cancelled')
    ),
    config_sha256 TEXT NOT NULL,
    input_artifact_sha256 TEXT,
    payload_json TEXT NOT NULL DEFAULT '{}',
    owner TEXT,
    lease_until REAL,
    lease_token TEXT,
    attempt INTEGER NOT NULL DEFAULT 0 CHECK(attempt >= 0),
    error_code TEXT,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS import_checkpoints (
    checkpoint_id INTEGER PRIMARY KEY AUTOINCREMENT,
    cache_key TEXT NOT NULL,
    stage TEXT NOT NULL CHECK(stage IN ('copy', 'probe')),
    job_id TEXT NOT NULL REFERENCES jobs(job_id),
    output_json TEXT NOT NULL,
    output_sha256 TEXT NOT NULL,
    completed_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS import_checkpoint_cache ON import_checkpoints(cache_key, checkpoint_id);
CREATE TABLE IF NOT EXISTS video_import_results (
    job_id TEXT PRIMARY KEY REFERENCES jobs(job_id),
    video_id TEXT NOT NULL REFERENCES videos(video_id),
    lease_owner TEXT NOT NULL,
    lease_token TEXT NOT NULL,
    receipt_json TEXT NOT NULL,
    receipt_sha256 TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS index_snapshots (
    snapshot_id TEXT PRIMARY KEY,
    artifact_id TEXT NOT NULL UNIQUE REFERENCES artifacts(artifact_id),
    matrix_sha256 TEXT NOT NULL,
    metadata_sha256 TEXT NOT NULL,
    rows INTEGER NOT NULL CHECK(rows > 0),
    dimension INTEGER NOT NULL CHECK(dimension > 0),
    created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS active_index_pointers (
    index_name TEXT PRIMARY KEY,
    snapshot_id TEXT NOT NULL REFERENCES index_snapshots(snapshot_id),
    updated_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS queries (
    query_id TEXT PRIMARY KEY,
    video_id TEXT NOT NULL REFERENCES videos(video_id),
    run_id TEXT NOT NULL,
    raw_query TEXT NOT NULL,
    spec_json TEXT NOT NULL,
    processing_status TEXT NOT NULL CHECK(
        processing_status IN ('complete', 'partial', 'failed')
    ),
    support_status TEXT NOT NULL CHECK(
        support_status IN ('supported', 'not_found', 'insufficient_evidence', 'unsupported_query')
    ),
    result_json TEXT NOT NULL,
    error_code TEXT,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS evidence (
    evidence_id TEXT NOT NULL,
    query_id TEXT NOT NULL REFERENCES queries(query_id) ON DELETE CASCADE,
    video_id TEXT NOT NULL REFERENCES videos(video_id),
    snapshot_id TEXT NOT NULL REFERENCES index_snapshots(snapshot_id),
    source_sha256 TEXT NOT NULL,
    frame_index INTEGER NOT NULL CHECK(frame_index >= 0),
    timestamp_seconds REAL NOT NULL CHECK(timestamp_seconds >= 0),
    label TEXT NOT NULL,
    bbox_json TEXT NOT NULL,
    score REAL NOT NULL CHECK(score >= 0 AND score <= 1),
    artifact_id TEXT NOT NULL,
    source_stage TEXT NOT NULL,
    role TEXT NOT NULL,
    provenance_json TEXT NOT NULL,
    created_at REAL NOT NULL,
    PRIMARY KEY (query_id, evidence_id)
);
CREATE TABLE IF NOT EXISTS observation_job_results (
    job_id TEXT PRIMARY KEY REFERENCES jobs(job_id),
    query_id TEXT NOT NULL REFERENCES queries(query_id),
    request_sha256 TEXT NOT NULL,
    lease_owner TEXT NOT NULL,
    lease_token TEXT NOT NULL,
    output_sha256 TEXT NOT NULL,
    output_json TEXT NOT NULL,
    created_at REAL NOT NULL
);
"""
MAX_ID_LENGTH = 256


def _identifier(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value.strip() != value
        or any(character.isspace() for character in value)
        or len(value) > MAX_ID_LENGTH
    ):
        raise ValueError(f"{name} must be a non-empty identifier")
    return value


def _sha256(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or value != value.lower()
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 hex digest")
    return value


def _relative_path(value: object) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError("relative_path must be a non-empty string")
    # Artifact paths are portable strings, so validate both path grammars even
    # when the current host only interprets one of them. In particular,
    # a leading backslash is rooted on Windows and C:outside is drive-relative;
    # neither is confined to the configured artifact root.
    posix_path = PurePosixPath(value)
    windows_path = PureWindowsPath(value)
    if (
        posix_path.is_absolute()
        or windows_path.is_absolute()
        or windows_path.drive
        or windows_path.root
        or ".." in posix_path.parts
        or ".." in windows_path.parts
    ):
        raise ValueError("relative_path must stay inside the artifact root")
    return value


def _positive_finite(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite positive number")
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"{name} must be a finite positive number")
    return result


def _expected_evidence_row(item, query_id: str) -> dict[str, object]:
    return {
        "query_id": query_id,
        "evidence_id": item.evidence_id,
        "video_id": item.frame_ref.video_id,
        "snapshot_id": item.snapshot_id,
        "source_sha256": item.source_sha256,
        "frame_index": item.frame_ref.frame_index,
        "timestamp_seconds": item.frame_ref.timestamp_seconds,
        "label": item.label,
        "bbox_json": list(item.bbox_xyxy),
        "score": item.score,
        "artifact_id": item.artifact_id,
        "source_stage": item.source_stage,
        "role": item.role,
        "provenance_json": item.provenance,
    }


def _actual_evidence_row(row) -> dict[str, object]:
    try:
        bbox = json.loads(row["bbox_json"])
        provenance = json.loads(row["provenance_json"])
    except (TypeError, json.JSONDecodeError) as error:
        raise ValueError("persisted evidence JSON is invalid") from error
    return {
        "query_id": row["query_id"],
        "evidence_id": row["evidence_id"],
        "video_id": row["video_id"],
        "snapshot_id": row["snapshot_id"],
        "source_sha256": row["source_sha256"],
        "frame_index": row["frame_index"],
        "timestamp_seconds": row["timestamp_seconds"],
        "label": row["label"],
        "bbox_json": bbox,
        "score": row["score"],
        "artifact_id": row["artifact_id"],
        "source_stage": row["source_stage"],
        "role": row["role"],
        "provenance_json": provenance,
    }


def _validate_normalized_evidence(rows, items, query_id: str) -> None:
    expected = {
        item.evidence_id: _expected_evidence_row(item, query_id)
        for item in items
    }
    actual: dict[str, dict[str, object]] = {}
    for row in rows:
        evidence_id = row["evidence_id"]
        if evidence_id in actual:
            raise ValueError("persisted evidence rows contain duplicate ids")
        actual[evidence_id] = _actual_evidence_row(row)
    if actual != expected:
        raise ValueError("persisted evidence rows do not match query result")


class LeaseFenceError(RuntimeError):
    """Raised when a worker attempts a side effect without its active lease."""


class SQLiteStore:
    """A single-writer-friendly SQLite store for local metadata and jobs."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).expanduser().resolve()
        self._connection: sqlite3.Connection | None = None

    def __enter__(self) -> SQLiteStore:
        self.open()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def open(self) -> None:
        if self._connection is not None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(str(self.path), timeout=5.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 5000")
        connection.execute("PRAGMA journal_mode = WAL")
        connection.executescript(SCHEMA)
        columns = {row["name"] for row in connection.execute("PRAGMA table_info(jobs)").fetchall()}
        if "payload_json" not in columns:
            connection.execute(
                "ALTER TABLE jobs ADD COLUMN payload_json TEXT NOT NULL DEFAULT '{}'"
            )
            connection.commit()
        if "lease_token" not in columns:
            connection.execute("ALTER TABLE jobs ADD COLUMN lease_token TEXT")
            connection.commit()
        self._connection = connection

    def close(self) -> None:
        if self._connection is not None:
            self._connection.close()
            self._connection = None

    def _require_connection(self) -> sqlite3.Connection:
        self.open()
        assert self._connection is not None
        return self._connection

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        connection = self._require_connection()
        connection.execute("BEGIN IMMEDIATE")
        try:
            yield connection
        except BaseException:
            connection.rollback()
            raise
        else:
            connection.commit()

    def register_video(self, record: VideoRecord) -> str:
        """Register a content-addressed video idempotently."""

        if record.source_sha256 is None:
            raise ValueError("video registration requires source_sha256")
        now = time.time()
        metadata_json = json.dumps(record.metadata.to_dict(), allow_nan=False, sort_keys=True)
        with self._transaction() as connection:
            existing = connection.execute(
                "SELECT source_sha256 FROM videos WHERE video_id = ?",
                (record.video_id,),
            ).fetchone()
            if existing is None:
                connection.execute(
                    """INSERT INTO videos
                    (video_id, source_path, source_sha256, metadata_json, created_at)
                    VALUES (?, ?, ?, ?, ?)""",
                    (
                        record.video_id,
                        str(record.source_path),
                        record.source_sha256,
                        metadata_json,
                        now,
                    ),
                )
            elif existing["source_sha256"] != record.source_sha256:
                raise ValueError("video_id is already registered with another source hash")
        return record.video_id

    def get_video(self, video_id: str) -> VideoRecord | None:
        """Load a registered video record without probing or opening the source."""

        video_id = _identifier(video_id, "video_id")
        row = (
            self._require_connection()
            .execute(
                """SELECT video_id, source_path, source_sha256, metadata_json
            FROM videos WHERE video_id = ?""",
                (video_id,),
            )
            .fetchone()
        )
        if row is None:
            return None
        metadata = VideoMetadata.from_dict(json.loads(row["metadata_json"]))
        return VideoRecord(
            video_id=row["video_id"],
            source_path=Path(row["source_path"]),
            metadata=metadata,
            source_sha256=row["source_sha256"],
        )

    def _import_fence(self, connection, job_id, owner, token) -> None:
        row = connection.execute("SELECT * FROM jobs WHERE job_id=?", (job_id,)).fetchone()
        if (row is None or row["stage"] != "video_import" or row["status"] != "running"
                or row["owner"] != owner or row["lease_token"] != token
                or row["lease_until"] is None or row["lease_until"] <= time.time()):
            raise LeaseFenceError("video import requires its current active lease")

    def retry_video_import(self, job_id: str) -> bool:
        """Explicit resume only; retain attempt count and completed stage records."""
        with self._transaction() as connection:
            return connection.execute(
                """UPDATE jobs SET status='pending', error_code=NULL, updated_at=?
                   WHERE job_id=? AND stage='video_import' AND status IN ('failed','cancelled')""",
                (time.time(), job_id),
            ).rowcount == 1

    def get_import_checkpoint(self, cache_key: str) -> dict | None:
        row = self._require_connection().execute(
            """SELECT * FROM import_checkpoints WHERE cache_key=?
               ORDER BY checkpoint_id DESC LIMIT 1""", (cache_key,),
        ).fetchone()
        if row is None:
            return None
        try:
            raw = row["output_json"].encode("utf-8")
            if hashlib.sha256(raw).hexdigest() != row["output_sha256"]:
                return None
            value = json.loads(raw)
            return value if isinstance(value, dict) else None
        except (TypeError, ValueError):
            return None

    def save_import_checkpoint(self, job_id, owner, token, *, stage, cache_key, output,
                               publish=None) -> None:
        """No checkpoint can admit a partial copy or a fenced-out worker.

        Filesystem create-if-absent runs inside the authoritative writer transaction.
        A crash after linking but before commit leaves an untrusted orphan; resume
        must hash it before admitting a checkpoint. Existing files are never replaced.
        """
        raw = json.dumps(output, sort_keys=True, allow_nan=False, separators=(",", ":"))
        with self._transaction() as connection:
            self._import_fence(connection, job_id, owner, token)
            if publish is not None:
                publish()
            self._import_fence(connection, job_id, owner, token)
            connection.execute(
                """INSERT INTO import_checkpoints
                   (cache_key,stage,job_id,output_json,output_sha256,completed_at)
                   VALUES (?,?,?,?,?,?)""",
                (cache_key, stage, job_id, raw,
                 hashlib.sha256(raw.encode()).hexdigest(), time.time()),
            )
            self._import_fence(connection, job_id, owner, token)

    def commit_video_import(self, job_id, owner, token, *, record, receipt, checkpoint) -> str:
        """Register location, receipt and job success atomically under the lease."""
        raw = json.dumps(receipt, sort_keys=True, allow_nan=False, separators=(",", ":"))
        digest = hashlib.sha256(raw.encode()).hexdigest()
        with self._transaction() as connection:
            self._import_fence(connection, job_id, owner, token)
            checkpoint()
            existing = connection.execute(
                "SELECT * FROM videos WHERE video_id=?", (record.video_id,),
            ).fetchone()
            metadata = json.dumps(record.metadata.to_dict(), sort_keys=True, allow_nan=False)
            if existing is None:
                connection.execute(
                    "INSERT INTO videos VALUES (?,?,?,?,?)",
                    (record.video_id, str(record.source_path), record.source_sha256,
                     metadata, time.time()),
                )
            elif (existing["source_sha256"] != record.source_sha256
                  or existing["source_path"] != str(record.source_path)
                  or existing["metadata_json"] != metadata):
                raise ValueError("video_id binds another source, location or probe metadata")
            connection.execute(
                "INSERT INTO video_import_results VALUES (?,?,?,?,?,?)",
                (job_id, record.video_id, owner, token, raw, digest),
            )
            checkpoint()
            self._import_fence(connection, job_id, owner, token)
            changed = connection.execute(
                """UPDATE jobs SET status='succeeded',owner=NULL,lease_until=NULL,lease_token=NULL,
                   error_code=NULL,updated_at=? WHERE job_id=? AND status='running'
                   AND owner=? AND lease_token=? AND lease_until>?""",
                (time.time(), job_id, owner, token, time.time()),
            ).rowcount
            if changed != 1:
                raise LeaseFenceError("video import lease expired before activation")
        return digest

    def get_video_import_result(self, job_id, *, owner=None, lease_token=None,
                                output_sha256=None) -> dict | None:
        row = self._require_connection().execute(
            "SELECT * FROM video_import_results WHERE job_id=?", (job_id,),
        ).fetchone()
        if row is None:
            return None
        raw = row["receipt_json"].encode()
        if (hashlib.sha256(raw).hexdigest() != row["receipt_sha256"]
                or (output_sha256 is not None and output_sha256 != row["receipt_sha256"])
                or (owner is not None and (row["lease_owner"], row["lease_token"])
                    != (owner, lease_token))):
            raise ValueError("video import receipt identity mismatch")
        return json.loads(raw)

    def register_artifact(
        self,
        *,
        artifact_id: str,
        kind: str,
        content_sha256: str,
        relative_path: str,
        size_bytes: int,
    ) -> str:
        """Register an immutable artifact pointer without claiming blob publication."""

        artifact_id = _identifier(artifact_id, "artifact_id")
        kind = _identifier(kind, "kind")
        content_sha256 = _sha256(content_sha256, "content_sha256")
        relative_path = _relative_path(relative_path)
        if isinstance(size_bytes, bool) or not isinstance(size_bytes, int) or size_bytes < 0:
            raise ValueError("size_bytes must be a non-negative integer")
        now = time.time()
        with self._transaction() as connection:
            existing = connection.execute(
                """SELECT kind, content_sha256, relative_path, size_bytes
                FROM artifacts WHERE artifact_id = ?""",
                (artifact_id,),
            ).fetchone()
            values = (kind, content_sha256, relative_path, size_bytes)
            if existing is None:
                connection.execute(
                    """INSERT INTO artifacts
                    (artifact_id, kind, content_sha256, relative_path, size_bytes, created_at)
                    VALUES (?, ?, ?, ?, ?, ?)""",
                    (artifact_id, *values, now),
                )
            elif tuple(existing) != values:
                raise ValueError("artifact_id is already registered with different metadata")
        return artifact_id

    def activate_index_snapshot(
        self,
        *,
        index_name: str,
        snapshot_id: str,
        artifact_id: str,
        relative_path: str,
        size_bytes: int,
        matrix_sha256: str,
        metadata_sha256: str,
        rows: int,
        dimension: int,
        lease_job_id: str | None = None,
        lease_owner: str | None = None,
        lease_token: str | None = None,
    ) -> sqlite3.Row:
        """Register a verified snapshot and switch its active pointer atomically.

        The caller must validate the immutable files before entering this method. If
        any artifact, snapshot, or pointer invariant fails, the transaction rolls
        back and no new active pointer is visible.
        """

        index_name = _identifier(index_name, "index_name")
        snapshot_id = _sha256(snapshot_id, "snapshot_id")
        artifact_id = _identifier(artifact_id, "artifact_id")
        relative_path = _relative_path(relative_path)
        matrix_sha256 = _sha256(matrix_sha256, "matrix_sha256")
        metadata_sha256 = _sha256(metadata_sha256, "metadata_sha256")
        if isinstance(size_bytes, bool) or not isinstance(size_bytes, int) or size_bytes < 0:
            raise ValueError("size_bytes must be a non-negative integer")
        if isinstance(rows, bool) or not isinstance(rows, int) or rows <= 0:
            raise ValueError("rows must be a positive integer")
        if isinstance(dimension, bool) or not isinstance(dimension, int) or dimension <= 0:
            raise ValueError("dimension must be a positive integer")
        lease_values = (lease_job_id, lease_owner, lease_token)
        if any(value is not None for value in lease_values) and not all(
            value is not None for value in lease_values
        ):
            raise ValueError("lease_job_id, lease_owner and lease_token must be provided together")
        if lease_job_id is not None:
            lease_job_id = _identifier(lease_job_id, "lease_job_id")
            lease_owner = _identifier(lease_owner, "lease_owner")
            lease_token = _identifier(lease_token, "lease_token")
        with self._transaction() as connection:
            # Read the clock after BEGIN IMMEDIATE: waiting for the write lock may
            # consume the entire lease and must not be hidden by an early timestamp.
            now = time.time()
            if lease_job_id is not None:
                lease = connection.execute(
                    """SELECT status, owner, lease_until, lease_token FROM jobs
                    WHERE job_id = ?""",
                    (lease_job_id,),
                ).fetchone()
                if (
                    lease is None
                    or lease["status"] != "running"
                    or lease["owner"] != lease_owner
                    or lease["lease_token"] != lease_token
                    or lease["lease_until"] is None
                    or lease["lease_until"] <= now
                ):
                    raise LeaseFenceError("job lease is not valid for artifact activation")
            artifact_values = ("index_snapshot", snapshot_id, relative_path, size_bytes)
            artifact = connection.execute(
                """SELECT kind, content_sha256, relative_path, size_bytes
                FROM artifacts WHERE artifact_id = ?""",
                (artifact_id,),
            ).fetchone()
            if artifact is None:
                connection.execute(
                    """INSERT INTO artifacts
                    (artifact_id, kind, content_sha256, relative_path, size_bytes, created_at)
                    VALUES (?, ?, ?, ?, ?, ?)""",
                    (artifact_id, *artifact_values, now),
                )
            elif tuple(artifact) != artifact_values:
                raise ValueError("artifact_id is already registered with different metadata")

            snapshot_values = (artifact_id, matrix_sha256, metadata_sha256, rows, dimension)
            snapshot = connection.execute(
                """SELECT artifact_id, matrix_sha256, metadata_sha256, rows, dimension
                FROM index_snapshots WHERE snapshot_id = ?""",
                (snapshot_id,),
            ).fetchone()
            if snapshot is None:
                connection.execute(
                    """INSERT INTO index_snapshots
                    (snapshot_id, artifact_id, matrix_sha256, metadata_sha256, rows, dimension,
                     created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (snapshot_id, *snapshot_values, now),
                )
            elif tuple(snapshot) != snapshot_values:
                raise ValueError("snapshot_id is already registered with different metadata")

            pointer = connection.execute(
                "SELECT index_name, snapshot_id FROM active_index_pointers WHERE index_name = ?",
                (index_name,),
            ).fetchone()
            if pointer is None:
                connection.execute(
                    """INSERT INTO active_index_pointers(index_name, snapshot_id, updated_at)
                    VALUES (?, ?, ?)""",
                    (index_name, snapshot_id, now),
                )
            elif pointer["snapshot_id"] != snapshot_id:
                connection.execute(
                    """UPDATE active_index_pointers SET snapshot_id = ?, updated_at = ?
                    WHERE index_name = ?""",
                    (snapshot_id, now, index_name),
                )
            return connection.execute(
                """SELECT p.index_name, p.snapshot_id, p.updated_at,
                s.artifact_id, s.matrix_sha256, s.metadata_sha256, s.rows, s.dimension,
                a.relative_path, a.size_bytes
                FROM active_index_pointers AS p
                JOIN index_snapshots AS s ON s.snapshot_id = p.snapshot_id
                JOIN artifacts AS a ON a.artifact_id = s.artifact_id
                WHERE p.index_name = ?""",
                (index_name,),
            ).fetchone()

    def get_active_index(self, index_name: str) -> sqlite3.Row | None:
        index_name = _identifier(index_name, "index_name")
        return (
            self._require_connection()
            .execute(
                """SELECT p.index_name, p.snapshot_id, p.updated_at,
            s.artifact_id, s.matrix_sha256, s.metadata_sha256, s.rows, s.dimension,
            a.relative_path, a.size_bytes
            FROM active_index_pointers AS p
            JOIN index_snapshots AS s ON s.snapshot_id = p.snapshot_id
            JOIN artifacts AS a ON a.artifact_id = s.artifact_id
            WHERE p.index_name = ?""",
                (index_name,),
            )
            .fetchone()
        )

    def get_index_snapshot(self, snapshot_id: str) -> sqlite3.Row | None:
        """Read a registered immutable snapshot, including its artifact metadata."""
        snapshot_id = _sha256(snapshot_id, "snapshot_id")
        return self._require_connection().execute(
            """SELECT s.*,a.kind,a.content_sha256,a.relative_path,a.size_bytes
               FROM index_snapshots s JOIN artifacts a ON a.artifact_id=s.artifact_id
               WHERE s.snapshot_id=?""", (snapshot_id,),
        ).fetchone()

    def save_query_result(self, result) -> str:
        """Persist a typed QueryResult and its evidence rows idempotently."""

        with self._transaction() as connection:
            return self._save_query_result_in_transaction(connection, result, time.time())

    def _save_query_result_in_transaction(self, connection, result, now: float) -> str:
        """Share query validation/writes without opening a nested transaction."""

        from video_intelligence.retrieval.schemas import QueryResult

        if not isinstance(result, QueryResult):
            raise TypeError("result must be a QueryResult")
        payload = result.to_dict()
        video_id = _identifier(result.video_id, "video_id")
        run_id = _identifier(result.run_id, "run_id")
        video = self.get_video(video_id)
        if video is None:
            raise ValueError("query persistence requires a registered video")
        result.validate_video(video)
        if result.evidence is not None and result.evidence.source_sha256 != video.source_sha256:
            raise ValueError("query evidence source hash differs from registered video")
        result_json = json.dumps(payload, ensure_ascii=False, allow_nan=False, sort_keys=True)
        spec_json = json.dumps(
            result.query.to_dict(), ensure_ascii=False, allow_nan=False, sort_keys=True
        )
        existing = connection.execute(
            "SELECT result_json FROM queries WHERE query_id = ?",
            (result.query.query_id,),
        ).fetchone()
        if existing is not None:
            if existing["result_json"] != result_json:
                raise ValueError("query_id is already persisted with different result")
            rows = connection.execute(
                "SELECT * FROM evidence WHERE query_id = ?",
                (result.query.query_id,),
            ).fetchall()
            _validate_normalized_evidence(
                rows,
                result.evidence.items if result.evidence is not None else (),
                result.query.query_id,
            )
            return result.query.query_id
        connection.execute(
            """INSERT INTO queries
                (query_id, video_id, run_id, raw_query, spec_json, processing_status,
                 support_status, result_json, error_code, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                    result.query.query_id,
                    video_id,
                    run_id,
                    result.query.raw_text,
                    spec_json,
                    result.processing_status.value,
                    result.support_status.value,
                    result_json,
                    result.error_code,
                    now,
                    now,
            ),
        )
        if result.evidence is not None:
            for item in result.evidence.items:
                connection.execute(
                    """INSERT INTO evidence
                        (evidence_id, query_id, video_id, snapshot_id, source_sha256,
                         frame_index, timestamp_seconds, label, bbox_json, score, artifact_id,
                         source_stage, role, provenance_json, created_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                            item.evidence_id,
                            result.query.query_id,
                            item.frame_ref.video_id,
                            item.snapshot_id,
                            item.source_sha256,
                            item.frame_ref.frame_index,
                            item.frame_ref.timestamp_seconds,
                            item.label,
                            json.dumps(list(item.bbox_xyxy), allow_nan=False),
                            item.score,
                            item.artifact_id,
                            item.source_stage,
                            item.role,
                            json.dumps(item.provenance, ensure_ascii=False, allow_nan=False),
                            now,
                    ),
                )
        return result.query.query_id

    def get_artifact(self, artifact_id: str) -> sqlite3.Row | None:
        artifact_id = _identifier(artifact_id, "artifact_id")
        return self._require_connection().execute(
            "SELECT * FROM artifacts WHERE artifact_id = ?", (artifact_id,),
        ).fetchone()

    def commit_observation_result(
        self, job_id: str, *, owner: str, lease_token: str, request_sha256: str,
        result, output: dict, artifacts: tuple[dict, ...] = (),
    ) -> str:
        """Publish output, artifacts, query and success under one authoritative fence."""
        from video_intelligence.retrieval.schemas import ProcessingStatus, QueryResult

        job_id = _identifier(job_id, "job_id")
        owner = _identifier(owner, "owner")
        lease_token = _identifier(lease_token, "lease_token")
        request_sha256 = _sha256(request_sha256, "request_sha256")
        if (not isinstance(result, QueryResult)
                or result.processing_status is not ProcessingStatus.COMPLETE):
            raise ValueError("only complete domain results may complete an observation job")
        result = QueryResult.from_dict(result.to_dict(), video=self.get_video(result.video_id))
        output_json = json.dumps(output, sort_keys=True, separators=(",", ":"),
                                 ensure_ascii=False, allow_nan=False)
        output = json.loads(output_json)
        if (
            output.get("job_id") != job_id or output.get("request_sha256") != request_sha256
            or output.get("query_result") != result.to_dict()
            or not isinstance(output.get("pipeline"), dict)
            or output["pipeline"].get("query_result") != result.to_dict()
            or output.get("artifacts") != list(artifacts)
        ):
            raise ValueError("observation output does not match its authoritative result")
        output_sha256 = hashlib.sha256(output_json.encode("utf-8")).hexdigest()
        with self._transaction() as connection:
            row = connection.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
            now = time.time()

            def ensure_fence() -> None:
                if (
                    row is None or row["status"] != "running" or row["owner"] != owner
                    or row["lease_token"] != lease_token or row["lease_until"] is None
                    or row["lease_until"] <= time.time()
                ):
                    raise LeaseFenceError(
                        "observation result requires the current active job lease",
                    )

            ensure_fence()
            if row["stage"] not in {"observation_query", "fixture_observe"}:
                raise ValueError("job stage is not an observation stage")
            payload = json.loads(row["payload_json"])
            payload_json = json.dumps(payload, sort_keys=True, separators=(",", ":"),
                                      ensure_ascii=False, allow_nan=False)
            if (
                row["config_sha256"] != request_sha256
                or hashlib.sha256(payload_json.encode("utf-8")).hexdigest() != request_sha256
                or payload.get("video_id") != result.video_id
                or payload.get("run_id") != result.run_id
                or payload.get("query_text") != result.query.raw_text
                or payload.get("snapshot_id") != result.query_config.get("snapshot_id")
            ):
                raise ValueError("observation result does not match the queued request")
            _validate_structured_answer(output, result, payload)
            for artifact in artifacts:
                if not isinstance(artifact, dict) or set(artifact) != {
                    "artifact_id", "kind", "content_sha256", "relative_path", "size_bytes",
                }:
                    raise ValueError("artifact registration requires exact immutable metadata")
                artifact_id = _identifier(artifact["artifact_id"], "artifact_id")
                kind = _identifier(artifact["kind"], "kind")
                digest = _sha256(artifact["content_sha256"], "content_sha256")
                relative = _relative_path(artifact["relative_path"])
                size = artifact["size_bytes"]
                if isinstance(size, bool) or not isinstance(size, int) or size < 0:
                    raise ValueError("artifact size_bytes must be a non-negative integer")
                values = (kind, digest, relative, size)
                existing = connection.execute(
                    "SELECT kind,content_sha256,relative_path,size_bytes FROM artifacts "
                    "WHERE artifact_id = ?", (artifact_id,),
                ).fetchone()
                if existing is None:
                    connection.execute(
                        "INSERT INTO artifacts VALUES (?, ?, ?, ?, ?, ?)",
                        (artifact_id, *values, now),
                    )
                elif tuple(existing) != values:
                    raise ValueError("artifact_id is already registered with different metadata")
            self._save_query_result_in_transaction(connection, result, now)
            connection.execute(
                "INSERT INTO observation_job_results VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (job_id, result.query.query_id, request_sha256, owner, lease_token,
                 output_sha256, output_json, now),
            )
            ensure_fence()
            finished = connection.execute(
                """UPDATE jobs SET status='succeeded', owner=NULL, lease_until=NULL,
                   lease_token=NULL,error_code=NULL,updated_at=?
                   WHERE job_id=? AND status='running' AND owner=? AND lease_token=?
                   AND lease_until > ?""",
                (time.time(), job_id, owner, lease_token, time.time()),
            ).rowcount
            if finished != 1:
                raise LeaseFenceError("observation lease expired before success admission")
            ensure_fence()
        return output_sha256

    def get_observation_result(
        self, job_id: str, *, owner: str | None = None, lease_token: str | None = None,
        output_sha256: str | None = None,
    ) -> dict | None:
        """Read only authoritative succeeded output and recheck its persisted bindings."""
        job_id = _identifier(job_id, "job_id")
        if (owner is None) != (lease_token is None):
            raise ValueError("owner and lease_token must be supplied together")
        row = self._require_connection().execute(
            """SELECT o.*,j.status,j.config_sha256,j.payload_json FROM observation_job_results o
               JOIN jobs j ON j.job_id=o.job_id WHERE o.job_id=?""", (job_id,),
        ).fetchone()
        if row is None or row["status"] != "succeeded":
            return None
        if owner is not None and (row["lease_owner"], row["lease_token"]) != (owner, lease_token):
            return None
        if output_sha256 is not None and row["output_sha256"] != output_sha256:
            raise ValueError("committed output receipt hash mismatch")
        if hashlib.sha256(row["output_json"].encode("utf-8")).hexdigest() != row["output_sha256"]:
            raise ValueError("persisted observation output hash mismatch")
        request_json = json.dumps(json.loads(row["payload_json"]), sort_keys=True,
                                  separators=(",", ":"), ensure_ascii=False, allow_nan=False)
        if (
            row["request_sha256"] != row["config_sha256"]
            or hashlib.sha256(request_json.encode("utf-8")).hexdigest() != row["request_sha256"]
        ):
            raise ValueError("persisted observation request binding mismatch")
        output = json.loads(row["output_json"])
        query = self.get_query_payload(row["query_id"])
        if (
            not isinstance(output, dict) or output.get("job_id") != job_id
            or output.get("request_sha256") != row["request_sha256"]
            or output.get("query_result") != query or not isinstance(output.get("pipeline"), dict)
            or output["pipeline"].get("query_result") != query
        ):
            raise ValueError("persisted observation output/query binding mismatch")
        from video_intelligence.retrieval.schemas import QueryResult

        _validate_structured_answer(
            output, QueryResult.from_dict(query, video=self.get_video(query["video_id"])),
            json.loads(row["payload_json"]),
        )
        return output

    def get_query_payload(self, query_id: str) -> dict | None:
        """Return the exact persisted JSON result for process-independent reload."""

        from video_intelligence.retrieval.schemas import QueryResult

        query_id = _identifier(query_id, "query_id")
        row = (
            self._require_connection()
            .execute("SELECT * FROM queries WHERE query_id = ?", (query_id,))
            .fetchone()
        )
        if row is None:
            return None
        payload = json.loads(row["result_json"])
        if not isinstance(payload, dict):
            raise ValueError("persisted query result must be a JSON object")
        video = self.get_video(row["video_id"])
        if video is None:
            raise ValueError("persisted query video is unavailable")
        result = QueryResult.from_dict(payload, video=video)
        if (
            result.query.query_id != query_id
            or result.run_id != row["run_id"]
            or result.query.raw_text != row["raw_query"]
            or result.query.to_dict() != json.loads(row["spec_json"])
            or result.processing_status.value != row["processing_status"]
            or result.support_status.value != row["support_status"]
            or result.error_code != row["error_code"]
        ):
            raise ValueError("persisted query columns do not match payload")
        _validate_normalized_evidence(
            self.get_query_evidence(query_id),
            result.evidence.items if result.evidence is not None else (),
            query_id,
        )
        json.dumps(payload, allow_nan=False)
        return payload

    def get_query_evidence(self, query_id: str) -> list[sqlite3.Row]:
        """Load normalized evidence rows associated with one persisted query."""

        query_id = _identifier(query_id, "query_id")
        return list(
            self._require_connection()
            .execute(
                """SELECT * FROM evidence WHERE query_id = ?
                ORDER BY timestamp_seconds, evidence_id""",
                (query_id,),
            )
            .fetchall()
        )

    def create_job(
        self,
        *,
        stage: str,
        config_sha256: str,
        input_artifact_sha256: str | None = None,
        payload: dict[str, object] | None = None,
    ) -> sqlite3.Row:
        """Create or return a deterministic deduplicated pending job."""

        stage = _identifier(stage, "stage")
        config_sha256 = _sha256(config_sha256, "config_sha256")
        if input_artifact_sha256 is not None:
            input_artifact_sha256 = _sha256(input_artifact_sha256, "input_artifact_sha256")
        if payload is None:
            payload = {}
        if not isinstance(payload, dict):
            raise ValueError("payload must be a JSON object")
        try:
            payload_json = json.dumps(
                payload,
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        except (TypeError, ValueError) as error:
            raise ValueError("payload must contain only JSON-compatible finite values") from error
        dedupe_parts = (stage, config_sha256, input_artifact_sha256 or "")
        dedupe_material = (
            "|".join(dedupe_parts) if not payload else "|".join((*dedupe_parts, payload_json))
        )
        dedupe_key = hashlib.sha256(dedupe_material.encode("utf-8")).hexdigest()
        job_id = f"job_{dedupe_key[:24]}"
        now = time.time()
        with self._transaction() as connection:
            connection.execute(
                """INSERT OR IGNORE INTO jobs
                (job_id, dedupe_key, stage, status, config_sha256, input_artifact_sha256,
                 payload_json, created_at, updated_at)
                VALUES (?, ?, ?, 'pending', ?, ?, ?, ?, ?)""",
                (
                    job_id,
                    dedupe_key,
                    stage,
                    config_sha256,
                    input_artifact_sha256,
                    payload_json,
                    now,
                    now,
                ),
            )
            row = connection.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
        assert row is not None
        return row

    def get_job(self, job_id: str) -> sqlite3.Row | None:
        job_id = _identifier(job_id, "job_id")
        return (
            self._require_connection()
            .execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,))
            .fetchone()
        )

    def claim_job(
        self, job_id: str, *, owner: str, lease_seconds: float = 300.0
    ) -> sqlite3.Row | None:
        job_id = _identifier(job_id, "job_id")
        owner = _identifier(owner, "owner")
        lease_seconds = _positive_finite(lease_seconds, "lease_seconds")
        lease_token = uuid.uuid4().hex
        with self._transaction() as connection:
            now = time.time()
            updated = connection.execute(
                """UPDATE jobs
                SET status = 'running', owner = ?, lease_until = ?, lease_token = ?,
                    attempt = attempt + 1, updated_at = ?
                WHERE job_id = ? AND (
                    status = 'pending' OR
                    (status = 'running' AND lease_until IS NOT NULL AND lease_until < ?)
                )""",
                (owner, now + lease_seconds, lease_token, now, job_id, now),
            ).rowcount
            if updated != 1:
                return None
            return connection.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)).fetchone()

    def claim_next_job(self, *, owner: str, lease_seconds: float = 300.0) -> sqlite3.Row | None:
        """Claim the oldest pending or expired running job atomically."""

        owner = _identifier(owner, "owner")
        lease_seconds = _positive_finite(lease_seconds, "lease_seconds")
        lease_token = uuid.uuid4().hex
        with self._transaction() as connection:
            now = time.time()
            candidate = connection.execute(
                """SELECT job_id FROM jobs
                WHERE status = 'pending' OR
                    (status = 'running' AND lease_until IS NOT NULL AND lease_until < ?)
                ORDER BY created_at, rowid, job_id LIMIT 1""",
                (now,),
            ).fetchone()
            if candidate is None:
                return None
            updated = connection.execute(
                """UPDATE jobs
                SET status = 'running', owner = ?, lease_until = ?, lease_token = ?,
                    attempt = attempt + 1, updated_at = ?
                WHERE job_id = ? AND (
                    status = 'pending' OR
                    (status = 'running' AND lease_until IS NOT NULL AND lease_until < ?)
                )""",
                (owner, now + lease_seconds, lease_token, now, candidate["job_id"], now),
            ).rowcount
            if updated != 1:
                return None
            return connection.execute(
                "SELECT * FROM jobs WHERE job_id = ?", (candidate["job_id"],)
            ).fetchone()

    def heartbeat(
        self,
        job_id: str,
        *,
        owner: str,
        lease_token: str | None = None,
        lease_seconds: float = 300.0,
    ) -> bool:
        job_id = _identifier(job_id, "job_id")
        owner = _identifier(owner, "owner")
        if lease_token is not None:
            lease_token = _identifier(lease_token, "lease_token")
        lease_seconds = _positive_finite(lease_seconds, "lease_seconds")
        with self._transaction() as connection:
            now = time.time()
            updated = connection.execute(
                """UPDATE jobs SET lease_until = ?, updated_at = ?
                WHERE job_id = ? AND status = 'running' AND owner = ? AND lease_token = ?
                  AND lease_until IS NOT NULL AND lease_until > ?""",
                (now + lease_seconds, now, job_id, owner, lease_token, now),
            ).rowcount
        return updated == 1

    def cancel_job(
        self,
        job_id: str,
        *,
        owner: str | None = None,
        lease_token: str | None = None,
    ) -> bool:
        """Cancel a pending job or an owned running job without marking success."""

        job_id = _identifier(job_id, "job_id")
        if owner is not None:
            owner = _identifier(owner, "owner")
        if lease_token is not None:
            lease_token = _identifier(lease_token, "lease_token")
        now = time.time()
        with self._transaction() as connection:
            if owner is None:
                updated = connection.execute(
                    """UPDATE jobs SET status = 'cancelled', owner = NULL, lease_until = NULL,
                    lease_token = NULL,
                    updated_at = ?, error_code = 'cancelled'
                    WHERE job_id = ? AND status IN ('pending', 'running')""",
                    (now, job_id),
                ).rowcount
            else:
                updated = connection.execute(
                    """UPDATE jobs SET status = 'cancelled', owner = NULL, lease_until = NULL,
                    lease_token = NULL,
                    updated_at = ?, error_code = 'cancelled'
                    WHERE job_id = ? AND status IN ('pending', 'running')
                    AND (status = 'pending' OR (owner = ? AND lease_token = ?))""",
                    (now, job_id, owner, lease_token),
                ).rowcount
        return updated == 1

    def finish_job(
        self,
        job_id: str,
        *,
        owner: str,
        lease_token: str | None = None,
        status: str,
        error_code: str | None = None,
    ) -> bool:
        job_id = _identifier(job_id, "job_id")
        owner = _identifier(owner, "owner")
        if lease_token is not None:
            lease_token = _identifier(lease_token, "lease_token")
        if status not in {"succeeded", "failed", "cancelled"}:
            raise ValueError("status must be succeeded, failed or cancelled")
        if error_code is not None:
            error_code = _identifier(error_code, "error_code")
        with self._transaction() as connection:
            now = time.time()
            updated = connection.execute(
                """UPDATE jobs SET status = ?, owner = NULL, lease_until = NULL,
                lease_token = NULL, updated_at = ?, error_code = ?
                WHERE job_id = ? AND status = 'running' AND owner = ? AND lease_token = ?
                  AND lease_until IS NOT NULL AND lease_until > ?""",
                (status, now, error_code, job_id, owner, lease_token, now),
            ).rowcount
        return updated == 1

    def recover_stale_jobs(self, *, now: float | None = None) -> int:
        current_time = time.time() if now is None else now
        with self._transaction() as connection:
            return connection.execute(
                """UPDATE jobs SET status = 'pending', owner = NULL, lease_until = NULL,
                lease_token = NULL,
                updated_at = ? WHERE status = 'running' AND lease_until < ?""",
                (current_time, current_time),
            ).rowcount
