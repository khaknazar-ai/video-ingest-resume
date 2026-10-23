"""Small local CLI with an explicitly named synthetic fixture path."""

from __future__ import annotations

import argparse
import json
import sqlite3
from dataclasses import asdict, is_dataclass
from pathlib import Path

from video_intelligence.core.paths import resolve_project_paths
from video_intelligence.index.snapshot import SnapshotStore
from video_intelligence.pipeline.index_registry import IndexRegistry
from video_intelligence.pipeline.ingestion import IngestionService, PixelFixtureAdapter
from video_intelligence.pipeline.jobs import JobQueue, JobStatus
from video_intelligence.pipeline.worker import JobWorker, default_handlers
from video_intelligence.retrieval.query import QueryService
from video_intelligence.storage.sqlite import SQLiteStore


def _common_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--db", type=Path, required=True, help="SQLite metadata path")
    parser.add_argument(
        "--snapshot-root", type=Path, required=True, help="immutable index snapshot root"
    )
    parser.add_argument(
        "--artifact-relative-root", help="must match the snapshot-root directory name"
    )


def _observation_paths(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--snapshot-root", type=Path, required=True)
    parser.add_argument(
        "--evidence-root", type=Path,
        help="artifact root; defaults to VI_ARTIFACTS_ROOT/ProjectPaths",
    )
    parser.add_argument("--enqueue-only", action="store_true")


def _semantic_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--semantic-config", type=Path,
                        help="opt-in local verifier config; judgments cannot promote an answer")
    parser.add_argument("--semantic-work-root", type=Path,
                        help="existing absolute directory for private create-only worker inputs")
    parser.add_argument("--semantic-timeout-seconds", type=float, default=60,
                        help="shared assessment deadline, at most 240 seconds; default 60")
    parser.add_argument("--semantic-cancel-file", type=Path,
                        help="stop assessment when this local cancellation marker exists")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="video-intelligence")
    commands = parser.add_subparsers(dest="command", required=True)

    inspect = commands.add_parser("inspect-frames", help="inspect source PTS and RGB hashes")
    inspect.add_argument("--video", type=Path, required=True)
    inspect.add_argument("--video-id", default="source")
    inspect.add_argument("--timestamp", type=float, action="append", required=True)
    inspect.add_argument(
        "--selection-policy", choices=("at_or_after", "nearest"), default="nearest"
    )
    inspect.add_argument("--max-decode-frames", type=int, default=100_000)
    inspect.add_argument("--timeout-seconds", type=float, default=30.)

    parse = commands.add_parser("parse-query", help="validate a query against explicit entity ids")
    parse.add_argument("--query", required=True)
    parse.add_argument("--registry-file", type=Path, required=True,
                       help="bounded prompt registry JSON, also used in observation requests")

    register = commands.add_parser("register", help="register a source video")
    register.add_argument("--db", type=Path, required=True)
    register.add_argument("--video", type=Path, required=True)
    register.add_argument("--video-id")

    local_import = commands.add_parser(
        "import-video", help="checkpoint local reference/managed import"
    )
    local_import.add_argument("--db", type=Path, required=True)
    local_import.add_argument("--video", type=Path, required=True)
    local_import.add_argument("--mode", choices=("reference", "managed"), default="reference")
    local_import.add_argument("--managed-root", type=Path)
    local_import.add_argument("--video-id")
    local_import.add_argument("--max-bytes", type=int, default=512*1024**2)
    local_import.add_argument("--timeout-seconds", type=float, default=60.)
    local_import.add_argument("--min-free-bytes", type=int, default=64*1024**2)
    local_import.add_argument("--max-pixels", type=int, default=2_073_600)
    local_import.add_argument("--max-duration-seconds", type=float, default=3600.)
    local_import.add_argument("--pipeline-version", default="local-import.v1")
    local_import.add_argument("--probe-profile", default="ffprobe-strict.v1")
    local_import.add_argument("--cancel-file", type=Path)
    for name in ("import-status", "import-resume"):
        command = commands.add_parser(name)
        command.add_argument("--db", type=Path, required=True)
        command.add_argument("--job-id", required=True)
        if name == "import-resume":
            command.add_argument("--cancel-file", type=Path)

    fixture_ingest = commands.add_parser(
        "fixture-ingest",
        help="synthetic-only: decode and index deterministic fixture pixels",
    )
    _common_parser(fixture_ingest)
    fixture_ingest.add_argument("--video-id", required=True)
    fixture_ingest.add_argument("--frame-budget", type=int, default=256)

    query = commands.add_parser("query", help="run a deterministic query")
    _common_parser(query)
    query.add_argument("--video-id", required=True)
    query_values = query.add_mutually_exclusive_group(required=True)
    query_values.add_argument("--query")
    query_values.add_argument("--query-id")
    query.add_argument("--run-id", default="run_cli")
    query.add_argument("--evidence-root", type=Path)
    _semantic_options(query)

    observe = commands.add_parser(
        "observe", help="run a typed observation request with an explicitly registered backend",
    )
    _observation_paths(observe)
    observe.add_argument("--request-file", type=Path, required=True)
    observe.add_argument(
        "--backend-config", type=Path, help="explicit local grounding backend config",
    )
    observe.add_argument("--sam2-config", type=Path)
    observe.add_argument("--vlm-config", type=Path)
    _semantic_options(observe)
    retrieve = commands.add_parser(
        "retrieve-observe", help="local SigLIP2 preparation followed by the observation worker",
    )
    _observation_paths(retrieve)
    retrieve.add_argument("--request-file", type=Path, required=True)
    retrieve.add_argument("--retrieval-config", type=Path, required=True)
    retrieve.add_argument("--backend-config", type=Path, required=True)
    retrieve.add_argument("--sam2-config", type=Path)
    retrieve.add_argument("--vlm-config", type=Path)
    _semantic_options(retrieve)
    retrieve.add_argument("--snapshot-id", help="reuse a matching registered learned snapshot")
    fixture_observe = commands.add_parser(
        "fixture-observe",
        help="synthetic/contract only: RGB retrieval, grounding, tracks and PNGs",
    )
    _observation_paths(fixture_observe)
    fixture_observe.add_argument("--video-id", required=True)
    fixture_observe.add_argument("--query", required=True)
    fixture_observe.add_argument("--run-id", default="run_fixture_observe_cli")
    fixture_observe.add_argument("--frame-budget", type=int, default=16)

    job = commands.add_parser("job", help="create, run, inspect or cancel a local job")
    job_commands = job.add_subparsers(dest="job_command", required=True)
    job_create = job_commands.add_parser("create")
    job_create.add_argument("--db", type=Path, required=True)
    job_create.add_argument("--stage", required=True)
    job_create.add_argument("--config-sha256", required=True)
    job_create.add_argument("--input-artifact-sha256")
    job_create.add_argument(
        "--payload-json",
        help="JSON object passed to the stage handler",
    )
    job_status = job_commands.add_parser("status")
    job_status.add_argument("--db", type=Path, required=True)
    job_status.add_argument("--job-id", required=True)
    job_cancel = job_commands.add_parser("cancel")
    job_cancel.add_argument("--db", type=Path, required=True)
    job_cancel.add_argument("--job-id", required=True)
    job_cancel.add_argument("--owner")
    job_cancel.add_argument(
        "--lease-token",
        help="fencing token for cancelling an owned running job",
    )
    job_run = job_commands.add_parser("run", help="run up to a bounded number of local jobs")
    job_run.add_argument("--db", type=Path, required=True)
    job_run.add_argument("--worker-id", default="local_worker")
    job_run.add_argument("--lease-seconds", type=float, default=300.0)
    job_run.add_argument("--max-jobs", type=int, default=1)
    job_run.add_argument("--snapshot-root", type=Path)
    job_run.add_argument("--artifact-relative-root")
    job_run.add_argument("--evidence-root", type=Path)
    job_run.add_argument("--backend-config", type=Path)
    job_run.add_argument("--sam2-config", type=Path)
    job_run.add_argument("--vlm-config", type=Path)
    job_result = job_commands.add_parser(
        "result", help="reload and verify saved observation output",
    )
    job_result.add_argument("--db", type=Path, required=True)
    job_result.add_argument("--job-id", required=True)
    job_result.add_argument("--evidence-root", type=Path)
    _semantic_options(job_result)
    job_answer = job_commands.add_parser("answer", help="verify and emit the final VLM answer")
    job_answer.add_argument("--db", type=Path, required=True)
    job_answer.add_argument("--job-id", required=True)
    job_answer.add_argument("--evidence-root", type=Path)
    job_recover = job_commands.add_parser("recover", help="return expired running jobs to pending")
    job_recover.add_argument("--db", type=Path, required=True)
    return parser


def _context(args: argparse.Namespace) -> tuple[SQLiteStore, IndexRegistry]:
    store = SQLiteStore(args.db)
    snapshot_store = SnapshotStore(args.snapshot_root)
    registry = IndexRegistry(
        store,
        snapshot_store,
        artifact_relative_root=args.artifact_relative_root,
    )
    return store, registry


def _emit(payload: dict) -> None:
    print(json.dumps(payload, ensure_ascii=False, allow_nan=False, sort_keys=True))


def _parse_payload(raw_payload: str | None) -> dict[str, object]:
    if raw_payload is None:
        return {}
    try:
        payload = json.loads(raw_payload)
    except json.JSONDecodeError as error:
        raise ValueError("payload-json must be valid JSON") from error
    if not isinstance(payload, dict):
        raise ValueError("payload-json must contain a JSON object")
    return payload


def _read_request_json(path: Path):
    def unique_pairs(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate request field: {key}")
            result[key] = value
        return result

    def invalid_constant(value):
        raise ValueError(f"non-finite request number: {value}")

    with path.open("rb") as stream:
        raw = stream.read(1024 * 1024 + 1)
    if len(raw) > 1024 * 1024:
        raise ValueError("observation request exceeds 1 MiB")
    payload = json.loads(
        raw.decode("utf-8"), object_pairs_hook=unique_pairs, parse_constant=invalid_constant,
    )
    return payload


def _read_observation_request(path: Path):
    from video_intelligence.pipeline.observation_jobs import ObservationJobRequest

    return ObservationJobRequest.from_dict(_read_request_json(path))


def _run_observation_command(args: argparse.Namespace, *, semantic=None) -> int:
    from video_intelligence.detection.backends import load_grounding_backends
    from video_intelligence.pipeline.fixture_observations import prepare_fixture_observation_request
    from video_intelligence.pipeline.observation_jobs import (
        ObservationJobRequest,
        create_observation_job,
    )

    evidence_root = args.evidence_root or resolve_project_paths().artifacts_root
    snapshots = SnapshotStore(args.snapshot_root)
    # Parse a generic request before opening/creating its metadata database.
    request = _read_observation_request(args.request_file) if args.command == "observe" else None
    preparation = None
    if args.command == "retrieve-observe":
        from video_intelligence.embeddings.local_siglip2 import load_retrieval_backend
        from video_intelligence.pipeline.retrieval_prepare import normalize_plan

        preparation, _ = normalize_plan(_read_request_json(args.request_file))
        retrieval_backend = load_retrieval_backend(args.retrieval_config)
    config_path = getattr(args, "backend_config", None)
    factories = load_grounding_backends(config_path) if config_path is not None else {}
    sam2_backend = None
    if getattr(args, "sam2_config", None) is not None:
        from video_intelligence.segmentation.sam2 import load_sam2_backend

        sam2_backend = load_sam2_backend(args.sam2_config)
        payload = preparation if preparation is not None else request.to_dict()
        declared = payload.get("sam2_backend_sha256")
        if declared is not None and declared != sam2_backend.identity_sha256:
            raise ValueError("request SAM2 backend digest differs from sam2-config")
        payload = {**payload, "sam2_backend_sha256": sam2_backend.identity_sha256}
        if preparation is None:
            request = ObservationJobRequest.from_dict(payload)
        else:
            preparation = payload
    vlm_backend = None
    if getattr(args, "vlm_config", None) is not None:
        from video_intelligence.reasoning.vlm import load_vlm_backend

        vlm_backend = load_vlm_backend(args.vlm_config)
        payload = preparation if preparation is not None else request.to_dict()
        declared = payload.get("vlm_backend_sha256")
        if declared is not None and declared != vlm_backend.identity_sha256:
            raise ValueError("request VLM backend digest differs from vlm-config")
        payload = {**payload, "vlm_backend_sha256": vlm_backend.identity_sha256}
        if preparation is None:
            request = ObservationJobRequest.from_dict(payload)
        else:
            preparation = payload
    if config_path is not None:
        payload = preparation if preparation is not None else request.to_dict()
        factory = factories.get(payload["grounding_backend"])
        if factory is None:
            raise ValueError("selected backend is absent from backend-config")
        declared = payload.get("grounding_backend_sha256")
        if declared is not None and declared != factory.identity_sha256:
            raise ValueError("request backend digest differs from backend-config")
        payload = {**payload, "grounding_backend_sha256": factory.identity_sha256}
        if preparation is None:
            request = ObservationJobRequest.from_dict(payload)
        else:
            factory.validate_plan(payload)
            preparation = payload
    with SQLiteStore(args.db) as store:
        if preparation is not None:
            from video_intelligence.pipeline.retrieval_prepare import prepare_retrieval_request

            request = prepare_retrieval_request(
                store, snapshots, plan=preparation, backend=retrieval_backend,
                snapshot_id=args.snapshot_id,
            )
        if args.command == "fixture-observe":
            request = prepare_fixture_observation_request(
                store, snapshots, video_id=args.video_id, query_text=args.query,
                run_id=args.run_id, frame_budget=args.frame_budget,
            )
        queue = JobQueue(store)
        job = create_observation_job(queue, request, fixture=args.command == "fixture-observe")
        if args.enqueue_only:
            _emit(job.to_dict())
            return 0
        worker = JobWorker(queue, handlers=default_handlers(
            snapshot_root=args.snapshot_root, evidence_root=evidence_root,
            backend_factories=factories,
            sam2_backend=sam2_backend,
            vlm_backend=vlm_backend,
        ))
        outcome = worker.run_once(job_id=job.job_id)
        if outcome is None:
            # A deduplicated completed job is replayed through file verification.
            from video_intelligence.pipeline.observation_jobs import load_observation_result
            from video_intelligence.storage.evidence_frames import EvidenceFrameStore

            saved = load_observation_result(store, job.job_id, EvidenceFrameStore(evidence_root))
            if saved is not None:
                code = 0
                if semantic is not None:
                    saved, code = semantic.attach(
                        saved, store.get_video(saved["query_result"]["video_id"]))
                _emit({"job_id": job.job_id, "status": "succeeded", "output": saved})
                return code
            current = queue.get(job.job_id)
            _emit({"job": current.to_dict() if current else None, "processed": 0})
            return 1
        emitted = outcome.to_dict()
        code = 0 if outcome.status is JobStatus.SUCCEEDED and outcome.error_code is None else 1
        if semantic is not None and code == 0 and emitted.get("output") is not None:
            saved = emitted["output"]
            emitted["output"], code = semantic.attach(
                saved, store.get_video(saved["query_result"]["video_id"]))
        _emit(emitted)
        return code


def main(argv: list[str] | None = None, *, semantic_services=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        semantic = None
        if getattr(args, "semantic_config", None) is not None:
            from video_intelligence.reasoning.semantic_cli import SemanticCLI

            if args.semantic_work_root is None:
                raise ValueError("semantic-config requires semantic-work-root")
            if getattr(args, "enqueue_only", False):
                raise ValueError("semantic assessment is inline; use job result after queued work")
            if args.command == "query" and args.query_id is None:
                raise ValueError("semantic query requires a persisted learned query-id")
            semantic = SemanticCLI.from_args(args, _read_request_json(args.semantic_config),
                                             semantic_services)
        elif (getattr(args, "semantic_work_root", None) is not None
              or getattr(args, "semantic_cancel_file", None) is not None
              or getattr(args, "semantic_timeout_seconds", 60) != 60):
            raise ValueError("semantic options require semantic-config")
        if args.command == "parse-query":
            from video_intelligence.pipeline.observation_jobs import _prompts
            from video_intelligence.retrieval.query_parser import parse_query

            registry = _prompts(_read_request_json(args.registry_file))
            result = parse_query(args.query, registry=registry)
            _emit(result.to_dict())
            return 0 if result.parse_status.value == "supported" else 2
        if args.command == "inspect-frames":
            from video_intelligence.video.source_frames import inspect_source_frames

            _emit(inspect_source_frames(
                args.video, args.timestamp, video_id=args.video_id,
                selection_policy=args.selection_policy, max_decode_frames=args.max_decode_frames,
                timeout_seconds=args.timeout_seconds,
            ))
            return 0
        if args.command in {"observe", "fixture-observe", "retrieve-observe"}:
            return _run_observation_command(args, semantic=semantic)
        if args.command in {"import-video", "import-status", "import-resume"}:
            from video_intelligence.pipeline.video_import import ImportConfig, LocalVideoImport
            from video_intelligence.storage.evidence_frames import _safe

            _safe(args.db.expanduser().absolute())
            with SQLiteStore(args.db) as store:
                if args.command == "import-video":
                    config = ImportConfig(**{name: getattr(args, name) for name in (
                        "max_bytes", "timeout_seconds", "min_free_bytes", "max_pixels",
                        "max_duration_seconds", "pipeline_version", "probe_profile",
                    )})
                    service = LocalVideoImport(store, config)
                    job = service.create(args.video, mode=args.mode, video_id=args.video_id,
                                         managed_root=args.managed_root)
                    _emit(service.run(job.job_id, cancel_file=args.cancel_file))
                elif args.command == "import-resume":
                    _emit(LocalVideoImport(store).run(args.job_id, cancel_file=args.cancel_file))
                else:
                    _emit(LocalVideoImport(store).status(args.job_id))
            return 0
        if args.command == "register":
            with SQLiteStore(args.db) as store:
                service = IngestionService(
                    store,
                    IndexRegistry(store, SnapshotStore(args.db.parent / "snapshots")),
                )
                record = service.register(args.video, video_id=args.video_id)
                _emit({"mode": "reference", "video": record.to_dict()})
            return 0
        if args.command == "job":
            factories = {}
            sam2_backend = None
            vlm_backend = None
            if args.job_command == "run" and args.vlm_config is not None:
                from video_intelligence.reasoning.vlm import load_vlm_backend

                vlm_backend = load_vlm_backend(args.vlm_config)
            if args.job_command == "run" and args.sam2_config is not None:
                from video_intelligence.segmentation.sam2 import load_sam2_backend

                sam2_backend = load_sam2_backend(args.sam2_config)
            if args.job_command == "run" and args.backend_config is not None:
                from video_intelligence.detection.backends import load_grounding_backends

                factories = load_grounding_backends(args.backend_config)
            with SQLiteStore(args.db) as store:
                queue = JobQueue(store)
                if args.job_command == "create":
                    _emit(queue.create(
                        stage=args.stage,
                        config_sha256=args.config_sha256,
                        input_artifact_sha256=args.input_artifact_sha256,
                        payload=_parse_payload(args.payload_json),
                    ).to_dict())
                    return 0
                if args.job_command == "status":
                    job = queue.get(args.job_id)
                    if job is None:
                        raise ValueError(f"job is not found: {args.job_id}")
                    _emit(job.to_dict())
                    return 0
                if args.job_command == "recover":
                    _emit({"recovered": queue.recover_stale()})
                    return 0
                if args.job_command in {"result", "answer"}:
                    from video_intelligence.pipeline.observation_jobs import load_observation_result
                    from video_intelligence.storage.evidence_frames import EvidenceFrameStore

                    evidence_root = args.evidence_root or resolve_project_paths().artifacts_root
                    payload = load_observation_result(
                        store, args.job_id, EvidenceFrameStore(evidence_root),
                    )
                    if payload is None:
                        raise ValueError(f"observation job has no completed output: {args.job_id}")
                    if args.job_command == "answer":
                        if "structured_answer" not in payload:
                            raise ValueError("observation job has no configured VLM answer")
                        _emit(payload["structured_answer"])
                    else:
                        code = 0
                        if semantic is not None:
                            payload, code = semantic.attach(
                                payload, store.get_video(payload["query_result"]["video_id"]))
                        _emit(payload)
                        return code
                    return 0
                if args.job_command == "run":
                    worker = JobWorker(
                        queue,
                        handlers=default_handlers(
                            snapshot_root=args.snapshot_root,
                            artifact_relative_root=args.artifact_relative_root,
                            evidence_root=(args.evidence_root
                                           or resolve_project_paths().artifacts_root),
                            backend_factories=factories,
                            sam2_backend=sam2_backend,
                            vlm_backend=vlm_backend,
                        ),
                        worker_id=args.worker_id,
                        lease_seconds=args.lease_seconds,
                    )
                    outcomes = worker.run_until_idle(max_jobs=args.max_jobs)
                    _emit({
                        "worker_id": args.worker_id,
                        "processed": len(outcomes),
                        "outcomes": [outcome.to_dict() for outcome in outcomes],
                    })
                    return 1 if any(
                        outcome.status is not JobStatus.SUCCEEDED
                        or outcome.error_code is not None
                        for outcome in outcomes
                    ) else 0
                lease_token = args.lease_token
                if args.owner is not None and lease_token is None:
                    # Preserve the legacy owner-only CLI contract while still
                    # passing the current fencing token through the storage
                    # guard. An owner mismatch never receives a token.
                    current = queue.get(args.job_id)
                    if current is not None and current.owner == args.owner:
                        lease_token = current.lease_token
                cancelled = queue.cancel(
                    args.job_id,
                    owner=args.owner,
                    lease_token=lease_token,
                )
                _emit({"job_id": args.job_id, "cancelled": cancelled})
                return 0 if cancelled else 1
        store, registry = _context(args)
        with store:
            if args.command == "fixture-ingest":
                service = IngestionService(store, registry)
                summary = service.ingest(
                    args.video_id,
                    adapter=PixelFixtureAdapter(frame_budget=args.frame_budget),
                )
                _emit(summary.to_dict())
                return 0 if summary.processing_status != "failed" else 1
            if args.command == "query":
                service = QueryService(store, registry)
                if args.query_id is not None:
                    payload = service.load_persisted(args.query_id)
                    if payload is None:
                        raise ValueError(f"query is not persisted: {args.query_id}")
                    if payload.get("video_id") != args.video_id:
                        raise ValueError("persisted query belongs to another video")
                    evidence = payload.get("evidence")
                    if evidence and any(
                        item.get("provenance", {}).get("image_persisted")
                        for item in evidence["items"]
                    ):
                        from video_intelligence.retrieval.schemas import QueryResult
                        from video_intelligence.storage.evidence_frames import EvidenceFrameStore

                        video = store.get_video(args.video_id)
                        EvidenceFrameStore(
                            args.evidence_root or resolve_project_paths().artifacts_root,
                        ).verify_result(QueryResult.from_dict(payload, video=video))
                    code = 0 if payload.get("processing_status") != "failed" else 1
                    if semantic is not None:
                        payload, semantic_code = semantic.attach(
                            payload, store.get_video(args.video_id))
                        code = max(code, semantic_code)
                    _emit(payload)
                    return code
                result = service.query(args.video_id, args.query, run_id=args.run_id)
                _emit(result.to_dict())
                return 0 if result.processing_status.value != "failed" else 1
    except (OSError, RuntimeError, ValueError, sqlite3.DatabaseError) as error:
        payload = {"error": str(error), "error_type": error.__class__.__name__}
        if hasattr(error, "code"):
            payload["error_code"] = error.code
        for name in ("job_id", "target"):
            if getattr(error, name, None) is not None:
                payload[name] = getattr(error, name)
        if getattr(error, "issue", None) is not None:
            payload["decode_issue"] = (
                asdict(error.issue) if is_dataclass(error.issue) else error.issue
            )
        _emit(payload)
        return 2
    raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
