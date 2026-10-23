"""Local import CPU contracts; generated media are synthetic, never quality data."""

import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import replace
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from video_intelligence.cli import main
from video_intelligence.data.dataset import DatasetManifest
from video_intelligence.pipeline.jobs import JobStatus
from video_intelligence.pipeline.video_import import ImportConfig, ImportFailure, LocalVideoImport
from video_intelligence.pipeline.worker import JobWorker, default_handlers
from video_intelligence.storage.sqlite import SQLiteStore
from video_intelligence.video import probe as probe_module


@pytest.fixture
def source(tmp_path):
    av = pytest.importorskip("av")
    path = tmp_path/"import-contract.mkv"
    with av.open(str(path), mode="w", format="matroska") as container:
        stream = container.add_stream("ffv1", rate=10)
        stream.width, stream.height, stream.pix_fmt = 32, 24, "bgr0"
        for ordinal in range(3):
            frame = av.VideoFrame.from_ndarray(
                np.full((24, 32, 3), ordinal*60, dtype=np.uint8), format="rgb24")
            frame.pts, frame.time_base = ordinal*100, Fraction(1, 1000)
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    manifest = DatasetManifest("import-contract", "synthetic", "contract", "contract",
        ("import-contract",), (hashlib.sha256(path.read_bytes()).hexdigest(),),
        synthetic_recipe="FFV1/Matroska/bgr0 32x24; PTS0/100/200 at1/1000; RGB=ordinal*60")
    path.with_suffix(".dataset.json").write_text(json.dumps(manifest.to_dict()), encoding="utf-8")
    return path


def count(store, table):
    assert table in {"videos", "import_checkpoints", "video_import_results"}
    return store._require_connection().execute(f"SELECT count(*) FROM {table}").fetchone()[0]


def test_reference_has_no_copy_and_repeated_registration_is_idempotent(
    source, tmp_path, monkeypatch
):
    before = source.read_bytes()
    with SQLiteStore(tmp_path/"state.sqlite") as store:
        service = LocalVideoImport(store)
        job = service.create(source)
        result = service.run(job.job_id)
        assert result["mode"] == "reference" and result["copy_key"] is None
        assert Path(result["video"]["source_path"]) == source
        assert count(store, "videos") == count(store, "video_import_results") == 1
        assert count(store, "import_checkpoints") == 1
        monkeypatch.setattr(probe_module, "probe_video", lambda *a, **k: pytest.fail("reprobe"))
        assert service.create(source).job_id == job.job_id
        assert service.run(job.job_id) == result
        assert not (tmp_path/"managed").exists()
        assert source.read_bytes() == before


def test_managed_copy_exact_once_repeated_and_pipeline_invalidates_only_probe(source, tmp_path,
                                                                            monkeypatch):
    original = source.read_bytes()
    calls = []
    probe = probe_module.probe_video
    def observed(*args, **kwargs):
        calls.append(args[0])
        return probe(*args, **kwargs)
    monkeypatch.setattr(probe_module, "probe_video", observed)
    with SQLiteStore(tmp_path/"state.sqlite") as store:
        service = LocalVideoImport(store)
        first = service.create(source, mode="managed", managed_root=tmp_path/"managed")
        result = service.run(first.job_id)
        target = Path(result["video"]["source_path"])
        assert target.read_bytes() == original == source.read_bytes()
        assert list(target.parent.iterdir()) == [target]
        assert not list((tmp_path/"managed/staging").iterdir())
        assert not result["reused_copy"] and not result["reused_probe"]
        monkeypatch.setattr(os, "link", lambda *a: pytest.fail("a second copy was published"))
        assert service.run(first.job_id) == result
        changed = LocalVideoImport(store, replace(ImportConfig(), pipeline_version="changed.v2"))
        second = changed.create(source, mode="managed", managed_root=tmp_path/"managed")
        refreshed = changed.run(second.job_id)
        assert refreshed["copy_key"] == result["copy_key"]
        assert refreshed["probe_key"] != result["probe_key"]
        assert refreshed["reused_copy"] and not refreshed["reused_probe"]
        assert len(calls) == 2
        assert count(store, "videos") == 1 and count(store, "video_import_results") == 2


def test_cli_reference_status_resume(source, tmp_path, capsys):
    db = str(tmp_path/"state.sqlite")
    assert main(["import-video", "--db", db, "--video", str(source)]) == 0
    result = json.loads(capsys.readouterr().out)
    for command in ("import-status", "import-resume"):
        assert main([command, "--db", db, "--job-id", result["job_id"]]) == 0
        payload = json.loads(capsys.readouterr().out)
        assert (payload["job"]["status"] == "succeeded" if command == "import-status"
                else payload == result)


def test_default_worker_uses_committed_import_receipt(source, tmp_path):
    with SQLiteStore(tmp_path/"state.sqlite") as store:
        service = LocalVideoImport(store)
        job = service.create(source)
        outcome = JobWorker(service.queue, handlers=default_handlers()).run_once(job_id=job.job_id)
        assert outcome.status == JobStatus.SUCCEEDED
        assert outcome.output == store.get_video_import_result(job.job_id)
        assert store.get_video(outcome.output["video"]["video_id"]) is not None


@pytest.mark.parametrize("event", ["copy_checkpoint", "probe_checkpoint"])
def test_completed_stage_survives_failure_and_new_service_reuses_it(source, tmp_path, event):
    db = tmp_path/"state.sqlite"
    with SQLiteStore(db) as store:
        service = LocalVideoImport(store)
        job = service.create(source, mode="managed", managed_root=tmp_path/"managed")
        def stop(name):
            if name == event:
                raise RuntimeError("contract interruption")
        with pytest.raises(ImportFailure):
            service.run(job.job_id, progress=stop)
        assert count(store, "videos") == 0
        assert service.queue.get(job.job_id).status == JobStatus.FAILED
    with SQLiteStore(db) as store:
        result = LocalVideoImport(store).run(job.job_id)
        assert result["reused_copy"]
        assert result["reused_probe"] == (event == "probe_checkpoint")
        assert count(store, "videos") == 1
        assert count(store, "video_import_results") == 1


@pytest.mark.parametrize("event", ["copy_chunk", "copy_checkpoint"])
def test_actual_process_crash_and_cli_restart(source, tmp_path, event):
    db, root = tmp_path/"state.sqlite", tmp_path/"managed"
    code = """
import os,sys
from pathlib import Path
from video_intelligence.storage.sqlite import SQLiteStore
from video_intelligence.pipeline.video_import import ImportConfig,LocalVideoImport
with SQLiteStore(sys.argv[1]) as store:
    service=LocalVideoImport(store,ImportConfig(chunk_bytes=64,timeout_seconds=1.))
    job=service.create(sys.argv[2],mode='managed',managed_root=Path(sys.argv[3]))
    print(job.job_id,flush=True)
    def crash(event):
        if event==sys.argv[4]: os._exit(77)
    service.run(job.job_id,progress=crash)
"""
    failed = subprocess.run(
        [sys.executable, "-B", "-c", code, str(db), str(source), str(root), event],
        capture_output=True, text=True, timeout=15)
    assert failed.returncode == 77, failed.stdout+failed.stderr
    job_id = failed.stdout.strip()
    time.sleep(1.1)  # Own bounded lease expiry, not a change to somebody else's job.
    completed = subprocess.run([
        sys.executable, "-B", "-m", "video_intelligence.cli", "import-resume",
        "--db", str(db), "--job-id", job_id], capture_output=True, text=True, timeout=15)
    assert completed.returncode == 0, completed.stdout+completed.stderr
    result = json.loads(completed.stdout)
    assert result["reused_copy"] == (event == "copy_checkpoint")
    assert Path(result["video"]["source_path"]).read_bytes() == source.read_bytes()
    with SQLiteStore(db) as store:
        assert count(store, "videos") == count(store, "video_import_results") == 1
    assert len(list((root/"objects").iterdir())) == 1
    if event == "copy_chunk":
        assert list((root/"staging").glob("*.partial")), "failed partial must be retained"


@pytest.mark.parametrize("moment", ["before", "copy_chunk", "before_activate"])
def test_original_source_change_never_registers(source, tmp_path, moment):
    with SQLiteStore(tmp_path/"state.sqlite") as store:
        service = LocalVideoImport(store, ImportConfig(chunk_bytes=64))
        job = service.create(source, mode="managed", managed_root=tmp_path/"managed")
        def mutate(name):
            if name == moment:
                source.write_bytes(b"changed contract input")
        if moment == "before":
            mutate(moment)
        with pytest.raises(ImportFailure) as failure:
            service.run(job.job_id, progress=mutate)
        assert failure.value.code == "source_changed"
        assert count(store, "videos") == count(store, "video_import_results") == 0


def test_source_mutation_during_probe_never_activates(source, tmp_path, monkeypatch):
    original_probe = probe_module.probe_video
    def mutate(*args, **kwargs):
        metadata = original_probe(*args, **kwargs)
        source.write_bytes(b"mutated during contract probe")
        return metadata
    monkeypatch.setattr(probe_module, "probe_video", mutate)
    with SQLiteStore(tmp_path/"state.sqlite") as store:
        service = LocalVideoImport(store)
        job = service.create(source, mode="managed", managed_root=tmp_path/"managed")
        with pytest.raises(ImportFailure, match="input differs"):
            service.run(job.job_id)
        assert count(store, "videos") == count(store, "video_import_results") == 0


def test_managed_target_changed_and_restored_during_probe_is_rejected(source, tmp_path,
                                                                    monkeypatch):
    original_probe = probe_module.probe_video
    def mutate(path, **kwargs):
        result = original_probe(path, **kwargs)
        path = Path(path)
        before, identity = path.read_bytes(), path.stat()
        path.write_bytes(b"temporary contract replacement")
        path.write_bytes(before)
        os.utime(path, ns=(identity.st_atime_ns, identity.st_mtime_ns+1_000_000))
        return result
    monkeypatch.setattr(probe_module, "probe_video", mutate)
    with SQLiteStore(tmp_path/"state.sqlite") as store:
        service = LocalVideoImport(store)
        job = service.create(source, mode="managed", managed_root=tmp_path/"managed")
        with pytest.raises(ImportFailure) as failure:
            service.run(job.job_id)
        assert failure.value.code == "copy_changed"
        assert count(store, "videos") == count(store, "video_import_results") == 0


def test_binary_identity_change_invalidates_old_request(source, tmp_path):
    with SQLiteStore(tmp_path/"state.sqlite") as store:
        service = LocalVideoImport(store)
        created = service.create(source)
        job = service.queue.claim(created.job_id, owner="contract_worker")
        payload = dict(job.payload, probe_tool={"path": "contract_changed", "sha256": "f"*64})
        with pytest.raises(ImportFailure) as failure:
            service.execute_claimed(replace(job, payload=payload))
        assert failure.value.code == "pipeline_changed"
        assert count(store, "videos") == 0


def test_completed_receipt_never_hides_later_source_change(source, tmp_path):
    with SQLiteStore(tmp_path/"state.sqlite") as store:
        service = LocalVideoImport(store)
        job = service.create(source)
        service.run(job.job_id)
        source.write_bytes(b"contract input changed after completion")
        with pytest.raises(ImportFailure) as failure:
            service.run(job.job_id)
        assert failure.value.code == "source_changed"


def test_stale_lease_cannot_link_or_checkpoint(source, tmp_path, monkeypatch):
    with SQLiteStore(tmp_path/"state.sqlite") as store:
        service = LocalVideoImport(store)
        job = service.create(source, mode="managed", managed_root=tmp_path/"managed")
        save = store.save_import_checkpoint
        def lose(*args, **kwargs):
            service.queue.cancel(job.job_id)
            return save(*args, **kwargs)
        monkeypatch.setattr(store, "save_import_checkpoint", lose)
        monkeypatch.setattr(os, "link", lambda *args: pytest.fail("stale worker linked"))
        with pytest.raises(ImportFailure) as failure:
            service.run(job.job_id)
        assert failure.value.code == "lost_lease"
        assert count(store, "import_checkpoints") == count(store, "videos") == 0
        assert not list((tmp_path/"managed/objects").iterdir())


def test_expiry_after_link_leaves_only_untrusted_orphan_then_resume_verifies(source, tmp_path,
                                                                           monkeypatch):
    with SQLiteStore(tmp_path/"state.sqlite") as store:
        service = LocalVideoImport(store)
        job = service.create(source, mode="managed", managed_root=tmp_path/"managed")
        link = os.link
        def expire(*args):
            link(*args)
            store._require_connection().execute(
                "UPDATE jobs SET lease_until=0 WHERE job_id=?", (job.job_id,))
        monkeypatch.setattr(os, "link", expire)
        with pytest.raises(ImportFailure):
            service.run(job.job_id)
        assert count(store, "import_checkpoints") == count(store, "videos") == 0
        monkeypatch.setattr(os, "link", lambda *a: pytest.fail("orphan was recopied"))
        result = service.run(job.job_id)
        assert result["reused_copy"] and count(store, "videos") == 1


def test_conflicting_destination_is_not_overwritten(source, tmp_path):
    with SQLiteStore(tmp_path/"state.sqlite") as store:
        service = LocalVideoImport(store)
        job = service.create(source, mode="managed", managed_root=tmp_path/"managed")
        target = tmp_path/"managed/objects"/job.payload["source_sha256"]
        target.parent.mkdir(parents=True)
        target.write_bytes(b"somebody else's contract object")
        with pytest.raises(ImportFailure) as failure:
            service.run(job.job_id)
        assert failure.value.code == "destination_conflict"
        assert target.read_bytes() == b"somebody else's contract object"
        assert count(store, "videos") == 0


def test_cancel_partial_then_explicit_resume(source, tmp_path):
    marker = tmp_path/"cancel"
    with SQLiteStore(tmp_path/"state.sqlite") as store:
        service = LocalVideoImport(store, ImportConfig(chunk_bytes=64))
        job = service.create(source, mode="managed", managed_root=tmp_path/"managed")
        def cancel(name):
            if name == "copy_chunk":
                marker.write_text("cancel", encoding="ascii")
        with pytest.raises(ImportFailure) as failure:
            service.run(job.job_id, cancel_file=marker, progress=cancel)
        assert failure.value.code == "import_cancelled"
        assert count(store, "import_checkpoints") == 0
        assert list((tmp_path/"managed/staging").glob("*.partial"))
        assert not list((tmp_path/"managed/objects").iterdir())
        assert service.run(job.job_id)["reused_copy"] is False
        assert marker.exists()


def test_invalid_probe_checkpoint_is_recomputed_without_erasing_bad_record(source, tmp_path):
    with SQLiteStore(tmp_path/"state.sqlite") as store:
        service = LocalVideoImport(store)
        job = service.create(source)
        def stop(name):
            if name == "probe_checkpoint":
                raise RuntimeError("contract stop")
        with pytest.raises(ImportFailure):
            service.run(job.job_id, progress=stop)
        store._require_connection().execute("UPDATE import_checkpoints SET output_json='{}'")
        store._require_connection().commit()
        result = service.run(job.job_id)
        assert not result["reused_probe"]
        assert count(store, "import_checkpoints") == 2


def test_space_and_size_limits_fail_before_publication(source, tmp_path, monkeypatch):
    with SQLiteStore(tmp_path/"state.sqlite") as store:
        with pytest.raises(ImportFailure) as failure:
            LocalVideoImport(store, ImportConfig(max_bytes=1)).create(source)
        assert failure.value.code == "source_size_limit"
        service = LocalVideoImport(store)
        job = service.create(source, mode="managed", managed_root=tmp_path/"managed")
        monkeypatch.setattr(shutil, "disk_usage", lambda _: SimpleNamespace(free=0))
        with pytest.raises(ImportFailure) as failure:
            service.run(job.job_id)
        assert failure.value.code == "insufficient_space"
        assert count(store, "videos") == 0


def test_timeout_and_untrusted_request_config_fail_closed(source, tmp_path):
    with SQLiteStore(tmp_path/"state.sqlite") as store:
        service = LocalVideoImport(store, ImportConfig(timeout_seconds=.05, chunk_bytes=64))
        job = service.create(source, mode="managed", managed_root=tmp_path/"managed")
        def slow(name):
            if name == "copy_chunk":
                time.sleep(.06)
        with pytest.raises(ImportFailure) as failure:
            service.run(job.job_id, progress=slow)
        assert failure.value.code == "import_timeout"
        assert count(store, "videos") == 0
        malformed = replace(job, config_sha256="f"*64)
        with pytest.raises(ValueError, match="identity mismatch"):
            service.execute_claimed(malformed)


@pytest.mark.parametrize("options", [{"max_bytes": True}, {"chunk_bytes": 0},
    {"timeout_seconds": float("nan")}, {"min_free_bytes": 0}, {"pipeline_version": " "}])
def test_config_rejects_invalid_budgets(options):
    with pytest.raises(ValueError):
        ImportConfig(**options)


def test_lexical_traversal_is_rejected(source, tmp_path):
    with SQLiteStore(tmp_path/"state.sqlite") as store:
        with pytest.raises(ValueError, match="traversal"):
            LocalVideoImport(store).create(source.parent/".."/source.parent.name/source.name)
