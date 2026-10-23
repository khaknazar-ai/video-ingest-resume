"""Synthetic portfolio demo for existing ingestion/resume/frame APIs; no models."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))


def _write(path, value):
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")


def _copy_worker(output):
    from video_intelligence.data.dataset import require_synthetic_contract
    from video_intelligence.pipeline.video_import import ImportConfig, LocalVideoImport
    from video_intelligence.storage.sqlite import SQLiteStore

    source = output/"fixtures/synthetic_fixture.mp4"
    require_synthetic_contract(source)
    with SQLiteStore(output/"state.sqlite") as store:
        service = LocalVideoImport(store, ImportConfig(
            timeout_seconds=2., pipeline_version="portfolio-demo.v1"
        ))
        job = service.create(source, mode="managed", managed_root=output/"managed",
                             video_id="fixture_managed")
        print(job.job_id, flush=True)

        def interrupt(stage):
            if stage == "copy_checkpoint":
                os._exit(77)

        service.run(job.job_id, progress=interrupt)
    raise RuntimeError("expected copy interruption was not reached")


def _frame_worker(output):
    import av
    import cv2

    from video_intelligence.core.types import FrameRequest, VideoRecord
    from video_intelligence.video.decoder import PyAVVideoDecoder

    receipt = json.loads((output/"managed.json").read_bytes())
    video = VideoRecord.from_dict(receipt["video"])
    reference = []
    with av.open(str(video.source_path)) as container:
        stream = container.streams.video[0]
        origin = None
        for frame in container.decode(stream):
            if origin is None:
                origin = frame.pts
            if len(reference) >= 100:
                raise RuntimeError("synthetic reference frame cap exceeded")
            reference.append(dict(pts=frame.pts, time_base=str(Fraction(frame.time_base)),
                seconds=float((frame.pts-origin)*Fraction(frame.time_base)),
                rgb_sha256=hashlib.sha256(frame.to_ndarray(format="rgb24").tobytes()).hexdigest()))
    matches = []
    for policy in ("at_or_after", "nearest"):
        inspected = json.loads((output/f"{policy}.json").read_bytes())
        for row in inspected["frames"]:
            requested = row["requested_seconds"]
            expected = (next(f for f in reference if f["seconds"] >= requested)
                if policy == "at_or_after" else min(reference, key=lambda f: (
                    abs(f["seconds"]-requested), f["seconds"])))
            ref = row["frame_ref"]
            valid = (ref["pts"] == expected["pts"] and ref["time_base"] == expected["time_base"]
                     and ref["timestamp_seconds"] == expected["seconds"]
                     and ref["source_sha256"] == video.source_sha256
                     and row["rgb_sha256"] == expected["rgb_sha256"])
            matches.append(valid)
    if len(matches) != 4 or not all(matches):
        raise RuntimeError("CLI frames differ from sequential exact-PTS/RGB reference")
    first = json.loads((output/"at_or_after.json").read_bytes())["frames"][0]
    with PyAVVideoDecoder(video, max_validation_frames=100, max_request_frames=100) as decoder:
        picked = decoder.read_request(FrameRequest(video.video_id, first["requested_seconds"]))
    if picked is None or hashlib.sha256(picked.image.tobytes()).hexdigest() != first["rgb_sha256"]:
        raise RuntimeError("saved frame differs from the inspected RGB")
    ok, png = cv2.imencode(".png", cv2.cvtColor(picked.image, cv2.COLOR_RGB2BGR))
    if not ok:
        raise RuntimeError("PNG encoding failed")
    restored = cv2.cvtColor(cv2.imdecode(png, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    if hashlib.sha256(restored.tobytes()).hexdigest() != first["rgb_sha256"]:
        raise RuntimeError("PNG roundtrip changed inspected RGB pixels")
    with (output/"frame.png").open("xb") as handle:
        handle.write(png.tobytes())
    print(json.dumps(dict(sequential_frames=len(reference), exact_frame_matches=matches,
                         png_rgb_matches_cli=True,
                         png_sha256=hashlib.sha256(png.tobytes()).hexdigest())))


def run_demo(output):
    from video_intelligence.storage.evidence_frames import _safe

    _safe(output)
    output.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ, PYTHONPATH=str(ROOT/"src"), PYTHONDONTWRITEBYTECODE="1",
        PYTHONIOENCODING="utf-8", CUDA_VISIBLE_DEVICES="", HF_HUB_OFFLINE="1",
        VI_ARTIFACTS_ROOT=str(output),
        VI_SYNTHETIC_HASH_REGISTRY=str(output/"synthetic_hash_registry.json"))
    commands = []

    def command(args, *, expected=0):
        argv = [sys.executable, "-B", *args]
        result = subprocess.run(argv, cwd=ROOT, env=env, capture_output=True, text=True, timeout=30)
        commands.append(dict(argv=argv, exit_code=result.returncode, expected_exit=expected,
                             stdout=result.stdout, stderr=result.stderr))
        if result.returncode != expected:
            raise RuntimeError(f"demo child failed: {result.stderr or result.stdout}")
        return result.stdout

    def cli(args, name):
        value = json.loads(command(["-m", "video_intelligence.cli", *args]))
        _write(output/name, value)
        return value

    report = dict(data_kind="synthetic", purpose="contract", inference=False,
                  quality_metrics=None, project_complete=False)
    try:
        command(["-c", "from video_intelligence.data.fixtures import generate_synthetic_fixture; "
            "import sys; generate_synthetic_fixture(sys.argv[1],width=320,height=240,"
            "fps=10,duration_seconds=1)", str(output/"fixtures")])
        source, db = output/"fixtures/synthetic_fixture.mp4", output/"state.sqlite"
        before = hashlib.sha256(source.read_bytes()).hexdigest()
        reference = cli(["import-video", "--db", str(db), "--video", str(source),
                         "--mode", "reference", "--video-id", "fixture_reference"],
                        "reference.json")
        job_id = command([str(Path(__file__).resolve()), "--output", str(output),
                          "--_copy-worker"], expected=77).strip()
        status = cli(["import-status", "--db", str(db), "--job-id", job_id], "interrupted.json")
        if status["job"]["status"] != "running" or status["receipt"] is not None:
            raise RuntimeError("interrupted import was incorrectly admitted")
        wait = max(0., status["job"]["lease_until"]-time.time()+.05)
        if wait > 3.:
            raise RuntimeError("unexpected demo lease duration")
        time.sleep(wait)
        managed = cli(["import-resume", "--db", str(db), "--job-id", job_id], "managed.json")
        repeated = cli(["import-resume", "--db", str(db), "--job-id", job_id], "repeated.json")
        path = Path(managed["video"]["source_path"])
        for policy in ("at_or_after", "nearest"):
            cli(["inspect-frames", "--video", str(path), "--video-id", "fixture_managed",
                 "--timestamp", "0.31", "--timestamp", "0.76", "--selection-policy", policy,
                 "--max-decode-frames", "100", "--timeout-seconds", "10"], f"{policy}.json")
        frames = json.loads(command([str(Path(__file__).resolve()), "--output", str(output),
                                    "--_frame-worker"]))
        checks = dict(reference_no_copy=Path(reference["video"]["source_path"]) == source,
            source_unchanged=hashlib.sha256(source.read_bytes()).hexdigest() == before,
            copy_byte_equal=hashlib.sha256(path.read_bytes()).hexdigest() == before,
            one_final_copy=len(list((output/"managed/objects").iterdir())) == 1,
            resumed_copy_checkpoint=managed["reused_copy"], idempotent_repeat=repeated == managed)
        if not all(checks.values()):
            raise RuntimeError("import/resume demo invariants failed")
        report.update(status="passed", checks=checks, frames=frames, source_sha256=before)
    except Exception as error:
        report.update(status="failed", error=str(error))
        raise
    finally:
        report["commands"] = commands
        _write(output/"report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("artifacts/portfolio-demo"))
    parser.add_argument("--_copy-worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--_frame-worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    output = args.output.expanduser().absolute()
    if args._copy_worker:
        _copy_worker(output)
    elif args._frame_worker:
        _frame_worker(output)
    else:
        report = run_demo(output)
        print(json.dumps(dict(status=report["status"], report=str(output/"report.json"),
                              frame_png=str(output/"frame.png"))))


if __name__ == "__main__":
    main()
