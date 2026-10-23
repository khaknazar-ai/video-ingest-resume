"""Decode-forward contracts, separate from actual dev-file qualification."""

import ctypes
import hashlib
import json
import multiprocessing
import os
import shutil
import subprocess
import time
from dataclasses import replace
from fractions import Fraction

import numpy as np
import pytest

from video_intelligence.core.types import FrameRequest, VideoMetadata, VideoRecord
from video_intelligence.data.dataset import DatasetManifest
from video_intelligence.video import decoder as decoder_module
from video_intelligence.video.decoder import DecodeError, PyAVVideoDecoder
from video_intelligence.video.source_frames import SourceInspectionError, _run_inspection


@pytest.fixture(params=[(0, 100, 200, 300), (0, 100, 400, 900), (500, 600, 900, 1400)])
def source(tmp_path, request):
    av = pytest.importorskip("av")
    pts_values = request.param
    path = tmp_path/"forward-contract.mkv"
    with av.open(str(path), mode="w", format="matroska") as container:
        stream = container.add_stream("ffv1", rate=10)
        stream.width, stream.height, stream.pix_fmt = 32, 24, "bgr0"
        stream.time_base = Fraction(1, 1000)
        for ordinal, pts in enumerate(pts_values):
            pixels = np.full((24, 32, 3), ordinal*50, dtype=np.uint8)
            frame = av.VideoFrame.from_ndarray(pixels, format="rgb24")
            frame.pts, frame.time_base = pts, Fraction(1, 1000)
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = DatasetManifest("forward-contract", "synthetic", "contract", "contract",
        ("forward-contract",), (digest,), synthetic_recipe=
        f"FFV1/Matroska/bgr0 32x24; PTS={pts_values} at 1/1000; RGB=ordinal*50")
    path.with_suffix(".dataset.json").write_text(json.dumps(manifest.to_dict()), encoding="utf-8")
    video = VideoRecord("forward-contract", path, VideoMetadata(2., 10., 4, 32, 24), digest)
    return video, pts_values


@pytest.mark.parametrize("policy", ["at_or_after", "nearest"])
def test_forward_matches_sequential_exact_pts_rgb_repeats_and_eof(source, policy):
    video, pts_values = source
    with PyAVVideoDecoder(video) as decoder:
        reference = list(decoder.iter_stream())
        assert [f.frame_ref.pts for f in reference] == list(pts_values)
        timestamps = [0., .05, .15, .5, .95, .15, 0., 2., 3.]
        for timestamp in timestamps:
            candidates = [f for f in reference if f.actual_timestamp_seconds >= timestamp]
            expected = (candidates[0] if candidates else None) if policy == "at_or_after" else min(
                reference, key=lambda f: (abs(f.actual_timestamp_seconds-timestamp),
                                         f.actual_timestamp_seconds))
            if timestamp >= 2.:
                expected = None
            actual = decoder.read_request(
                FrameRequest(video.video_id, timestamp), selection_policy=policy
            )
            if expected is None:
                assert actual is None
                assert decoder.issues[-1].code == "out_of_range"
            else:
                assert actual.frame_ref.identity_key == expected.frame_ref.identity_key
                assert actual.frame_ref.timestamp_seconds == expected.frame_ref.timestamp_seconds
                assert actual.frame_ref.source_sha256 == video.source_sha256
                assert actual.frame_ref.source_path == video.source_path
                np.testing.assert_array_equal(actual.image, expected.image)


def test_frame_budget_stops_before_extra_decode_and_recovers(source, monkeypatch):
    video, _ = source
    with PyAVVideoDecoder(video, max_request_frames=1) as decoder:
        converted = []
        original = decoder._convert_frame
        def convert(frame):
            converted.append(frame.pts)
            return original(frame)
        monkeypatch.setattr(decoder, "_convert_frame", convert)
        with pytest.raises(DecodeError) as failure:
            decoder.read_request(FrameRequest(video.video_id, .8))
        assert failure.value.issue.code == "decode_frame_limit"
        assert len(converted) == 1
        recovered = decoder.read_request(FrameRequest(video.video_id, 0.))
        assert recovered.frame_ref.pts == reference_origin(video)


def reference_origin(video):
    import av
    with av.open(str(video.source_path)) as container:
        return next(container.decode(container.streams.video[0])).pts


def test_cooperative_timeout_and_codec_error_are_structured(source, monkeypatch):
    video, _ = source
    with PyAVVideoDecoder(video, max_request_seconds=.01) as decoder:
        tick = [0.]
        monkeypatch.setattr(decoder_module.time, "monotonic", lambda: tick[0])
        original = decoder._convert_frame
        def slow(frame):
            tick[0] += 1.
            return original(frame)
        monkeypatch.setattr(decoder, "_convert_frame", slow)
        with pytest.raises(DecodeError) as failure:
            decoder.read_request(FrameRequest(video.video_id, 0.))
        assert failure.value.issue.code == "decode_timeout"
        def broken():
            raise RuntimeError("contract codec failure")
        monkeypatch.setattr(decoder, "_restart_stream", broken)
        with pytest.raises(DecodeError) as failure:
            decoder.read_request(FrameRequest(video.video_id, 0.))
        assert failure.value.issue.code == "decode_failed"


def _native_block(connection, pid_path):
    """Actual native blocking call, not a cooperative Python deadline check."""
    pid_path.write_text(str(os.getpid()), encoding="ascii")
    if os.name == "nt":
        ctypes.windll.kernel32.Sleep(30_000)
    else:
        ctypes.CDLL(None).sleep(30)
    connection.send((True, {"should_not_escape": True}))


def _crash(connection):
    os._exit(7)


def test_outer_deadline_kills_and_reaps_native_block(tmp_path):
    pid_path = tmp_path/"worker.pid"
    started = time.monotonic()
    with pytest.raises(SourceInspectionError) as failure:
        _run_inspection(_native_block, (pid_path,), 3.)
    assert failure.value.code == "inspection_timeout"
    assert pid_path.exists(), "worker must reach native call before the deadline"
    assert time.monotonic()-started < 6.
    assert int(pid_path.read_text()) not in {p.pid for p in multiprocessing.active_children()}


def test_abnormal_worker_exit_is_structured():
    with pytest.raises(SourceInspectionError) as failure:
        _run_inspection(_crash, (), 10.)
    assert failure.value.code == "inspection_worker_failed"


@pytest.mark.parametrize("options", [{"max_request_frames": 0}, {"max_request_frames": True},
                                      {"max_request_seconds": float("nan")},
                                      {"max_request_seconds": False}])
def test_invalid_forward_limits(source, options):
    video, _ = source
    with pytest.raises(ValueError):
        PyAVVideoDecoder(video, **options)


def test_unknown_duration_eof_uses_decoded_timeline(source):
    video, _ = source
    video = replace(video, metadata=replace(video.metadata, duration_seconds=None))
    with PyAVVideoDecoder(video) as decoder:
        assert decoder.read_request(FrameRequest(video.video_id, 5.)) is None
        assert decoder.read_request(
            FrameRequest(video.video_id, 5.), selection_policy="nearest"
        ) is not None


def test_corrupt_source_open_fails_structurally_without_frames(tmp_path):
    path = tmp_path/"corrupt-contract.mkv"
    path.write_bytes(b"deliberately invalid Matroska contract fixture")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = DatasetManifest("corrupt-contract", "synthetic", "contract", "contract",
        ("corrupt-contract",), (digest,), synthetic_recipe="invalid literal Matroska bytes")
    path.with_suffix(".dataset.json").write_text(json.dumps(manifest.to_dict()), encoding="utf-8")
    video = VideoRecord("corrupt-contract", path, VideoMetadata(1., 10., 1, 32, 24), digest)
    decoder = PyAVVideoDecoder(video)
    with pytest.raises(DecodeError) as failure:
        decoder.read_request(FrameRequest(video.video_id, 0.))
    assert failure.value.issue.code == "open_failed"
    assert decoder._container is None


@pytest.mark.parametrize("source", [(0, 100, 100, 400), (0, 0, 100, 400)], indirect=True)
@pytest.mark.parametrize("operation", ["stream", "request"])
def test_duplicate_real_codec_pts_fail_before_any_frame(source, operation):
    video, _ = source
    decoder = PyAVVideoDecoder(video)
    with pytest.raises(DecodeError) as failure:
        if operation == "stream":
            next(decoder.iter_stream(max_frames=1))
        else:
            decoder.read_request(FrameRequest(video.video_id, .4))
    assert failure.value.issue.code == "duplicate_pts_unsupported"
    assert decoder._container is None


@pytest.mark.parametrize("source", [(0, 100, 200, 300)], indirect=True)
def test_actual_rotated_display_is_rejected_by_decoder_probe_and_cli(source, capsys):
    from video_intelligence.cli import main
    from video_intelligence.video.probe import probe_video

    video, _ = source
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        pytest.skip("existing FFmpeg required for actual display-transform contract")
    base = video.source_path.with_suffix(".mp4")
    rotated = video.source_path.with_name("rotated.mp4")
    for target, arguments, recipe in (
        (base, ["-i", str(video.source_path), "-c:v", "mpeg4"],
         "transcode synthetic FFV1 to MPEG4"),
        (rotated, ["-display_rotation:v:0", "90", "-i", str(base), "-c", "copy"],
         "synthetic MPEG4 remux with display_rotation=90"),
    ):
        subprocess.run([ffmpeg, "-v", "error", *arguments, str(target)],
                       check=True, capture_output=True, timeout=20)
        digest = hashlib.sha256(target.read_bytes()).hexdigest()
        manifest = DatasetManifest("rotation-contract", "synthetic", "contract", "contract",
            (target.stem,), (digest,), synthetic_recipe=recipe+"; source="+video.source_sha256)
        target.with_suffix(".dataset.json").write_text(
            json.dumps(manifest.to_dict()), encoding="utf-8"
        )
    with pytest.raises(ValueError, match="display"):
        probe_video(rotated, allow_fallback=False)
    record = replace(video, source_path=rotated, source_sha256=digest)
    with pytest.raises(DecodeError) as failure:
        PyAVVideoDecoder(record).open()
    assert failure.value.issue.code == "unsupported_display_transform"
    assert main(["inspect-frames", "--video", str(rotated), "--timestamp", "0"]) == 2
    result = json.loads(capsys.readouterr().out)
    assert result["error_code"] == "unsupported_display_transform"
    assert "frames" not in result
