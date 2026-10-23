"""Real codec inspection contracts; generated pixels are never quality evidence."""

import hashlib
import json
from fractions import Fraction

import numpy as np
import pytest

from video_intelligence.cli import main
from video_intelligence.data.dataset import DatasetManifest
from video_intelligence.video import source_frames


@pytest.fixture
def contract_source(tmp_path):
    av = pytest.importorskip("av")
    source = tmp_path / "source-inspection-contract.mkv"
    pixels = np.full((24, 32, 3), 100, dtype=np.uint8)
    with av.open(str(source), mode="w", format="matroska") as container:
        stream = container.add_stream("ffv1", rate=10)
        stream.width, stream.height, stream.pix_fmt = 32, 24, "bgr0"
        for pts in (0, 100, 200):
            frame = av.VideoFrame.from_ndarray(pixels, format="rgb24")
            frame.pts, frame.time_base = pts, Fraction(1, 1000)
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    manifest = DatasetManifest("source-inspection-contract", "synthetic", "contract", "contract",
        ("source-inspection-contract",), (digest,), synthetic_recipe=
        "ffv1/matroska/bgr0, 32x24 RGB=100; requested PTS0/100/200 at1/1000")
    source.with_suffix(".dataset.json").write_text(json.dumps(manifest.to_dict()), encoding="utf-8")
    return source, pixels, digest


def test_cli_inspects_actual_frames_without_publishing_or_mutating(contract_source, capsys):
    source, pixels, digest = contract_source
    names = set(source.parent.iterdir())
    assert main(["inspect-frames", "--video", str(source), "--video-id", "contract",
                 "--timestamp", "0.1", "--timestamp", "0.2"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["source_sha256"] == digest == hashlib.sha256(source.read_bytes()).hexdigest()
    assert result["inference"] is result["published_evidence"] is False
    assert result["quality_status"] == "unmeasured"
    assert [r["frame_ref"]["pts"] for r in result["frames"]] == [100, 200]
    for row in result["frames"]:
        assert row["frame_ref"]["timestamp_origin"] == "first_displayed_pts"
        assert row["frame_ref"]["source_sha256"] == digest
        assert row["frame_ref"]["source_path"] is None
        assert row["rgb_sha256"] == hashlib.sha256(pixels.tobytes()).hexdigest()
    assert set(source.parent.iterdir()) == names


@pytest.mark.parametrize("timestamps", [[], [-1], [float("nan")], [float("inf")],
                                          [True], [0] * 9])
def test_invalid_budget_rejected_before_source_access(tmp_path, timestamps):
    with pytest.raises(ValueError, match="timestamps"):
        source_frames.inspect_source_frames(tmp_path / "does-not-exist", timestamps)


def test_source_change_returns_no_accepted_partial_output(contract_source, monkeypatch):
    source, _, digest = contract_source
    hashes = iter([digest, "0" * 64])
    monkeypatch.setattr(source_frames, "_source_hash", lambda *args: next(hashes))
    # Inject the hash race inside the worker body, not the spawned parent process.
    with pytest.raises(ValueError, match="changed during decode"):
        source_frames._inspect_source_frames(source, [0.1])


def test_cli_out_of_range_has_structured_issue_no_partial_frames(contract_source, capsys):
    source, _, _ = contract_source
    assert main(["inspect-frames", "--video", str(source), "--timestamp", "0.1",
                 "--timestamp", "99", "--selection-policy", "at_or_after"]) == 2
    result = json.loads(capsys.readouterr().out)
    assert result["error_code"] == result["decode_issue"]["code"] == "out_of_range"
    assert "frames" not in result


def test_cli_tiny_deadline_returns_timeout(contract_source, capsys):
    source, _, _ = contract_source
    assert main(["inspect-frames", "--video", str(source), "--timestamp", "0.1",
                 "--timeout-seconds", "0.001"]) == 2
    result = json.loads(capsys.readouterr().out)
    assert result["error_code"] == "inspection_timeout"
    assert "frames" not in result


@pytest.mark.parametrize("options", [{"max_decode_frames": 0}, {"max_decode_frames": True},
                                      {"timeout_seconds": float("inf")},
                                      {"timeout_seconds": 61}, {"selection_policy": "previous"}])
def test_invalid_options_rejected_before_worker_start(tmp_path, options):
    with pytest.raises(ValueError):
        source_frames.inspect_source_frames(tmp_path/"missing", [0.], **options)


def test_stale_preflight_hash_is_rejected_before_decode(contract_source):
    source, _, _ = contract_source
    with pytest.raises(ValueError, match="changed before decode"):
        source_frames._inspect_source_frames(source, [0.1], expected_source_sha256="0"*64)


def test_cli_rejects_unmapped_display_before_worker(contract_source, monkeypatch, capsys):
    source, _, _ = contract_source
    def reject(*args, **kwargs):
        raise source_frames.UnsupportedVideoTransformError("contract nonidentity display")
    monkeypatch.setattr(source_frames, "probe_video", reject)
    assert main(["inspect-frames", "--video", str(source), "--timestamp", "0.1"]) == 2
    result = json.loads(capsys.readouterr().out)
    assert result["error_code"] == "unsupported_display_transform"
    assert "frames" not in result
