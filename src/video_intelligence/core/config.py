"""Minimal configuration loading for the foundation slice."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from numbers import Real
from pathlib import Path

from .paths import ProjectPaths, resolve_project_paths


@dataclass(frozen=True, slots=True)
class FoundationConfig:
    """Small, dependency-light config used by M0 scripts."""

    sample_fps: float = 1.0
    max_sample_frames: int | None = None
    decoder_backend: str = "ffprobe_then_opencv"

    def __post_init__(self) -> None:
        if isinstance(self.sample_fps, bool) or not isinstance(self.sample_fps, Real):
            raise ValueError("sample_fps must be a real number")
        sample_fps = float(self.sample_fps)
        if not math.isfinite(sample_fps) or sample_fps <= 0:
            raise ValueError("sample_fps must be a finite positive number")
        object.__setattr__(self, "sample_fps", sample_fps)
        if self.max_sample_frames is not None and (
            isinstance(self.max_sample_frames, bool)
            or not isinstance(self.max_sample_frames, int)
            or self.max_sample_frames <= 0
            or self.max_sample_frames > 1_000_000
        ):
            raise ValueError("max_sample_frames must be an integer in [1, 1000000]")
        if not isinstance(self.decoder_backend, str) or not self.decoder_backend.strip():
            raise ValueError("decoder_backend must be a non-empty string")

    @classmethod
    def from_json(cls, path: Path) -> FoundationConfig:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("foundation config must be a JSON object")
        unknown = set(data) - {"sample_fps", "max_sample_frames", "decoder_backend"}
        if unknown:
            raise ValueError(f"foundation config contains unknown fields: {sorted(unknown)!r}")
        return cls(**data)


def default_config_path(paths: ProjectPaths | None = None) -> Path:
    resolved = paths or resolve_project_paths()
    return resolved.project_root / "configs" / "ingestion" / "default.json"
