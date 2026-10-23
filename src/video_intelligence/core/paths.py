"""Canonical project paths with environment-variable overrides.

The project keeps source code and small fixtures in the repository. Large, immutable
assets are resolved through this module and can live on the E: drive outside Git.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _project_root() -> Path:
    # paths.py -> core -> video_intelligence -> src -> repository root
    return Path(__file__).resolve().parents[3]


def _env_path(name: str, default: Path) -> Path:
    value = os.environ.get(name)
    return Path(value).expanduser() if value else default


@dataclass(frozen=True, slots=True)
class ProjectPaths:
    """Resolved paths used by the application and experiment reports."""

    project_root: Path
    data_root: Path
    datasets_root: Path
    models_root: Path
    cache_root: Path
    temp_root: Path
    artifacts_root: Path
    reports_root: Path

    @classmethod
    def from_environment(cls, project_root: Path | None = None) -> ProjectPaths:
        root = (project_root or _project_root()).resolve()
        # Keep defaults on E: when it exists; the project remains usable elsewhere.
        external = Path("E:/AI") if Path("E:/").exists() else root / ".external"
        return cls(
            project_root=root,
            data_root=_env_path("VI_DATA_ROOT", root / "data"),
            datasets_root=_env_path("VI_DATASETS_ROOT", external / "datasets"),
            models_root=_env_path("VI_MODELS_ROOT", external / "models"),
            cache_root=_env_path("VI_CACHE_ROOT", external / "cache"),
            temp_root=_env_path("VI_TEMP_ROOT", external / "temp"),
            # Small generated fixtures are safe in the repo. Point this to E:\\AI\\artifacts
            # explicitly for large experiment outputs via VI_ARTIFACTS_ROOT.
            artifacts_root=_env_path("VI_ARTIFACTS_ROOT", root / "artifacts"),
            reports_root=_env_path("VI_REPORTS_ROOT", root / "reports"),
        )

    def create_small_project_layout(self) -> None:
        """Create only repository-local, small directories.

        External roots are reported but intentionally not created automatically: they may
        point to a shared model or dataset store and should be provisioned deliberately.
        """

        for path in (
            self.data_root,
            self.data_root / "manifests",
            self.data_root / "fixtures",
            self.reports_root / "environment",
            self.reports_root / "experiments",
            self.reports_root / "benchmarks",
            self.reports_root / "figures",
            self.artifacts_root / "fixtures",
        ):
            path.mkdir(parents=True, exist_ok=True)


def resolve_project_paths(project_root: Path | None = None) -> ProjectPaths:
    """Return paths without creating directories."""

    return ProjectPaths.from_environment(project_root)
