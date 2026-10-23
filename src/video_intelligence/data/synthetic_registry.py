"""Content-addressed quarantine for generated synthetic video sources."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

from video_intelligence.core.paths import resolve_project_paths
from video_intelligence.data.dataset import (
    DatasetKind,
    DatasetManifestError,
    load_dataset_manifest,
)

REGISTRY_SCHEMA = "video_intelligence.synthetic_hash_registry.v1"
REGISTRY_FILENAME = "synthetic_hash_registry.json"
_REGISTERED_HASHES: set[str] = set()


def synthetic_registry_path() -> Path:
    """Return the configured small registry location without creating it."""

    configured = os.environ.get("VI_SYNTHETIC_HASH_REGISTRY")
    if configured:
        return Path(configured).expanduser().resolve()
    return resolve_project_paths().artifacts_root / REGISTRY_FILENAME


def _validate_hash(value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or value != value.lower()
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise DatasetManifestError("synthetic registry contains an invalid SHA-256 hash")
    return value


def _load_registry(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise DatasetManifestError(f"invalid synthetic hash registry: {path}") from error
    if not isinstance(payload, dict) or set(payload) != {"schema", "source_sha256"}:
        raise DatasetManifestError(f"synthetic hash registry fields mismatch: {path}")
    if payload["schema"] != REGISTRY_SCHEMA:
        raise DatasetManifestError(f"unsupported synthetic hash registry schema: {path}")
    values = payload["source_sha256"]
    if isinstance(values, (str, bytes)) or not isinstance(values, list):
        raise DatasetManifestError("synthetic registry source_sha256 must be a list")
    hashes = {_validate_hash(value) for value in values}
    if len(hashes) != len(values):
        raise DatasetManifestError("synthetic registry source_sha256 must be unique")
    return hashes


def _discover_fixture_hashes() -> set[str]:
    """Discover retained fixture manifests under the configured artifact root."""

    root = (resolve_project_paths().artifacts_root / "fixtures").resolve()
    if not root.is_dir():
        return set()
    discovered: set[str] = set()
    for sidecar in root.rglob("*.dataset.json"):
        try:
            manifest = load_dataset_manifest(sidecar)
        except (DatasetManifestError, OSError, UnicodeError):
            continue
        if manifest.kind is DatasetKind.SYNTHETIC:
            discovered.update(manifest.source_sha256)
    return discovered


def known_synthetic_hashes(*, source_path: str | Path | None = None) -> frozenset[str]:
    """Return known synthetic hashes, independent of source path or filename."""

    hashes = set(_REGISTERED_HASHES)
    hashes.update(_load_registry(synthetic_registry_path()))
    hashes.update(_discover_fixture_hashes())
    if source_path is not None:
        asset = Path(source_path).expanduser().resolve()
        for candidate in (
            asset.with_suffix(".dataset.json"),
            asset.parent / "synthetic_fixture.dataset.json",
        ):
            if not candidate.is_file():
                continue
            try:
                manifest = load_dataset_manifest(candidate)
            except (DatasetManifestError, OSError, UnicodeError):
                continue
            if manifest.kind is DatasetKind.SYNTHETIC:
                hashes.update(manifest.source_sha256)
    return frozenset(hashes)


def register_synthetic_hashes(
    hashes: tuple[str, ...] | list[str] | set[str],
    *,
    persist_path: str | Path | None = None,
) -> None:
    """Register hashes in memory and optionally persist the small registry."""

    validated = {_validate_hash(value) for value in hashes}
    _REGISTERED_HASHES.update(validated)
    if persist_path is None or not validated:
        return
    destination = Path(persist_path).expanduser().resolve()
    existing = _load_registry(destination)
    payload = {
        "schema": REGISTRY_SCHEMA,
        "source_sha256": sorted(existing | validated),
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, destination)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise
