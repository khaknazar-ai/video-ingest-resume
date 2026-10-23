"""Dataset provenance and fail-closed separation of synthetic and real data."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

SCHEMA_VERSION = "0.1"
REAL_QUALITY_SPLITS = frozenset({"train", "validation", "test"})


def _identifier(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value.strip() != value
        or any(character.isspace() for character in value)
    ):
        raise ValueError(f"{name} must be a non-empty identifier")
    return value


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ValueError(f"{name} must be a non-empty trimmed string")
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


class DatasetKind(StrEnum):
    SYNTHETIC = "synthetic"
    REAL = "real"


class DatasetPurpose(StrEnum):
    CONTRACT = "contract"
    QUALITY = "quality"


class DatasetManifestError(ValueError):
    """Invalid or unsafe dataset provenance."""


class SyntheticDataUsageError(DatasetManifestError):
    """Raised when synthetic data is sent to a real-quality evaluator."""


@dataclass(frozen=True, slots=True)
class DatasetManifest:
    """Immutable provenance required before a dataset can enter evaluation."""

    dataset_id: str
    kind: DatasetKind | str
    purpose: DatasetPurpose | str
    split: str
    source_video_ids: tuple[str, ...] = field(default_factory=tuple)
    source_sha256: tuple[str, ...] = field(default_factory=tuple)
    annotation_schema: str | None = None
    synthetic_recipe: str | None = None
    annotation_sha256: str | None = None
    source_group_ids: tuple[str, ...] = field(default_factory=tuple)
    source_provenance: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "dataset_id", _identifier(self.dataset_id, "dataset_id"))
        object.__setattr__(self, "split", _identifier(self.split, "split"))
        try:
            kind = DatasetKind(self.kind)
        except (TypeError, ValueError) as error:
            raise ValueError(f"unsupported dataset kind: {self.kind!r}") from error
        try:
            purpose = DatasetPurpose(self.purpose)
        except (TypeError, ValueError) as error:
            raise ValueError(f"unsupported dataset purpose: {self.purpose!r}") from error
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "purpose", purpose)
        if isinstance(self.source_video_ids, (str, bytes)) or not isinstance(
            self.source_video_ids, Sequence
        ):
            raise ValueError("source_video_ids must be a sequence")
        video_ids = tuple(_identifier(value, "source_video_id") for value in self.source_video_ids)
        if len(set(video_ids)) != len(video_ids):
            raise ValueError("source_video_ids must be unique")
        object.__setattr__(self, "source_video_ids", video_ids)
        if isinstance(self.source_sha256, (str, bytes)) or not isinstance(
            self.source_sha256, Sequence
        ):
            raise ValueError("source_sha256 must be a sequence")
        hashes = tuple(_sha256(value, "source_sha256") for value in self.source_sha256)
        if len(set(hashes)) != len(hashes):
            raise ValueError("source_sha256 values must be unique")
        if len(video_ids) != len(hashes):
            raise DatasetManifestError(
                "source_video_ids and source_sha256 must contain aligned members"
            )
        object.__setattr__(self, "source_sha256", hashes)
        if isinstance(self.source_group_ids, (str, bytes)) or not isinstance(
            self.source_group_ids, Sequence
        ):
            raise ValueError("source_group_ids must be a sequence")
        source_groups = tuple(
            _identifier(value, "source_group_id") for value in self.source_group_ids
        )
        if source_groups and len(video_ids) != len(source_groups):
            raise DatasetManifestError(
                "source_video_ids and source_group_ids must contain aligned members"
            )
        object.__setattr__(self, "source_group_ids", source_groups)
        if self.annotation_schema is not None:
            object.__setattr__(
                self,
                "annotation_schema",
                _identifier(self.annotation_schema, "annotation_schema"),
            )
        if self.synthetic_recipe is not None:
            object.__setattr__(
                self,
                "synthetic_recipe",
                _text(self.synthetic_recipe, "synthetic_recipe"),
            )
        if self.annotation_sha256 is not None:
            object.__setattr__(
                self,
                "annotation_sha256",
                _sha256(self.annotation_sha256, "annotation_sha256"),
            )
        if self.source_provenance is not None:
            object.__setattr__(
                self,
                "source_provenance",
                _text(self.source_provenance, "source_provenance"),
            )
        if kind is DatasetKind.SYNTHETIC:
            if purpose is not DatasetPurpose.CONTRACT:
                raise DatasetManifestError(
                    "synthetic datasets are contract-only and cannot be quality datasets"
                )
            if not self.synthetic_recipe:
                raise DatasetManifestError(
                    "synthetic datasets require an explicit synthetic_recipe"
                )
        if purpose is DatasetPurpose.QUALITY:
            if kind is not DatasetKind.REAL:
                raise DatasetManifestError("quality purpose requires kind=real")
            if not self.source_video_ids or not self.source_sha256:
                raise DatasetManifestError("quality datasets require source video ids and hashes")
            if not self.annotation_schema:
                raise DatasetManifestError("quality datasets require an annotation_schema")
            if not self.annotation_sha256:
                raise DatasetManifestError("quality datasets require an annotation_sha256")
            if self.split not in REAL_QUALITY_SPLITS:
                raise DatasetManifestError(
                    f"quality split must be one of {sorted(REAL_QUALITY_SPLITS)!r}"
                )
            if not self.source_group_ids:
                raise DatasetManifestError("quality datasets require source_group_ids")
            if not self.source_provenance:
                raise DatasetManifestError("quality datasets require source_provenance")

    @property
    def quality_eligible(self) -> bool:
        """Return declared eligibility; use require_quality_eligible for live admission."""
        return self.kind is DatasetKind.REAL and self.purpose is DatasetPurpose.QUALITY

    @property
    def source_members(self) -> tuple[tuple[str, str], ...]:
        """Return content-addressed source members as bound id/hash pairs."""

        return tuple(zip(self.source_video_ids, self.source_sha256, strict=True))

    @property
    def source_group_members(self) -> tuple[tuple[str, str, str], ...]:
        """Return source id/hash/group triples for split leakage checks."""

        groups = self.source_group_ids or self.source_video_ids
        return tuple(zip(self.source_video_ids, self.source_sha256, groups, strict=True))

    def require_quality_eligible(self) -> DatasetManifest:
        """Recheck source quarantine on every admission, including already loaded manifests."""
        if self.kind is DatasetKind.SYNTHETIC:
            raise SyntheticDataUsageError(
                "synthetic dataset is contract-only and cannot be used for quality metrics"
            )
        if not self.quality_eligible:
            raise DatasetManifestError(
                "dataset manifest is not eligible for real-quality evaluation"
            )
        # Lazy import keeps sidecar parsing independent of this runtime admission
        # check: the registry itself discovers manifests using the same data class.
        from video_intelligence.data.synthetic_registry import known_synthetic_hashes

        if set(self.source_sha256) & known_synthetic_hashes():
            raise SyntheticDataUsageError(
                "known synthetic source cannot be used for real-quality metrics"
            )
        return self

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": SCHEMA_VERSION,
            "dataset_id": self.dataset_id,
            "kind": self.kind.value,
            "purpose": self.purpose.value,
            "split": self.split,
            "source_video_ids": list(self.source_video_ids),
            "source_sha256": list(self.source_sha256),
            "annotation_schema": self.annotation_schema,
            "synthetic_recipe": self.synthetic_recipe,
            "annotation_sha256": self.annotation_sha256,
            "source_group_ids": list(self.source_group_ids),
            "source_provenance": self.source_provenance,
        }

    @classmethod
    def from_dict(cls, value: object) -> DatasetManifest:
        if not isinstance(value, dict):
            raise DatasetManifestError("dataset manifest must be an object")
        legacy_expected = {
            "schema_version",
            "dataset_id",
            "kind",
            "purpose",
            "split",
            "source_video_ids",
            "source_sha256",
            "annotation_schema",
            "synthetic_recipe",
        }
        current_expected = legacy_expected | {
            "annotation_sha256",
            "source_group_ids",
            "source_provenance",
        }
        if set(value) not in (legacy_expected, current_expected):
            raise DatasetManifestError("dataset manifest fields mismatch")
        if value["schema_version"] != SCHEMA_VERSION:
            raise DatasetManifestError(
                f"unsupported dataset manifest schema: {value['schema_version']!r}"
            )
        return cls(
            dataset_id=value["dataset_id"],
            kind=value["kind"],
            purpose=value["purpose"],
            split=value["split"],
            source_video_ids=value["source_video_ids"],
            source_sha256=value["source_sha256"],
            annotation_schema=value["annotation_schema"],
            synthetic_recipe=value["synthetic_recipe"],
            annotation_sha256=value.get("annotation_sha256"),
            source_group_ids=value.get("source_group_ids", ()),
            source_provenance=value.get("source_provenance"),
        )


def write_dataset_manifest(
    manifest: DatasetManifest,
    path: str | Path,
    *,
    overwrite: bool = False,
) -> Path:
    """Persist a small provenance manifest atomically; never copies dataset assets."""

    if not isinstance(manifest, DatasetManifest):
        raise TypeError("manifest must be a DatasetManifest")
    destination = Path(path).expanduser().resolve()
    if destination.exists() and not overwrite:
        raise FileExistsError(f"dataset manifest already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(manifest.to_dict(), indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        if overwrite:
            os.replace(temporary_name, destination)
        else:
            # A hard link is an atomic create-if-absent operation on the same
            # filesystem. Unlike exists()+replace, a concurrent writer wins
            # the destination race without being clobbered by this publisher.
            try:
                os.link(temporary_name, destination)
            except FileExistsError as error:
                raise FileExistsError(
                    f"dataset manifest already exists; pass overwrite=True: {destination}"
                ) from error
            # Keep the published destination inode stable while retiring the
            # private temporary link. This also makes a writer that replaces
            # the destination path concurrently observable without allowing
            # our cleanup operation to replace that writer's path.
            alias_descriptor, alias_name = tempfile.mkstemp(
                prefix=f".{destination.name}.commit-", dir=destination.parent
            )
            os.close(alias_descriptor)
            os.unlink(alias_name)
            try:
                os.link(destination, alias_name)
                os.replace(temporary_name, alias_name)
            finally:
                try:
                    os.unlink(alias_name)
                except FileNotFoundError:
                    pass
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise
    return destination


def load_dataset_manifest(path: str | Path) -> DatasetManifest:
    source = Path(path).expanduser().resolve()
    try:
        payload = json.loads(
            source.read_text(encoding="utf-8"),
            parse_constant=_reject_json_constant,
        )
    except json.JSONDecodeError as error:
        raise DatasetManifestError(f"invalid dataset manifest JSON: {source}") from error
    return DatasetManifest.from_dict(payload)


def require_real_quality(manifest: DatasetManifest) -> DatasetManifest:
    """Fail closed before metrics if the caller did not provide a real quality dataset."""

    if not isinstance(manifest, DatasetManifest):
        raise TypeError("manifest must be a DatasetManifest")
    return manifest.require_quality_eligible()


def require_synthetic_contract(asset_path: str | Path) -> DatasetManifest:
    """Allow fixture algorithms only for a generated asset with a verified sidecar."""

    asset = Path(asset_path).expanduser().resolve()
    candidates = (
        asset.with_suffix(".dataset.json"),
        asset.parent / "synthetic_fixture.dataset.json",
    )
    manifests = tuple(
        load_dataset_manifest(candidate) for candidate in candidates if candidate.is_file()
    )
    if not manifests:
        raise DatasetManifestError(
            f"synthetic contract manifest is required beside fixture asset: {asset}"
        )
    if len(manifests) > 1 and manifests[0] != manifests[1]:
        raise DatasetManifestError("multiple conflicting dataset manifests found for fixture asset")
    manifest = manifests[0]
    if (
        manifest.kind is not DatasetKind.SYNTHETIC
        or manifest.purpose is not DatasetPurpose.CONTRACT
    ):
        raise DatasetManifestError("fixture asset must have kind=synthetic and purpose=contract")
    if not asset.is_file():
        raise FileNotFoundError(f"fixture asset is missing: {asset}")
    from video_intelligence.video.manifest import sha256_file

    if sha256_file(asset) not in manifest.source_sha256:
        raise DatasetManifestError("fixture asset hash does not match its dataset manifest")
    return manifest


def _reject_json_constant(value: str) -> None:
    raise DatasetManifestError(f"non-finite JSON constant is not allowed: {value}")
