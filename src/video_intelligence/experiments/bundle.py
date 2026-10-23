"""Strict, content-addressed experiment bundle manifest utilities."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from video_intelligence.data.dataset import (
    DatasetKind,
    load_dataset_manifest,
    require_synthetic_contract,
)
from video_intelligence.data.synthetic_registry import known_synthetic_hashes
from video_intelligence.models import ModelSpec
from video_intelligence.video.manifest import sha256_file

SCHEMA_VERSION = "0.1"
_FILE_KINDS = frozenset({"code", "config", "video", "query", "dataset_manifest"})
_MODEL_SPEC_FIELDS = frozenset(ModelSpec.__dataclass_fields__)


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _identifier(value: object, name: str) -> str:
    value = _text(value, name)
    if any(character.isspace() for character in value) or len(value) > 256:
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


def _relative_path(value: object, name: str = "relative_path") -> str:
    value = _text(value, name)
    path = PurePosixPath(value.replace("\\", "/"))
    if path.is_absolute() or "." in path.parts or ".." in path.parts:
        raise ValueError(f"{name} must be a normalized relative path")
    return path.as_posix()


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _non_negative_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _positive_float(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a positive number")
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"{name} must be a finite positive number")
    return result


@dataclass(frozen=True, slots=True)
class BundleFile:
    """A transferred file reference; the bundle never embeds file contents."""

    kind: str
    relative_path: str
    sha256: str
    size_bytes: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", _identifier(self.kind, "kind"))
        if self.kind not in _FILE_KINDS:
            raise ValueError(f"unsupported bundle file kind: {self.kind!r}")
        object.__setattr__(self, "relative_path", _relative_path(self.relative_path))
        object.__setattr__(self, "sha256", _sha256(self.sha256, "sha256"))
        object.__setattr__(
            self,
            "size_bytes",
            _non_negative_int(self.size_bytes, "size_bytes"),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "relative_path": self.relative_path,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
        }

    @classmethod
    def from_dict(cls, value: object) -> BundleFile:
        if not isinstance(value, dict):
            raise ValueError("bundle file must be an object")
        expected = {"kind", "relative_path", "sha256", "size_bytes"}
        unknown = set(value) - expected
        if unknown or set(value) != expected:
            raise ValueError(f"bundle file fields mismatch: {sorted(unknown)!r}")
        return cls(**value)


@dataclass(frozen=True, slots=True)
class BundleJob:
    """A planned bounded job, listed before any external compute starts."""

    job_id: str
    stage: str
    config_sha256: str
    input_paths: tuple[str, ...] = ()
    model_ids: tuple[str, ...] = ()
    payload: dict[str, object] | None = None

    def __post_init__(self) -> None:
        for name in ("job_id", "stage"):
            object.__setattr__(self, name, _identifier(getattr(self, name), name))
        object.__setattr__(self, "config_sha256", _sha256(self.config_sha256, "config_sha256"))
        paths = tuple(_relative_path(path, "input_path") for path in self.input_paths)
        object.__setattr__(self, "input_paths", paths)
        model_ids = tuple(_identifier(model_id, "model_id") for model_id in self.model_ids)
        object.__setattr__(self, "model_ids", model_ids)
        payload = {} if self.payload is None else self.payload
        if not isinstance(payload, dict):
            raise ValueError("job payload must be a JSON object")
        if _contains_ground_truth(payload):
            raise ValueError("ground-truth payloads are not allowed in worker jobs")
        try:
            encoded = json.dumps(payload, ensure_ascii=False, allow_nan=False, sort_keys=True)
        except (TypeError, ValueError) as error:
            raise ValueError("job payload must contain finite JSON values") from error
        # Nested values must not remain aliased to a caller's annotation objects.
        object.__setattr__(self, "payload", json.loads(encoded))

    def to_dict(self) -> dict[str, object]:
        if _contains_ground_truth(self.payload):
            raise ValueError("ground-truth payloads are not allowed in worker jobs")
        return {
            "job_id": self.job_id,
            "stage": self.stage,
            "config_sha256": self.config_sha256,
            "input_paths": list(self.input_paths),
            "model_ids": list(self.model_ids),
            "payload": self.payload,
        }

    @classmethod
    def from_dict(cls, value: object) -> BundleJob:
        if not isinstance(value, dict):
            raise ValueError("bundle job must be an object")
        expected = {"job_id", "stage", "config_sha256", "input_paths", "model_ids", "payload"}
        if set(value) != expected:
            raise ValueError("bundle job fields mismatch")
        return cls(
            job_id=value["job_id"],
            stage=value["stage"],
            config_sha256=value["config_sha256"],
            input_paths=tuple(value["input_paths"]),
            model_ids=tuple(value["model_ids"]),
            payload=value["payload"],
        )


@dataclass(frozen=True, slots=True)
class BundleCommand:
    """A bounded command declaration, not an instruction to execute it."""

    name: str
    argv: tuple[str, ...]
    timeout_seconds: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _identifier(self.name, "name"))
        argv = tuple(_text(argument, "argv") for argument in self.argv)
        if not argv:
            raise ValueError("argv must contain at least one argument")
        object.__setattr__(self, "argv", argv)
        object.__setattr__(
            self,
            "timeout_seconds",
            _positive_float(self.timeout_seconds, "timeout_seconds"),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "argv": list(self.argv),
            "timeout_seconds": self.timeout_seconds,
        }

    @classmethod
    def from_dict(cls, value: object) -> BundleCommand:
        if not isinstance(value, dict) or set(value) != {"name", "argv", "timeout_seconds"}:
            raise ValueError("bundle command fields mismatch")
        return cls(
            name=value["name"],
            argv=tuple(value["argv"]),
            timeout_seconds=value["timeout_seconds"],
        )


@dataclass(frozen=True, slots=True)
class ExperimentBundle:
    """Content-addressed manifest for a reproducible, bounded experiment run."""

    schema_version: str
    bundle_id: str
    code: tuple[BundleFile, ...]
    configs: tuple[BundleFile, ...]
    inputs: tuple[BundleFile, ...]
    model_specs: tuple[ModelSpec, ...]
    jobs: tuple[BundleJob, ...]
    output_schema: str
    commands: tuple[BundleCommand, ...]
    package_constraints: tuple[str, ...]
    deadline_seconds: float | None

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"unsupported bundle schema: {self.schema_version!r}")
        object.__setattr__(self, "bundle_id", _identifier(self.bundle_id, "bundle_id"))
        object.__setattr__(self, "output_schema", _text(self.output_schema, "output_schema"))
        object.__setattr__(
            self,
            "package_constraints",
            tuple(_text(item, "package_constraint") for item in self.package_constraints),
        )
        if self.deadline_seconds is not None:
            object.__setattr__(
                self,
                "deadline_seconds",
                _positive_float(self.deadline_seconds, "deadline_seconds"),
            )
        all_files = (*self.code, *self.configs, *self.inputs)
        for item in all_files:
            if not isinstance(item, BundleFile):
                raise ValueError("bundle file collections must contain BundleFile values")
        paths = [item.relative_path for item in all_files]
        if len(paths) != len(set(paths)):
            raise ValueError("bundle file paths must be unique")
        known_paths = set(paths)
        for spec in self.model_specs:
            if not isinstance(spec, ModelSpec):
                raise ValueError("model_specs must contain ModelSpec values")
        model_ids = [spec.model_id for spec in self.model_specs]
        if len(model_ids) != len(set(model_ids)):
            raise ValueError("model_specs must have unique model IDs")
        for job in self.jobs:
            if not isinstance(job, BundleJob):
                raise ValueError("jobs must contain BundleJob values")
            missing = set(job.input_paths) - known_paths
            if missing:
                raise ValueError(
                    f"job references files outside transfer manifest: {sorted(missing)!r}"
                )
            missing_models = set(job.model_ids) - set(model_ids)
            if missing_models:
                raise ValueError(f"job references undeclared models: {sorted(missing_models)!r}")
        for command in self.commands:
            if not isinstance(command, BundleCommand):
                raise ValueError("commands must contain BundleCommand values")

    @property
    def transfer_manifest(self) -> tuple[BundleFile, ...]:
        return tuple(
            sorted(
                (*self.code, *self.configs, *self.inputs),
                key=lambda item: item.relative_path,
            )
        )

    def _body(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "code": [item.to_dict() for item in self.code],
            "configs": [item.to_dict() for item in self.configs],
            "inputs": [item.to_dict() for item in self.inputs],
            "model_specs": [spec.to_dict() for spec in self.model_specs],
            "jobs": [job.to_dict() for job in self.jobs],
            "output_schema": self.output_schema,
            "commands": [command.to_dict() for command in self.commands],
            "package_constraints": list(self.package_constraints),
            "deadline_seconds": self.deadline_seconds,
            "transfer_manifest": [item.to_dict() for item in self.transfer_manifest],
        }

    def to_dict(self) -> dict[str, object]:
        payload = self._body()
        payload["bundle_id"] = self.bundle_id
        return payload

    def verify_files(self, root: str | Path) -> None:
        """Verify every referenced file in-place without copying or deleting anything."""

        base = Path(root).expanduser().resolve()
        for item in self.transfer_manifest:
            candidate = (base / PurePosixPath(item.relative_path)).resolve()
            try:
                candidate.relative_to(base)
            except ValueError as error:
                raise ValueError(f"bundle file escapes root: {item.relative_path}") from error
            if not candidate.is_file():
                raise FileNotFoundError(f"bundle file is missing: {candidate}")
            if candidate.stat().st_size != item.size_bytes:
                raise ValueError(f"bundle size mismatch: {item.relative_path}")
            if sha256_file(candidate) != item.sha256:
                raise ValueError(f"bundle hash mismatch: {item.relative_path}")

    @classmethod
    def create(
        cls,
        *,
        code: Sequence[BundleFile],
        configs: Sequence[BundleFile],
        inputs: Sequence[BundleFile],
        model_specs: Sequence[ModelSpec],
        jobs: Sequence[BundleJob],
        output_schema: str,
        commands: Sequence[BundleCommand],
        package_constraints: Sequence[str] = (),
        deadline_seconds: float | None = None,
    ) -> ExperimentBundle:
        bundle = cls(
            schema_version=SCHEMA_VERSION,
            bundle_id="unassigned",
            code=tuple(code),
            configs=tuple(configs),
            inputs=tuple(inputs),
            model_specs=tuple(model_specs),
            jobs=tuple(jobs),
            output_schema=output_schema,
            commands=tuple(commands),
            package_constraints=tuple(package_constraints),
            deadline_seconds=deadline_seconds,
        )
        canonical = json.dumps(bundle._body(), ensure_ascii=False, allow_nan=False, sort_keys=True)
        digest = hashlib.sha256(canonical.encode()).hexdigest()[:24]
        object.__setattr__(bundle, "bundle_id", f"bundle_{digest}")
        return bundle

    @classmethod
    def from_dict(cls, value: object) -> ExperimentBundle:
        if not isinstance(value, dict):
            raise ValueError("experiment bundle must be an object")
        expected = {
            "schema_version",
            "bundle_id",
            "code",
            "configs",
            "inputs",
            "model_specs",
            "jobs",
            "output_schema",
            "commands",
            "package_constraints",
            "deadline_seconds",
            "transfer_manifest",
        }
        if set(value) != expected:
            raise ValueError("experiment bundle fields mismatch")
        model_specs = []
        for raw in value["model_specs"]:
            if not isinstance(raw, dict) or set(raw) != _MODEL_SPEC_FIELDS:
                raise ValueError("model spec fields mismatch")
            model_specs.append(ModelSpec(**raw))
        bundle = cls(
            schema_version=value["schema_version"],
            bundle_id=value["bundle_id"],
            code=tuple(BundleFile.from_dict(item) for item in value["code"]),
            configs=tuple(BundleFile.from_dict(item) for item in value["configs"]),
            inputs=tuple(BundleFile.from_dict(item) for item in value["inputs"]),
            model_specs=tuple(model_specs),
            jobs=tuple(BundleJob.from_dict(item) for item in value["jobs"]),
            output_schema=value["output_schema"],
            commands=tuple(BundleCommand.from_dict(item) for item in value["commands"]),
            package_constraints=tuple(value["package_constraints"]),
            deadline_seconds=value["deadline_seconds"],
        )
        if value["transfer_manifest"] != [item.to_dict() for item in bundle.transfer_manifest]:
            raise ValueError("transfer_manifest does not match bundle files")
        canonical = json.dumps(bundle._body(), ensure_ascii=False, allow_nan=False, sort_keys=True)
        expected_id = f"bundle_{hashlib.sha256(canonical.encode()).hexdigest()[:24]}"
        if bundle.bundle_id != expected_id:
            raise ValueError("bundle_id does not match canonical bundle content")
        return bundle


def _file_entry(root: Path, path: str | Path, *, kind: str) -> BundleFile:
    base = root.expanduser().resolve()
    candidate = Path(path).expanduser().resolve()
    try:
        relative = candidate.relative_to(base).as_posix()
    except ValueError as error:
        raise ValueError(f"bundle path is outside root: {candidate}") from error
    if not candidate.is_file():
        raise FileNotFoundError(f"bundle path is not a file: {candidate}")
    return BundleFile(kind, relative, sha256_file(candidate), candidate.stat().st_size)


def _collect_files(root: Path, paths: Iterable[str | Path], *, kind: str) -> tuple[BundleFile, ...]:
    return tuple(_file_entry(root, path, kind=kind) for path in paths)


def _reject_known_synthetic_input(path: str | Path) -> None:
    asset = Path(path).expanduser().resolve()
    if not asset.is_file():
        raise FileNotFoundError(f"bundle path is not a file: {asset}")
    if sha256_file(asset) in known_synthetic_hashes(source_path=asset):
        raise ValueError(
            "synthetic dataset input is blocked by default; pass "
            "allow_synthetic_inputs=True for contract-only bundles"
        )
    manifest_path = asset.with_suffix(".dataset.json")
    if not manifest_path.is_file():
        return
    manifest = load_dataset_manifest(manifest_path)
    if manifest.kind is DatasetKind.SYNTHETIC:
        raise ValueError(
            "synthetic dataset input is blocked by default; pass "
            "allow_synthetic_inputs=True for contract-only bundles"
        )


def _adjacent_dataset_manifest(path: str | Path) -> Path | None:
    asset = Path(path).expanduser().resolve()
    candidate = asset.with_suffix(".dataset.json")
    return candidate if candidate.is_file() else None


def _contains_ground_truth(payload: object) -> bool:
    if isinstance(payload, (list, tuple)):
        return any(_contains_ground_truth(item) for item in payload)
    if not isinstance(payload, dict):
        return False
    keys = set(payload)
    if (
        "ground_truth" in keys
        or "relevant_row_ids" in keys
        or "forbidden_row_ids" in keys
        or {"objects", "events", "queries"} <= keys
        or {"frame_ref", "boxes"} <= keys
        or ("schema" in keys and "videos" in keys and bool({"queries", "frames"} & keys))
    ):
        return True
    return any(_contains_ground_truth(value) for value in payload.values())


def _looks_like_ground_truth(path: Path) -> bool:
    if path.suffix.lower() not in {".json", ".jsonl"}:
        return False
    try:
        if path.suffix.lower() == ".jsonl":
            with path.open("r", encoding="utf-8") as stream:
                seen = False
                for line in stream:
                    if not line.strip():
                        continue
                    seen = True
                    if _contains_ground_truth(
                        json.loads(line, parse_constant=_reject_json_constant)
                    ):
                        return True
                if not seen:
                    raise ValueError("empty JSONL worker input")
            return False
        payload = json.loads(path.read_text(encoding="utf-8"), parse_constant=_reject_json_constant)
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"JSON worker input cannot be safely classified: {path}") from error
    return _contains_ground_truth(payload)


def build_experiment_bundle(
    root: str | Path,
    *,
    code_paths: Sequence[str | Path],
    config_paths: Sequence[str | Path],
    input_paths: Sequence[tuple[str, str | Path]],
    model_specs: Sequence[ModelSpec],
    jobs: Sequence[BundleJob],
    output_schema: str,
    commands: Sequence[BundleCommand],
    package_constraints: Sequence[str] = (),
    deadline_seconds: float | None = None,
    allow_synthetic_inputs: bool = False,
) -> ExperimentBundle:
    """Build a manifest from explicit files; no source or asset is copied."""

    base = Path(root).expanduser().resolve()
    if not isinstance(allow_synthetic_inputs, bool):
        raise ValueError("allow_synthetic_inputs must be a boolean")
    inputs: list[BundleFile] = []
    manifest_paths: set[Path] = set()
    for kind, path in input_paths:
        if kind not in {"video", "query"}:
            raise ValueError("input kind must be video or query")
        candidate = Path(path).expanduser().resolve()
        known_synthetic = candidate.is_file() and (
            sha256_file(candidate) in known_synthetic_hashes(source_path=candidate)
        )
        if known_synthetic:
            if not allow_synthetic_inputs:
                _reject_known_synthetic_input(path)
            require_synthetic_contract(path)
        manifest_path = _adjacent_dataset_manifest(path)
        if manifest_path is not None:
            manifest = load_dataset_manifest(manifest_path)
            if manifest.kind is DatasetKind.SYNTHETIC:
                if not allow_synthetic_inputs:
                    _reject_known_synthetic_input(path)
                require_synthetic_contract(path)

        if _looks_like_ground_truth(candidate):
            raise ValueError("ground-truth payloads are not allowed in worker inputs")
        inputs.append(_file_entry(base, path, kind=kind))
        if manifest_path is not None and manifest_path not in manifest_paths:
            inputs.append(_file_entry(base, manifest_path, kind="dataset_manifest"))
            manifest_paths.add(manifest_path)
    return ExperimentBundle.create(
        code=_collect_files(base, code_paths, kind="code"),
        configs=_collect_files(base, config_paths, kind="config"),
        inputs=tuple(inputs),
        model_specs=model_specs,
        jobs=jobs,
        output_schema=output_schema,
        commands=commands,
        package_constraints=package_constraints,
        deadline_seconds=deadline_seconds,
    )


def write_bundle(bundle: ExperimentBundle, path: str | Path, *, overwrite: bool = False) -> Path:
    destination = Path(path).expanduser().resolve()
    if destination.exists() and not overwrite:
        raise FileExistsError(f"bundle already exists; pass overwrite=True: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(bundle.to_dict(), indent=2, ensure_ascii=False, allow_nan=False) + "\n"
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
            os.link(temporary_name, destination)
            os.unlink(temporary_name)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise
    return destination


def load_bundle(path: str | Path) -> ExperimentBundle:
    source = Path(path).expanduser().resolve()
    payload = json.loads(source.read_text(encoding="utf-8"), parse_constant=_reject_json_constant)
    return ExperimentBundle.from_dict(payload)


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant is not allowed: {value}")
