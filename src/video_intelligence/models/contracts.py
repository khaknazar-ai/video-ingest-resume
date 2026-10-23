"""Typed model lifecycle contracts with no heavyweight ML dependencies."""

from __future__ import annotations

import hashlib
import json
import math
import stat
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from dataclasses import dataclass, fields, is_dataclass
from enum import StrEnum
from pathlib import Path
from threading import Event, Lock, Thread
from typing import Any, Generic, TypeVar

InputT = TypeVar("InputT")
OutputT = TypeVar("OutputT")
WEIGHTS_DIRECTORY_HASH_SCHEMA = "video_intelligence.weights_directory.v2"


def _identifier(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value.strip() != value
        or any(character.isspace() for character in value)
        or len(value) > 256
    ):
        raise ValueError(f"{name} must be a non-empty identifier")
    return value


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _sha256(value: object, name: str) -> str | None:
    if value is None:
        return None
    if (
        not isinstance(value, str)
        or len(value) != 64
        or value != value.lower()
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 hex digest or null")
    return value


def _weights_inventory(location: Path) -> tuple[Path, ...]:
    """List regular members without following nested links or Windows junctions."""

    files: list[Path] = []
    pending = [location]
    while pending:
        for item in pending.pop().iterdir():
            info = item.lstat()
            if stat.S_ISLNK(info.st_mode) or (
                getattr(info, "st_file_attributes", 0)
                & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
            ):
                raise OSError("weights directory contains a nested symlink or junction")
            if stat.S_ISDIR(info.st_mode):
                pending.append(item)
            elif stat.S_ISREG(info.st_mode):
                files.append(item)
            else:
                raise OSError("weights directory contains a non-regular member")
    return tuple(sorted(files, key=lambda item: item.relative_to(location).as_posix()))


def _hash_weights_location(location: Path) -> str:
    digest = hashlib.sha256()
    if location.is_file():
        with location.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    if not location.is_dir():
        raise FileNotFoundError(f"weights_location is not a file or directory: {location}")
    files = _weights_inventory(location)
    # v1 concatenated unframed content and could alias different directory trees.
    # Do not accept that legacy digest: existing directory declarations must be
    # regenerated. Single-file SHA-256 remains the ordinary hash of its bytes.
    digest.update(WEIGHTS_DIRECTORY_HASH_SCHEMA.encode("ascii") + b"\0")
    digest.update(len(files).to_bytes(8, "big"))
    for item in files:
        relative = item.relative_to(location).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        size = item.stat().st_size
        digest.update(size.to_bytes(8, "big"))
        content_digest = hashlib.sha256()
        read_bytes = 0
        with item.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                content_digest.update(chunk)
                read_bytes += len(chunk)
        if read_bytes != size:
            raise OSError("weights member changed size while hashing")
        digest.update(content_digest.digest())
    if tuple(item.relative_to(location).as_posix() for item in _weights_inventory(location)) != (
        tuple(item.relative_to(location).as_posix() for item in files)
    ):
        raise OSError("weights directory inventory changed while hashing")
    return digest.hexdigest()


def hash_weights_location(location: str | Path) -> str:
    """Hash file bytes or framed v2 directory records; no legacy directory fallback.

    Directory records bind member count, name, size and SHA-256 of its bytes.
    Nested symlinks/junctions are rejected and member inventory is rechecked;
    this is not an atomic filesystem snapshot. Keep assets immutable during use.
    All pre-v2 directory digests must be recalculated before loading local weights.
    """

    return _hash_weights_location(Path(location).expanduser().resolve())


def _verify_declared_weights(spec: ModelSpec) -> None:
    if spec.mode is not ModelMode.INFERENCE:
        return
    if spec.weights_location is None:
        if spec.weights_sha256 is None and not spec.allow_download:
            return
        raise ModelAdapterError(
            ModelErrorCode.MODEL_UNAVAILABLE,
            "inference requires a local weights_location to verify its SHA-256",
            model_id=spec.model_id,
        )
    if spec.weights_sha256 is None:
        raise ModelAdapterError(
            ModelErrorCode.MODEL_UNAVAILABLE,
            "local inference weights require a declared weights_sha256",
            model_id=spec.model_id,
        )
    location = Path(spec.weights_location).expanduser().resolve()
    try:
        actual = _hash_weights_location(location)
    except OSError as error:
        raise ModelAdapterError(
            ModelErrorCode.MODEL_UNAVAILABLE,
            "declared weights_location could not be hashed",
            model_id=spec.model_id,
        ) from error
    if actual != spec.weights_sha256:
        raise ModelAdapterError(
            ModelErrorCode.MODEL_UNAVAILABLE,
            "declared weights SHA-256 does not match weights_location",
            model_id=spec.model_id,
        )


def _positive_int(value: object, name: str, *, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 < value <= maximum:
        raise ValueError(f"{name} must be an integer in [1, {maximum}]")
    return value


def _positive_float(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite positive number")
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"{name} must be a finite positive number")
    return result


class ModelMode(StrEnum):
    FIXTURE = "fixture"
    MOCK = "mock"
    REPLAY = "replay"
    INFERENCE = "inference"


class ModelErrorCode(StrEnum):
    MODEL_UNAVAILABLE = "model_unavailable"
    OOM = "oom"
    INVALID_OUTPUT = "invalid_output"
    CANCELLED = "cancelled"


class AdapterState(StrEnum):
    UNLOADED = "unloaded"
    LOADING = "loading"
    LOADED = "loaded"
    CLOSING = "closing"
    POISONED = "poisoned"
    CLOSED = "closed"


@dataclass(frozen=True, slots=True)
class ModelSpec:
    """Immutable model provenance and resource policy for one adapter load."""

    model_id: str
    role: str
    source: str
    revision: str
    weights_sha256: str | None = None
    weights_location: str | Path | None = None
    dependency_profile: str = "cpu"
    device: str = "cpu"
    dtype: str = "float32"
    max_image_side: int = 1024
    batch_size: int = 1
    timeout_seconds: float = 60.0
    allow_download: bool = False
    mode: ModelMode | str = ModelMode.INFERENCE

    def __post_init__(self) -> None:
        for name in ("model_id", "role", "dependency_profile", "device", "dtype"):
            object.__setattr__(self, name, _identifier(getattr(self, name), name))
        for name in ("source", "revision"):
            object.__setattr__(self, name, _text(getattr(self, name), name))
        object.__setattr__(self, "weights_sha256", _sha256(self.weights_sha256, "weights_sha256"))
        if self.weights_location is not None:
            object.__setattr__(self, "weights_location", str(self.weights_location))
            _text(self.weights_location, "weights_location")
        object.__setattr__(
            self,
            "max_image_side",
            _positive_int(self.max_image_side, "max_image_side", maximum=16384),
        )
        object.__setattr__(
            self,
            "batch_size",
            _positive_int(self.batch_size, "batch_size", maximum=1024),
        )
        object.__setattr__(
            self,
            "timeout_seconds",
            _positive_float(self.timeout_seconds, "timeout_seconds"),
        )
        if not isinstance(self.allow_download, bool):
            raise ValueError("allow_download must be a boolean")
        try:
            object.__setattr__(self, "mode", ModelMode(self.mode))
        except (TypeError, ValueError) as error:
            raise ValueError(f"unsupported model mode: {self.mode!r}") from error

    def to_dict(self) -> dict[str, object]:
        return {
            "model_id": self.model_id,
            "role": self.role,
            "source": self.source,
            "revision": self.revision,
            "weights_sha256": self.weights_sha256,
            "weights_location": self.weights_location,
            "dependency_profile": self.dependency_profile,
            "device": self.device,
            "dtype": self.dtype,
            "max_image_side": self.max_image_side,
            "batch_size": self.batch_size,
            "timeout_seconds": self.timeout_seconds,
            "allow_download": self.allow_download,
            "mode": self.mode.value,
        }


@dataclass(frozen=True, slots=True)
class ModelCapabilities:
    """Static capability report available before model weights are loaded."""

    model_id: str
    role: str
    available: bool
    mode: ModelMode | str
    supports_cpu: bool = True
    supports_cuda: bool = False
    max_batch_size: int = 1
    reason: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "model_id", _identifier(self.model_id, "model_id"))
        object.__setattr__(self, "role", _identifier(self.role, "role"))
        if not isinstance(self.available, bool):
            raise ValueError("available must be a boolean")
        for name in ("supports_cpu", "supports_cuda"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be a boolean")
        object.__setattr__(
            self,
            "max_batch_size",
            _positive_int(self.max_batch_size, "max_batch_size", maximum=1024),
        )
        if self.reason is not None:
            object.__setattr__(self, "reason", _text(self.reason, "reason"))
        try:
            object.__setattr__(self, "mode", ModelMode(self.mode))
        except (TypeError, ValueError) as error:
            raise ValueError(f"unsupported model mode: {self.mode!r}") from error

    def to_dict(self) -> dict[str, object]:
        return {
            "model_id": self.model_id,
            "role": self.role,
            "available": self.available,
            "mode": self.mode.value,
            "supports_cpu": self.supports_cpu,
            "supports_cuda": self.supports_cuda,
            "max_batch_size": self.max_batch_size,
            "reason": self.reason,
        }


@dataclass(frozen=True, slots=True)
class ResourceMeasurement:
    """Measured resource fields; unknown measurements remain explicit nulls."""

    phase: str
    duration_seconds: float
    input_count: int
    peak_rss_bytes: int | None = None
    peak_vram_bytes: int | None = None
    metadata: dict[str, object] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "phase", _identifier(self.phase, "phase"))
        object.__setattr__(
            self,
            "duration_seconds",
            _positive_float(self.duration_seconds, "duration_seconds"),
        )
        object.__setattr__(
            self,
            "input_count",
            _positive_int(self.input_count, "input_count", maximum=1_000_000),
        )
        for name in ("peak_rss_bytes", "peak_vram_bytes"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(
                    self,
                    name,
                    _positive_int(value, name, maximum=10**15),
                )
        if self.metadata is not None and not isinstance(self.metadata, dict):
            raise ValueError("metadata must be a JSON object or null")

    def to_dict(self) -> dict[str, object]:
        return {
            "phase": self.phase,
            "duration_seconds": self.duration_seconds,
            "input_count": self.input_count,
            "peak_rss_bytes": self.peak_rss_bytes,
            "peak_vram_bytes": self.peak_vram_bytes,
            "metadata": self.metadata,
        }


@dataclass(frozen=True, slots=True)
class ModelResult(Generic[OutputT]):
    """Successful typed outputs with explicit mode, provenance and measurements."""

    model_id: str
    mode: ModelMode
    outputs: tuple[OutputT, ...]
    measurement: ResourceMeasurement
    provenance: dict[str, object]

    def to_dict(self) -> dict[str, object]:
        payload = {
            "model_id": self.model_id,
            "mode": self.mode.value,
            "outputs": [_jsonable(output) for output in self.outputs],
            "measurement": self.measurement.to_dict(),
            "provenance": _jsonable(self.provenance),
        }
        json.dumps(payload, ensure_ascii=False, allow_nan=False)
        return payload


class ModelAdapterError(RuntimeError):
    """A typed adapter failure that callers can handle without parsing messages."""

    def __init__(
        self,
        code: ModelErrorCode | str,
        message: str,
        *,
        model_id: str | None = None,
    ) -> None:
        try:
            self.code = ModelErrorCode(code)
        except (TypeError, ValueError) as error:
            raise ValueError(f"unsupported model error code: {code!r}") from error
        self.message = _text(message, "message")
        self.model_id = None if model_id is None else _identifier(model_id, "model_id")
        super().__init__(self.message)

    def to_dict(self) -> dict[str, object]:
        return {"code": self.code.value, "message": self.message, "model_id": self.model_id}


class ModelAdapter(Generic[InputT, OutputT]):
    """Base lifecycle contract for lazy, resource-aware model backends."""

    def __init__(self, capabilities: ModelCapabilities) -> None:
        self._capabilities = capabilities
        self._state = AdapterState.UNLOADED
        self._spec: ModelSpec | None = None
        self._inference_lock = Lock()
        self._active_future: Future[tuple[OutputT, ...]] | None = None
        self._active_executor: ThreadPoolExecutor | None = None
        self._loading = False
        self._load_done = Event()
        self._load_done.set()
        self._closing = False
        self._needs_cleanup = False
        self._cleanup_future: Future[None] | None = None

    @property
    def state(self) -> AdapterState:
        return self._state

    @property
    def spec(self) -> ModelSpec | None:
        return self._spec

    def capabilities(self) -> ModelCapabilities:
        return self._capabilities

    def availability(self, spec: ModelSpec) -> tuple[bool, str | None]:
        """Check static and spec-specific availability without loading weights."""

        if not isinstance(spec, ModelSpec):
            raise TypeError("spec must be a ModelSpec")
        capabilities = self.capabilities()
        return capabilities.available, capabilities.reason

    def load(self, spec: ModelSpec) -> None:
        with self._inference_lock:
            if self._closing or self._state in {AdapterState.POISONED, AdapterState.CLOSED}:
                raise ModelAdapterError(
                    ModelErrorCode.MODEL_UNAVAILABLE,
                    "adapter is poisoned, closing or closed; create a fresh adapter",
                    model_id=self._capabilities.model_id,
                )
            if self._loading or self._state is AdapterState.LOADED:
                raise ValueError("adapter is already loading or loaded")
            if spec.model_id != self._capabilities.model_id or spec.role != self._capabilities.role:
                raise ValueError("model spec does not match adapter capabilities")
            if spec.mode is not self._capabilities.mode:
                raise ValueError("model spec mode does not match adapter capabilities")
            if spec.batch_size > self._capabilities.max_batch_size:
                raise ValueError("model spec batch_size exceeds adapter capability")
            if spec.device.startswith("cuda") and not self._capabilities.supports_cuda:
                raise ModelAdapterError(
                    ModelErrorCode.MODEL_UNAVAILABLE,
                    "adapter does not support the requested CUDA device",
                    model_id=spec.model_id,
                )
            if spec.device == "cpu" and not self._capabilities.supports_cpu:
                raise ModelAdapterError(
                    ModelErrorCode.MODEL_UNAVAILABLE,
                    "adapter does not support CPU execution",
                    model_id=spec.model_id,
                )
            self._loading = True
            self._load_done.clear()
            self._state = AdapterState.LOADING
            self._spec = spec
        loaded = False
        try:
            available, reason = self.availability(spec)
            if not available:
                raise ModelAdapterError(
                    ModelErrorCode.MODEL_UNAVAILABLE,
                    reason or "model backend is unavailable",
                    model_id=spec.model_id,
                )
            _verify_declared_weights(spec)
            self._needs_cleanup = True
            self._load(spec)
            loaded = True
        except MemoryError as error:
            raise ModelAdapterError(
                ModelErrorCode.OOM,
                "model load ran out of memory",
                model_id=spec.model_id,
            ) from error
        finally:
            with self._inference_lock:
                self._loading = False
                if self._closing:
                    self._state = AdapterState.POISONED
                else:
                    self._state = (
                        AdapterState.LOADED if loaded
                        else AdapterState.POISONED if self._needs_cleanup
                        else AdapterState.UNLOADED
                    )
                if not loaded and not self._needs_cleanup:
                    self._spec = None
                self._load_done.set()
        if self._closing:
            raise ModelAdapterError(
                ModelErrorCode.CANCELLED,
                "model load completed after close began; adapter remains unavailable",
                model_id=spec.model_id,
            )

    def infer(
        self,
        inputs: Sequence[InputT],
        *,
        cancel_check: Callable[[], bool] | None = None,
    ) -> ModelResult[OutputT]:
        batch = tuple(inputs)
        if not batch:
            raise ValueError("inputs must contain at least one item")
        if cancel_check is not None and cancel_check():
            raise ModelAdapterError(
                ModelErrorCode.CANCELLED,
                "inference was cancelled",
                model_id=self._capabilities.model_id,
            )
        started = time.perf_counter()
        executor: ThreadPoolExecutor | None = None
        future: Future[tuple[OutputT, ...]] | None = None

        def invoke() -> tuple[OutputT, ...]:
            return tuple(self._infer(batch))

        with self._inference_lock:
            # Admission and close's state transition must be atomic. Input
            # collection and caller callbacks happen outside this lock.
            spec = self._require_loaded()
            if len(batch) > spec.batch_size:
                raise ValueError("input count exceeds model spec batch_size")
            if self._active_future is not None and not self._active_future.done():
                raise ModelAdapterError(
                    ModelErrorCode.MODEL_UNAVAILABLE,
                    "adapter already has an active inference",
                    model_id=spec.model_id,
                )
            executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="model-infer")
            try:
                future = executor.submit(invoke)
            except BaseException:
                executor.shutdown(wait=True, cancel_futures=True)
                raise
            self._active_future = future
            self._active_executor = executor
        timed_out = False
        try:
            assert future is not None and executor is not None
            outputs = future.result(timeout=spec.timeout_seconds)
        except FutureTimeoutError as error:
            timed_out = True
            with self._inference_lock:
                if self._state is not AdapterState.CLOSED:
                    self._state = AdapterState.POISONED
            executor.shutdown(wait=False, cancel_futures=True)
            raise TimeoutError(
                f"model inference exceeded deadline of {spec.timeout_seconds:.6g} seconds; "
                "adapter was quarantined because the backend worker may still be running"
            ) from error
        except MemoryError as error:
            raise ModelAdapterError(
                ModelErrorCode.OOM,
                "inference ran out of memory",
                model_id=spec.model_id,
            ) from error
        except ModelAdapterError:
            raise
        except (TypeError, ValueError) as error:
            raise ModelAdapterError(
                ModelErrorCode.INVALID_OUTPUT,
                "adapter returned invalid output",
                model_id=spec.model_id,
            ) from error
        finally:
            if not timed_out:
                executor.shutdown(wait=False, cancel_futures=True)
                with self._inference_lock:
                    if self._active_future is future:
                        self._active_future = None
                        self._active_executor = None
        if cancel_check is not None and cancel_check():
            raise ModelAdapterError(
                ModelErrorCode.CANCELLED,
                "inference was cancelled",
                model_id=spec.model_id,
            )
        measurement = ResourceMeasurement(
            phase="inference",
            duration_seconds=max(time.perf_counter() - started, math.ulp(1.0)),
            input_count=len(batch),
        )
        return ModelResult(
            model_id=spec.model_id,
            mode=spec.mode,
            outputs=outputs,
            measurement=measurement,
            provenance=spec.to_dict(),
        )

    def close(self, *, timeout_seconds: float = 0.1) -> bool:
        """Bound this caller's wait, returning whether cleanup actually completed.

        An unfinished load/inference stays quarantined with its resources intact;
        call close again after it finishes. Python threads cannot be forcibly
        stopped. Inference workers may keep the process alive: use an external
        process deadline for heavy runs. CLOSED never means pod billing stopped.
        """

        if (
            isinstance(timeout_seconds, bool)
            or not isinstance(timeout_seconds, (int, float))
            or not math.isfinite(timeout_seconds)
            or timeout_seconds < 0
        ):
            raise ValueError("close timeout_seconds must be finite and non-negative")
        deadline = time.perf_counter() + timeout_seconds

        def remaining() -> float:
            return max(0.0, deadline - time.perf_counter())

        def quarantine() -> bool:
            with self._inference_lock:
                if self._state is AdapterState.CLOSED:
                    return True
                self._state = AdapterState.POISONED
            return False

        with self._inference_lock:
            if self._state is AdapterState.CLOSED:
                return True
            self._closing = True
            if self._state is not AdapterState.POISONED:
                self._state = AdapterState.CLOSING
            future = self._active_future
        if not self._load_done.wait(remaining()):
            return quarantine()
        if future is not None and not future.done():
            try:
                future.result(timeout=remaining())
            except FutureTimeoutError:
                return quarantine()
            except BaseException:
                pass
        with self._inference_lock:
            cleanup = self._cleanup_future
            start_cleanup = cleanup is None
            if start_cleanup:
                cleanup = Future()
                self._cleanup_future = cleanup
            executor = self._active_executor
            self._active_future = None
            self._active_executor = None
        if executor is not None:
            executor.shutdown(wait=False, cancel_futures=True)

        def finish_cleanup() -> None:
            assert cleanup is not None
            try:
                if self._needs_cleanup:
                    self._close()
            except BaseException as error:
                quarantine()
                cleanup.set_exception(error)
            else:
                with self._inference_lock:
                    self._spec = None
                    self._needs_cleanup = False
                    self._state = AdapterState.CLOSED
                cleanup.set_result(None)

        if start_cleanup:
            # A stalled subclass cleanup hook must not stall this caller either.
            Thread(target=finish_cleanup, name="model-close", daemon=True).start()
        assert cleanup is not None
        try:
            cleanup.result(timeout=remaining())
        except FutureTimeoutError:
            return quarantine()
        return True

    def _require_loaded(self) -> ModelSpec:
        if self._state is AdapterState.POISONED:
            raise ModelAdapterError(
                ModelErrorCode.MODEL_UNAVAILABLE,
                "adapter is poisoned after a timeout; create a fresh adapter",
                model_id=self._capabilities.model_id,
            )
        if self._closing:
            raise ModelAdapterError(
                ModelErrorCode.MODEL_UNAVAILABLE,
                "adapter is closing or closed; create a fresh adapter",
                model_id=self._capabilities.model_id,
            )
        if self._state is not AdapterState.LOADED or self._spec is None:
            raise ModelAdapterError(
                ModelErrorCode.MODEL_UNAVAILABLE,
                "adapter must be loaded before inference",
                model_id=self._capabilities.model_id,
            )
        return self._spec

    def _load(self, spec: ModelSpec) -> None:
        del spec

    def _infer(self, inputs: tuple[InputT, ...]) -> Iterable[OutputT]:
        del inputs
        raise ModelAdapterError(
            ModelErrorCode.MODEL_UNAVAILABLE,
            "adapter has no inference implementation",
            model_id=self._capabilities.model_id,
        )

    def _close(self) -> None:
        return None


class MockAdapter(ModelAdapter[InputT, OutputT]):
    """Explicit mock mode for contract tests; it never represents real-model quality."""

    def __init__(
        self,
        *,
        model_id: str,
        role: str,
        infer_fn: Callable[[tuple[InputT, ...]], Iterable[OutputT]],
    ) -> None:
        if not callable(infer_fn):
            raise ValueError("infer_fn must be callable")
        super().__init__(
            ModelCapabilities(
                model_id=model_id,
                role=role,
                available=True,
                mode=ModelMode.MOCK,
                max_batch_size=1024,
            )
        )
        self._infer_fn = infer_fn

    def _infer(self, inputs: tuple[InputT, ...]) -> Iterable[OutputT]:
        outputs = self._infer_fn(inputs)
        if isinstance(outputs, (str, bytes)):
            raise TypeError("mock output must be an iterable of domain outputs")
        return outputs


class UnavailableModelAdapter(ModelAdapter[Any, Any]):
    """Explicit no-model adapter used when weights/backend are not installed."""

    def __init__(self, *, model_id: str, role: str, reason: str) -> None:
        super().__init__(
            ModelCapabilities(
                model_id=model_id,
                role=role,
                available=False,
                mode=ModelMode.INFERENCE,
                reason=reason,
            )
        )

    def load(self, spec: ModelSpec) -> None:
        raise ModelAdapterError(
            ModelErrorCode.MODEL_UNAVAILABLE,
            self.capabilities().reason or "model backend is unavailable",
            model_id=spec.model_id,
        )


def _jsonable(value: object) -> object:
    if value is None or isinstance(value, (str, int, bool, float)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, StrEnum):
        return value.value
    if is_dataclass(value):
        return {field.name: _jsonable(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    raise TypeError(f"value is not JSON-serializable: {type(value).__name__}")
