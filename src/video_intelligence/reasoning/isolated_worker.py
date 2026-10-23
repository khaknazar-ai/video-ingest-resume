"""Bounded local Python worker supervision; no shell, retry or descendant launching."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class WorkerCommand:
    python: Path
    python_sha256: str
    script: Path
    script_sha256: str

    def validate(self):
        for path, digest in ((self.python, self.python_sha256), (self.script, self.script_sha256)):
            if not path.is_absolute() or path.is_symlink() or not path.is_file():
                raise ValueError("worker executable/script must be absolute regular files")
            if file_hash(path) != digest:
                raise ValueError("worker executable/script hash differs")

    def run(self, request: Path, directory: Path, *, timeout_seconds=60, checkpoint=lambda: None):
        self.validate()
        request, directory = Path(request).absolute(), Path(directory).absolute()
        if not 0 < timeout_seconds <= 60 or not directory.is_dir():
            raise ValueError("worker directory/deadline differs")
        env = dict(os.environ)
        for key in ("PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP", "PYTHONOPTIMIZE"):
            env.pop(key, None)
        env.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", PYTHONDONTWRITEBYTECODE="1")
        command = [str(self.python), "-I", "-B", str(self.script), "--request", str(request)]
        started = time.monotonic()
        child = None
        reason = None
        errors = []
        result = {"command": command, "status": "not_started", "own_process_exited": None}
        try:
            checkpoint()
            with ((directory / "stdout.log").open("xb") as out,
                  (directory / "stderr.log").open("xb") as err):
                child = subprocess.Popen(command, cwd=directory, env=env, stdout=out, stderr=err)
                result["pid"] = child.pid
                while child.poll() is None:
                    try:
                        checkpoint()
                    except Exception as error:
                        reason = "cancelled:" + type(error).__name__
                        break
                    if time.monotonic()-started >= timeout_seconds:
                        reason = "deadline"
                        break
                    if any((directory / name).stat().st_size > 131_072
                           for name in ("stdout.log", "stderr.log")):
                        reason = "log_budget"
                        break
                    time.sleep(.02)
        except Exception as error:
            errors.append({"phase": "launch_or_poll", "type": type(error).__name__})
        finally:
            if child is not None and child.poll() is None:
                try:
                    child.kill()
                    child.wait(timeout=10)
                except Exception as error:
                    errors.append({"phase": "kill_or_wait_own_child", "type": type(error).__name__})
            # Polling is a soft size limit; final validation also catches fast exits.
            try:
                result["log_bytes"] = {name: (directory / name).stat().st_size
                    for name in ("stdout.log", "stderr.log") if (directory / name).exists()}
                if any(size > 131_072 for size in result["log_bytes"].values()):
                    reason = reason or "log_budget"
            except Exception as error:
                errors.append({"phase": "final_log_validation", "type": type(error).__name__})
            result["stop_reason"] = reason
            result["returncode"] = child.poll() if child else None
            result["own_process_exited"] = child.poll() is not None if child else None
            result["wall_seconds"] = time.monotonic()-started
            result["errors"] = errors
            result["status"] = ("passed" if child and child.poll() == 0 and not reason
                                and not errors else "failed")
            with (directory / "worker-lifecycle.json").open("x", encoding="utf-8") as stream:
                json.dump(result, stream, indent=2)
        return result
