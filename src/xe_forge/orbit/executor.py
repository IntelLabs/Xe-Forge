"""
Process execution (plan §20).

Local execution only. Running inside an interactive Slurm allocation is local
execution from Orbit's perspective — `salloc`, then `xe-orbit` — so there is exactly
one implementation here. The protocol exists so a batch backend can be added later
without touching call sites, not because one is being built now.
"""

from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, Field


class RunResult(BaseModel):
    """Outcome of one process execution."""

    command: list[str] = Field(default_factory=list)
    returncode: int = 0
    stdout: str = ""
    stderr: str = ""
    duration_s: float = 0.0
    timed_out: bool = False

    @property
    def ok(self) -> bool:
        return self.returncode == 0 and not self.timed_out


@runtime_checkable
class Executor(Protocol):
    """How Orbit runs a workload. The only implementation is `LocalExecutor`."""

    def run(
        self,
        cmd: list[str],
        env: dict[str, str] | None = None,
        cwd: Path | None = None,
        timeout: float = 1800.0,
    ) -> RunResult: ...


class LocalExecutor:
    """Runs a command as a subprocess on this machine."""

    def __init__(self, inherit_env: bool = True, capture_output: bool = True) -> None:
        self.inherit_env = inherit_env
        self.capture_output = capture_output

    def build_env(self, env: dict[str, str] | None) -> dict[str, str]:
        """Overlay `env` onto the ambient environment (or replace it entirely)."""
        base = dict(os.environ) if self.inherit_env else {}
        if env:
            base.update({str(k): str(v) for k, v in env.items()})
        return base

    def run(
        self,
        cmd: list[str],
        env: dict[str, str] | None = None,
        cwd: Path | None = None,
        timeout: float = 1800.0,
    ) -> RunResult:
        if not cmd:
            raise ValueError("LocalExecutor.run requires a non-empty command")

        resolved_env = self.build_env(env)
        workdir = Path(cwd) if cwd else Path.cwd()

        start = time.perf_counter()
        try:
            proc = subprocess.run(
                cmd,
                env=resolved_env,
                cwd=str(workdir),
                capture_output=self.capture_output,
                text=True,
                timeout=timeout,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            return RunResult(
                command=list(cmd),
                returncode=-1,
                stdout=_as_text(exc.stdout),
                stderr=_as_text(exc.stderr),
                duration_s=time.perf_counter() - start,
                timed_out=True,
            )
        except FileNotFoundError as exc:
            return RunResult(
                command=list(cmd),
                returncode=127,
                stdout="",
                stderr=str(exc),
                duration_s=time.perf_counter() - start,
            )

        return RunResult(
            command=list(cmd),
            returncode=proc.returncode,
            stdout=proc.stdout or "",
            stderr=proc.stderr or "",
            duration_s=time.perf_counter() - start,
        )


def _as_text(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value
