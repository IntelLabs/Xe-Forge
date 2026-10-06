"""Run a command in its own process group and kill the group on timeout.

A kernel that never completes leaves the process blocked inside the driver, where no
signal handler runs; only a parent can end it and say why. Every caller that runs
untrusted device code (the built-in benchmark, the vISA lowering runner) goes through
here so the kill is the same everywhere: SIGKILL to the whole group, so a child that
spawned its own helpers (a compiler, a launcher) does not outlive it.
"""

from __future__ import annotations

import os
import signal
import subprocess
from dataclasses import dataclass


@dataclass
class WatchedResult:
    returncode: int | None
    timed_out: bool
    stdout: str = ""
    stderr: str = ""

    @property
    def signal(self) -> int | None:
        """The signal that ended the child, if one did (negative return code)."""
        if self.returncode is not None and self.returncode < 0:
            return -self.returncode
        return None


def run_watched(
    cmd: list[str],
    timeout: float,
    *,
    env: dict[str, str] | None = None,
    cwd: str | None = None,
    capture: bool = False,
) -> WatchedResult:
    """Run ``cmd``; on expiry of ``timeout`` seconds kill its process group.

    With ``capture`` the child's stdout and stderr are returned as text;
    otherwise they are inherited.
    """
    pipe = subprocess.PIPE if capture else None
    child = subprocess.Popen(
        cmd,
        env=env,
        cwd=cwd,
        stdout=pipe,
        stderr=pipe,
        text=True,
        start_new_session=True,
    )
    try:
        out, err = child.communicate(timeout=timeout)
        return WatchedResult(child.returncode, False, out or "", err or "")
    except subprocess.TimeoutExpired:
        try:
            os.killpg(child.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        out, err = child.communicate()
        return WatchedResult(child.returncode, True, out or "", err or "")
