"""A ledger of what earlier optimization runs measured on this part.

A workspace is scratch: generated per kernel, thrown away, regenerated. Everything
a session learns dies with it -- which strategy this part rewarded, which one it
punished, which benchmark variant is too small to resolve a difference on -- so the
next session starts from the vendor's priors and its own guesses, and re-runs
experiments that have already been paid for in GPU time.

The knowledge base is not that ledger and must not become one. Its entries are
patterns recorded on the machines the vendor ran, which is why the workspace
template tells a session not to read their ``expected_speedup``. What is missing is
the other half: what *this* part returned, from ``benchmark``, in a run we did.

So a host that wants its sessions to accumulate names a directory. Every session
reads every file in it and writes only its own. That is what makes a shared ledger
safe: two sessions optimizing two kernels at once never touch the same file, and no
session can rewrite another's record of what it measured.

Unset, nothing renders and the workspace is exactly what it was.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

__all__ = ["LessonsLog", "lessons_filename", "load_lessons_log"]

_UNSAFE = re.compile(r"[^A-Za-z0-9._-]+")


def lessons_filename(kernel_name: str) -> str:
    """The file a session optimizing *kernel_name* owns within the ledger.

    Derived from the kernel name rather than from the run, so a second session on
    the same kernel appends to its own earlier record instead of starting a file
    nobody will correlate with it.
    """
    stem = _UNSAFE.sub("_", str(kernel_name)).strip("._-")
    return f"{stem or 'session'}.md"


@dataclass(frozen=True)
class LessonsLog:
    """Where the session reads from, and the one file it may write."""

    # Absolute: the session's working directory is the workspace, and the ledger
    # is deliberately somewhere else -- a path relative to the caller's shell
    # would resolve to a directory inside the scratch it is meant to outlive.
    dir: str
    own_file: str


def load_lessons_log(dir_path: str | Path | None, kernel_name: str) -> LessonsLog | None:
    """Resolve the ledger directory, creating it, or ``None`` when none was named."""
    if not dir_path:
        return None

    path = Path(dir_path).expanduser()
    if path.exists() and not path.is_dir():
        raise NotADirectoryError(f"lessons ledger is not a directory: {path}")
    path.mkdir(parents=True, exist_ok=True)
    resolved = path.resolve()

    return LessonsLog(
        dir=str(resolved),
        own_file=str(resolved / lessons_filename(kernel_name)),
    )
