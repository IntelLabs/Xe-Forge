"""One headless Claude session for work the frontend delegates: preparing a model for its
framework, naming its ops, building the stacks a patch is measured in. The prompts are
templates (``claude/templates/frontend-*.md.j2``); per-model and per-repository specifics
live there, not here. The session answers in a JSON file it writes."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import time
from pathlib import Path

TIMEOUT = int(os.environ.get("XE_FORGE_AGENT_TIMEOUT", 6 * 3600))  # seconds per session
RESUMES = int(os.environ.get("XE_FORGE_AGENT_RESUMES", 5))


def run(template: str, context: dict, cwd: Path, answer: str, python: str, max_turns: int) -> dict:
    """Render ``template`` (claude/templates) with ``context`` plus the shared rules, run
    the session in ``cwd``, and return the JSON it wrote to ``answer`` (raises if none)."""
    from xe_forge.claude.generator import render
    from xe_forge.config import get_config
    from xe_forge.engines.claude_engine import claude_env

    claude = shutil.which("claude")
    if not claude:
        raise RuntimeError("no 'claude' CLI on PATH")
    cwd.mkdir(parents=True, exist_ok=True)
    path = cwd / answer
    path.unlink(missing_ok=True)  # a previous session's answer is not this one's
    (cwd / "session.log").unlink(missing_ok=True)
    env = claude_env(get_config())
    # Headless claude kills background work after this ceiling: let a build run its course.
    env["CLAUDE_CODE_PRINT_BG_WAIT_CEILING_MS"] = str(TIMEOUT * 1000)
    prompt = render(template, **context) + render(
        "frontend-rules.md.j2", python=python, answer=answer
    )
    log_path, deadline = cwd / "session.log", time.monotonic() + TIMEOUT
    cmd = [claude, "-p", prompt]
    for _ in range(1 + RESUMES):
        # A session can stop while work it started detached is still running ("I'll check
        # back"): resume it, with the same context, until the answer is written.
        with open(log_path, "a") as log:
            try:
                proc = subprocess.run(
                    [*cmd, "--dangerously-skip-permissions", "--max-turns", str(max_turns),
                     "--output-format", "stream-json", "--verbose"],
                    cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT,
                    timeout=max(1.0, deadline - time.monotonic()),
                )  # fmt: skip
            except subprocess.TimeoutExpired:
                raise RuntimeError(f"agent timed out after {TIMEOUT} s (see {cwd})") from None
        session = re.findall(r'"session_id":"([^"]+)"', log_path.read_text())
        if path.is_file() or not session:
            break
        cmd = [
            claude,
            "--resume",
            session[-1],
            "-p",
            render("frontend-resume.md.j2", answer=answer),
        ]
    got = json.loads(path.read_text()) if path.is_file() else {}
    if "error" in got or not got:
        raise RuntimeError(f"agent: {got.get('error', f'no {answer}')} (see {cwd})")
    if proc.returncode != 0:
        raise RuntimeError(f"agent exited {proc.returncode} (see {cwd}/session.log)")
    return got
