"""One headless Claude session for work the frontend delegates: preparing a model for its
framework, naming its ops, building the stacks a patch is measured in. The prompts are
templates (``claude/templates/frontend-*.md.j2``); per-model and per-repository specifics
live there, not here. The session answers in a JSON file it writes."""

from __future__ import annotations

import json
import os
import re
import shutil
import signal
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
    (cwd / "prompt.md").write_text(prompt)  # what the session was asked, for the record
    log_path, deadline = cwd / "session.log", time.monotonic() + TIMEOUT
    cmd = [claude, "-p", prompt]
    for _ in range(1 + RESUMES):
        start = log_path.stat().st_size if log_path.is_file() else 0
        with open(log_path, "a") as log:
            # Its own process group: on timeout the detached builds a session started go too.
            proc = subprocess.Popen(
                [*cmd, "--dangerously-skip-permissions", "--max-turns", str(max_turns),
                 "--output-format", "stream-json", "--verbose"],
                cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
            )  # fmt: skip
            try:
                proc.wait(timeout=max(1.0, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
                raise RuntimeError(f"agent timed out after {TIMEOUT} s (see {cwd})") from None
        if path.is_file():
            break
        text = log_path.read_text()[start:]
        result = _result(text)
        # A session that stopped cleanly (or ran out of turns) while detached work it started
        # was still running ("I'll check back") is resumed, with its context, until it
        # answers. An error -- the API, the CLI -- is not something resuming fixes.
        failed = proc.returncode != 0 and result.get("subtype") != "error_max_turns"
        session = re.findall(r'"session_id":"([^"]+)"', text)
        if failed or not session:
            why = result.get("result") or result.get("subtype") or f"exit {proc.returncode}"
            raise RuntimeError(f"agent stopped: {str(why)[:300]} (see {log_path})")
        cmd = [
            claude,
            "--resume",
            session[-1],
            "-p",
            render("frontend-resume.md.j2", answer=answer),
        ]
    got = json.loads(path.read_text()) if path.is_file() else {}
    if not isinstance(got, dict):
        raise RuntimeError(f"agent: {answer} is not a JSON object (see {cwd})")
    if "error" in got or not got:
        raise RuntimeError(f"agent: {got.get('error', f'no {answer}')} (see {cwd})")
    return got


def _result(stream: str) -> dict:
    """The last ``result`` event of a stream-json session log (subtype, result text)."""
    for line in reversed(stream.splitlines()):
        if '"type":"result"' in line or '"type": "result"' in line:
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                return {}
    return {}
