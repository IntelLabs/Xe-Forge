"""One headless Claude session for work the frontend delegates: preparing a model for its
framework, building the stacks a patch is measured in. Per-model and per-repository
specifics live in the prompt, not here. The session answers in a JSON file it writes."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import time
from pathlib import Path

TIMEOUT = 6 * 3600
RESUMES = 5
RESUME = (
    "{answer} is not written yet. Wait for the work you started -- check it is still running "
    "and restart it if it died -- and stop only once {answer} is written."
)

RULES = """
The framework's interpreter is {python}; its venv must not change -- install nothing into
it, never use uv. Anything missing goes into a venv you create here, layered over it (a .pth
calling site.addsitedir on its site-packages, after the venv's own). Run builds and exports
with the ANTHROPIC_* and OPENAI_* variables unset, so no log carries them. Long builds and
exports are expected: this session ends when you stop and anything still running is killed,
so wait for all of it, and stop only once {answer} is written.
"""


def run(prompt: str, cwd: Path, answer: str, python: str, max_turns: int) -> dict:
    """Run the session in ``cwd``; return the JSON it wrote to ``answer`` (raises if none)."""
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
    prompt += RULES.format(python=python, answer=answer)
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
                raise RuntimeError(
                    f"agent timed out after {TIMEOUT // 3600} h (see {cwd})"
                ) from None
        session = re.findall(r'"session_id":"([^"]+)"', log_path.read_text())
        if path.is_file() or not session:
            break
        cmd = [claude, "--resume", session[-1], "-p", RESUME.format(answer=answer)]
    got = json.loads(path.read_text()) if path.is_file() else {}
    if "error" in got or not got:
        raise RuntimeError(f"agent: {got.get('error', f'no {answer}')} (see {cwd})")
    if proc.returncode != 0:
        raise RuntimeError(f"agent exited {proc.returncode} (see {cwd}/session.log)")
    return got
