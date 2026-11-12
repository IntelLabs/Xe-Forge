"""One headless agent session (claude, codex: ``xe_forge.agent_cli``) for delegated frontend
work; it answers in a JSON file."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import time
from pathlib import Path

TIMEOUT = int(os.environ.get("XE_FORGE_AGENT_TIMEOUT", 6 * 3600))
RESUMES = int(os.environ.get("XE_FORGE_AGENT_RESUMES", 5))


def run(template: str, context: dict, cwd: Path, answer: str, python: str, max_turns: int) -> dict:
    """Run ``template`` as a session in ``cwd``; return the JSON it wrote to ``answer``."""
    from xe_forge import agent_cli as agents
    from xe_forge.claude.generator import render
    from xe_forge.config import get_config

    backend, config = agents.get(), get_config()
    backend.path()
    cwd.mkdir(parents=True, exist_ok=True)
    path = cwd / answer
    path.unlink(missing_ok=True)
    (cwd / "session.log").unlink(missing_ok=True)
    env = backend.env(config)
    # Headless claude kills background work after this ceiling; builds need longer.
    env["CLAUDE_CODE_PRINT_BG_WAIT_CEILING_MS"] = str(TIMEOUT * 1000)
    prompt = render(template, **{"python": python, "answer": answer, **context}) + render(
        "frontend-rules.md.j2", python=python, answer=answer
    )
    (cwd / "prompt.md").write_text(prompt)
    log_path, deadline = cwd / "session.log", time.monotonic() + TIMEOUT
    cmd = backend.start(prompt, max_turns, config)
    groups: list[int] = []  # every session's process group, for the work it detached
    try:
        for _ in range(1 + RESUMES):
            start = log_path.stat().st_size if log_path.is_file() else 0
            with open(log_path, "a") as log:
                # Own process group, so the detached builds it started can be killed.
                proc = subprocess.Popen(
                    cmd,
                    cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT,
                    start_new_session=True,
                )  # fmt: skip
                groups.append(proc.pid)
                try:
                    proc.wait(timeout=max(1.0, deadline - time.monotonic()))
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait()
                    raise RuntimeError(f"agent timed out after {TIMEOUT} s (see {cwd})") from None
            if path.is_file():
                break
            # st_size is a byte offset: slice the bytes, then decode.
            text = log_path.read_bytes()[start:].decode(errors="replace")
            # Resume a session that stopped while its detached work ran; not one that errored.
            session, failed, why = backend.parse(text, proc.returncode)
            if failed or not session:
                raise RuntimeError(f"agent stopped: {why[:300]} (see {log_path})")
            resume = render("frontend-resume.md.j2", answer=answer)
            cmd = backend.resume(session, resume, max_turns, config)
    finally:
        # Detached work must outlive a resume, but not the step: left running, an export or
        # build holds the device memory the capture that follows needs.
        _kill_groups(groups)
    got = json.loads(path.read_text()) if path.is_file() else {}
    if not isinstance(got, dict):
        raise RuntimeError(f"agent: {answer} is not a JSON object (see {cwd})")
    if "error" in got or not got:
        hint = f"; next: {got['next_step']}" if got.get("next_step") else ""
        raise RuntimeError(f"agent: {got.get('error', f'no {answer}')}{hint} (see {cwd})")
    return got


def oneshot(prompt: str, inputs: dict[str, str], schema: dict, profile: dict | None = None,
            timeout: float = 300) -> dict:  # fmt: skip
    """One answer to ``prompt`` matching ``schema``, from a session confined to a directory
    holding ``inputs`` (name -> text, under ``inputs/``) with the tools ``profile`` names
    (default read-only). The caller checks the answer; this only returns the object."""
    import tempfile

    from xe_forge import agent_cli as agents
    from xe_forge.config import get_config

    backend, config = agents.get(), get_config()
    with tempfile.TemporaryDirectory(prefix="xf-oneshot-") as tmp:
        work = Path(tmp) / "work"
        for name, text in inputs.items():
            f = work / "inputs" / name
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_text(text)
        work.mkdir(exist_ok=True)
        schema_file, answer = Path(tmp) / "schema.json", Path(tmp) / "answer.json"
        schema_file.write_text(json.dumps(schema))
        argv = backend.oneshot(prompt, schema_file, answer, profile or agents.READ_ONLY, config)
        proc = subprocess.run(argv, cwd=work, env=backend.env(config), capture_output=True,
                              text=True, timeout=timeout, start_new_session=True)  # fmt: skip
        if any(backend.auth_failed(line) for line in (proc.stdout + proc.stderr).splitlines()):
            raise agents.AgentError(f"the LLM endpoint rejected the {backend.name} key")
        if proc.returncode != 0:
            raise agents.AgentError(f"{backend.name} exited {proc.returncode}: "
                                    f"{(proc.stderr or proc.stdout)[-300:]}")  # fmt: skip
        return backend.oneshot_answer(proc.stdout, answer)


def _kill_groups(groups: list[int], grace: float = 10.0) -> None:
    """SIGTERM each process group, then SIGKILL what is left after ``grace`` seconds."""

    def signal_all(sig) -> list[int]:
        alive = []
        for pg in groups:
            try:
                os.killpg(pg, sig)
                alive.append(pg)
            except (ProcessLookupError, PermissionError):
                pass
        return alive

    if not signal_all(signal.SIGTERM):
        return
    stop = time.monotonic() + grace
    while time.monotonic() < stop and signal_all(0):
        time.sleep(0.2)
    signal_all(signal.SIGKILL)
