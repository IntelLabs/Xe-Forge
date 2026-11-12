"""Agent CLIs Xe-Forge can drive headless: how to start, resume and read each one.

Everything else -- the resume loop, the deadline, process-group cleanup, the answer file
and the gates -- is one implementation that only asks a backend for argv, env and what
its log says. Adding a CLI is adding a class here.

A one-shot call names its tools as a profile instead of assuming a default:
``{"read": True, "write": False, "shell": False, "network": False}``. The session runs in
a directory staged with its inputs; ``read`` and ``write`` are confined to it.
"""

from __future__ import annotations

import json
import os
import re
import shutil
from pathlib import Path

READ_ONLY = {"read": True, "write": False, "shell": False, "network": False}


class AgentError(RuntimeError):
    pass


class Claude:
    name, binary, instructions = "claude", "claude", "CLAUDE.md"

    def env(self, config) -> dict[str, str]:
        from xe_forge.engines.claude_engine import claude_env

        return claude_env(config)

    def _stream(self, max_turns: int) -> list[str]:
        return ["--dangerously-skip-permissions", "--max-turns", str(max_turns),
                "--output-format", "stream-json", "--verbose"]  # fmt: skip

    def start(self, prompt: str, max_turns: int, config=None) -> list[str]:
        return [self.path(), "-p", prompt, *self._stream(max_turns)]

    def resume(self, session: str, prompt: str, max_turns: int, config=None) -> list[str]:
        return [self.path(), "--resume", session, "-p", prompt, *self._stream(max_turns)]

    def task(self, workspace: Path, kernel_name: str) -> str:
        return f"/optimize-kernel {kernel_name}"

    def prepare(self, workspace: Path) -> None:
        """Make a generated workspace (CLAUDE.md, .claude/) readable by this CLI."""

    def oneshot(self, prompt: str, schema: Path, answer: Path, profile: dict, config=None):
        tools = (["Read", "Glob", "Grep"] if profile.get("read") else []) + (
            ["Write", "Edit"] if profile.get("write") else []) + (
            ["Bash"] if profile.get("shell") else []) + (
            ["WebFetch", "WebSearch"] if profile.get("network") else [])  # fmt: skip
        # --restricted confines file tools to the working directory; dontAsk refuses
        # anything not listed instead of waiting for a person.
        argv = [self.path(), "-p", prompt, "--restricted", "--strict-mcp-config",
                "--permission-mode", "dontAsk", "--tools", ",".join(tools),
                "--json-schema", schema.read_text(), "--output-format", "json"]  # fmt: skip
        if tools:
            argv += ["--allowedTools", *tools]
        return argv

    def oneshot_answer(self, stdout: str, answer: Path) -> dict:
        try:
            got = json.loads(stdout).get("structured_output")
        except (json.JSONDecodeError, AttributeError):
            got = None
        if not isinstance(got, dict):
            raise AgentError("claude gave no structured answer")
        return got

    def parse(self, text: str, returncode: int) -> tuple[str | None, bool, str]:
        """(session id, whether it failed rather than stopped, why) from a session's log."""
        result = {}
        for line in reversed(text.splitlines()):
            if '"type":"result"' in line or '"type": "result"' in line:
                try:
                    result = json.loads(line)
                except json.JSONDecodeError:
                    pass
                break
        sessions = re.findall(r'"session_id":\s*"([^"]+)"', text)
        failed = returncode != 0 and result.get("subtype") != "error_max_turns"
        why = result.get("result") or result.get("subtype") or f"exit {returncode}"
        return (sessions[-1] if sessions else None), failed, str(why)

    def auth_failed(self, line: str) -> bool:
        return '"api_retry"' in line and re.search(r'"error_status":\s*40[13]\b', line) is not None

    def path(self) -> str:
        found = shutil.which(self.binary)
        if not found:
            raise AgentError(f"no {self.binary!r} CLI on PATH")
        return found


class Codex(Claude):
    """OpenAI Codex CLI (``codex exec``). It has no turn limit (the deadline bounds a
    session), reads AGENTS.md, and reads files through its shell, so ``read`` is a
    read-only sandboxed shell."""

    name, binary, instructions = "codex", "codex", "AGENTS.md"
    _EXEC = ("--json", "--skip-git-repo-check")

    def env(self, config) -> dict[str, str]:
        env = os.environ.copy()
        if config is not None and config.llm.api_key:
            env.setdefault("CODEX_API_KEY", config.llm.api_key)
        return env

    def _endpoint(self, config) -> list[str]:
        if config is None:
            return []
        llm, out = config.llm, []
        if llm.api_base:
            out += ["-c", f"openai_base_url={json.dumps(llm.api_base)}"]
        if llm.model:
            out += ["-m", llm.model.split("/")[-1]]
        return out

    def start(self, prompt: str, max_turns: int, config=None) -> list[str]:
        return [self.path(), "exec", *self._EXEC, "--dangerously-bypass-approvals-and-sandbox",
                *self._endpoint(config), prompt]  # fmt: skip

    def resume(self, session: str, prompt: str, max_turns: int, config=None) -> list[str]:
        return [self.path(), "exec", *self._EXEC, "--dangerously-bypass-approvals-and-sandbox",
                *self._endpoint(config), "resume", session, prompt]  # fmt: skip

    def task(self, workspace: Path, kernel_name: str) -> str:
        text = (workspace / ".claude" / "commands" / "optimize-kernel.md").read_text()
        return text.replace("$ARGUMENTS", kernel_name)

    def prepare(self, workspace: Path) -> None:
        """AGENTS.md from CLAUDE.md, and each ``.claude/agents/<name>.md`` as a codex agent
        (``.codex/agents/<name>.toml``); where it cannot dispatch one, it follows the file."""
        helpers = sorted((workspace / ".claude" / "agents").glob("*.md"))
        note = (
            (
                "The agents these instructions dispatch are defined in `.claude/agents/<name>.md`"
                f" ({', '.join(f.stem for f in helpers)}). Where you cannot dispatch one, read "
                "its file and carry it out yourself as one step, then continue.\n\n"
            )
            if helpers
            else ""
        )
        (workspace / "AGENTS.md").write_text(note + (workspace / "CLAUDE.md").read_text())
        out = workspace / ".codex" / "agents"
        for f in helpers:
            head, _, body = f.read_text().removeprefix("---\n").partition("\n---\n")
            meta = dict(ln.split(":", 1) for ln in head.splitlines() if ":" in ln)
            name = meta.get("name", f.stem).strip()
            desc = meta.get("description", "").strip().strip('"')
            out.mkdir(parents=True, exist_ok=True)
            (out / f"{name}.toml").write_text(  # JSON strings are valid TOML basic strings
                f"name = {json.dumps(name)}\ndescription = {json.dumps(desc)}\n"
                f"developer_instructions = {json.dumps(body.strip())}\n"
            )

    def oneshot(self, prompt: str, schema: Path, answer: Path, profile: dict, config=None):
        shell = bool(profile.get("shell") or profile.get("read"))
        net = bool(profile.get("network"))
        argv = [self.path(), "exec", *self._EXEC, "--ephemeral", "--ignore-user-config",
                "--sandbox", "workspace-write" if profile.get("write") else "read-only",
                "-c", f"features.shell_tool={str(shell).lower()}",
                "-c", f"sandbox_workspace_write.network_access={str(net).lower()}",
                "-c", f"web_search={'live' if net else 'disabled'}",
                "--output-schema", str(schema), "-o", str(answer), *self._endpoint(config)]  # fmt: skip
        return [*argv, prompt]

    def oneshot_answer(self, stdout: str, answer: Path) -> dict:
        try:
            got = json.loads(answer.read_text())
        except (OSError, json.JSONDecodeError):
            got = None
        if not isinstance(got, dict):
            raise AgentError("codex gave no structured answer")
        return got

    def parse(self, text: str, returncode: int) -> tuple[str | None, bool, str]:
        sessions = re.findall(r'"thread_id":\s*"([^"]+)"', text)
        why = f"exit {returncode}"
        failed = returncode != 0
        for line in text.splitlines():
            if '"turn.failed"' in line or '"type":"error"' in line or '"type": "error"' in line:
                failed, why = True, line[:300]
        return (sessions[-1] if sessions else None), failed, why

    def auth_failed(self, line: str) -> bool:
        return ('"turn.failed"' in line or '"error"' in line) and re.search(
            r"\b40[13]\b|unauthori[sz]ed|forbidden|invalid api key", line, re.I) is not None  # fmt: skip


BACKENDS = {b.name: b for b in (Claude(), Codex())}


def get(name: str | None = None) -> Claude:
    """The backend named, else ``XE_FORGE_AGENT``, else claude."""
    name = name or os.environ.get("XE_FORGE_AGENT") or "claude"
    if name not in BACKENDS:
        raise AgentError(f"unknown agent CLI {name!r} (one of {', '.join(BACKENDS)})")
    return BACKENDS[name]


def available() -> list[str]:
    return [n for n, b in BACKENDS.items() if shutil.which(b.binary)]
