"""Agent CLIs X-Forge can drive headless: how to start, resume and read each one.

Everything else -- the resume loop, the deadline, process-group cleanup, the answer file
and the gates -- is one implementation that only asks a backend for argv, env and what
its log says. Adding a CLI is adding a class here.

Agents a session may dispatch are written once, in the ``.claude/agents/<name>.md`` form
(:func:`agent_definition`), and each backend makes them its own: a workspace's in its
directory (:meth:`Claude.install_agents`, and :meth:`Claude.agent_args` for the session that
runs there), one session's for that session alone (:meth:`Claude.session_agents`). Both give
arguments that :meth:`Claude.start` and :meth:`Claude.resume` place where the CLI takes them. An agent's ``model`` names a tier, never a model:
:func:`agent_models` turns the tiers into the provider's model ids.

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
# A session's tools, by what they allow, as a one-shot call's are: ``agents`` dispatches helper
# agents, ``background`` runs and waits on long commands, ``commands`` runs slash commands
# (a kernel session starts with one). Whatever a backend offers beyond these stays out of the
# session, and of the context every one of its calls reads again.
SESSION = {"read": True, "write": True, "shell": True, "network": True, "agents": True,
           "background": True, "commands": False}  # fmt: skip
KERNEL_SESSION = {**SESSION, "commands": True}


class AgentError(RuntimeError):
    pass


def bare(model: str | None) -> str | None:
    """``model`` without the provider litellm prefixes (``openai/gpt-4o``): what a CLI takes."""
    return model.split("/")[-1] if model else None


def agent_models(config) -> dict[str, str]:
    """The ``model`` an agent of each tier is written with: ``fast`` (``LLM_FAST_MODEL``),
    else ``inherit``, the session's own model."""
    return {"fast": bare(config.llm.fast_model) or "inherit"}


def agent_definition(text: str) -> dict:
    """An agent written as ``.claude/agents/<name>.md``: name, description, tools, model
    (None for the session's own) and its instructions."""
    head, _, body = text.removeprefix("---\n").partition("\n---\n")
    meta = {k.strip(): v.strip() for k, v in
            (ln.split(":", 1) for ln in head.splitlines() if ":" in ln)}  # fmt: skip
    model = meta.get("model", "")
    return {"name": meta.get("name", ""), "description": meta.get("description", "").strip('"'),
            "tools": [t.strip() for t in meta.get("tools", "").split(",") if t.strip()],
            "model": None if model in ("", "inherit") else model,
            "prompt": body.strip()}  # fmt: skip


class Claude:
    name, binary, instructions = "claude", "claude", "CLAUDE.md"

    def env(self, config) -> dict[str, str]:
        from x_forge.engines.claude_engine import claude_env

        return claude_env(config)

    def _stream(self, max_turns: int) -> list[str]:
        return ["--dangerously-skip-permissions", "--max-turns", str(max_turns),
                "--output-format", "stream-json", "--verbose"]  # fmt: skip

    @staticmethod
    def _tools(profile: dict) -> list[str]:
        """Claude's tools for a profile (:data:`READ_ONLY`, :data:`SESSION`)."""
        return ((["Read", "Glob", "Grep"] if profile.get("read") else [])
                + (["Write", "Edit"] if profile.get("write") else [])
                + (["Bash"] if profile.get("shell") else [])
                + (["WebFetch", "WebSearch"] if profile.get("network") else [])
                + (["Task", "SendMessage"] if profile.get("agents") else [])
                + (["Monitor", "TaskStop"] if profile.get("background") else [])
                + (["Skill"] if profile.get("commands") else []))  # fmt: skip

    def _session(self, profile: dict | None) -> list[str]:
        if profile is None:
            return []
        out = ["--tools", ",".join(self._tools(profile)), "--strict-mcp-config"]
        return out + ([] if profile.get("commands") else ["--disable-slash-commands"])

    def start(self, prompt: str, max_turns: int, config=None, agents=(),
              profile: dict | None = None) -> list[str]:  # fmt: skip
        return [self.path(), "-p", prompt, *self._stream(max_turns), *self._session(profile),
                *agents]  # fmt: skip

    def resume(self, session: str, prompt: str, max_turns: int, config=None,
               agents=(), profile: dict | None = None) -> list[str]:  # fmt: skip
        return [self.path(), "--resume", session, "-p", prompt, *self._stream(max_turns),
                *self._session(profile), *agents]  # fmt: skip

    def interactive(self, kernel_name: str) -> str:
        """The command a person runs in the workspace to start the loop."""
        return f"claude /optimize-kernel {kernel_name}"

    def task(self, workspace: Path, kernel_name: str) -> str:
        return f"/optimize-kernel {kernel_name}"

    def prepare(self, workspace: Path) -> None:
        """Make a generated workspace (CLAUDE.md, .claude/) readable by this CLI."""

    def install_agents(self, cwd: Path) -> list[str]:
        """Make the agents in ``cwd/.claude/agents/`` dispatchable here; their names. Claude
        reads them as they are."""
        return sorted(f.stem for f in (cwd / ".claude" / "agents").glob("*.md"))

    def agent_args(self, cwd: Path) -> list[str]:
        """The arguments a session in ``cwd`` needs to dispatch its installed agents: none."""
        return []

    def session_agents(self, cwd: Path, agents: list[str]) -> list[str]:
        """The arguments that give one session ``agents`` (``.claude/agents`` texts) and no
        other: claude would also load a ``.claude/agents/`` in ``cwd`` into every session
        under it, so they go to ``--agents`` as a file."""
        defs = {}
        for text in agents:
            d = agent_definition(text)
            defs[d["name"]] = {"description": d["description"], "prompt": d["prompt"],
                               "tools": d["tools"], **({"model": d["model"]} if d["model"] else {})}  # fmt: skip
        path = cwd / "session-agents.json"
        path.write_text(json.dumps(defs, indent=1) + "\n")
        return ["--agents", str(path)]

    def oneshot(self, prompt: str, schema: Path, answer: Path, profile: dict, config=None):
        tools = self._tools(profile)
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
            out += ["-m", bare(llm.model)]
        return out

    def start(self, prompt: str, max_turns: int, config=None, agents=(),
              profile: dict | None = None) -> list[str]:  # fmt: skip
        """``profile`` is not mapped: codex exec offers its own fixed tool set."""
        return [self.path(), "exec", *self._EXEC, "--dangerously-bypass-approvals-and-sandbox",
                *self._endpoint(config), *agents, prompt]  # fmt: skip

    def resume(self, session: str, prompt: str, max_turns: int, config=None,
               agents=(), profile: dict | None = None) -> list[str]:  # fmt: skip
        return [self.path(), "exec", *self._EXEC, "--dangerously-bypass-approvals-and-sandbox",
                *self._endpoint(config), *agents, "resume", session, prompt]  # fmt: skip

    def interactive(self, kernel_name: str) -> str:
        """The command a person runs in the workspace to start the loop."""
        return (f'codex "Follow .claude/commands/optimize-kernel.md with $ARGUMENTS = '
                f'{kernel_name}"')  # fmt: skip

    def task(self, workspace: Path, kernel_name: str) -> str:
        text = (workspace / ".claude" / "commands" / "optimize-kernel.md").read_text()
        return text.replace("$ARGUMENTS", kernel_name)

    def prepare(self, workspace: Path) -> None:
        """AGENTS.md from CLAUDE.md, and each ``.claude/agents/<name>.md`` as a codex agent
        (``.codex/agents/<name>.toml``); where it cannot dispatch one, it follows the file."""
        helpers = self.install_agents(workspace)
        note = (
            (
                "The agents these instructions dispatch are defined in `.claude/agents/<name>.md`"
                f" ({', '.join(helpers)}). Where you cannot dispatch one, read "
                "its file and carry it out yourself as one step, then continue.\n\n"
            )
            if helpers
            else ""
        )
        (workspace / "AGENTS.md").write_text(note + (workspace / "CLAUDE.md").read_text())

    def install_agents(self, cwd: Path) -> list[str]:
        """Each ``.claude/agents/<name>.md`` as a codex agent, ``.codex/agents/<name>.toml``,
        with its model when it names one; their names."""
        files = sorted((cwd / ".claude" / "agents").glob("*.md"))
        return self._write_agents(cwd, [f.read_text() for f in files], [f.stem for f in files])

    def agent_args(self, cwd: Path) -> list[str]:
        """Each agent in ``cwd/.codex/agents/`` named to the session as configuration: codex
        reads no project-local configuration in a directory it does not trust."""
        out = []
        for f in sorted((cwd / ".codex" / "agents").glob("*.toml")):
            name = f.stem
            out += ["-c", f"agents.{name}.config_file={json.dumps(str(f.resolve()))}"]
        return out

    def session_agents(self, cwd: Path, agents: list[str]) -> list[str]:
        """``agents`` written as codex agents in ``cwd`` and named to the session."""
        self._write_agents(cwd, agents)
        return self.agent_args(cwd)

    @staticmethod
    def _write_agents(cwd: Path, texts: list[str], stems: list[str] | None = None) -> list[str]:
        names, out = [], cwd / ".codex" / "agents"
        for i, text in enumerate(texts):
            d = agent_definition(text)
            name = d["name"] or (stems[i] if stems else f"agent{i}")
            out.mkdir(parents=True, exist_ok=True)
            (out / f"{name}.toml").write_text(  # JSON strings are valid TOML basic strings
                f"name = {json.dumps(name)}\ndescription = {json.dumps(d['description'])}\n"
                + (f"model = {json.dumps(d['model'])}\n" if d["model"] else "")
                + f"developer_instructions = {json.dumps(d['prompt'])}\n"
            )
            names.append(name)
        return names

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
    """The backend named, else ``X_FORGE_AGENT``, else claude."""
    name = name or os.environ.get("X_FORGE_AGENT") or "claude"
    if name not in BACKENDS:
        raise AgentError(f"unknown agent CLI {name!r} (one of {', '.join(BACKENDS)})")
    return BACKENDS[name]


def available() -> list[str]:
    return [n for n, b in BACKENDS.items() if shutil.which(b.binary)]
