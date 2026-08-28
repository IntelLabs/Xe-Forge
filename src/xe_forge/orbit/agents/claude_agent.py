"""
Claude Code as the repository agent: shells out to the `claude` CLI in print mode, so
Orbit depends on a binary rather than an SDK and its analysis path never imports an
LLM client. Design rationale: docs/DESIGN.md.
"""

from __future__ import annotations

import os
import shutil
import subprocess

from xe_forge.orbit.agents.base import AgentAnswer, AgentTask, BaseRepoAgent, RepoAgentError

CLAUDE_BIN = "claude"

# A resolution question is a few greps and a file read. Beyond this something is wrong —
# usually the search root is far too broad — and waiting longer will not fix it.
DEFAULT_TIMEOUT_S = 180.0


class ClaudeRepoAgent(BaseRepoAgent):
    """Answers repository questions by invoking the Claude Code CLI in print mode."""

    name = "claude"

    def __init__(
        self,
        binary: str = CLAUDE_BIN,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        extra_args: tuple[str, ...] = (),
    ) -> None:
        self.binary = binary
        self.timeout_s = timeout_s
        self.extra_args = extra_args

    def available(self) -> bool:
        return shutil.which(self.binary) is not None

    def ask(self, task: AgentTask) -> AgentAnswer:
        if not self.available():
            raise RepoAgentError(
                f"{self.binary!r} is not on PATH. Install Claude Code, or select a "
                f"different RepoAgent provider by config."
            )

        prompt = self.build_prompt(task)
        argv = [self.binary, "-p", prompt, *self.extra_args]

        # Run inside the tree being searched so the agent's own file tools are scoped to
        # it. An agent pointed at the whole filesystem is slow and its answers are not
        # reproducible.
        cwd = str(task.search_root) if task.search_root and task.search_root.is_dir() else None

        try:
            result = subprocess.run(
                argv,
                capture_output=True,
                text=True,
                timeout=self.timeout_s,
                cwd=cwd,
                env={**os.environ, "CLAUDE_NONINTERACTIVE": "1"},
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise RepoAgentError(
                f"claude did not answer within {self.timeout_s:.0f}s; narrow the search "
                f"root or reduce the candidate list"
            ) from exc
        except OSError as exc:
            raise RepoAgentError(f"could not run {self.binary!r}: {exc}") from exc

        if result.returncode != 0:
            tail = (result.stderr or "").strip().splitlines()
            raise RepoAgentError(
                f"claude exited {result.returncode}: {tail[-1] if tail else 'no stderr'}"
            )

        return self.parse_answer(result.stdout)
