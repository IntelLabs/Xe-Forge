"""DSPy agent engine: reusing Claude workspace, driven by a DSPy tool-calling agent."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import dspy
from dspy.utils.callback import BaseCallback

from xe_forge.engines.claude_engine import ClaudeEngine

logger = logging.getLogger(__name__)

LOG_NAME = "agent-session.log"


class _ToolTrace(BaseCallback):
    """Print each tool call as it happens -- an unattended run's only live view."""

    def __init__(self, log):
        self.log = log
        self.calls: dict[str, str] = {}

    def on_tool_start(self, call_id, instance, inputs):
        name = getattr(instance, "name", "tool")
        args = json.dumps(inputs.get("kwargs", inputs), default=str)
        self.calls[call_id] = name
        self._emit(f">>> {name} {args[:300]}")

    def on_tool_end(self, call_id, outputs, exception=None):
        name = self.calls.pop(call_id, "tool")
        text = f"!!! {exception}" if exception is not None else str(outputs)
        self._emit(f"<<< {name}: {text[:600]}")

    def _emit(self, line):
        print(line, flush=True)
        self.log.write(line + "\n")


class AgentEngine(ClaudeEngine):
    """Generate the Claude workspace and run :class:`EngineerAgent` in it."""

    always_launch = True
    log_name = LOG_NAME

    def _launch_claude(self, workspace: Path, kernel_name: str) -> tuple[int, str | None]:
        from xe_forge.agents.engineer import EngineerAgent
        from xe_forge.agents.workspace_tools import WorkspacePolicy, make_workspace_tools
        from xe_forge.cli import _setup_dspy
        from xe_forge.core.lessons import load_lessons_log

        _setup_dspy(self.config)
        external = self.config.external
        lessons = load_lessons_log(external.lessons, kernel_name)
        policy = WorkspacePolicy.for_workspace(
            workspace,
            kernel_repo=external.kernel_repo,
            integration_repos=external.integration_repos,
            lessons_file=lessons.own_file if lessons else None,
            given=("CLAUDE.md", ".claude/commands/optimize-kernel.md"),
        )
        instructions = (workspace / "CLAUDE.md").read_text()
        task = (workspace / ".claude" / "commands" / "optimize-kernel.md").read_text()
        task = task.replace("$ARGUMENTS", kernel_name)
        files = sorted(
            str(p.relative_to(workspace)) for p in (workspace / "test_kernels").glob("*")
        )
        brief = "\n".join(
            [
                f"kernel_name: {kernel_name}",
                f"test_kernels: {', '.join(files) or '(empty)'}",
                f"kernel_repo (read-only): {external.kernel_repo or 'none'}",
                f"lessons file (yours to append to): {lessons.own_file if lessons else 'none'}",
            ]
        )

        log_path = workspace / self.log_name
        max_iters = self.config.engine.max_turns
        print(f"\nRunning DSPy agent in {workspace} (max {max_iters} steps)...")
        print(f"  session log: {log_path}")
        with open(log_path, "w", buffering=1) as log:
            agent = EngineerAgent(
                make_workspace_tools(policy),
                instructions,
                max_iters,
                unfinished=lambda: self._unfinished(workspace, kernel_name),
            )
            try:
                with dspy.context(callbacks=[_ToolTrace(log)]), dspy.track_usage() as usage:
                    result = agent(goal=task, workspace_brief=brief)
            except Exception as exc:
                msg = f"DSPy agent failed: {exc}; see {log_path}"
                log.write(msg + "\n")
                logger.error(msg)
                return 1, msg
            log.write(f"\ntermination_reason: {getattr(result, 'termination_reason', None)}\n")
            log.write(f"usage: {json.dumps(usage.get_total_tokens(), default=str)}\n")
            log.write(f"summary:\n{getattr(result, 'summary', '')}\n")
        print(f"\nAgent summary:\n{getattr(result, 'summary', '')}")
        return 0, None

    def _unfinished(self, workspace: Path, kernel_name: str) -> str | None:
        """Why a session that submitted is not done: nothing but the baseline was measured."""
        from xe_forge.core.trial_manager import TrialManager

        mgr = TrialManager(self._trials_dir(workspace))
        if not mgr.exists(kernel_name):
            return "the trial tree was never initialized (`trial init`)."
        if mgr.has_measured_attempt(kernel_name):
            return None
        return (
            "no trial other than a copy of the baseline has a correct result recorded by "
            "`benchmark --trial-id` yet. A failed profile is not a halting verdict."
        )
