"""Claude Code engine: generates an agent-driven workspace for optimization."""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
from pathlib import Path

from xe_forge.engines.base import BaseEngine
from xe_forge.models import OptimizationResult, OptimizationStage

logger = logging.getLogger(__name__)

LOG_NAME = "claude-session.log"


class ClaudeEngine(BaseEngine):
    """Generate a Claude Code workspace and optionally launch ``claude``.

    When ``auto_launch`` is set the loop runs itself: nobody is at the terminal,
    so what this returns is the only account of what happened. It therefore
    waits for the session, propagates its exit status, and keeps its output --
    a launcher that reports success it did not observe is worse than one that
    does not launch at all, because a stage downstream will consume it.
    """

    def optimize(
        self,
        kernel_code: str,
        reference_code: str = "",
        kernel_name: str | None = None,
        input_shapes: list[tuple[int, ...]] | None = None,
        spec_path: str | None = None,
        variant_type: str = "bench-gpu",
        target_dtype: str | None = None,
        rtol: float | None = None,
        atol: float | None = None,
        stages: list[OptimizationStage] | None = None,
    ) -> OptimizationResult:
        from xe_forge.claude.generator import generate_workspace

        kernel_name = kernel_name or "kernel"
        workspace = Path(self.config.engine.workspace).resolve()
        workspace.mkdir(parents=True, exist_ok=True)

        generate_workspace(
            workspace=workspace,
            config=self.config,
            kernel_name=kernel_name,
            kernel_code=kernel_code,
            reference_code=reference_code,
            spec_path=spec_path,
            variant_type=variant_type,
            target_dtype=target_dtype,
        )

        print(f"\nClaude Code workspace ready at: {workspace}")
        print("Run:")
        print(f"  cd {workspace}")
        print(f"  claude /optimize-kernel {kernel_name}")

        if not self.config.engine.auto_launch:
            # Nothing was attempted, so there is nothing to report as done.
            return OptimizationResult(
                kernel_name=kernel_name,
                original_code=kernel_code,
                success=True,
            )

        returncode, error = self._launch_claude(workspace, kernel_name)

        # With trials disabled there is no record of what the session did, so
        # its exit status is the only thing there is to report.
        observable = self.config.trial.enabled
        best = self._best_trial(workspace, kernel_name) if observable else None
        if returncode == 0 and observable and best is None:
            # The session ended cleanly having produced no correct, faster
            # trial -- a kernel that never built, or one that never beat its
            # baseline. Exiting zero is what the CLI does when it runs out of
            # things to say; it is not evidence that anything was optimized.
            error = (
                f"claude session completed but no correct trial was recorded for "
                f"{kernel_name!r}; see {workspace / LOG_NAME}"
            )
            logger.error(error)

        return OptimizationResult(
            kernel_name=kernel_name,
            original_code=kernel_code,
            optimized_code=(best or {}).get("code"),
            total_speedup=(best or {}).get("speedup"),
            success=returncode == 0 and (best is not None or not observable),
            error_message=error,
        )

    def _best_trial(self, workspace: Path, kernel_name: str) -> dict | None:
        """The best correct trial the session recorded, if any."""
        try:
            from xe_forge.core.trial_manager import TrialManager

            trials_dir = Path(self.config.trial.trials_dir)
            if not trials_dir.is_absolute():
                trials_dir = workspace / trials_dir
            mgr = TrialManager(trials_dir)
            if not mgr.exists(kernel_name):
                # A session that never ran `trial init` -- the caller is told
                # by the absent result, not by a warning about a missing file.
                logger.debug("No trial tree under %s for %r", trials_dir, kernel_name)
                return None
            best = mgr.get_best(kernel_name)
        except Exception as exc:
            logger.warning("Could not read trial tree for %r: %s", kernel_name, exc)
            return None
        if best is None:
            return None
        path = best.get("file_path")
        if path and Path(path).exists():
            best["code"] = Path(path).read_text()
        return best

    def _child_env(self) -> dict[str, str]:
        """Environment for the CLI, mapped from ``config.llm``.

        The config names an endpoint the litellm way (``OPENAI_API_BASE`` and
        friends, which is what the DSPy engine wants); the CLI reads the
        ``ANTHROPIC_*`` spelling. Mapping here means one configured endpoint has
        two consumers and the key exists in no second place -- not in a file
        under the workspace, not in the rendered instructions, not in the log.
        """
        env = os.environ.copy()
        llm = self.config.llm
        if llm.api_base:
            env.setdefault("ANTHROPIC_BASE_URL", llm.api_base)
        if llm.api_key:
            env.setdefault("ANTHROPIC_AUTH_TOKEN", llm.api_key)
        if llm.model:
            # litellm prefixes a provider ("openai/gpt-4o"); the CLI wants the
            # bare model id.
            env.setdefault("ANTHROPIC_MODEL", llm.model.split("/")[-1])
        return env

    def _launch_claude(self, workspace: Path, kernel_name: str) -> tuple[int, str | None]:
        """Run one headless ``claude`` session to completion.

        Returns ``(returncode, error_message)``.
        """
        claude_bin = shutil.which("claude")
        if not claude_bin:
            # An error, not a warning: auto_launch was asked for and did not
            # happen, and this is the failure a caller is least able to notice
            # on its own.
            msg = (
                "'claude' CLI not found in PATH; auto-launch requested but the workspace "
                f"at {workspace} was only generated, not optimized."
            )
            logger.error(msg)
            return 1, msg

        max_turns = self.config.engine.max_turns
        log_path = workspace / LOG_NAME
        cmd = [
            claude_bin,
            "-p",
            f"/optimize-kernel {kernel_name}",
            "--dangerously-skip-permissions",
            "--max-turns",
            str(max_turns),
        ]

        print(f"\nLaunching Claude Code in {workspace} (max {max_turns} turns)...")
        print(f"  session log: {log_path}")

        try:
            with open(log_path, "w") as log:
                proc = subprocess.Popen(
                    cmd,
                    cwd=str(workspace),
                    env=self._child_env(),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                )
                assert proc.stdout is not None
                for line in proc.stdout:
                    # Teed rather than redirected: an unattended run needs the
                    # file, and a watched one needs the terminal.
                    print(line, end="")
                    log.write(line)
                returncode = proc.wait()
        except OSError as exc:
            msg = f"could not run 'claude': {exc}"
            logger.error(msg)
            return 1, msg

        if returncode != 0:
            # A turn limit reached mid-search shows up here too. That is a
            # stopping condition like any other, and the caller is told which
            # by the session's own finalize summary, not by a raised limit.
            msg = f"claude session exited {returncode}; see {log_path}"
            logger.error(msg)
            return returncode, msg

        print(f"\nClaude session finished. Log: {log_path}")
        return 0, None
