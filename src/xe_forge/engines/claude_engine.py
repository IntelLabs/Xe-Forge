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


def _find_claude_bin(explicit: str = "") -> str | None:
    """Locate the claude CLI binary.

    Resolution order:
    1. Explicit path from config / CLAUDE_BIN env var (if non-empty and executable).
    2. PATH lookup via shutil.which.
    3. Glob for the latest VS Code extension installation:
       ~/.vscode-server/extensions/anthropic.claude-code-*/resources/native-binary/claude
       The newest version (by directory name sort) is preferred.
    """
    if explicit:
        p = Path(explicit)
        if p.is_file() and os.access(p, os.X_OK):
            return str(p)
        logger.warning("CLAUDE_BIN='%s' not found or not executable", explicit)

    found = shutil.which("claude")
    if found:
        return found

    # VS Code extension glob — version directories sort lexicographically so the
    # largest string is the newest release (e.g. 2.1.241 > 2.1.237).
    vscode_base = Path.home() / ".vscode-server" / "extensions"
    candidates = sorted(
        vscode_base.glob("anthropic.claude-code-*/resources/native-binary/claude"),
        key=lambda p: p.parts[-4],  # sort on the extension directory name
        reverse=True,
    )
    for candidate in candidates:
        if candidate.is_file() and os.access(candidate, os.X_OK):
            logger.info("Found claude binary: %s", candidate)
            return str(candidate)

    return None


def _build_claude_env(config) -> dict[str, str]:
    """Build the subprocess environment for the claude CLI.

    Maps xe-forge LLM config (OPENAI_API_KEY / OPENAI_API_BASE / LLM_MODEL,
    which may point at an Azure Anthropic endpoint) to the ANTHROPIC_* variables
    that the claude CLI expects. Existing ANTHROPIC_* vars in the process
    environment take priority so they are not accidentally overwritten.
    """
    env = dict(os.environ)

    api_key = (
        env.get("ANTHROPIC_API_KEY")
        or config.llm.api_key
        or env.get("OPENAI_API_KEY", "")
    )
    api_base = (
        env.get("ANTHROPIC_BASE_URL")
        or config.llm.api_base
        or env.get("OPENAI_API_BASE", "")
    )
    model = (
        env.get("ANTHROPIC_MODEL")
        or env.get("LLM_MODEL")
        or config.llm.model
    )

    if api_key:
        env["ANTHROPIC_API_KEY"] = api_key
    if api_base:
        env["ANTHROPIC_BASE_URL"] = api_base
    if model:
        env["ANTHROPIC_MODEL"] = model

    return env


class ClaudeEngine(BaseEngine):
    """Generate a Claude Code workspace and optionally launch ``claude``."""

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

        claude_bin = _find_claude_bin(self.config.engine.claude_bin)

        mcp_servers = []
        if self.config.mcp.intel_perf_enabled:
            mcp_servers.append("intel-perf")
        if self.config.mcp.intel_profiler_mcp_enabled:
            mcp_servers.append("intel-profiler")

        print(f"\nClaude Code workspace ready at: {workspace}")
        if mcp_servers:
            print(f"  MCP servers configured: {', '.join(mcp_servers)}")
        if claude_bin:
            print(f"  Claude binary: {claude_bin}")
            print("Run:")
            print(f"  cd {workspace}")
            print(f"  {claude_bin} /optimize-kernel {kernel_name}")
        else:
            print("  WARNING: claude binary not found (set CLAUDE_BIN or install Claude Code)")
            print("Run manually:")
            print(f"  cd {workspace}")
            print(f"  claude /optimize-kernel {kernel_name}")

        if self.config.engine.auto_launch:
            self._launch_claude(workspace, kernel_name, claude_bin)

        return OptimizationResult(
            kernel_name=kernel_name,
            original_code=kernel_code,
            success=True,
        )

    def _launch_claude(
        self,
        workspace: Path,
        kernel_name: str,
        claude_bin: str | None,
    ) -> None:
        """Launch the claude CLI in the workspace with API credentials set."""
        if not claude_bin:
            logger.warning(
                "claude binary not found — workspace generated at %s but not launched. "
                "Set CLAUDE_BIN to the full path of the claude executable.",
                workspace,
            )
            return

        env = _build_claude_env(self.config)

        cmd = [
            claude_bin,
            "-p",
            f"/optimize-kernel {kernel_name}",
            "--dangerously-skip-permissions",
            "--max-turns",
            "80",
        ]
        model = env.get("ANTHROPIC_MODEL", "")
        if model:
            cmd += ["--model", model]

        print(f"\nLaunching Claude Code in {workspace}...")
        subprocess.Popen(cmd, cwd=str(workspace), env=env)
