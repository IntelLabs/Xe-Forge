"""Claude Code engine: generates an agent-driven workspace for optimization."""

from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path

from xe_forge.engines.base import BaseEngine
from xe_forge.models import OptimizationResult, OptimizationStage

logger = logging.getLogger(__name__)


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

        print(f"\nClaude Code workspace ready at: {workspace}")
        print("Run:")
        print(f"  cd {workspace}")
        print(f"  claude /optimize-kernel {kernel_name}")

        if self.config.engine.auto_launch:
            self._launch_claude(workspace, kernel_name)

        return OptimizationResult(
            kernel_name=kernel_name,
            original_code=kernel_code,
            success=True,
        )

    def _launch_claude(self, workspace: Path, kernel_name: str) -> None:
        """Launch ``claude`` CLI in the workspace."""
        claude_bin = shutil.which("claude")
        if not claude_bin:
            logger.warning("'claude' CLI not found in PATH. Workspace generated but not launched.")
            return
        print(f"\nLaunching Claude Code in {workspace}...")
        subprocess.Popen(
            [
                claude_bin,
                "-p",
                f"/optimize-kernel {kernel_name}",
                "--dangerously-skip-permissions",
                "--max-turns",
                "80",
            ],
            cwd=str(workspace),
        )
