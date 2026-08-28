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
    """Generate a Claude Code workspace and optionally launch ``claude``.

    Two result paths (plan §9.9):

    - Asynchronous (default): generate the workspace, optionally spawn
      ``claude -p`` fire-and-forget, and return immediately with the
      historical unconditional ``success=True``.
    - Synchronous (``engine.synchronous=True``): block on the ``claude`` run,
      read back the kernel the session finalized, and return it with
      ``success=False``. Plan §19 forbids reporting a candidate successful on
      generated reasoning alone, and this engine performs no measurement, so
      even a returned kernel is only a candidate — the caller (orbit's
      ``optimize_kernel_dir`` or the pipeline) benchmarks it and decides.
    """

    # One invocation shared by both paths so their flags cannot drift apart.
    _CLAUDE_FLAGS = ("--dangerously-skip-permissions", "--max-turns", "80")

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

        if self.config.engine.synchronous:
            return self._run_claude_sync(workspace, kernel_name, kernel_code)

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

    def _run_claude_sync(
        self, workspace: Path, kernel_name: str, kernel_code: str
    ) -> OptimizationResult:
        """Run ``claude`` to completion and read back the kernel it produced.

        Every return carries ``success=False``: this engine never benchmarks,
        and plan §19 forbids reporting success without a measured, verified
        result. ``error_message`` is used descriptively — it states why the
        result is unproven (unmeasured code, no edit, timeout, missing CLI),
        not necessarily that something failed.
        """
        result = OptimizationResult(
            kernel_name=kernel_name,
            original_code=kernel_code,
            success=False,
        )

        claude_bin = shutil.which("claude")
        if not claude_bin:
            result.error_message = (
                "'claude' CLI not found in PATH; workspace was generated at "
                f"{workspace} but no synchronous run happened."
            )
            return result

        timeout_s = self.config.engine.claude_timeout_s
        print(f"\nRunning Claude Code synchronously in {workspace} (timeout {timeout_s:.0f}s)...")
        try:
            proc = subprocess.run(
                [claude_bin, "-p", f"/optimize-kernel {kernel_name}", *self._CLAUDE_FLAGS],
                cwd=str(workspace),
                capture_output=True,
                text=True,
                timeout=timeout_s,
                check=False,
            )
        except subprocess.TimeoutExpired:
            result.error_message = (
                f"claude run timed out after {timeout_s:.0f}s; no result was read back."
            )
            return result

        optimized = self._read_back_kernel(workspace, kernel_name, kernel_code)
        if optimized is None:
            detail = f"exit code {proc.returncode}"
            stderr_tail = (proc.stderr or "").strip()[-500:]
            if stderr_tail:
                detail += f"; stderr tail: {stderr_tail}"
            result.error_message = f"claude session produced no edited kernel ({detail})."
            return result

        result.optimized_code = optimized
        result.error_message = (
            "Optimized code was produced synchronously but has NOT been measured "
            "or verified; success stays False per plan §19 (never report success "
            "on generated reasoning alone). The caller must benchmark this code."
        )
        return result

    def _read_back_kernel(self, workspace: Path, kernel_name: str, kernel_code: str) -> str | None:
        """Locate the kernel the session produced, or None if nothing changed.

        The layout comes from the generator's templates: CLAUDE.md step 4
        instructs the session to finalize the best trial to
        ``output/<name>_optimized.py``, and the baseline it starts from is
        written to ``test_kernels/<name>.py``. The finalize target is the
        authoritative location; an in-place edit of the baseline is accepted
        as a fallback. A candidate identical to the input counts as no edit.
        """
        candidates = [
            workspace / "output" / f"{kernel_name}_optimized.py",
            workspace / "test_kernels" / f"{kernel_name}.py",
        ]
        for candidate in candidates:
            if not candidate.is_file():
                continue
            code = candidate.read_text()
            if code.strip() and code != kernel_code:
                return code
        return None

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
                *self._CLAUDE_FLAGS,
            ],
            cwd=str(workspace),
        )
