"""CPU-only tests for ClaudeEngine's synchronous result path (plan §9.9).

No claude binary, GPU, or jinja rendering is involved: the workspace
generator is replaced by a stub module that reproduces the real layout,
and the claude CLI is simulated by monkeypatching ``shutil.which`` and
``subprocess.run``. The invariant under test is plan §19's: the engine
never reports success, because it never measures — even when the session
hands back code.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import types
from pathlib import Path

from xe_forge.config import Config
from xe_forge.engines.claude_engine import ClaudeEngine

BASELINE = "def kernel():\n    return 'baseline'\n"
OPTIMIZED = "def kernel():\n    return 'optimized'\n"


def _stub_generator(monkeypatch) -> None:
    """Install a stub ``xe_forge.claude.generator`` so no templates render.

    Pre-seeding ``sys.modules`` means the engine's lazy import never touches
    the real module, so jinja2 is not needed. The stub reproduces the one
    layout fact the sync path depends on: the baseline kernel lands at
    ``test_kernels/<name>.py`` inside the workspace.
    """
    stub = types.ModuleType("xe_forge.claude.generator")

    def generate_workspace(workspace, config, kernel_name, kernel_code, **kwargs):
        tk_dir = Path(workspace) / "test_kernels"
        tk_dir.mkdir(parents=True, exist_ok=True)
        (tk_dir / f"{kernel_name}.py").write_text(kernel_code)

    stub.generate_workspace = generate_workspace
    monkeypatch.setitem(sys.modules, "xe_forge.claude.generator", stub)


def _sync_config(tmp_path) -> Config:
    cfg = Config()
    cfg.engine.engine = "claude"
    cfg.engine.workspace = str(tmp_path / "ws")
    cfg.engine.synchronous = True
    return cfg


def _claude_on_path(monkeypatch, found: bool = True) -> None:
    monkeypatch.setattr(
        shutil, "which", lambda name: "/usr/bin/claude" if (found and name == "claude") else None
    )


def test_sync_session_edit_returns_unmeasured_code(tmp_path, monkeypatch):
    """A session that finalizes a kernel yields the code, but success stays False."""
    _stub_generator(monkeypatch)
    _claude_on_path(monkeypatch)
    cfg = _sync_config(tmp_path)
    calls = {}

    def fake_run(cmd, cwd=None, capture_output=False, text=False, timeout=None, check=True):
        calls["cmd"] = cmd
        calls["timeout"] = timeout
        # Simulate the documented finalize step: CLAUDE.md step 4 writes the
        # best trial to output/<name>_optimized.py in the workspace root.
        out_dir = Path(cwd) / "output"
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "demo_optimized.py").write_text(OPTIMIZED)
        return subprocess.CompletedProcess(cmd, 0, stdout="done", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    result = ClaudeEngine(cfg).optimize(BASELINE, kernel_name="demo")

    assert result.success is False
    assert result.optimized_code == OPTIMIZED
    assert result.original_code == BASELINE
    assert "measur" in result.error_message.lower()  # "NOT been measured"
    # The blocking run must be non-interactive and honor the configured timeout.
    assert "-p" in calls["cmd"]
    assert calls["timeout"] == cfg.engine.claude_timeout_s == 1800.0


def test_sync_no_edit_reports_failure(tmp_path, monkeypatch):
    """A session that exits cleanly without editing anything is not a success."""
    _stub_generator(monkeypatch)
    _claude_on_path(monkeypatch)
    cfg = _sync_config(tmp_path)

    def fake_run(cmd, cwd=None, **kwargs):
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    result = ClaudeEngine(cfg).optimize(BASELINE, kernel_name="demo")

    assert result.success is False
    assert result.optimized_code is None
    assert "no edited kernel" in result.error_message


def test_sync_timeout_reports_failure(tmp_path, monkeypatch):
    """A hung claude run surfaces as a timeout, not as success."""
    _stub_generator(monkeypatch)
    _claude_on_path(monkeypatch)
    cfg = _sync_config(tmp_path)
    cfg.engine.claude_timeout_s = 5.0

    def fake_run(cmd, cwd=None, timeout=None, **kwargs):
        raise subprocess.TimeoutExpired(cmd=cmd, timeout=timeout)

    monkeypatch.setattr(subprocess, "run", fake_run)

    result = ClaudeEngine(cfg).optimize(BASELINE, kernel_name="demo")

    assert result.success is False
    assert result.optimized_code is None
    assert "timed out" in result.error_message
    assert "5" in result.error_message


def test_sync_claude_missing_from_path(tmp_path, monkeypatch):
    """Without the claude CLI the sync path fails honestly instead of hanging."""
    _stub_generator(monkeypatch)
    _claude_on_path(monkeypatch, found=False)
    cfg = _sync_config(tmp_path)

    def fake_run(*args, **kwargs):
        raise AssertionError("subprocess.run must not be called when claude is missing")

    monkeypatch.setattr(subprocess, "run", fake_run)

    result = ClaudeEngine(cfg).optimize(BASELINE, kernel_name="demo")

    assert result.success is False
    assert result.optimized_code is None
    assert "PATH" in result.error_message


def test_default_async_path_unchanged(tmp_path, monkeypatch, capsys):
    """The default config keeps the legacy fire-and-forget behavior exactly."""
    _stub_generator(monkeypatch)
    cfg = Config()
    cfg.engine.workspace = str(tmp_path / "ws")
    assert cfg.engine.synchronous is False  # the sync path is strictly opt-in

    def forbidden(*args, **kwargs):
        raise AssertionError("the async path without auto_launch must not spawn anything")

    monkeypatch.setattr(subprocess, "run", forbidden)
    monkeypatch.setattr(subprocess, "Popen", forbidden)

    result = ClaudeEngine(cfg).optimize(BASELINE, kernel_name="demo")

    # Legacy contract: immediate unconditional success, no code, no message,
    # and the manual launch instructions printed for the user.
    assert result.success is True
    assert result.optimized_code is None
    assert result.error_message is None
    assert "claude /optimize-kernel demo" in capsys.readouterr().out
