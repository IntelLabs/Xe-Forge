"""Rung 3 of the enablement ladder: the attempt-scoped runtime (plan §5.6, §24 Tier C).

The venvs here are real (`uv venv` is fast and needs no network); the *install*
step is a stub installer, because `uv pip install` would hit an index. The gate
discipline under test is the same as `runnable_gate`'s: a climb earns KEEP only
when the workload boots under the scoped python AND re-passes the eval — and the
environment survives only a KEEP.
"""

from __future__ import annotations

import sys
from pathlib import Path

from xe_forge.orbit.enablement import (
    IMPLEMENTED_RUNGS,
    CapabilityGap,
    Rung,
    climb_missing_package,
    create_scoped_runtime,
    diagnose,
)
from xe_forge.orbit.executor import RunResult


def recording_installer(returncode: int = 0, stderr: str = ""):
    """An installer stub that records its calls instead of touching an index."""
    calls: list[tuple[Path, list[str]]] = []

    def installer(python: Path, packages: list[str]) -> tuple[int, str, str]:
        calls.append((python, list(packages)))
        return returncode, "", stderr

    return installer, calls


class RecordingExecutor:
    """Records the exact command the climb runs, without running it."""

    def __init__(self, returncode: int = 0, stdout: str = "serving", stderr: str = ""):
        self.commands: list[list[str]] = []
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr

    def run(self, cmd, env=None, cwd=None, timeout=1800.0) -> RunResult:
        self.commands.append(list(cmd))
        return RunResult(
            command=list(cmd),
            returncode=self.returncode,
            stdout=self.stdout,
            stderr=self.stderr,
        )


def missing_package_gaps(*modules: str) -> list[CapabilityGap]:
    """Real gaps, produced by the real classifier — not hand-built lookalikes."""
    stderr = "\n".join(f"ModuleNotFoundError: No module named '{m}'" for m in modules)
    gaps = diagnose(1, "", stderr)
    assert {g.kind for g in gaps} == {"missing_package"}
    return gaps


class TestScopedRuntime:
    def test_creates_a_real_venv_with_its_own_interpreter(self, tmp_path):
        installer, _calls = recording_installer()
        runtime = create_scoped_runtime(tmp_path, ["torch"], installer=installer)
        assert runtime.created
        assert runtime.venv_path.name.startswith("venv-")
        assert runtime.python.exists()
        assert runtime.installed == ["torch"]

    def test_installer_receives_the_venv_python_and_the_packages(self, tmp_path):
        installer, calls = recording_installer()
        runtime = create_scoped_runtime(tmp_path, ["torch", "pyyaml"], installer=installer)
        assert calls == [(runtime.python, ["torch", "pyyaml"])]

    def test_repeat_attempt_reuses_the_environment_and_says_so(self, tmp_path):
        installer, calls = recording_installer()
        first = create_scoped_runtime(tmp_path, ["torch"], installer=installer)
        second = create_scoped_runtime(tmp_path, ["torch"], installer=installer)
        assert second.venv_path == first.venv_path
        assert not second.created
        assert "reusing" in second.reason
        # The installer still runs on reuse — a reused venv is re-verified, and a
        # previously broken one is healed rather than trusted.
        assert len(calls) == 2

    def test_package_order_does_not_change_the_environment(self, tmp_path):
        installer, _calls = recording_installer()
        first = create_scoped_runtime(tmp_path, ["torch", "triton"], installer=installer)
        second = create_scoped_runtime(tmp_path, ["triton", "torch"], installer=installer)
        assert second.venv_path == first.venv_path

    def test_different_package_sets_get_different_environments(self, tmp_path):
        installer, _calls = recording_installer()
        first = create_scoped_runtime(tmp_path, ["torch"], installer=installer)
        second = create_scoped_runtime(tmp_path, ["vllm"], installer=installer)
        assert first.venv_path != second.venv_path

    def test_installer_failure_is_named_not_raised(self, tmp_path):
        installer, _calls = recording_installer(
            returncode=1, stderr="error: No matching distribution found for nonesuch"
        )
        runtime = create_scoped_runtime(tmp_path, ["nonesuch"], installer=installer)
        assert runtime.installed == []
        assert "install failed: nonesuch" in runtime.reason
        assert "No matching distribution found" in runtime.reason


class TestClimb:
    """The rung-3 climb, gated exactly as runnable_gate gates."""

    def test_boot_plus_passing_eval_is_kept_and_the_environment_survives(self, tmp_path):
        installer, _calls = recording_installer()
        result = climb_missing_package(
            missing_package_gaps("torch"),
            ["python", "-c", "pass"],
            tmp_path,
            quality=lambda: True,
            installer=installer,
        )
        assert result.rung is Rung.SCOPED_RUNTIME
        assert result.kept and result.gate.kept
        assert result.runtime.venv_path.exists()
        assert "retained" in result.reason

    def test_boot_without_an_eval_is_booted_not_kept(self, tmp_path):
        installer, _calls = recording_installer()
        result = climb_missing_package(
            missing_package_gaps("torch"),
            ["python", "-c", "pass"],
            tmp_path,
            installer=installer,
        )
        assert result.gate.booted and not result.gate.evaluated
        assert not result.kept
        assert "boot alone does not" in result.gate.reason
        # Not kept means discarded — an unproven venv is not left behind.
        assert not result.runtime.venv_path.exists()

    def test_boot_plus_failing_eval_is_not_a_keep(self, tmp_path):
        installer, _calls = recording_installer()
        result = climb_missing_package(
            missing_package_gaps("torch"),
            ["python", "-c", "pass"],
            tmp_path,
            quality=lambda: False,
            installer=installer,
        )
        assert result.gate.booted and result.gate.eval_passed is False
        assert not result.kept
        assert not result.runtime.venv_path.exists()

    def test_boot_failure_attaches_the_gate_diagnosis(self, tmp_path):
        installer, _calls = recording_installer()
        boot_cmd = [
            "python",
            "-c",
            "import sys; sys.stderr.write('RuntimeError: xpu is not available'); sys.exit(1)",
        ]
        result = climb_missing_package(
            missing_package_gaps("torch"), boot_cmd, tmp_path, installer=installer
        )
        assert not result.kept and not result.gate.booted
        assert result.gate.gaps and result.gate.gaps[0].kind == "missing_device"
        assert not result.runtime.venv_path.exists()

    def test_install_failure_discards_and_never_boots(self, tmp_path):
        installer, _calls = recording_installer(returncode=1, stderr="resolution error")
        executor = RecordingExecutor()
        result = climb_missing_package(
            missing_package_gaps("torch"),
            ["python", "-c", "pass"],
            tmp_path,
            executor=executor,
            installer=installer,
        )
        assert not result.kept
        assert result.gate is None
        assert executor.commands == []
        assert "install failed" in result.reason and "discarded" in result.reason
        assert not result.runtime.venv_path.exists()

    def test_gaps_without_missing_packages_get_an_honest_refusal(self, tmp_path):
        gaps = diagnose(1, "", "ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY on allocation")
        result = climb_missing_package(gaps, ["python", "-c", "pass"], tmp_path)
        assert not result.kept
        assert result.runtime is None and result.gate is None
        assert "rung 3 addresses missing packages" in result.reason
        assert "oom" in result.reason

    def test_module_name_maps_to_its_distribution_name(self, tmp_path):
        installer, calls = recording_installer()
        climb_missing_package(
            missing_package_gaps("dotenv"),
            ["python", "-c", "pass"],
            tmp_path,
            executor=RecordingExecutor(),
            installer=installer,
        )
        assert calls[0][1] == ["python-dotenv"]

    def test_submodule_maps_through_its_top_level_module(self, tmp_path):
        installer, calls = recording_installer()
        climb_missing_package(
            missing_package_gaps("yaml.loader"),
            ["python", "-c", "pass"],
            tmp_path,
            executor=RecordingExecutor(),
            installer=installer,
        )
        assert calls[0][1] == ["pyyaml"]

    def test_unknown_module_passes_through_with_a_note(self, tmp_path):
        installer, calls = recording_installer()
        result = climb_missing_package(
            missing_package_gaps("frobnicator"),
            ["python", "-c", "pass"],
            tmp_path,
            executor=RecordingExecutor(),
            installer=installer,
        )
        assert calls[0][1] == ["frobnicator"]
        assert "not in the distribution map" in result.reason

    def test_python_argv0_is_replaced_with_the_scoped_interpreter(self, tmp_path):
        installer, _calls = recording_installer()
        executor = RecordingExecutor()
        result = climb_missing_package(
            missing_package_gaps("torch"),
            [sys.executable, "-c", "pass"],
            tmp_path,
            executor=executor,
            installer=installer,
        )
        assert executor.commands == [[str(result.runtime.python), "-c", "pass"]]

    def test_non_python_argv0_gets_the_scoped_interpreter_prepended(self, tmp_path):
        installer, _calls = recording_installer()
        executor = RecordingExecutor()
        result = climb_missing_package(
            missing_package_gaps("torch"),
            ["bench.py", "--fast"],
            tmp_path,
            executor=executor,
            installer=installer,
        )
        assert executor.commands == [[str(result.runtime.python), "bench.py", "--fast"]]

    def test_the_scoped_interpreter_actually_runs_the_boot(self, tmp_path):
        # No executor stub: the real LocalExecutor runs the real venv python. The
        # boot exits 0 only when it finds itself inside a venv, so a pass proves
        # the workload ran under the scoped interpreter, not the ambient one.
        installer, _calls = recording_installer()
        result = climb_missing_package(
            missing_package_gaps("torch"),
            [
                "python",
                "-c",
                "import sys; raise SystemExit(0 if sys.prefix != sys.base_prefix else 1)",
            ],
            tmp_path,
            quality=lambda: True,
            installer=installer,
        )
        assert result.gate.booted
        assert result.kept


class TestLadderState:
    def test_scoped_runtime_is_now_implemented_and_rungs_4_5_are_not(self):
        assert Rung.SCOPED_RUNTIME in IMPLEMENTED_RUNGS
        assert Rung.SOURCE_LOCALIZE not in IMPLEMENTED_RUNGS
        assert Rung.COMPILED_BUILD not in IMPLEMENTED_RUNGS

    def test_missing_package_gap_is_no_longer_deferred(self):
        gaps = missing_package_gaps("vllm")
        assert not gaps[0].deferred
        assert "actionable now" in gaps[0].format()
