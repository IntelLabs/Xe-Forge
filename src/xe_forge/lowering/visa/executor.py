"""Parent side of execution: finalize, run in a watched child, classify.

The parent never touches the device. It compiles with :class:`VisaCompiler`,
hands the binary to :mod:`xe_forge.lowering.visa.runner` in a fresh process under
:func:`~xe_forge.core.watchdog.run_watched`, and turns what comes back into an
:class:`~xe_forge.lowering.visa.diagnostics.Evaluation`.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any

from xe_forge.core.watchdog import run_watched
from xe_forge.lowering.visa.compiler import AbiBuild, VisaCompiler
from xe_forge.lowering.visa.diagnostics import (
    CorrectnessResult,
    Evaluation,
    FailureCategory,
    PerfResult,
    RuntimeResult,
    sanitize,
)
from xe_forge.lowering.visa.verifier import VerificationCase

# A correct candidate must be this much faster to count as faster: below it, a
# difference between two medians is not distinguished from noise.
MIN_SPEEDUP = 1.02


class DeviceUnhealthy(RuntimeError):
    """The device stopped answering after an attempt; the run cannot continue."""


class VisaExecutor:
    def __init__(
        self,
        compiler: VisaCompiler,
        workroot: str | Path,
        *,
        timeout_s: float = 180,
        python: str = sys.executable,
    ):
        self.compiler = compiler
        self.workroot = Path(workroot)
        self.workroot.mkdir(parents=True, exist_ok=True)
        self.timeout_s = timeout_s
        self.python = python
        self._jobs = 0

    def _run_job(self, job: dict[str, Any], env: dict[str, str] | None = None) -> tuple[dict | None, Any]:
        self._jobs += 1
        # Never reuse a job directory: it holds that job's Triton cache, and a cache
        # from another run would serve a stale binary.
        jobdir = Path(tempfile.mkdtemp(prefix=f"job_{self._jobs:04d}_{job['mode']}_", dir=self.workroot))
        job.setdefault("triton_cache", str(jobdir / "triton_cache"))
        (jobdir / "job.json").write_text(json.dumps(job, indent=2, default=str))
        proc = run_watched(
            [self.python, "-m", "xe_forge.lowering.visa.runner", str(jobdir / "job.json")],
            self.timeout_s,
            env={**os.environ, **(env or {})},
            capture=True,
        )
        (jobdir / "stdout.txt").write_text(proc.stdout)
        (jobdir / "stderr.txt").write_text(proc.stderr)
        result_path = jobdir / "result.json"
        result = json.loads(result_path.read_text()) if result_path.exists() else None
        return result, proc

    def capture(self, module_path: str, spec_path: str | None, variant: str | None, kernel: str | None) -> dict:
        job = {
            "mode": "capture",
            "module_path": str(Path(module_path).resolve()),
            "spec_path": str(Path(spec_path).resolve()) if spec_path else None,
            "variant": variant,
            "kernel": kernel,
            "workdir": str(self.workroot / "capture"),
        }
        # The target's own IGC output is dumped into a sealed directory: it feeds the
        # artifact guard and the corpus similarity screen, and is deleted afterwards.
        # It never reaches a prompt.
        sealed = Path(tempfile.mkdtemp(prefix="sealed_igc_", dir=self.workroot))
        result, proc = self._run_job(job, env={
            "IGC_ShaderDumpEnable": "1", "IGC_DumpToCustomDir": str(sealed) + "/", "NEO_CACHE_PERSISTENT": "0",
        })
        if result is None or not result.get("ok"):
            detail = (result or {}).get("traceback") or proc.stderr[-4000:]
            raise RuntimeError(f"capturing the kernel launch failed:\n{detail}")
        result["triton_cache"] = job["triton_cache"]
        result["sealed_igc_dump"] = str(sealed)
        return result

    def build_abi(self, captured: dict, config, guard=None):
        """ABI stub for a launch configuration: (AbiBuild, stub_py)."""
        from xe_forge.lowering.triton_analyzer import LaunchRecord

        cache = self.workroot / "abi_cache" / config.key()
        job = {"mode": "stub", "launch": captured["launch"], "config": config.to_json(), "workdir": str(cache)}
        result, proc = self._run_job(job)
        if result is None or not result.get("ok"):
            detail = (result or {}).get("error") or proc.stderr[-2000:]
            raise RuntimeError(f"building the ABI stub for {config} failed: {detail}")
        record = LaunchRecord.from_json(captured["launch"])
        abi_build = self.compiler.discover_abi(Path(result["stub_spv"]), record.kernel, guard)
        return abi_build, result["stub_py"]

    def health_check(self) -> bool:
        probe = (
            "import torch; x = torch.ones(1024, device='xpu'); "
            "assert float((x + 1).sum()) == 2048.0"
        )
        proc = run_watched([self.python, "-c", probe], 60, capture=True)
        return proc.returncode == 0 and not proc.timed_out

    def evaluate(
        self,
        visa_text: str,
        abi_build: AbiBuild,
        captured: dict,
        cases: list[VerificationCase],
        *,
        module_path: str,
        spec_path: str | None,
        tolerance: dict[str, float] | None = None,
        transcendental: bool = False,
        measure_perf: bool = False,
        keep_dir: Path | None = None,
        forbidden_lines: set[str] | None = None,
        config=None,
        stub_py: str | None = None,
    ) -> Evaluation:
        build = self.compiler.compile(visa_text, abi_build, keep_dir=keep_dir)
        if not build.result.success:
            return Evaluation(FailureCategory.FINALIZER_ERROR, build.result)
        zebin_dir = keep_dir or (self.workroot / "zebin")
        zebin_dir.mkdir(parents=True, exist_ok=True)
        zebin_path = zebin_dir / "kernel.zebin"
        zebin_path.write_bytes(build.zebin)
        job = {
            "mode": "evaluate",
            "module_path": str(Path(module_path).resolve()),
            "spec_path": str(Path(spec_path).resolve()) if spec_path else None,
            "variant": captured.get("variant"),
            "launch": captured["launch"],
            "stub_py": stub_py or captured["stub_py"],
            "config": config.to_json() if config is not None else None,
            "zebin_path": str(zebin_path),
            "cases": [asdict(c) for c in cases],
            "tolerance": tolerance or {},
            "transcendental": transcendental,
            "measure_perf": measure_perf,
        }
        result, proc = self._run_job(job)
        if proc.timed_out:
            if not self.health_check():
                raise DeviceUnhealthy("the device did not answer after a timed-out attempt")
            return Evaluation(
                FailureCategory.TIMEOUT,
                build.result,
                RuntimeResult(False, error=f"the kernel did not finish within {self.timeout_s:.0f}s and was killed "
                              "(a loop that never exits, or a wait on something that never arrives)", timed_out=True),
            )
        if result is None or not result.get("ok"):
            healthy = self.health_check()
            if not healthy:
                raise DeviceUnhealthy("the device did not answer after a crashed attempt")
            detail = (result or {}).get("error") or sanitize(proc.stderr[-3000:], forbidden_lines=forbidden_lines)
            if proc.signal:
                detail = f"the process was killed by signal {proc.signal}\n{detail}"
            return Evaluation(FailureCategory.RUNTIME_ERROR, build.result, RuntimeResult(False, error=detail))
        runtime = RuntimeResult(**result["runtime"])
        if not runtime.success:
            runtime.error = sanitize(runtime.error, forbidden_lines=forbidden_lines)
            return Evaluation(FailureCategory.RUNTIME_ERROR, build.result, runtime,
                              detail=json.dumps(result.get("failing_case")))
        correctness = CorrectnessResult(**result["correctness"])
        if not correctness.success:
            return Evaluation(FailureCategory.INCORRECT, build.result, runtime, correctness)
        if "performance" not in result:
            return Evaluation(FailureCategory.CORRECT, build.result, runtime, correctness)
        perf = PerfResult(**result["performance"])
        category = FailureCategory.CORRECT_FASTER if perf.speedup >= MIN_SPEEDUP else FailureCategory.CORRECT_SLOWER
        return Evaluation(category, build.result, runtime, correctness, perf)
