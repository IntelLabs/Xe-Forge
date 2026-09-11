"""The default build backend: ``ai_bench``'s SYCL compiler, unchanged.

This is a wrapper, not a rewrite. It compiles with
``ai_bench.sycl.compiler.SYCLCompiler`` and runs the resulting standalone binary
as a subprocess, parsing the CUTLASS-style stdout -- exactly what
:class:`~xe_forge.core.sycl_executor.SyclExecutor` did before the backend seam
existed, so a workspace that names no backend sees no change.

Its limits are the reason the seam exists and are worth stating where someone
choosing a backend will read them: no AOT device target, no SPIR-V extension
control, no register mode, and ``BuildSpec.dependencies`` is accepted and then
ignored, because there is nowhere in this toolchain path to put a library.

Taken together those make this backend unsuitable for optimizing a real SYCL
kernel: without an AOT target the kernel is JIT-compiled from generic SPIR-V and
drops off the DPAS and 2D-block-IO paths silently -- correct, and slow, with
nothing in the result saying so. It is kept as the default only so that a
workspace which names no backend behaves exactly as it did before the seam
existed. For SYCL, name a backend that derives these from the part. Of
``ai_bench``, what SYCL work should take is the *spec* -- see
:mod:`xe_forge.core.spec_loader` -- and not this build-and-run path.
"""

from __future__ import annotations

import logging
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from xe_forge.core.build_backend import BuildSpec, BuiltKernel
from xe_forge.models import ExecutionResult

logger = logging.getLogger(__name__)

DEFAULT_RUN_TIMEOUT = 600


def parse_raw_output(output: str) -> ExecutionResult:
    """Parse stdout from a CUTLASS SYCL kernel (GEMM or FA runner)."""
    passed = None
    disp = re.search(r"Disposition:\s*(Passed|Failed)", output)
    if disp:
        passed = disp.group(1) == "Passed"

    tflops = None
    time_ms = None
    perf = re.search(r"\[([0-9.]+)\]\s*TFlop/s\s+\(([0-9.]+)\)\s*ms", output)
    if not perf:
        perf = re.search(r"([0-9.]+)\s+TFlop/s.*?([0-9.]+)\s+ms", output)
    if perf:
        tflops = float(perf.group(1))
        time_ms = float(perf.group(2))

    if passed is False:
        return ExecutionResult(
            success=False,
            output_correct=False,
            execution_time_ms=time_ms,
            tflops=tflops,
            error_message="Correctness verification failed (Disposition: Failed)",
        )

    return ExecutionResult(
        success=True,
        execution_time_ms=time_ms,
        tflops=tflops,
        output_correct=passed,
    )


def run_binary(
    binary_path: str,
    args: dict[str, int | float | bool | str] | None = None,
    args_str: str | None = None,
    timeout: int = DEFAULT_RUN_TIMEOUT,
) -> ExecutionResult:
    """Run an already-compiled binary with arbitrary CLI args."""
    cmd = [binary_path]
    if args:
        for k, v in args.items():
            cmd.append(f"--{k}={v}")
    elif args_str:
        cmd.extend(args_str.split())

    logger.info("Running kernel: %s", " ".join(cmd))

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return ExecutionResult(success=False, error_message="Execution timed out")
    except Exception as e:
        return ExecutionResult(success=False, error_message=str(e))

    output = result.stdout + result.stderr
    if result.returncode != 0:
        return ExecutionResult(
            success=False,
            error_message=f"Exit code {result.returncode}\n{output[-2000:]}",
        )

    return parse_raw_output(output)


class _BinaryKernel:
    """A compiled standalone binary, invoked as a subprocess."""

    def __init__(self, binary_path: str, timeout: int = DEFAULT_RUN_TIMEOUT):
        self.binary_path = binary_path
        self.timeout = timeout

    def __call__(self, **run_args: Any) -> ExecutionResult:
        # ``dims`` is the generic form; this toolchain's binaries take flat
        # --key=value CLI args, so anything not flat is dropped rather than
        # guessed at.
        args: dict[str, int | float | bool | str] = {}
        dims = run_args.pop("dims", None) or {}
        for key, value in dims.items():
            args[str(key).lower()] = value
        for key, value in run_args.items():
            if value is None:
                continue
            if isinstance(value, (int, float, bool, str)):
                args[key] = value
            else:
                logger.debug("ai_bench backend ignoring non-scalar run arg %r", key)
        return run_binary(self.binary_path, args=args, timeout=self.timeout)


class AiBenchSyclBackend:
    """Compile SYCL source with ai_bench's SYCLCompiler."""

    name = "ai_bench"

    def __init__(
        self,
        include_dirs: list[str] | None = None,
        device_target: str | None = None,
        run_timeout: int = DEFAULT_RUN_TIMEOUT,
    ):
        self._include_dirs = list(include_dirs or [])
        self._device_target = device_target
        self._run_timeout = run_timeout
        self._build_dir: str | None = None
        self.last_error: str = ""

    @property
    def build_dir(self) -> str:
        if self._build_dir is None:
            self._build_dir = tempfile.mkdtemp(prefix="sycl_build_")
        return self._build_dir

    def build(self, source: str, spec: BuildSpec, device: str) -> BuiltKernel:
        from ai_bench.sycl.compiler import SYCLCompiler

        from xe_forge.core.build_backend import BuildError

        if spec.dependencies:
            # Said rather than silently dropped: a kernel that declares a
            # library and then links against nothing fails at link time with a
            # message about a missing symbol, which reads like a bug in the
            # kernel rather than a limit of the toolchain path it took.
            logger.warning(
                "ai_bench backend cannot resolve library dependencies %s for %r; "
                "the link will proceed without them. Use a build backend that maps "
                "dependency names to linker flags.",
                list(spec.dependencies),
                spec.name,
            )

        workdir = Path(spec.workdir) if spec.workdir else Path(self.build_dir)
        workdir.mkdir(parents=True, exist_ok=True)
        src_path = workdir / f"{spec.name}.cpp"
        src_path.write_text(source)

        include_dirs = list(self._include_dirs)
        for extra in (*spec.include_dirs, str(src_path.parent)):
            if extra and extra not in include_dirs:
                include_dirs.append(extra)

        target = spec.extra.get("device_target", self._device_target)
        compiler = SYCLCompiler(include_dirs=include_dirs, target_device=target or None)

        logger.info("Compiling SYCL kernel: %s", src_path)
        binary = compiler.compile(src_path)
        if binary is None:
            self.last_error = compiler.last_compile_error or "Compilation failed (no details)"
            raise BuildError(f"Compilation failed:\n{self.last_error[-2000:]}")
        logger.info("Compilation succeeded: %s", binary)
        return _BinaryKernel(str(binary), timeout=self._run_timeout)

    def __del__(self):
        if self._build_dir is not None:
            try:
                shutil.rmtree(self._build_dir)
            except Exception:
                pass


__all__ = ["AiBenchSyclBackend", "parse_raw_output", "run_binary"]
