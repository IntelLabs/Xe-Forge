"""
SYCL Kernel Executor - Compiles and benchmarks SYCL/XeTLA C++ kernels.

Wraps ai_bench.sycl.compiler.SYCLCompiler for compile/run/parse, adding:
- Source string → temp file conversion
- Generic dims dict support (not just m/n/k)
- ExecutionResult / SyclComparisonResult conversion
- compare_kernels() with feedback messages for the optimization loop
- Auto-detection of AOT device target via torch.xpu
"""

import logging
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

from ai_bench.sycl.compiler import SYCLCompiler, SYCLRunResult

from xe_forge.models import ExecutionResult

logger = logging.getLogger(__name__)

SYCL_TLA_DIR = os.environ.get("SYCL_TLA_DIR", "/data/nfs_home/mspoczy/temp/sycl-tla")

MKL_INCLUDE = "/swtools/intel/mkl/latest/include"

_DEVICE_NAME_TO_TARGET: dict[str, str] = {
    "b580": "bmg-g31",
    "b570": "bmg-g31",
    "battlemage": "bmg-g31",
    "bmg": "bmg-g31",
    "a770": "acm-g10",
    "a750": "acm-g10",
    "a580": "acm-g11",
    "a380": "acm-g11",
    "a310": "acm-g12",
    "max 1550": "pvc",
    "max 1100": "pvc",
    "ponte vecchio": "pvc",
    "data center gpu max": "pvc",
    "lunar lake": "lnl-m",
}


def _detect_device_target() -> str:
    """Auto-detect the icpx AOT device target from torch.xpu."""
    try:
        import torch

        if not hasattr(torch, "xpu") or not torch.xpu.is_available():
            return ""
        name = torch.xpu.get_device_name(torch.xpu.current_device()).lower()
        for key, target in _DEVICE_NAME_TO_TARGET.items():
            if key in name:
                logger.info("Auto-detected SYCL device target: %s (from '%s')", target, name)
                return target
        logger.warning("Unknown XPU device '%s', skipping AOT target", name)
        return ""
    except Exception as e:
        logger.debug("Could not auto-detect SYCL device target: %s", e)
        return ""


def _include_dirs(sycl_tla_dir: str) -> list[str]:
    dirs = [
        f"{sycl_tla_dir}/include",
        f"{sycl_tla_dir}/tools/util/include",
        f"{sycl_tla_dir}/examples/common",
    ]
    if Path(MKL_INCLUDE).exists():
        dirs.append(MKL_INCLUDE)
    return dirs


@dataclass
class SyclComparisonResult:
    """Result of comparing original vs optimized SYCL kernel performance."""

    original_time_ms: float
    optimized_time_ms: float
    speedup: float
    original_tflops: float | None = None
    optimized_tflops: float | None = None
    original_correct: bool = True
    optimized_correct: bool = True
    is_slower: bool = False
    feedback_message: str = ""

    @property
    def original_time_us(self) -> float:
        return self.original_time_ms * 1000

    @property
    def optimized_time_us(self) -> float:
        return self.optimized_time_ms * 1000


class SyclExecutor:
    """
    Compiles and runs SYCL C++ kernels, measures performance.

    Wraps ai_bench.sycl.compiler.SYCLCompiler for the underlying compile/run
    pipeline. Adds source-string input, generic dims, and comparison feedback.
    """

    def __init__(
        self,
        sycl_tla_dir: str = SYCL_TLA_DIR,
        device_target: str | None = None,
        compile_timeout: int = 300,
        run_timeout: int = 120,
        iterations: int = 20,
        verify: bool = True,
    ):
        if device_target is None:
            device_target = _detect_device_target()
        self._compiler = SYCLCompiler(
            include_dirs=_include_dirs(sycl_tla_dir),
            target_device=device_target or None,
        )
        self.iterations = iterations
        self.verify = verify
        self._build_dir = None

    @property
    def build_dir(self) -> str:
        if self._build_dir is None:
            self._build_dir = tempfile.mkdtemp(prefix="sycl_build_")
        return self._build_dir

    def compile(
        self,
        source_code: str | None = None,
        source_path: str | None = None,
        output_name: str = "kernel_sycl",
    ) -> tuple[bool, str, str]:
        """Compile SYCL C++ source to a binary.

        Args:
            source_code: C++ source code string (written to temp file)
            source_path: Path to existing .cpp file (used if source_code is None)
            output_name: Name for the output binary

        Returns:
            (success, binary_path, error_message)
        """
        if source_code is not None:
            src_path = Path(self.build_dir) / f"{output_name}.cpp"
            src_path.write_text(source_code)
        elif source_path is not None:
            src_path = Path(source_path)
        else:
            return False, "", "No source code or path provided"

        logger.info(f"Compiling SYCL kernel: {src_path}")
        binary = self._compiler.compile(src_path)
        if binary is None:
            err = self._compiler.last_compile_error or "Compilation failed (no details)"
            return False, "", err
        logger.info(f"Compilation succeeded: {binary}")
        return True, str(binary), ""

    @staticmethod
    def _dims_to_mnk(
        dims: dict[str, int | float] | None,
        m: int = 1024,
        n: int = 1024,
        k: int = 1024,
    ) -> tuple[int, int, int]:
        """Extract m, n, k from a dims dict or fall back to explicit values."""
        if not dims:
            return m, n, k
        em = int(dims.get("M", dims.get("N", m)))
        en = int(dims.get("N", em))
        ek = int(dims.get("K", em))
        return em, en, ek

    def execute(
        self,
        kernel_code: str | None = None,
        kernel_path: str | None = None,
        m: int = 1024,
        n: int = 1024,
        k: int = 1024,
        dims: dict[str, int | float] | None = None,
        output_name: str = "kernel_sycl",
    ) -> ExecutionResult:
        """Compile and run a SYCL kernel, returning structured results.

        Args:
            kernel_code: C++ source code string
            kernel_path: Path to existing .cpp file
            m, n, k: GEMM dimensions (backward compat)
            dims: Generic dimension dict from spec (takes precedence over m/n/k)
            output_name: Name for compiled binary

        Returns:
            ExecutionResult with timing and correctness info
        """
        success, binary_path, err = self.compile(
            source_code=kernel_code,
            source_path=kernel_path,
            output_name=output_name,
        )
        if not success:
            return ExecutionResult(
                success=False,
                error_message=f"Compilation failed:\n{err[-2000:]}",
            )

        em, en, ek = self._dims_to_mnk(dims, m, n, k)
        logger.info(f"Running SYCL kernel: {binary_path} (M={em}, N={en}, K={ek})")

        result: SYCLRunResult = self._compiler.run(
            Path(binary_path),
            m=em,
            n=en,
            k=ek,
            iterations=self.iterations,
            verify=1 if self.verify else 0,
        )
        return self._to_execution_result(result)

    @staticmethod
    def _to_execution_result(r: SYCLRunResult) -> ExecutionResult:
        if not r.success:
            return ExecutionResult(
                success=False,
                error_message=f"Execution failed: {r.error}",
            )
        if r.passed is False:
            return ExecutionResult(
                success=False,
                output_correct=False,
                execution_time_ms=r.time_ms,
                tflops=r.tflops,
                error_message="Correctness verification failed (Disposition: Failed)",
            )
        return ExecutionResult(
            success=True,
            execution_time_ms=r.time_ms,
            tflops=r.tflops,
            output_correct=r.passed,
        )

    def compare_kernels(
        self,
        original_code: str | None = None,
        optimized_code: str | None = None,
        original_path: str | None = None,
        optimized_path: str | None = None,
        m: int = 1024,
        n: int = 1024,
        k: int = 1024,
        dims: dict[str, int | float] | None = None,
    ) -> SyclComparisonResult:
        """Compare performance of original vs optimized SYCL kernel.

        Args:
            original_code/path: Original kernel source
            optimized_code/path: Optimized kernel source
            m, n, k: GEMM dimensions (backward compat)
            dims: Generic dimension dict from spec (takes precedence)

        Returns:
            SyclComparisonResult with speedup and feedback
        """
        orig_result = self.execute(
            kernel_code=original_code,
            kernel_path=original_path,
            m=m,
            n=n,
            k=k,
            dims=dims,
            output_name="original_sycl",
        )
        opt_result = self.execute(
            kernel_code=optimized_code,
            kernel_path=optimized_path,
            m=m,
            n=n,
            k=k,
            dims=dims,
            output_name="optimized_sycl",
        )

        if not orig_result.success:
            return SyclComparisonResult(
                original_time_ms=float("inf"),
                optimized_time_ms=float("inf"),
                speedup=0.0,
                original_correct=False,
                feedback_message=f"FAILURE: Original kernel failed: {orig_result.error_message}",
            )

        if not opt_result.success:
            return SyclComparisonResult(
                original_time_ms=orig_result.execution_time_ms or float("inf"),
                optimized_time_ms=float("inf"),
                speedup=0.0,
                optimized_correct=False,
                feedback_message=(
                    f"FAILURE: Optimized kernel failed: {opt_result.error_message}. "
                    "Fix compilation or runtime errors."
                ),
            )

        orig_ms = orig_result.execution_time_ms or float("inf")
        opt_ms = opt_result.execution_time_ms or float("inf")
        speedup = orig_ms / opt_ms if opt_ms > 0 else 0.0

        is_slower = speedup < 1.0
        orig_tflops = orig_result.tflops
        opt_tflops = opt_result.tflops

        if is_slower:
            slowdown = 1.0 / speedup if speedup > 0 else float("inf")
            msg = (
                f"PERFORMANCE REGRESSION: Optimized kernel is {slowdown:.2f}x SLOWER. "
                f"Original: {orig_ms:.4f}ms ({orig_tflops:.3f} TFlop/s), "
                f"Optimized: {opt_ms:.4f}ms ({opt_tflops:.3f} TFlop/s). "
                "Try a different approach."
            )
        elif speedup >= 2.0:
            msg = (
                f"SUCCESS: Excellent! {speedup:.2f}x speedup. "
                f"Original: {orig_ms:.4f}ms ({orig_tflops:.3f} TFlop/s), "
                f"Optimized: {opt_ms:.4f}ms ({opt_tflops:.3f} TFlop/s)."
            )
        elif speedup >= 1.2:
            msg = (
                f"SUCCESS: Good {speedup:.2f}x speedup. "
                f"Original: {orig_ms:.4f}ms, Optimized: {opt_ms:.4f}ms. "
                "Consider further optimizations."
            )
        else:
            msg = (
                f"MARGINAL: Only {speedup:.2f}x speedup. "
                f"Original: {orig_ms:.4f}ms, Optimized: {opt_ms:.4f}ms. "
                "Try more aggressive optimizations."
            )

        return SyclComparisonResult(
            original_time_ms=orig_ms,
            optimized_time_ms=opt_ms,
            speedup=speedup,
            original_tflops=orig_tflops,
            optimized_tflops=opt_tflops,
            original_correct=orig_result.output_correct is not False,
            optimized_correct=opt_result.output_correct is not False,
            is_slower=is_slower,
            feedback_message=msg,
        )

    def __del__(self):
        if self._build_dir is not None:
            try:
                shutil.rmtree(self._build_dir)
            except Exception:
                pass
