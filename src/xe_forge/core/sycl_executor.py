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
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import numpy as np
import torch
from ai_bench.harness.runner.benchmark_compare import set_all_seeds
from ai_bench.sycl.compiler import SYCLCompiler, SYCLRunResult

from xe_forge.models import ExecutionResult

logger = logging.getLogger(__name__)

SYCL_TLA_DIR = os.environ.get("SYCL_TLA_DIR", "")

MKL_INCLUDE = os.environ.get("MKL_INCLUDE", "/swtools/intel/mkl/latest/include")

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


class KernelType(Enum):
    GEMM = "gemm"
    FA = "fa"
    DUAL_GEMM = "dual_gemm"
    GROUPED_GEMM = "grouped_gemm"
    MOE_GEMM = "moe_gemm"


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


def _include_dirs(sycl_tla_dir: str, kernel_type: KernelType = KernelType.GEMM) -> list[str]:
    dirs = [
        f"{sycl_tla_dir}/include",
        f"{sycl_tla_dir}/tools/util/include",
        f"{sycl_tla_dir}/examples/common",
    ]
    if kernel_type in (KernelType.FA, KernelType.DUAL_GEMM):
        dirs.append(f"{sycl_tla_dir}/applications")
    if kernel_type == KernelType.FA:
        dirs.append(f"{sycl_tla_dir}/examples/06_bmg_flash_attention")
        dirs.append(f"{sycl_tla_dir}/benchmarks/flash_attention")
    if kernel_type == KernelType.MOE_GEMM:
        dirs.append(f"{sycl_tla_dir}/examples/12_xe20_moe_gemm_cute_interface")
    if Path(MKL_INCLUDE).exists():
        dirs.append(MKL_INCLUDE)
    return dirs


def _save_tensor(t: torch.Tensor, path: str) -> None:
    """Write a tensor to a binary file, handling bfloat16 (unsupported by NumPy)."""
    t = t.contiguous()
    if t.dtype == torch.bfloat16:
        t.view(torch.int16).numpy().tofile(path)
    else:
        t.numpy().tofile(path)


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
        kernel_type: KernelType | str = KernelType.GEMM,
    ):
        if isinstance(kernel_type, str):
            kernel_type = KernelType(kernel_type)
        self.kernel_type = kernel_type
        if device_target is None:
            device_target = _detect_device_target()
        self._compiler = SYCLCompiler(
            include_dirs=_include_dirs(sycl_tla_dir, kernel_type),
            target_device=device_target or None,
        )
        self.iterations = iterations
        self.verify = verify
        self._build_dir = None
        self._cached_input_dir: str | None = None
        self._cached_input_key: tuple | None = None

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

        src_parent = str(src_path.parent)
        if src_parent not in self._compiler.include_dirs:
            self._compiler.include_dirs.append(src_parent)

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

    def generate_inputs(
        self,
        dims: dict[str, int | float],
        output_dir: str,
        dtype: torch.dtype = torch.bfloat16,
        seed: int | None = None,
    ) -> None:
        """Generate random input tensors and write them as binary files.

        Writes A.bin, B0.bin, B1.bin (for dual GEMM) or A.bin, B.bin (for GEMM)
        to the given directory. The binary expects raw row-major tensor data.
        """
        if seed is not None:
            set_all_seeds(seed)
        os.makedirs(output_dir, exist_ok=True)
        m = int(dims.get("M", dims.get("N", 1024)))
        n = int(dims.get("N", m))
        k = int(dims.get("K", m))

        A = torch.randn(m, k, dtype=dtype)
        _save_tensor(A, os.path.join(output_dir, "A.bin"))

        B0 = torch.randn(k, n, dtype=dtype)
        _save_tensor(B0, os.path.join(output_dir, "B0.bin"))

        B1 = torch.randn(k, n, dtype=dtype)
        _save_tensor(B1, os.path.join(output_dir, "B1.bin"))

        logger.info(
            f"Generated inputs: A[{m},{k}], B0[{k},{n}], B1[{k},{n}] ({dtype}) -> {output_dir}"
        )

    def get_or_create_inputs(
        self,
        dims: dict[str, int | float],
        seed: int = 42,
        dtype: torch.dtype = torch.bfloat16,
    ) -> str:
        """Return a directory with deterministic input tensors, caching across calls."""
        key = (tuple(sorted(dims.items())), seed)
        if self._cached_input_dir is not None and self._cached_input_key == key:
            return self._cached_input_dir
        if self._cached_input_dir is not None:
            try:
                shutil.rmtree(self._cached_input_dir)
            except Exception:
                pass
        input_dir = tempfile.mkdtemp(prefix="sycl_inputs_")
        self.generate_inputs(dims, input_dir, dtype=dtype, seed=seed)
        self._cached_input_dir = input_dir
        self._cached_input_key = key
        return input_dir

    @staticmethod
    def load_output(path: str, dtype: np.dtype = np.float32) -> np.ndarray:
        """Load a binary tensor file dumped by the SYCL kernel."""
        return np.fromfile(path, dtype=dtype)

    @staticmethod
    def compare_outputs(
        output_a: np.ndarray,
        output_b: np.ndarray,
        rtol: float = 1e-2,
        atol: float = 1e-3,
    ) -> tuple[bool, str]:
        """Compare two output tensors element-wise.

        Returns (passed, message).
        """
        if output_a.shape != output_b.shape:
            return False, f"Shape mismatch: {output_a.shape} vs {output_b.shape}"
        close = np.allclose(output_a, output_b, rtol=rtol, atol=atol)
        if close:
            return True, "Outputs match"
        diff = np.abs(output_a - output_b)
        max_diff = float(np.max(diff))
        mean_diff = float(np.mean(diff))
        num_mismatch = int(np.sum(~np.isclose(output_a, output_b, rtol=rtol, atol=atol)))
        total = output_a.size
        return False, (
            f"Outputs differ: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}, "
            f"mismatched={num_mismatch}/{total} ({100 * num_mismatch / total:.1f}%)"
        )

    def execute(
        self,
        kernel_code: str | None = None,
        kernel_path: str | None = None,
        m: int = 1024,
        n: int = 1024,
        k: int = 1024,
        dims: dict[str, int | float] | None = None,
        output_name: str = "kernel_sycl",
        input_dir: str | None = None,
        output_dir: str | None = None,
    ) -> ExecutionResult:
        """Compile and run a SYCL kernel, returning structured results.

        Args:
            kernel_code: C++ source code string
            kernel_path: Path to existing .cpp file
            m, n, k: GEMM dimensions (backward compat)
            dims: Generic dimension dict from spec (takes precedence over m/n/k)
            output_name: Name for compiled binary
            input_dir: Directory with input tensor files (A.bin, B0.bin, B1.bin)
            output_dir: Directory to dump output tensor (D2.bin)

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

        # Skip internal Disposition check when using file-based I/O — the
        # reference path reinitializes C randomly so it always mismatches.
        # We verify correctness in Python via compare_outputs() instead.
        use_verify = 0 if input_dir else (1 if self.verify else 0)

        extra_cli: dict[str, int | float | bool | str] = {}
        if input_dir:
            extra_cli["input_dir"] = input_dir
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            extra_cli["output_dir"] = output_dir

        if extra_cli:
            args_dict: dict[str, int | float | bool | str] = {
                "m": em,
                "n": en,
                "k": ek,
                "iterations": self.iterations,
                "verify": use_verify,
                **extra_cli,
            }
            return self._run_binary(binary_path, args=args_dict)

        result: SYCLRunResult = self._compiler.run(
            Path(binary_path),
            m=em,
            n=en,
            k=ek,
            iterations=self.iterations,
            verify=use_verify,
        )
        return self._to_execution_result(result)

    def execute_raw(
        self,
        kernel_code: str | None = None,
        kernel_path: str | None = None,
        output_name: str = "kernel_sycl",
        args: dict[str, int | float | bool | str] | None = None,
        args_str: str | None = None,
        timeout: int = 600,
    ) -> ExecutionResult:
        """Compile and run a kernel with arbitrary CLI args.

        Use for kernels that don't follow GEMM --m/--n/--k convention
        (e.g. Flash Attention, dual GEMM). Pass args as a dict
        (converted to --key=value) or a raw argument string.
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
        return self._run_binary(binary_path, args=args, args_str=args_str, timeout=timeout)

    def _run_binary(
        self,
        binary_path: str,
        args: dict[str, int | float | bool | str] | None = None,
        args_str: str | None = None,
        timeout: int = 600,
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
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
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

        return self._parse_raw_output(output)

    @staticmethod
    def _parse_raw_output(output: str) -> ExecutionResult:
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
        rtol: float = 1e-2,
        atol: float = 1e-3,
        input_dir: str | None = None,
        seed: int = 42,
    ) -> SyclComparisonResult:
        """Compare performance and correctness of original vs optimized SYCL kernel.

        Both kernels are run with the same input tensors (generated once, shared
        via files). Outputs are compared in Python using numpy.allclose.

        Args:
            original_code/path: Original kernel source
            optimized_code/path: Optimized kernel source
            m, n, k: GEMM dimensions (backward compat)
            dims: Generic dimension dict from spec (takes precedence)
            rtol: Relative tolerance for output comparison
            atol: Absolute tolerance for output comparison

        Returns:
            SyclComparisonResult with speedup, correctness, and feedback
        """
        caller_owns_inputs = input_dir is not None
        io_dir = tempfile.mkdtemp(prefix="sycl_compare_")
        if not caller_owns_inputs:
            effective_dims = dims or {"M": m, "N": n, "K": k}
            input_dir = self.get_or_create_inputs(effective_dims, seed=seed)
        orig_output_dir = os.path.join(io_dir, "orig_out")
        opt_output_dir = os.path.join(io_dir, "opt_out")

        orig_result = self.execute(
            kernel_code=original_code,
            kernel_path=original_path,
            m=m,
            n=n,
            k=k,
            dims=dims,
            output_name="original_sycl",
            input_dir=input_dir,
            output_dir=orig_output_dir,
        )
        opt_result = self.execute(
            kernel_code=optimized_code,
            kernel_path=optimized_path,
            m=m,
            n=n,
            k=k,
            dims=dims,
            output_name="optimized_sycl",
            input_dir=input_dir,
            output_dir=opt_output_dir,
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

        # Correctness: compare outputs via dumped files
        orig_correct = True
        opt_correct = True
        correctness_msg = ""
        orig_d2 = os.path.join(orig_output_dir, "D2.bin")
        opt_d2 = os.path.join(opt_output_dir, "D2.bin")
        if os.path.exists(orig_d2) and os.path.exists(opt_d2):
            orig_out = self.load_output(orig_d2)
            opt_out = self.load_output(opt_d2)
            passed, detail = self.compare_outputs(orig_out, opt_out, rtol=rtol, atol=atol)
            opt_correct = passed
            if not passed:
                correctness_msg = f" CORRECTNESS FAILED: {detail}."
            else:
                correctness_msg = " Correctness: PASSED."
            logger.info(f"Output comparison (rtol={rtol}, atol={atol}): {detail}")
        else:
            correctness_msg = " (no output files for comparison)"
            logger.warning("Output dump files not found — skipping correctness check")

        # Clean up temp I/O dir
        try:
            shutil.rmtree(io_dir)
        except Exception:
            pass

        if not opt_correct:
            msg = (
                f"CORRECTNESS FAILURE: Optimized kernel produces wrong results. "
                f"{correctness_msg.strip()} "
                f"Original: {orig_ms:.4f}ms, Optimized: {opt_ms:.4f}ms. "
                "Fix numerical correctness before optimizing for speed."
            )
        elif is_slower:
            slowdown = 1.0 / speedup if speedup > 0 else float("inf")
            msg = (
                f"PERFORMANCE REGRESSION: Optimized kernel is {slowdown:.2f}x SLOWER. "
                f"Original: {orig_ms:.4f}ms ({orig_tflops:.3f} TFlop/s), "
                f"Optimized: {opt_ms:.4f}ms ({opt_tflops:.3f} TFlop/s). "
                f"{correctness_msg.strip()} Try a different approach."
            )
        elif speedup >= 2.0:
            msg = (
                f"SUCCESS: Excellent! {speedup:.2f}x speedup. "
                f"Original: {orig_ms:.4f}ms ({orig_tflops:.3f} TFlop/s), "
                f"Optimized: {opt_ms:.4f}ms ({opt_tflops:.3f} TFlop/s)."
                f"{correctness_msg}"
            )
        elif speedup >= 1.2:
            msg = (
                f"SUCCESS: Good {speedup:.2f}x speedup. "
                f"Original: {orig_ms:.4f}ms, Optimized: {opt_ms:.4f}ms. "
                f"{correctness_msg.strip()} Consider further optimizations."
            )
        else:
            msg = (
                f"MARGINAL: Only {speedup:.2f}x speedup. "
                f"Original: {orig_ms:.4f}ms, Optimized: {opt_ms:.4f}ms. "
                f"{correctness_msg.strip()} Try more aggressive optimizations."
            )

        return SyclComparisonResult(
            original_time_ms=orig_ms,
            optimized_time_ms=opt_ms,
            speedup=speedup,
            original_tflops=orig_tflops,
            optimized_tflops=opt_tflops,
            original_correct=orig_correct,
            optimized_correct=opt_correct,
            is_slower=is_slower,
            feedback_message=msg,
        )

    def __del__(self):
        if self._build_dir is not None:
            try:
                shutil.rmtree(self._build_dir)
            except Exception:
                pass
        if self._cached_input_dir is not None:
            try:
                shutil.rmtree(self._cached_input_dir)
            except Exception:
                pass
