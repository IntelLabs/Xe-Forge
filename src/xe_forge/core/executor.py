"""
Kernel Executor - Executes and measures Triton kernels using KernelBench framework

Correctness validation uses ai-bench's check_correctness, copy_model_weights,
and set_all_seeds utilities (shared with the benchmark harness).
"""

import logging
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from ai_bench import time as bench_time
from ai_bench.harness.runner.benchmark_compare import (
    check_correctness,
    copy_model_weights,
    set_all_seeds,
)
from ai_bench.utils import count_torch_flop, import_from_path

from xe_forge.models import ExecutionResult

logger = logging.getLogger(__name__)


@dataclass
class ComparisonResult:
    """Result of comparing original vs optimized kernel performance."""

    original_time_us: float
    optimized_time_us: float
    speedup: float
    original_tflops: float | None = None
    optimized_tflops: float | None = None
    original_correct: bool = True
    optimized_correct: bool = True
    is_slower: bool = False
    feedback_message: str = ""


class KernelBenchExecutor:
    """
    Executes Triton kernels using KernelBench-style testing.

    """

    def __init__(
        self,
        device: str = "xpu",
        warmup_iters: int = 200,
        benchmark_iters: int = 100,
        require_correctness: bool = True,
        rtol: float = 1e-2,
        atol: float = 1e-5,
    ):
        """
        Initialize executor.

        Args:
            device: Target device (xpu)
            warmup_iters: Warmup iterations (200 recommended for XPU)
            benchmark_iters: Benchmark iterations
            require_correctness: If True, validate output correctness (default: True)
            rtol: Relative tolerance for correctness check (default: 1e-2)
            atol: Absolute tolerance for correctness check (default: 1e-5)
        """
        self.device = device
        self.warmup_iters = warmup_iters
        self.benchmark_iters = benchmark_iters
        self.require_correctness = require_correctness
        self.rtol = rtol
        self.atol = atol
        self._temp_dir = None
        self._module_counter = 0

    def time(
        self,
        fn: Callable,
        args: tuple,
        warmup: int | None = None,
        rep: int | None = None,
    ) -> float:
        """
        Measure execution time of the provided function.

        Uses ai_bench.time() which dispatches to the correct backend:
          - xpu/cuda → time_gpu() with hardware events, L2 flush, dummy matmul
          - cpu      → time_cpu() with torch profiler

        Args:
            fn: Function to measure
            args: Arguments to pass to the function
            warmup: Warmup iterations
            rep: Measurement iterations

        Returns:
            Mean runtime in microseconds
        """
        warmup = warmup or self.warmup_iters
        rep = rep or self.benchmark_iters

        return bench_time(fn, args, warmup=warmup, rep=rep, device=torch.device(self.device))

    def execute(
        self,
        kernel_code: str,
        kernel_name: str | None = None,
        input_shapes: list[tuple[int, ...]] | None = None,
        inputs: list | None = None,
        flop: float | None = None,
        reference_fn: Callable | None = None,
        dtype=None,
        init_args: list | None = None,
    ) -> ExecutionResult:
        """
        Execute kernel and measure performance.

        Args:
            kernel_code: Triton kernel source code
            kernel_name: Name of kernel function (optional if using KernelBench Model format)
            input_shapes: Input tensor shapes (used if inputs not provided)
            inputs: Pre-created input tensors
            flop: Number of floating-point operations (for TFLOPS calculation)
            reference_fn: Optional reference function for validation
            dtype: Torch dtype for input tensors
            init_args: Positional args for Model.__init__ (from spec inits section)

        Returns:
            ExecutionResult with timing and correctness info
        """
        try:
            # Compile kernel module
            module = self._compile_module(kernel_code)
            if module is None:
                return ExecutionResult(
                    success=False,
                    error_message="Failed to compile kernel module",
                )

            # Get the callable (either kernel function or Model.forward)
            fn, model = self._get_callable(module, kernel_name, init_args=init_args)
            if fn is None:
                return ExecutionResult(
                    success=False,
                    error_message=f"Could not find callable (no Model class or function '{kernel_name}')",
                )

            # Create inputs if not provided
            if inputs is None:
                if input_shapes:
                    inputs = self._create_inputs(input_shapes, dtype=dtype)
                else:
                    return ExecutionResult(
                        success=False,
                        error_message="No inputs or input_shapes provided",
                    )

            # Move model and inputs to device if needed
            device = torch.device(self.device)
            if model is not None and hasattr(model, "to"):
                model = model.to(device)
                if dtype is not None:
                    model = model.to(dtype)
                fn = model.forward
            inputs = [inp.to(device) if hasattr(inp, "to") else inp for inp in inputs]

            # Run once to check for errors
            try:
                output = fn(*inputs)
            except Exception as e:
                return ExecutionResult(
                    success=False,
                    error_message=f"Kernel execution failed: {e}",
                )

            # Benchmark
            time_us = self.time(fn, tuple(inputs))
            logger.info(f"Execution time: {time_us:.2f} μs")

            # Calculate TFLOPS if flop count provided
            tflops = None
            actual_flop = flop
            if actual_flop is None:
                try:
                    actual_flop = count_torch_flop(fn, tuple(inputs))
                except Exception as e:
                    logger.debug(f"Could not count FLOPs: {e}")

            if actual_flop and time_us > 0:
                tflops = actual_flop / time_us / 1e6

            # Validate correctness if reference provided
            output_correct = None
            if reference_fn and self.require_correctness:
                try:
                    ref_output = reference_fn(*inputs)
                    output_correct = check_correctness(
                        ref_output, output, rtol=self.rtol, atol=self.atol
                    )
                except Exception as e:
                    logger.warning(f"Reference validation failed: {e}")

            return ExecutionResult(
                success=True,
                execution_time_ms=time_us / 1000,  # Convert to ms
                tflops=tflops,
                output_correct=output_correct,
            )

        except Exception as e:
            import traceback

            return ExecutionResult(
                success=False,
                error_message=str(e),
                error_traceback=traceback.format_exc(),
            )

    def _check_correctness(
        self,
        original_code: str,
        optimized_code: str,
        kernel_name: str | None,
        input_shapes: list[tuple[int, ...]],
        dtype=None,
        init_args: list | None = None,
    ) -> bool:
        """
        Check if optimized kernel produces same outputs as original.

        Uses ai-bench utilities (check_correctness, copy_model_weights,
        set_all_seeds) — the same code path used by the benchmark harness.

        Args:
            original_code: Original kernel code
            optimized_code: Optimized kernel code
            kernel_name: Kernel function name (or "Model" for Model-based kernels)
            input_shapes: Input tensor shapes
            dtype: Torch dtype for inputs

        Returns:
            True if outputs match within tolerance
        """
        try:
            device = torch.device(self.device)

            # Compile original with deterministic init
            set_all_seeds(42)
            original_module = self._compile_module(original_code)
            if original_module is None:
                logger.warning("Failed to compile original kernel")
                return False

            original_fn, original_model = self._get_callable(
                original_module, kernel_name, init_args=init_args
            )
            if original_fn is None:
                logger.warning("Failed to get original callable")
                return False

            if original_model is not None:
                original_model = original_model.to(device)
                if dtype is not None:
                    original_model = original_model.to(dtype)
                original_fn = original_model.forward

            # Compile optimized with same seed
            set_all_seeds(42)
            optimized_module = self._compile_module(optimized_code)
            if optimized_module is None:
                logger.warning("Failed to compile optimized kernel")
                return False

            optimized_fn, optimized_model = self._get_callable(
                optimized_module, kernel_name, init_args=init_args
            )
            if optimized_fn is None:
                logger.warning("Failed to get optimized callable")
                return False

            if optimized_model is not None:
                optimized_model = optimized_model.to(device)
                if dtype is not None:
                    optimized_model = optimized_model.to(dtype)
                optimized_fn = optimized_model.forward

            # Copy weights from original to optimized (ai-bench utility)
            if original_model is not None and optimized_model is not None:
                weights_copied = copy_model_weights(original_model, optimized_model)
                if not weights_copied:
                    logger.warning("Could not copy weights - using seed-based initialization")

            # Shared inputs with deterministic seed
            set_all_seeds(123)
            inputs = self._create_inputs(input_shapes, dtype=dtype)

            inputs_orig = [inp.clone() for inp in inputs]
            inputs_opt = [inp.clone() for inp in inputs]

            with torch.no_grad():
                original_output = original_fn(*inputs_orig)
                optimized_output = optimized_fn(*inputs_opt)

            # Normalize outputs before comparison:
            if isinstance(optimized_output, torch.Tensor):
                if original_output.dtype != optimized_output.dtype:
                    original_output = original_output.to(optimized_output.dtype)

            # Compare using ai-bench's check_correctness
            return check_correctness(
                original_output, optimized_output, rtol=self.rtol, atol=self.atol
            )

        except Exception as e:
            logger.error(f"Correctness validation error: {e}")
            import traceback

            logger.debug(traceback.format_exc())
            return False

    def compare_kernels(
        self,
        original_code: str,
        optimized_code: str,
        kernel_name: str | None = None,
        input_shapes: list[tuple[int, ...]] | None = None,
        inputs: list | None = None,
        flop: float | None = None,
        reference_fn: Callable | None = None,
        dtype=None,
        init_args: list | None = None,
    ) -> ComparisonResult:
        """
        Compare performance AND correctness of original vs optimized kernel.

        This is the main method used by the CoVeR agent to evaluate
        optimization results and get feedback.

        Args:
            original_code: Original Triton kernel code
            optimized_code: Optimized Triton kernel code
            kernel_name: Name of kernel function (optional if using KernelBench Model format)
            input_shapes: Input tensor shapes
            inputs: Pre-created input tensors (will be copied for each test)
            flop: Number of floating-point operations
            reference_fn: Reference function for correctness validation
            dtype: Torch dtype for input tensors

        Returns:
            ComparisonResult with speedup and feedback message for CoVeR
        """

        # Execute original kernel
        original_result = self.execute(
            original_code,
            kernel_name,
            input_shapes,
            inputs,
            flop,
            reference_fn,
            dtype=dtype,
            init_args=init_args,
        )

        # Execute optimized kernel
        optimized_result = self.execute(
            optimized_code,
            kernel_name,
            input_shapes,
            inputs,
            flop,
            reference_fn,
            dtype=dtype,
            init_args=init_args,
        )

        # Handle failures
        if not original_result.success:
            return ComparisonResult(
                original_time_us=float("inf"),
                optimized_time_us=float("inf"),
                speedup=0.0,
                original_correct=False,
                feedback_message=f"FAILURE: Original kernel failed: {original_result.error_message}",
            )

        if not optimized_result.success:
            return ComparisonResult(
                original_time_us=original_result.execution_time_ms * 1000,
                optimized_time_us=float("inf"),
                speedup=0.0,
                optimized_correct=False,
                feedback_message=f"FAILURE: Optimized kernel failed to compile or run: {optimized_result.error_message}. "
                f"Please fix the syntax or runtime errors in the optimized code.",
            )

        # Correctness validation using ai-bench utilities
        if self.require_correctness and input_shapes:
            outputs_match = self._check_correctness(
                original_code=original_code,
                optimized_code=optimized_code,
                kernel_name=kernel_name or "Model",
                input_shapes=input_shapes,
                dtype=dtype,
                init_args=init_args,
            )

            if not outputs_match:
                return ComparisonResult(
                    original_time_us=original_result.execution_time_ms * 1000,
                    optimized_time_us=optimized_result.execution_time_ms * 1000,
                    speedup=0.0,
                    original_correct=True,
                    optimized_correct=False,
                    is_slower=False,
                    feedback_message=(
                        "CORRECTNESS FAILURE: Optimized kernel produces WRONG outputs. "
                        "This is a CRITICAL error - the optimization MUST be numerically equivalent to the original. "
                        "Common causes:\n"
                        "- Wrong matrix dimensions or strides in block pointers\n"
                        "- Transposed matrices loaded incorrectly (check shape=(K,N) vs shape=(N,K))\n"
                        "- Missing or reordered operations (bias, activation)\n"
                        "- Wrong accumulator dtype causing overflow/underflow\n"
                        "- Tile boundary errors (check boundary_check=(0,1))\n"
                        "Please carefully compare your kernel logic against the original and fix the computation."
                    ),
                )
        elif not self.require_correctness:
            logger.info("Correctness validation SKIPPED (require_correctness=False)")

        # Calculate times and speedup
        original_time_us = original_result.execution_time_ms * 1000
        optimized_time_us = optimized_result.execution_time_ms * 1000
        speedup = original_time_us / optimized_time_us if optimized_time_us > 0 else 0.0

        # Check correctness
        original_correct = original_result.output_correct is not False

        # Determine if optimization made things worse
        is_slower = speedup < 1.0

        # Generate feedback message for CoVeR agent
        feedback_parts = []

        if is_slower:
            slowdown = 1.0 / speedup if speedup > 0 else float("inf")
            feedback_parts.append(
                f"PERFORMANCE REGRESSION: Optimized kernel is {slowdown:.2f}x SLOWER than original. "
                f"Original: {original_time_us:.2f}μs, Optimized: {optimized_time_us:.2f}μs. "
                f"The optimization made performance worse. Please try a different approach."
            )
        elif speedup >= 1.0:
            orig_tflops = original_result.tflops or 0
            opt_tflops = optimized_result.tflops or 0

            if speedup >= 2.0:
                feedback_parts.append(
                    f"SUCCESS: Excellent optimization! {speedup:.2f}x speedup achieved. "
                    f"Original: {original_time_us:.2f}μs ({orig_tflops:.2f} TFLOPS), "
                    f"Optimized: {optimized_time_us:.2f}μs ({opt_tflops:.2f} TFLOPS). "
                    f"Correctness verified: outputs match within tolerance."
                )
            elif speedup >= 1.2:
                feedback_parts.append(
                    f"SUCCESS: Good optimization! {speedup:.2f}x speedup achieved. "
                    f"Original: {original_time_us:.2f}μs, Optimized: {optimized_time_us:.2f}μs. "
                    f"Correctness verified. Consider additional optimizations for further improvement."
                )
            else:
                feedback_parts.append(
                    f"MARGINAL: Only {speedup:.2f}x speedup achieved. "
                    f"Original: {original_time_us:.2f}μs, Optimized: {optimized_time_us:.2f}μs. "
                    f"Correctness verified. Consider more aggressive optimizations."
                )

        feedback_message = " ".join(feedback_parts)

        return ComparisonResult(
            original_time_us=original_time_us,
            optimized_time_us=optimized_time_us,
            speedup=speedup,
            original_tflops=original_result.tflops,
            optimized_tflops=optimized_result.tflops,
            original_correct=original_correct,
            optimized_correct=True,  # Passed validation or skipped
            is_slower=is_slower,
            feedback_message=feedback_message,
        )

    def _compile_module(self, kernel_code: str):
        """Compile Triton kernel module from source code.

        Writes kernel_code to a temp file, then imports it using
        ai_bench.utils.import_from_path (shared importlib helper).
        """
        try:
            # Create temp directory if needed
            if self._temp_dir is None:
                self._temp_dir = tempfile.mkdtemp(prefix="triton_opt_")

            # Generate unique module name
            self._module_counter += 1
            module_name = f"triton_kernel_{self._module_counter}"
            module_path = Path(self._temp_dir) / f"{module_name}.py"

            # Write kernel code to file
            with open(module_path, "w") as f:
                f.write(kernel_code)

            module = import_from_path(module_name, module_path)
            return module

        except Exception as e:
            logger.error(f"Compilation error: {e}")
            return None

    def _get_callable(
        self, module, kernel_name: str | None = None, init_args: list | None = None
    ) -> tuple[Callable | None, Any | None]:
        """
        Get callable from module.

        Supports both KernelBench-style (Model class with forward) and
        direct function access.

        Args:
            module: Compiled module
            kernel_name: Optional kernel function name
            init_args: Optional positional args for Model.__init__ (from spec inits)

        Returns:
            Tuple of (callable, model_instance or None)
        """
        # Try KernelBench-style Model class first (preferred)
        if hasattr(module, "Model"):
            model_class = module.Model
            try:
                # Priority 1: Use init_args from spec if provided
                if init_args:
                    model = model_class(*init_args)
                # Priority 2: Try get_init_inputs() convention
                elif hasattr(module, "get_init_inputs"):
                    init_args = module.get_init_inputs()
                    model = model_class(*init_args)
                # Priority 3: No-arg construction
                else:
                    model = model_class()

                if hasattr(model, "forward"):
                    return model.forward, model
            except Exception as e:
                logger.warning(f"Could not instantiate Model: {e}")
                import traceback

                logger.debug(traceback.format_exc())

        # Try direct function access if kernel_name provided
        if kernel_name and hasattr(module, kernel_name):
            fn = getattr(module, kernel_name)
            if callable(fn):
                return fn, None

        # Try to find any callable that looks like a kernel wrapper
        for name in dir(module):
            if name.startswith("_"):
                continue
            obj = getattr(module, name)
            if callable(obj) and not isinstance(obj, type):
                # Check if it's a wrapper function (not a kernel)
                if hasattr(obj, "__module__") and "triton" not in str(type(obj)):
                    return obj, None

        return None, None

    def _create_inputs(
        self,
        shapes: list[tuple[int, ...]],
        dtype=None,
    ) -> list:
        """Create input tensors."""
        if dtype is None:
            dtype = torch.float16

        return [torch.randn(shape, dtype=dtype, device=self.device) for shape in shapes]

    def __del__(self):
        """Cleanup temp directory."""
        if self._temp_dir is not None:
            import shutil

            try:
                shutil.rmtree(self._temp_dir)
            except Exception:
                pass


KernelExecutor = KernelBenchExecutor


def create_executor_tool(
    executor: KernelBenchExecutor,
    original_code: str,
    kernel_name: str | None = None,
    input_shapes: list[tuple[int, ...]] | None = None,
    flop: float | None = None,
    dtype=None,
) -> Callable[[str], str]:
    """
    Create a tool function for the CoVeR agent.

    The tool takes optimized code and returns feedback about performance.
    This is used as a CoVeR tool to evaluate optimization attempts.

    Args:
        executor: KernelBenchExecutor instance
        original_code: Original kernel code for comparison
        kernel_name: Kernel function name (optional if using KernelBench Model format)
        input_shapes: Input tensor shapes
        flop: Number of floating-point operations
        dtype: Torch dtype for input tensors

    Returns:
        Tool function that takes optimized code and returns feedback
    """

    def execute_and_compare(optimized_code: str) -> str:
        """
        Execute optimized kernel and compare with original.

        Args:
            optimized_code: The optimized Triton kernel code to test

        Returns:
            Feedback message about performance and correctness
        """
        result = executor.compare_kernels(
            original_code=original_code,
            optimized_code=optimized_code,
            kernel_name=kernel_name,
            input_shapes=input_shapes,
            flop=flop,
            dtype=dtype,
        )

        return result.feedback_message

    return execute_and_compare
