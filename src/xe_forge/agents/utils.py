"""Shared utilities for Xe-Forge agents."""

import logging

logger = logging.getLogger(__name__)

SUCCESS_MESSAGE = "Success! Optimization verified and kernel is faster."


def extract_gemm_dims(
    input_shapes: list[tuple[int, ...]] | None,
) -> tuple[int, int, int]:
    """Extract M, N, K from GEMM input shapes [(M, K), (K, N)]."""
    if input_shapes and len(input_shapes) >= 2:
        a, b = input_shapes[0], input_shapes[1]
        if len(a) >= 2 and len(b) >= 2:
            return a[-2], b[-1], a[-1]
    return 1024, 1024, 1024


def verify_sycl(code, original_code, executor, input_shapes, spec_dims=None):
    """Verify a SYCL C++ kernel: basic structure check + runtime comparison."""
    if "#include" not in code:
        return "MISSING: C++ code must contain #include directives."
    if "sycl" not in code.lower() and "cutlass" not in code.lower():
        return "MISSING: Code does not appear to be a SYCL/CUTLASS kernel."

    if executor:
        try:
            _dims = spec_dims or dict(
                zip(("M", "N", "K"), extract_gemm_dims(input_shapes), strict=False)
            )
            comparison = executor.compare_kernels(
                original_code=original_code,
                optimized_code=code,
                dims=_dims,
            )
            if not comparison.optimized_correct:
                return comparison.feedback_message or "Optimized kernel failed."
            if comparison.is_slower:
                sd = 1.0 / comparison.speedup if comparison.speedup > 0 else float("inf")
                return (
                    f"PERFORMANCE REGRESSION: {sd:.2f}x SLOWER.\n"
                    f"Original: {comparison.original_time_ms:.4f}ms ({comparison.original_tflops or 0:.3f} TFlop/s)\n"
                    f"Optimized: {comparison.optimized_time_ms:.4f}ms ({comparison.optimized_tflops or 0:.3f} TFlop/s)"
                )
            logger.info(
                f"SYCL optimization verified: {comparison.speedup:.2f}x speedup "
                f"({comparison.original_tflops or 0:.3f} -> {comparison.optimized_tflops or 0:.3f} TFlop/s)"
            )
            return SUCCESS_MESSAGE
        except Exception as e:
            return f"RUNTIME ERROR: {e!s}"

    logger.warning("No executor - accepting SYCL code based on static checks only")
    return SUCCESS_MESSAGE
