"""
C++ template generator for CUTLASS GEMM tile tuning benchmarks.

Generates complete, compilable SYCL C++ source files parameterized by tile
shape. Based on sycl-tla/examples/01_bmg_gemm_with_collective_builder.
Uses CollectiveBuilder so the MMA atom and subgroup layout are auto-derived.

The output matches the format expected by ai_bench.sycl.compiler.SYCLCompiler:
  - "Disposition: Passed" / "Disposition: Failed"
  - "Cutlass GEMM Performance:     [X]TFlop/s  (Y)ms"
"""

from __future__ import annotations

_DTYPE_MAP = {
    "bf16": ("bfloat16_t", "float"),
    "f16": ("half_t", "float"),
    "tf32": ("tfloat32_t", "float"),
}


def generate_gemm_source(
    wg_m: int,
    wg_n: int,
    wg_k: int,
    dtype: str = "bf16",
    layout_a: str = "RowMajor",
    layout_b: str = "RowMajor",
) -> str:
    """Generate a complete CUTLASS GEMM C++ source for the given tile config."""
    from xe_forge.core.tile_search.templates import render

    input_type, acc_type = _DTYPE_MAP.get(dtype, _DTYPE_MAP["bf16"])
    return render(
        "gemm.cpp.j2",
        wg_m=wg_m,
        wg_n=wg_n,
        wg_k=wg_k,
        input_type=input_type,
        acc_type=acc_type,
        layout_a=layout_a,
        layout_b=layout_b,
    )
