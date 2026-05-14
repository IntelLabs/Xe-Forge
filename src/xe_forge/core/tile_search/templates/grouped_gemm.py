"""
C++ template generator for CUTLASS Grouped GEMM tile tuning benchmarks.

Generates complete, compilable SYCL C++ source files parameterized by tile
shape. Based on sycl-tla/examples/04_bmg_grouped_gemm.
Uses explicit TiledMMA construction with MainloopXeL1StagedGroup dispatch.

Output format matches _parse_raw_output:
  - "Disposition: Passed" / "Disposition: Failed"
  - "Cutlass GEMM Performance:     [X]TFlop/s  (Y)ms"
"""

from __future__ import annotations

_DTYPE_MAP = {
    "bf16": ("bfloat16_t", "float"),
    "f16": ("half_t", "float"),
    "tf32": ("tfloat32_t", "float"),
}


def generate_grouped_gemm_source(
    wg_m: int,
    wg_n: int,
    wg_k: int,
    sg_m: int,
    sg_n: int,
    dtype: str = "bf16",
    layout_a: str = "RowMajor",
    layout_b: str = "RowMajor",
    pipeline_stages: int = 2,
) -> str:
    """Generate a complete CUTLASS Grouped GEMM C++ source for the given tile config."""
    from xe_forge.core.tile_search.templates import render

    input_type, acc_type = _DTYPE_MAP.get(dtype, _DTYPE_MAP["bf16"])
    return render(
        "grouped_gemm.cpp.j2",
        wg_m=wg_m,
        wg_n=wg_n,
        wg_k=wg_k,
        sg_m=sg_m,
        sg_n=sg_n,
        input_type=input_type,
        acc_type=acc_type,
        layout_a=layout_a,
        layout_b=layout_b,
        pipeline_stages=pipeline_stages,
    )
