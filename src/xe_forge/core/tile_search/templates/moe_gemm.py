"""
C++ template generator for CUTLASS MoE GEMM tile tuning benchmarks.

Generates complete, compilable SYCL C++ source files parameterized by tile
shape. Based on sycl-tla/examples/12_xe20_moe_gemm_cute_interface.
Uses manual TiledMMA construction with PersistentTileSchedulerXeMoE.

The MoE GEMM runs multiple grouped GEMMs where each expert has a different
number of tokens (M dimension). N and K are shared across all experts.

Output format matches _parse_raw_output:
  - "Disposition: Passed" / "Disposition: Failed"
  - "Cutlass GEMM Performance:     [X]TFlop/s  (Y)ms"
"""

from __future__ import annotations

_DTYPE_MAP = {
    "bf16": ("bfloat16_t", "float"),
    "f16": ("half_t", "float"),
}


def generate_moe_gemm_source(
    wg_m: int,
    wg_n: int,
    wg_k: int,
    sg_m: int,
    sg_n: int,
    dtype: str = "bf16",
    iterations: int = 20,
) -> str:
    """Generate a complete CUTLASS MoE GEMM C++ source for the given tile config."""
    from xe_forge.core.tile_search.templates import render

    input_type, acc_type = _DTYPE_MAP.get(dtype, _DTYPE_MAP["bf16"])
    return render(
        "moe_gemm.cpp.j2",
        wg_m=wg_m,
        wg_n=wg_n,
        wg_k=wg_k,
        sg_m=sg_m,
        sg_n=sg_n,
        input_type=input_type,
        acc_type=acc_type,
        iterations_default=iterations,
    )
