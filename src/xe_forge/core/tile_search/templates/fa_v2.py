"""
C++ template generator for Flash Attention V2 tile tuning using FMHAConfigGenWithTileShape.

Uses the sycl-tla benchmarks/flash_attention/fmha_configuration.hpp infrastructure
with explicit tile shape specification (WgTileQ, WgTileK, WgTileV, SgTileQ, SgTileK,
HeadDimQK, HeadDimV). Subgroup layouts for both QK and PV are derived automatically.

Output format matches _parse_raw_output:
  - "Disposition: Passed" / "Disposition: Failed"
  - "Performance:   X.XXX  GB/s,    Y.YYY  TFlop/s,   Z.ZZZZ  ms"
"""

from __future__ import annotations

_DTYPE_MAP = {
    "bf16": "bfloat16_t",
    "f16": "half_t",
    "fp8_e5m2": "cutlass::float_e5m2_t",
    "fp8_e4m3": "cutlass::float_e4m3_t",
}


def generate_fa_v2_source(
    wg_tile_q: int,
    wg_tile_k: int,
    wg_tile_v: int,
    sg_tile_q: int,
    sg_tile_k: int,
    head_dim_qk: int,
    head_dim_v: int,
    dtype: str = "bf16",
    causal: bool = False,
    mode: str = "prefill",
    persistent: bool = False,
    iterations: int = 100,
) -> str:
    """Generate a complete FA V2 C++ source using FMHAConfigGenWithTileShape."""
    from xe_forge.core.tile_search.templates import render

    element_type = _DTYPE_MAP.get(dtype, _DTYPE_MAP["bf16"])
    mode_cpp = "FMHAMode::Decode" if mode == "decode" else "FMHAMode::Prefill"

    return render(
        "fa_v2.cpp.j2",
        wg_tile_q=wg_tile_q,
        wg_tile_k=wg_tile_k,
        wg_tile_v=wg_tile_v,
        sg_tile_q=sg_tile_q,
        sg_tile_k=sg_tile_k,
        head_dim_qk=head_dim_qk,
        head_dim_v=head_dim_v,
        element_type=element_type,
        causal="true" if causal else "false",
        mode=mode_cpp,
        persistent="true" if persistent else "false",
        iterations_default=iterations,
    )
