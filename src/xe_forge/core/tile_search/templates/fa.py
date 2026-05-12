"""
C++ template generator for Flash Attention V2 tile tuning benchmarks.

Generates complete, compilable SYCL C++ source files parameterized by
FA tile shapes (ShapeQK, ShapePV, ShapeOut, SubgroupLayoutQK).

Based on sycl-tla/examples/06_bmg_flash_attention. Uses the same
FMHAConfig -> ExampleRunner pipeline, but with tile shapes injected
as template parameters instead of hardcoded #if HEAD_DIM blocks.

Output format matches the FA runner:
  "Disposition: Passed" / "Disposition: Failed"
  "Performance:   X.XXX  GB/s,    Y.YYY  TFlop/s,   Z.ZZZZ  ms"
"""

from __future__ import annotations

_DTYPE_MAP = {
    "bf16": "bfloat16_t",
    "f16": "half_t",
    "fp8_e5m2": "cutlass::float_e5m2_t",
    "fp8_e4m3": "cutlass::float_e4m3_t",
}


def generate_fa_source(
    qk_m: int,
    qk_n: int,
    qk_k: int,
    pv_m: int,
    pv_n: int,
    pv_k: int,
    head_dim: int,
    sg_q: int,
    dtype: str = "bf16",
    pipeline_stages: int = 2,
    split_v_groups: int = 1,
) -> str:
    """Generate a complete FA V2 C++ source for the given tile config.

    When split_v_groups > 1, the kernel processes VTiles in groups to
    reduce output accumulator register pressure. ShapeOut uses the FULL
    head_dim -- the runner internally divides VTiles by SPLIT_V_GROUPS.
    """
    from xe_forge.core.tile_search.templates import render

    element_type = _DTYPE_MAP.get(dtype, _DTYPE_MAP["bf16"])

    out_n = head_dim
    if split_v_groups > 1:
        split_v_define = f"#define SPLIT_V_GROUPS {split_v_groups}"
    else:
        split_v_define = ""

    return render(
        "fa.cpp.j2",
        qk_m=qk_m,
        qk_n=qk_n,
        qk_k=qk_k,
        pv_m=pv_m,
        pv_n=pv_n,
        pv_k=pv_k,
        out_m=qk_m,
        out_n=out_n,
        sg_q=sg_q,
        element_type=element_type,
        pipeline_stages=pipeline_stages,
        head_dim=head_dim,
        split_v_define=split_v_define,
    )
