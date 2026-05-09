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

_TEMPLATE = '''\
#ifndef HEAD_DIM
#define HEAD_DIM {head_dim}
#endif
#ifndef PREFILL
#define PREFILL
#endif
{split_v_define}
#include "xe_fmha_fwd_runner.hpp"

int main(int argc, const char **argv) {{
  Options options;
  options.parse(argc, argv);

  if (options.help) {{
    options.print_usage(std::cout) << std::endl;
    return 0;
  }}
  if (options.error) {{
    std::cerr << "Aborting execution." << std::endl;
    return -1;
  }}

  // Tile shapes from tile tuning search
  using ShapeQK  = Shape<_{qk_m}, _{qk_n}, _{qk_k}>;
  using ShapePV  = Shape<_{pv_m}, _{pv_n}, _{pv_k}>;
  using ShapeOut = Shape<_{out_m}, _{out_n}>;
  using SubgroupLayoutQK = Layout<Shape<_{sg_q}, _1, _1>>;

  using ElementQ = {element_type};
  using ElementK = {element_type};
  using ElementV = {element_type};

  constexpr int PipelineStages = {pipeline_stages};

  return options.is_causal
    ? FMHAConfig<true,  ShapeQK, ShapePV, ShapeOut, SubgroupLayoutQK,
                 void, PipelineStages, false,
                 ElementQ, ElementK, ElementV>::run(options)
    : FMHAConfig<false, ShapeQK, ShapePV, ShapeOut, SubgroupLayoutQK,
                 void, PipelineStages, false,
                 ElementQ, ElementK, ElementV>::run(options);
}}
'''


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
    element_type = _DTYPE_MAP.get(dtype, _DTYPE_MAP["bf16"])

    out_n = head_dim
    if split_v_groups > 1:
        split_v_define = f"#define SPLIT_V_GROUPS {split_v_groups}"
    else:
        split_v_define = ""

    return _TEMPLATE.format(
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
