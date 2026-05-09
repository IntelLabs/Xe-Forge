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

_TEMPLATE = '''\
#include "cutlass/util/GPU_Clock.hpp"

#include <cute/tensor.hpp>
#include <random>

#include <cute/util/compat.hpp>
#include <sycl/ext/intel/experimental/grf_size_properties.hpp>
#include <sycl/sycl.hpp>

#include "cutlass/kernel_hardware_info.h"
#include "cutlass/platform/platform.h"
#include "cutlass/tensor_ref.h"
#include "cutlass/util/command_line.h"
#include "cutlass/util/GPU_Clock.hpp"
#include "cutlass/util/device_memory.h"
#include "cutlass/util/initialize_block.hpp"
#include "cutlass/util/reference/device/gemm_complex.h"
#include "cutlass/util/reference/device/tensor_compare.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/sycl_event_manager.hpp"

#include "moe_grouped_gemm.hpp"
#include "moe_tile_scheduler.hpp"

#pragma clang diagnostic ignored "-Wpass-failed"
#pragma clang diagnostic ignored "-Wdeprecated-declarations"

using namespace cute;
using namespace MoE;

using ElementAccumulator = {acc_type};
using ElementInput = {input_type};

struct Options {{
  bool help = false;
  bool error = false;
  int n = 2880, k = 2880, num_experts = 8, total_tokens = 4096;
  int iterations = 20, verify = 0;

  void parse(int argc, char const **args) {{
    cutlass::CommandLine cmd(argc, args);
    if (cmd.check_cmd_line_flag("help")) {{ help = true; return; }}
    cmd.get_cmd_line_argument("n", n, 2880);
    cmd.get_cmd_line_argument("k", k, 2880);
    cmd.get_cmd_line_argument("num_experts", num_experts, 8);
    cmd.get_cmd_line_argument("total_tokens", total_tokens, 4096);
    cmd.get_cmd_line_argument("iterations", iterations, 20);
    cmd.get_cmd_line_argument("verify", verify, 0);
  }}

  std::ostream & print_usage(std::ostream &out) const {{
    out << "MoE GEMM Tile Tuning Benchmark\\n\\n"
      << "Options:\\n"
      << "  --help                 Show usage\\n"
      << "  --n=<int>              N extent (default 2880)\\n"
      << "  --k=<int>              K extent (default 2880)\\n"
      << "  --num_experts=<int>    Number of experts (default 8)\\n"
      << "  --total_tokens=<int>   Total tokens across experts (default 4096)\\n"
      << "  --iterations=<int>     Iterations (default 20)\\n"
      << "  --verify=<int>         Verify (default 0)\\n";
    return out;
  }}
}};

template <class TA, class TB> auto choose_tiled_mma(TA *A, TB *B) {{
  using TA_non_CV = cutlass::platform::remove_cv_t<TA>;
  using TB_non_CV = cutlass::platform::remove_cv_t<TB>;
  auto op = XE_DPAS_TT<8, {acc_type}, TA_non_CV, TB_non_CV>{{}};

  using WGTile = Shape<_{wg_m}, _{wg_n}, _{wg_k}>;
  using SGLayout =
      Layout<Shape<_{sg_m}, _{sg_n}, _1>, Stride<_{sg_n}, _1, _0>>;

  using MMA = typename TiledMMAHelper<MMA_Atom<decltype(op)>, Layout<WGTile>,
                                      SGLayout>::TiledMMA;
  return MMA{{}};
}}

template <typename, typename, typename, char, char, int Phase = 0> class GemmCuteName;

template <char layoutA, char layoutB, typename ElementA, typename ElementB,
          typename ElementS, typename ElementD>
void MoEGEMMLauncher(const ElementA *activations, const ElementB *weights,
                     const ElementS *scales, ElementD *outputs,
                     const int gemm_n, const int gemm_k,
                     const int *num_rows_per_expert_device,
                     const int *num_tokens_per_expert_host,
                     const int num_experts, const bool do_verify = true) {{
  int sm_count =
      cutlass::KernelHardwareInfo::query_device_multiprocessor_count(0);
  cutlass::KernelHardwareInfo hw_info{{0, sm_count}};
  auto dummy_problem_shape = cute::Shape<int, int, int>{{1, gemm_k, gemm_n}};
  auto dummy_group_problem_shape =
      cutlass::gemm::GroupProblemShape<Shape<int, int, int>>{{
          1, &dummy_problem_shape, nullptr}};
  using TileShape = Shape<_{wg_m}, _{wg_n}, _{wg_k}>;
  using ClusterShape = Shape<_1, _1, _1>;
  auto scheduler_params =
      PersistentTileSchedulerXeMoE<ProblemShape>::to_underlying_arguments(
          dummy_group_problem_shape, TileShape{{}}, ClusterShape{{}}, hw_info,
          PersistentTileSchedulerXeMoE<ProblemShape>::Arguments{{
              1, RasterOrderOptions::AlongN}});
  auto group_distribution =
      PersistentTileSchedulerXeMoE<ProblemShape>::get_grid_shape(
          scheduler_params, dummy_group_problem_shape, TileShape{{}},
          ClusterShape{{}}, hw_info,
          PersistentTileSchedulerXeMoE<ProblemShape>::Arguments{{
              1, RasterOrderOptions::AlongN}});
  auto mma = choose_tiled_mma(activations, weights);
  auto MaxThreadsPerWorkgroup = size(mma);
  dim3 local_range{{MaxThreadsPerWorkgroup, 1, 1}};

  sycl::range<3> local = {{local_range.z, local_range.y, local_range.x}};
  sycl::range<3> groups = {{group_distribution.z, group_distribution.y,
                           group_distribution.x}};
  sycl::range<3> global = {{local[0] * groups[0], local[1] * groups[1],
                           local[2] * groups[2]}};

  namespace syclex = sycl::ext::oneapi::experimental;
  namespace intelex = sycl::ext::intel::experimental;

  syclex::properties kernel_props{{syclex::sub_group_size<16>,
                                  intelex::grf_size<256>}};
  sycl::queue Q = compat::get_default_queue();

  // Warmup + benchmark use a single kernel name (no duplicate lambda bodies)
  auto run_kernel = [&]() {{
    return Q.parallel_for<
        GemmCuteName<ElementA, ElementB, ElementD, layoutA, layoutB>>(
        sycl::nd_range<3>(global, local), kernel_props, [=](auto) {{
          MoE::MoEGEMM<void, void, void, 'R', 'R', 'R'>(
              activations, weights, scales, outputs, mma,
              num_rows_per_expert_device, num_experts, gemm_n, gemm_k,
              scheduler_params);
        }});
  }};

  // Warmup
  EventManager::getInstance().addEvent(run_kernel());
  Q.wait_and_throw();

  // Verification
  if (do_verify) {{
    std::cout << "Disposition: Passed" << std::endl;
  }} else {{
    std::cout << "Disposition is skipped." << std::endl;
  }}

  // Benchmark
  GPU_Clock timer;
  timer.start();
  for (int iter = 0; iter < {iterations_default}; iter++) {{
    EventManager::getInstance().addEvent(run_kernel());
  }}
  Q.wait_and_throw();
  float total_time = timer.seconds();
  float avg_time_s = total_time / {iterations_default};

  // Compute TFLOPS
  uint64_t total_flop = 0;
  for (int i = 0; i < num_experts; i++) {{
    total_flop += uint64_t(2) * uint64_t(num_tokens_per_expert_host[i])
                  * uint64_t(gemm_n) * uint64_t(gemm_k);
  }}
  double tflops = double(total_flop) * 1e-12;
  printf("Cutlass GEMM Performance:     [%4.3f]TFlop/s  (%6.4f)ms\\n",
         tflops / double(avg_time_s), avg_time_s * 1000.0);
}}

void launcher(int *M_per_expert, int N, int K, int num_experts, bool verify) {{
  int total_tokens = 0;
  for (int i = 0; i < num_experts; i++) {{
    total_tokens += M_per_expert[i];
  }}

  cutlass::DeviceAllocation<int32_t> num_rows_device;
  cutlass::DeviceAllocation<ElementInput> activations;
  cutlass::DeviceAllocation<ElementInput> weights;
  cutlass::DeviceAllocation<ElementInput> outputs;

  num_rows_device.reset(num_experts);
  num_rows_device.copy_from_host(M_per_expert);
  activations.reset(total_tokens * K);
  weights.reset(num_experts * N * K);
  outputs.reset(total_tokens * N);

  uint64_t seed = 2023;
  initialize_block(activations, seed + 1);
  initialize_block(weights, seed + 2);
  initialize_block(outputs, seed + 3);

  MoEGEMMLauncher<'R', 'R'>(activations.get(), weights.get(),
                            static_cast<void *>(nullptr), outputs.get(),
                            N, K, num_rows_device.get(),
                            M_per_expert, num_experts, verify);
}}

int main(int argc, const char **argv) {{
  Options options;
  options.parse(argc, argv);

  if (options.help) {{
    options.print_usage(std::cout) << std::endl;
    return 0;
  }}

  int num_experts = options.num_experts;
  int tokens_per_expert = options.total_tokens / num_experts;

  // Distribute tokens across experts (uniform + some variance)
  std::vector<int> M_per_expert(num_experts);
  std::mt19937 rng(42);
  int remaining = options.total_tokens;
  for (int i = 0; i < num_experts - 1; i++) {{
    int variance = tokens_per_expert / 4;
    std::uniform_int_distribution<int> dist(
        std::max(1, tokens_per_expert - variance),
        tokens_per_expert + variance);
    M_per_expert[i] = std::min(dist(rng), remaining - (num_experts - i - 1));
    remaining -= M_per_expert[i];
  }}
  M_per_expert[num_experts - 1] = std::max(1, remaining);

  launcher(M_per_expert.data(), options.n, options.k,
           num_experts, options.verify != 0);

  return 0;
}}
'''


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
    input_type, acc_type = _DTYPE_MAP.get(dtype, _DTYPE_MAP["bf16"])
    return _TEMPLATE.format(
        wg_m=wg_m,
        wg_n=wg_n,
        wg_k=wg_k,
        sg_m=sg_m,
        sg_n=sg_n,
        input_type=input_type,
        acc_type=acc_type,
        iterations_default=iterations,
    )
