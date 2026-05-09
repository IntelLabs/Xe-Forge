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

_TEMPLATE = '''\
#include "cutlass/epilogue/collective/default_epilogue.hpp"
#include "cutlass/epilogue/collective/xe_array_epilogue.hpp"
#include "cutlass/epilogue/fusion/xe_callbacks.hpp"
#include "cutlass/gemm/group_array_problem_shape.hpp"
#include "cutlass/gemm/device/gemm_universal.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/collective/collective_mma.hpp"
#include "cutlass/util/GPU_Clock.hpp"

#include <cute/tensor.hpp>
#include <random>

#include "cutlass/util/command_line.h"
#include "cutlass/util/device_memory.h"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass/util/reference/device/gemm_complex.h"
#include "cutlass/util/reference/device/tensor_compare.h"
#include "sycl_common.hpp"
#include "helper.h"

#include <cfloat>

using namespace cute;
using ProblemShape = cutlass::gemm::GroupProblemShape<Shape<int,int,int>>;

using ElementAccumulator = {acc_type};
using ElementComputeEpilogue = {acc_type};
using ElementA = {input_type};
using ElementB = {input_type};
using ElementOutput = {acc_type};

struct Options {{
  bool error = false;
  bool help = false;
  float alpha = 1.f, beta = 0.f;
  int m = 5120, n = 4096, k = 4096, groups = 2, iterations = 100, verify = 1;

  std::vector<typename ProblemShape::UnderlyingProblemShape> problem_sizes_host;

  Options() {{
    problem_sizes_host.reserve(groups);
    for(int i = 0; i < groups; i++) {{
      problem_sizes_host.push_back({{m, n, k}});
    }}
  }}

  void parse(int argc, char const **args) {{
    cutlass::CommandLine cmd(argc, args);
    if (cmd.check_cmd_line_flag("help")) {{ help = true; return; }}
    cmd.get_cmd_line_argument("m", m, 5120);
    cmd.get_cmd_line_argument("n", n, 4096);
    cmd.get_cmd_line_argument("k", k, 4096);
    cmd.get_cmd_line_argument("groups", groups, 2);
    cmd.get_cmd_line_argument("alpha", alpha, 1.f);
    cmd.get_cmd_line_argument("beta", beta, 0.f);
    cmd.get_cmd_line_argument("iterations", iterations, 100);
    cmd.get_cmd_line_argument("verify", verify, 1);

    problem_sizes_host.clear();
    problem_sizes_host.reserve(groups);
    for(int i = 0; i < groups; i++) {{
      problem_sizes_host.push_back({{m, n, k}});
    }}
  }}

  std::ostream & print_usage(std::ostream &out) const {{
    out << "Tile Tuning Grouped GEMM Benchmark\\n\\n"
      << "Options:\\n"
      << "  --help         Show usage\\n"
      << "  --m=<int>      M extent per group (default 5120)\\n"
      << "  --n=<int>      N extent per group (default 4096)\\n"
      << "  --k=<int>      K extent per group (default 4096)\\n"
      << "  --groups=<int> Number of groups (default 2)\\n"
      << "  --iterations=<int>  Iterations (default 100)\\n"
      << "  --verify=<int> Verify correctness (default 1)\\n";
    return out;
  }}

  double gflops(double runtime_s) const {{
    uint64_t fmas = 0;
    for (auto const & problem : problem_sizes_host) {{
      fmas += static_cast<uint64_t>(get<0>(problem)) *
              static_cast<uint64_t>(get<1>(problem)) *
              static_cast<uint64_t>(get<2>(problem));
    }}
    uint64_t flop = uint64_t(2) * fmas;
    return double(flop) / double(1.0e9) / runtime_s;
  }}
}};

///////////////////////////////////////////////////////////////////////////////////////////////////

template <class Gemm>
struct ExampleRunner {{

  using LayoutA = typename Gemm::LayoutA;
  using LayoutB = typename Gemm::LayoutB;
  using LayoutC = typename Gemm::LayoutC;
  using LayoutD = typename Gemm::LayoutD;

  using CollectiveEpilogue = typename Gemm::CollectiveEpilogue;
  using ElementOutput_ = typename CollectiveEpilogue::ElementOutput;
  using ElementAccumulator_ = ElementOutput_;

  using StrideA = typename Gemm::GemmKernel::InternalStrideA;
  using StrideB = typename Gemm::GemmKernel::InternalStrideB;
  using StrideC = typename Gemm::GemmKernel::InternalStrideC;
  using StrideD = typename Gemm::GemmKernel::InternalStrideD;

  std::vector<int64_t> offset_A, offset_B, offset_C, offset_D;
  std::vector<StrideA> stride_A_host;
  std::vector<StrideB> stride_B_host;
  std::vector<StrideC> stride_C_host;
  std::vector<StrideD> stride_D_host;
  std::vector<ElementAccumulator_> alpha_host, beta_host;

  cutlass::DeviceAllocation<typename ProblemShape::UnderlyingProblemShape> problem_sizes;
  cutlass::DeviceAllocation<ElementA> block_A;
  cutlass::DeviceAllocation<ElementB> block_B;
  cutlass::DeviceAllocation<ElementOutput> block_C;
  cutlass::DeviceAllocation<ElementOutput> block_D;
  cutlass::DeviceAllocation<ElementOutput> block_ref_D;

  cutlass::DeviceAllocation<const ElementA *> ptr_A;
  cutlass::DeviceAllocation<const ElementB *> ptr_B;
  cutlass::DeviceAllocation<const ElementOutput *> ptr_C;
  cutlass::DeviceAllocation<ElementOutput *> ptr_D;

  cutlass::DeviceAllocation<StrideA> stride_A;
  cutlass::DeviceAllocation<StrideB> stride_B;
  cutlass::DeviceAllocation<StrideC> stride_C;
  cutlass::DeviceAllocation<StrideD> stride_D;

  cutlass::DeviceAllocation<ElementAccumulator_*> alpha_device;
  cutlass::DeviceAllocation<ElementAccumulator_*> beta_device;
  cutlass::DeviceAllocation<ElementAccumulator_> block_alpha;
  cutlass::DeviceAllocation<ElementAccumulator_> block_beta;

  uint64_t seed = 0;

  bool verify(const Options &options) {{
    bool passed = true;
    for (int32_t i = 0; i < options.groups; ++i) {{
      auto problem = options.problem_sizes_host.at(i);
      auto M = get<0>(problem);
      auto N = get<1>(problem);
      auto K = get<2>(problem);
      cutlass::TensorRef ref_A(block_A.get() + offset_A.at(i), LayoutA::packed({{M, K}}));
      cutlass::TensorRef ref_B(block_B.get() + offset_B.at(i), LayoutB::packed({{K, N}}));
      cutlass::TensorRef ref_C(block_C.get() + offset_C.at(i), LayoutC::packed({{M, N}}));
      cutlass::TensorRef ref_D(block_ref_D.get() + offset_D.at(i), LayoutD::packed({{M, N}}));

      cutlass::reference::device::GemmComplex(
            {{M, N, K}},
            alpha_host.at(i),
            ref_A, cutlass::ComplexTransform::kNone,
            ref_B, cutlass::ComplexTransform::kNone,
            beta_host.at(i),
            ref_C, ref_D,
            ElementAccumulator_(0),
            1, M * K, K * N, M * N, M * N);

      compat::wait();

      passed &= cutlass::reference::device::BlockCompareEqual(
          block_ref_D.get() + offset_D.at(i), block_D.get() + offset_D.at(i), M * N);
      if(!passed) break;
    }}
    return passed;
  }}

  void allocate(const Options &options) {{
    int64_t total_A = 0, total_B = 0, total_C = 0, total_D = 0;
    for (int32_t i = 0; i < options.groups; ++i) {{
      auto problem = options.problem_sizes_host.at(i);
      auto M = get<0>(problem); auto N = get<1>(problem); auto K = get<2>(problem);
      offset_A.push_back(total_A);
      offset_B.push_back(total_B);
      offset_C.push_back(total_C);
      offset_D.push_back(total_D);
      total_A += M * K; total_B += K * N; total_C += M * N; total_D += M * N;
      stride_A_host.push_back(cutlass::make_cute_packed_stride(StrideA{{}}, {{M, K, 1}}));
      stride_B_host.push_back(cutlass::make_cute_packed_stride(StrideB{{}}, {{N, K, 1}}));
      stride_C_host.push_back(cutlass::make_cute_packed_stride(StrideC{{}}, {{M, N, 1}}));
      stride_D_host.push_back(cutlass::make_cute_packed_stride(StrideD{{}}, {{M, N, 1}}));
    }}
    block_A.reset(total_A); block_B.reset(total_B);
    block_C.reset(total_C); block_D.reset(total_D);
    block_ref_D.reset(total_D);
    block_alpha.reset(options.groups); block_beta.reset(options.groups);
  }}

  void initialize(const Options &options) {{
    problem_sizes.reset(options.groups);
    problem_sizes.copy_from_host(options.problem_sizes_host.data());

    std::vector<ElementA *> ptr_A_host(options.groups);
    std::vector<ElementB *> ptr_B_host(options.groups);
    std::vector<ElementOutput *> ptr_C_host(options.groups);
    std::vector<ElementOutput *> ptr_D_host(options.groups);
    std::vector<ElementAccumulator_ *> ptr_alpha_host(options.groups);
    std::vector<ElementAccumulator_ *> ptr_beta_host(options.groups);

    for (int32_t i = 0; i < options.groups; ++i) {{
      ptr_A_host.at(i) = block_A.get() + offset_A.at(i);
      ptr_B_host.at(i) = block_B.get() + offset_B.at(i);
      ptr_C_host.at(i) = block_C.get() + offset_C.at(i);
      ptr_D_host.at(i) = block_D.get() + offset_D.at(i);
      alpha_host.push_back(options.alpha);
      beta_host.push_back(options.beta);
      ptr_alpha_host.at(i) = block_alpha.get() + i;
      ptr_beta_host.at(i) = block_beta.get() + i;
    }}

    ptr_A.reset(options.groups); ptr_A.copy_from_host(ptr_A_host.data());
    ptr_B.reset(options.groups); ptr_B.copy_from_host(ptr_B_host.data());
    ptr_C.reset(options.groups); ptr_C.copy_from_host(ptr_C_host.data());
    ptr_D.reset(options.groups); ptr_D.copy_from_host(ptr_D_host.data());

    stride_A.reset(options.groups); stride_A.copy_from_host(stride_A_host.data());
    stride_B.reset(options.groups); stride_B.copy_from_host(stride_B_host.data());
    stride_C.reset(options.groups); stride_C.copy_from_host(stride_C_host.data());
    stride_D.reset(options.groups); stride_D.copy_from_host(stride_D_host.data());

    alpha_device.reset(options.groups); alpha_device.copy_from_host(ptr_alpha_host.data());
    beta_device.reset(options.groups); beta_device.copy_from_host(ptr_beta_host.data());

    initialize_block(block_A, seed + 2023);
    initialize_block(block_B, seed + 2022);
    initialize_block(block_C, seed + 2021);
    block_alpha.copy_from_host(alpha_host.data());
    block_beta.copy_from_host(beta_host.data());
  }}

  typename Gemm::Arguments args_from_options(
      const Options &options, const cutlass::KernelHardwareInfo& hw_info) {{
    using RasterOrderOptions = typename cutlass::gemm::kernel::detail::PersistentTileSchedulerXeGroup<ProblemShape>::RasterOrderOptions;

    decltype(typename Gemm::Arguments{{}}.epilogue.thread) fusion_args;
    fusion_args.alpha = options.alpha;
    fusion_args.beta = options.beta;
    fusion_args.alpha_ptr = nullptr;
    fusion_args.beta_ptr = nullptr;
    fusion_args.alpha_ptr_array = nullptr;
    fusion_args.beta_ptr_array = nullptr;
    fusion_args.dAlpha = {{cute::_0{{}}, cute::_0{{}}, 0}};
    fusion_args.dBeta = {{cute::_0{{}}, cute::_0{{}}, 0}};

    return typename Gemm::Arguments {{
      cutlass::gemm::GemmUniversalMode::kGrouped,
      {{options.groups, problem_sizes.get(), options.problem_sizes_host.data()}},
      {{ptr_A.get(), stride_A.get(), ptr_B.get(), stride_B.get()}},
      {{fusion_args, ptr_C.get(), stride_C.get(), ptr_D.get(), stride_D.get()}},
      hw_info,
      {{1, RasterOrderOptions::AlongN}}
    }};
  }}

  cutlass::Status run(const Options& options, const cutlass::KernelHardwareInfo& hw_info) {{
    allocate(options);
    initialize(options);

    Gemm gemm_op;
    auto arguments = args_from_options(options, hw_info);

    size_t workspace_size = Gemm::get_workspace_size(arguments);
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

    CUTLASS_CHECK(gemm_op.can_implement(arguments));
    CUTLASS_CHECK(gemm_op.initialize(arguments, workspace.get()));
    CUTLASS_CHECK(gemm_op.run());

    compat::wait();

    if (options.verify != 0) {{
      bool passed = verify(options);
      std::cout << "Disposition: " << (passed ? "Passed" : "Failed") << std::endl;
      if (!passed) return cutlass::Status::kErrorInternal;
    }} else {{
      std::cout << "Disposition is skipped." << std::endl;
    }}

    if (options.iterations > 0) {{
      GPU_Clock timer;
      timer.start();
      for (int i = 0; i < options.iterations; ++i) {{
        CUTLASS_CHECK(gemm_op.run());
      }}
      compat::wait();

      float cute_time = timer.seconds() / options.iterations;
      double tflops = 0;
      for (auto const & p : options.problem_sizes_host) {{
        tflops += (2.0 * get<0>(p) * get<1>(p) * get<2>(p)) * 1e-12;
      }}
      std::cout << "Problem Size: " << options.m << 'x' << options.n << 'x' << options.k
                << " groups=" << options.groups << std::endl;
      printf("Cutlass GEMM Performance:     [%4.3f]TFlop/s  (%6.4f)ms\\n", tflops / cute_time, cute_time*1000);
    }}

    return cutlass::Status::kSuccess;
  }}
}};

int main(int argc, const char** argv)
{{
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

  cutlass::KernelHardwareInfo hw_info;
  hw_info.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(hw_info.device_id);

  using LayoutA = cutlass::layout::{layout_a};
  using LayoutB = cutlass::layout::{layout_b};
  using LayoutC = cutlass::layout::RowMajor;
  using LayoutD = cutlass::layout::RowMajor;

  using GmemTiledCopyA = void;
  using GmemTiledCopyB = void;

  using TileShape = Shape<_{wg_m}, _{wg_n}, _{wg_k}>;

  using TiledMma = typename TiledMMAHelper<
      MMA_Atom<XE_DPAS_TT<8, ElementAccumulator, ElementA>>,
      Layout<TileShape>,
      Layout<Shape<_{sg_m}, _{sg_n}, _1>, Stride<_{sg_n}, _1, _0>>
  >::TiledMMA;

  constexpr int PipelineStages = {pipeline_stages};
  using GEMMDispatchPolicy = cutlass::gemm::MainloopXeL1StagedGroup<PipelineStages>;
  using EpilogueDispatchPolicy = cutlass::epilogue::IntelXeGenericGroup;

  using EpilogueOp = cutlass::epilogue::fusion::LinearCombination<
      ElementOutput, ElementComputeEpilogue, ElementAccumulator, ElementAccumulator,
      cutlass::FloatRoundStyle::round_to_nearest>;

  using FusionCallBacks = cutlass::epilogue::fusion::FusionCallbacks<
      EpilogueDispatchPolicy, EpilogueOp, TileShape, decltype(tile_shape(TiledMma()))>;

  using CollectiveEpilogue = cutlass::epilogue::collective::CollectiveEpilogue<
      EpilogueDispatchPolicy,
      TileShape,
      void,
      ElementAccumulator,
      cutlass::gemm::TagToStrideC_t<LayoutC*>,
      ElementOutput,
      cutlass::gemm::TagToStrideC_t<LayoutD*>,
      FusionCallBacks,
      void, void>;

  using CollectiveMainloop = cutlass::gemm::collective::CollectiveMma<
      GEMMDispatchPolicy,
      TileShape,
      ElementA,
      cutlass::gemm::TagToStrideA_t<LayoutA*>,
      ElementB,
      cutlass::gemm::TagToStrideB_t<LayoutB*>,
      TiledMma,
      GmemTiledCopyA, void, void, cute::identity,
      GmemTiledCopyB, void, void, cute::identity
  >;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      ProblemShape,
      CollectiveMainloop,
      CollectiveEpilogue,
      cutlass::gemm::GroupScheduler
  >;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

  ExampleRunner<Gemm> runner;
  CUTLASS_CHECK(runner.run(options, hw_info));

  return 0;
}}
'''


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
    input_type, acc_type = _DTYPE_MAP.get(dtype, _DTYPE_MAP["bf16"])
    return _TEMPLATE.format(
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
