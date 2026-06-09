/*
  Example SYCL trial kernel (t1) for the Xe-Forge Claude engine — gemm.cpp with
  a smaller workgroup tile (128x128x32 instead of 256x256x32). On an Intel Arc
  Pro B70 at M=N=K=1024 this is ~1.7x faster than the t0 baseline while still
  matching the PyTorch golden reference. Illustrates one optimization step the
  agent takes; the file-IO scaffolding is unchanged from gemm.cpp.

  File-IO contract (see knowledge_base/sycl/xpu/sycl_io_contract.yaml):
    --m/--n/--k            problem dims
    --input_dir=<dir>      read A.bin [M,K], B0.bin [K,N] (bf16 as raw int16 bits)
    --output_dir=<dir>     write D2.bin [M,N] as float32, row-major
    --iterations=<int>     timed iterations
    --verify=<int>         ignored (correctness checked in Python vs golden ref)

  Prints a "[<tflops>]TFlop/s  (<ms>)ms" line that SyclExecutor parses.

  Derived from sycl-tla/examples/00_bmg_gemm/00_bmg_gemm.cpp.
*/

#include "cutlass/epilogue/collective/default_epilogue.hpp"
#include "cutlass/epilogue/collective/xe_epilogue.hpp"
#include "cutlass/epilogue/fusion/xe_callbacks.hpp"
#include "cutlass/gemm/device/gemm_universal.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/collective/collective_mma.hpp"
#include "cutlass/util/GPU_Clock.hpp"

#include <cute/tensor.hpp>
#include <fstream>
#include <vector>
#include <string>

#include "cutlass/util/command_line.h"
#include "cutlass/util/device_memory.h"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass/util/reference/device/gemm_complex.h"
#include "cutlass/util/reference/device/tensor_compare.h"
#include "sycl_common.hpp"
#include "helper.h"

using namespace cute;

struct Options {
  bool help = false;
  bool error = false;
  int m = 5120, n = 4096, k = 4096, l = 1, iterations = 20, verify = 1;
  float alpha = 1.f, beta = 0.f;
  std::string input_dir;
  std::string output_dir;

  void parse(int argc, char const **args) {
    cutlass::CommandLine cmd(argc, args);
    if (cmd.check_cmd_line_flag("help")) { help = true; return; }
    cmd.get_cmd_line_argument("m", m, 5120);
    cmd.get_cmd_line_argument("n", n, 4096);
    cmd.get_cmd_line_argument("k", k, 4096);
    cmd.get_cmd_line_argument("l", l, 1);
    cmd.get_cmd_line_argument("alpha", alpha, 1.f);
    cmd.get_cmd_line_argument("beta", beta, 0.f);
    cmd.get_cmd_line_argument("iterations", iterations, 20);
    cmd.get_cmd_line_argument("verify", verify, 1);
    cmd.get_cmd_line_argument("input_dir", input_dir, std::string(""));
    cmd.get_cmd_line_argument("output_dir", output_dir, std::string(""));
  }
};

template <typename T>
static std::vector<T> read_bin(const std::string& dir, const std::string& name, size_t count) {
  std::string path = dir + "/" + name;
  std::ifstream f(path, std::ios::binary);
  if (!f) { std::cerr << "Could not open " << path << std::endl; std::exit(2); }
  std::vector<T> buf(count);
  f.read(reinterpret_cast<char*>(buf.data()), count * sizeof(T));
  if (static_cast<size_t>(f.gcount()) != count * sizeof(T)) {
    std::cerr << "Short read on " << path << ": got " << f.gcount()
              << " want " << count * sizeof(T) << std::endl;
    std::exit(2);
  }
  return buf;
}

template <typename T>
static void write_bin(const std::string& dir, const std::string& name, const std::vector<T>& buf) {
  std::string path = dir + "/" + name;
  std::ofstream f(path, std::ios::binary);
  if (!f) { std::cerr << "Could not open for write " << path << std::endl; std::exit(2); }
  f.write(reinterpret_cast<const char*>(buf.data()), buf.size() * sizeof(T));
}

template <class Gemm>
struct ExampleRunner {
  using StrideA = typename Gemm::GemmKernel::StrideA;
  using StrideB = typename Gemm::GemmKernel::StrideB;
  using StrideC = typename Gemm::GemmKernel::StrideC;
  using StrideD = typename Gemm::GemmKernel::StrideD;

  using LayoutA = typename Gemm::LayoutA;
  using LayoutB = typename Gemm::LayoutB;
  using LayoutC = typename Gemm::LayoutC;
  using LayoutD = typename Gemm::LayoutD;

  using ElementA = typename Gemm::ElementA;
  using ElementB = typename Gemm::ElementB;
  using ElementAccumulator = typename Gemm::ElementAccumulator;

  using CollectiveEpilogue = typename Gemm::CollectiveEpilogue;
  using ElementC = typename Gemm::ElementC;
  using ElementOutput = typename CollectiveEpilogue::ElementOutput;
  using ElementCompute = typename CollectiveEpilogue::ElementCompute;

  using ProblemShapeType = typename Gemm::GemmKernel::ProblemShape;

  StrideA stride_A;
  StrideB stride_B;
  StrideC stride_C;
  StrideD stride_D;
  uint64_t seed = 0;

  cutlass::DeviceAllocation<ElementA> block_A;
  cutlass::DeviceAllocation<ElementB> block_B;
  cutlass::DeviceAllocation<ElementC> block_C;
  cutlass::DeviceAllocation<ElementOutput> block_D;

  void initialize(const ProblemShapeType& problem_size, const Options& options) {
    auto problem_shape_MNKL = cute::append<4>(problem_size, 1);
    auto [M, N, K, L] = problem_shape_MNKL;

    stride_A = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(M, K, L));
    stride_B = cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(N, K, L));
    stride_C = cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(M, N, L));
    stride_D = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(M, N, L));

    block_A.reset(static_cast<std::size_t>(M) * K * L);
    block_B.reset(static_cast<std::size_t>(K) * N * L);
    block_C.reset(static_cast<std::size_t>(M) * N * L);
    block_D.reset(static_cast<std::size_t>(M) * N * L);

    // A [M,K] and B0 [K,N] are raw bf16 bits (bit-identical to torch bf16).
    auto host_A = read_bin<ElementA>(options.input_dir, "A.bin",
                                     static_cast<size_t>(M) * K * L);
    auto host_B = read_bin<ElementB>(options.input_dir, "B0.bin",
                                     static_cast<size_t>(K) * N * L);
    block_A.copy_from_host(host_A.data());
    block_B.copy_from_host(host_B.data());

    // C unused (beta = 0); zero-fill so the epilogue reads valid memory.
    std::vector<ElementC> host_C(static_cast<size_t>(M) * N * L, ElementC(0));
    block_C.copy_from_host(host_C.data());
  }

  void dump_output(const ProblemShapeType& problem_size, const Options& options) {
    auto problem_shape_MNKL = cute::append<4>(problem_size, 1);
    auto [M, N, K, L] = problem_shape_MNKL;
    std::vector<ElementOutput> host_D(static_cast<size_t>(M) * N * L);
    block_D.copy_to_host(host_D.data());
    write_bin(options.output_dir, "D2.bin", host_D);  // ElementOutput = float32
  }

  cutlass::Status run(const Options& options, const cutlass::KernelHardwareInfo& hw_info) {
    ProblemShapeType problem_size = ProblemShapeType{options.m, options.n, options.k, options.l};

    initialize(problem_size, options);

    typename Gemm::GemmKernel::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGemm,
      problem_size,
      {block_A.get(), stride_A, block_B.get(), stride_B},
      {{options.alpha, options.beta}, block_C.get(), stride_C, block_D.get(), stride_D},
      hw_info
    };

    Gemm gemm_op;

    size_t workspace_size = Gemm::get_workspace_size(arguments);
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

    if (gemm_op.can_implement(arguments) != cutlass::Status::kSuccess) {
      std::cout << "Invalid Problem Size: " << options.m << 'x' << options.n << 'x'
                << options.k << 'x' << options.l << std::endl;
      std::exit(1);
    }

    CUTLASS_CHECK(gemm_op.initialize(arguments, workspace.get()));
    CUTLASS_CHECK(gemm_op.run());
    compat::wait();

    if (!options.output_dir.empty()) {
      dump_output(problem_size, options);
    }
    std::cout << "Disposition: Passed" << std::endl;

    if (options.iterations > 0) {
      GPU_Clock timer;
      timer.start();
      for (int i = 0; i < options.iterations; ++i) {
        gemm_op.run();
      }
      compat::wait();

      float cute_time = timer.seconds() / options.iterations;
      double tflops = (2.0 * options.m * options.n * options.k * options.l) * 1e-12;
      std::cout << "Problem Size: " << options.m << 'x' << options.n << 'x'
                << options.k << 'x' << options.l << std::endl;
      printf("Cutlass GEMM Performance:     [%4.3f]TFlop/s  (%6.4f)ms\n",
             tflops / cute_time, cute_time * 1000);
    }

    return cutlass::Status::kSuccess;
  }
};

int main(int argc, const char** argv) {
  Options options;
  options.parse(argc, argv);
  if (options.help) { std::cout << "Xe-Forge SYCL starter GEMM\n"; return 0; }
  if (options.error) { std::cerr << "Aborting execution." << std::endl; return -1; }

  cutlass::KernelHardwareInfo hw_info;
  hw_info.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(hw_info.device_id);

  using ElementAccumulator = float;
  using ElementComputeEpilogue = float;
  using ElementInputA = bfloat16_t;
  using ElementInputB = bfloat16_t;
  using ElementOutput = float;

  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::RowMajor;
  using LayoutC = cutlass::layout::RowMajor;
  using LayoutD = cutlass::layout::RowMajor;

  using GmemTiledCopyA = void;
  using GmemTiledCopyB = void;

  // Workgroup tile — the primary thing to tune.
  using TileShape = Shape<_128, _128, _32>;

  using TiledMma = typename TiledMMAHelper<MMA_Atom<XE_DPAS_TT<8, float, cute::bfloat16_t>>,
        Layout<TileShape>, Layout<Shape<_8, _4, _1>, Stride<_4, _1, _0>>>::TiledMMA;

  constexpr int PipelineStages = 2;
  using GEMMDispatchPolicy = cutlass::gemm::MainloopXeL1Staged<PipelineStages>;
  using EpilogueDispatchPolicy = cutlass::epilogue::IntelXeGeneric;

  using EpilogueOp = cutlass::epilogue::fusion::LinearCombination<ElementOutput, ElementComputeEpilogue,
          ElementAccumulator, ElementAccumulator, cutlass::FloatRoundStyle::round_to_nearest>;

  using FusionCallbacks = cutlass::epilogue::fusion::FusionCallbacks<EpilogueDispatchPolicy, EpilogueOp, TileShape,
          decltype(tile_shape(TiledMma()))>;

  using CollectiveEpilogue = cutlass::epilogue::collective::CollectiveEpilogue<
          EpilogueDispatchPolicy,
          TileShape,
          void,
          ElementAccumulator,
          cutlass::gemm::TagToStrideC_t<LayoutC>,
          ElementOutput,
          cutlass::gemm::TagToStrideC_t<LayoutD>,
          FusionCallbacks,
          void,
          void>;

  using CollectiveMainloop = cutlass::gemm::collective::CollectiveMma<
          GEMMDispatchPolicy,
          TileShape,
          ElementInputA,
          cutlass::gemm::TagToStrideA_t<LayoutA>,
          ElementInputB,
          cutlass::gemm::TagToStrideB_t<LayoutB>,
          TiledMma,
          GmemTiledCopyA, void, void, cute::identity,
          GmemTiledCopyB, void, void, cute::identity
  >;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
          Shape<int, int, int, int>,
          CollectiveMainloop,
          CollectiveEpilogue
  >;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

  ExampleRunner<Gemm> runner;
  CUTLASS_CHECK(runner.run(options, hw_info));

  return 0;
}
