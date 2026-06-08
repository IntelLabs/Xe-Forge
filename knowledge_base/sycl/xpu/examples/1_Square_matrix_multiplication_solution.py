import os
import re
import sys
from pathlib import Path

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# IGC workaround: raise vector alias analysis threshold for large CUTLASS kernels
# ---------------------------------------------------------------------------
os.environ.setdefault("IGC_VectorAliasBBThreshold", "100000000000")

# ---------------------------------------------------------------------------
# Tuning knobs — override these to match your problem size.
# Run onednn_gemm_tuning.py to find optimal values.
# ---------------------------------------------------------------------------
TILE_M = 256
TILE_N = 256
TILE_K = 32
GRF_COUNT = 256
SCHEDULER_TYPE = 1  # 0=default, 1=Persistent, 2=StreamK
EPILOGUE_TYPE = 0   # 0=LinComb, 1=ReLU, 2=GELU

# ---------------------------------------------------------------------------
# Inline SYCL kernel source — sycl_tla_gemm_template.cpp
# All tile/GRF/scheduler/epilogue knobs are set via -D flags at compile time.
# ---------------------------------------------------------------------------
_KERNEL_SYCL = r"""
#include <torch/extension.h>
#include <c10/xpu/XPUStream.h>
#include <optional>

#include "cute/tensor.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/device/gemm_universal.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/default_epilogue.hpp"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/epilogue/thread/activation.h"
#include "cutlass/epilogue/fusion/operations.hpp"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/kernel/tile_scheduler.hpp"
#include "cutlass/util/packed_stride.hpp"

using namespace cute;

// ---- Tunable parameters (set via -D flags from oneDNN tuning output) ----

#ifndef TILE_M
#define TILE_M 256
#endif
#ifndef TILE_N
#define TILE_N 256
#endif
#ifndef TILE_K
#define TILE_K 32
#endif
#ifndef GRF_COUNT
#define GRF_COUNT 256
#endif
static_assert(GRF_COUNT == 128 || GRF_COUNT == 256,
              "GRF_COUNT must be 128 or 256.");
#ifndef SCHEDULER_TYPE
#define SCHEDULER_TYPE 1
#endif
#ifndef ELEMENT_INPUT
#define ELEMENT_INPUT cutlass::bfloat16_t
#endif
#ifndef ELEMENT_ACC
#define ELEMENT_ACC float
#endif
#ifndef ELEMENT_OUTPUT
#define ELEMENT_OUTPUT cutlass::bfloat16_t
#endif
#ifndef TORCH_OUTPUT_DTYPE
#define TORCH_OUTPUT_DTYPE torch::kBFloat16
#endif
#ifndef TORCH_OUTPUT_CTYPE
#define TORCH_OUTPUT_CTYPE at::BFloat16
#endif
#ifndef ALIGNMENT
#define ALIGNMENT 8
#endif
#ifndef LAYOUT_A
#define LAYOUT_A 0
#endif
#ifndef LAYOUT_B
#define LAYOUT_B 0
#endif
#ifndef EPILOGUE_TYPE
#define EPILOGUE_TYPE 0
#endif

#define CUTLASS_CHECK(status)                                                   \
  {                                                                             \
    cutlass::Status _err = (status);                                            \
    if (_err != cutlass::Status::kSuccess) {                                    \
      TORCH_CHECK(false, "[CUTLASS] ", cutlassGetStatusString(_err),            \
                  " at ", __FILE__, ":", __LINE__);                             \
    }                                                                           \
  }

// ---- GEMM kernel type definitions ----

using TileShape   = cute::Shape<cute::Int<TILE_M>, cute::Int<TILE_N>, cute::Int<TILE_K>>;
using ClusterShape = cute::Shape<cute::_1, cute::_1, cute::_1>;

#if LAYOUT_A == 1
using LayoutA = cutlass::layout::ColumnMajor;
#else
using LayoutA = cutlass::layout::RowMajor;
#endif
#if LAYOUT_B == 1
using LayoutB = cutlass::layout::ColumnMajor;
#else
using LayoutB = cutlass::layout::RowMajor;
#endif

using StageCount = cutlass::gemm::collective::StageCountAuto;

#if SCHEDULER_TYPE == 2
using KernelSchedule = cutlass::gemm::KernelXeCooperative;
#else
using KernelSchedule = cutlass::gemm::collective::KernelScheduleAuto;
#endif

// Epilogue schedule must match kernel schedule for cooperative mode
#if SCHEDULER_TYPE == 2
using EpilogueSchedule = cutlass::epilogue::XeCooperative;
#else
using EpilogueSchedule = cutlass::epilogue::collective::EpilogueScheduleAuto;
#endif

#if EPILOGUE_TYPE == 1
using EpilogueFusionOp = cutlass::epilogue::fusion::LinCombEltAct<
    cutlass::epilogue::thread::ReLu, ELEMENT_OUTPUT, ELEMENT_ACC, ELEMENT_OUTPUT, ELEMENT_ACC>;
#elif EPILOGUE_TYPE == 2
using EpilogueFusionOp = cutlass::epilogue::fusion::LinCombEltAct<
    cutlass::epilogue::thread::GELU, ELEMENT_OUTPUT, ELEMENT_ACC, ELEMENT_OUTPUT, ELEMENT_ACC>;
#else
using EpilogueFusionOp = cutlass::epilogue::fusion::LinearCombination<
    ELEMENT_OUTPUT, ELEMENT_ACC, ELEMENT_OUTPUT, ELEMENT_ACC>;
#endif

using gemm_epilogue =
  typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Xe20, cutlass::arch::OpClassTensorOp,
    TileShape, ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ELEMENT_ACC, ELEMENT_ACC,
    ELEMENT_OUTPUT, cutlass::layout::RowMajor, ALIGNMENT,
    ELEMENT_OUTPUT, cutlass::layout::RowMajor, ALIGNMENT,
    EpilogueSchedule,
    EpilogueFusionOp
  >::CollectiveOp;

using gemm_mainloop =
  typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Xe20, cutlass::arch::OpClassTensorOp,
    ELEMENT_INPUT, LayoutA, ALIGNMENT,
    ELEMENT_INPUT, LayoutB, ALIGNMENT,
    ELEMENT_ACC,
    TileShape, ClusterShape,
    StageCount, KernelSchedule
  >::CollectiveOp;

#if SCHEDULER_TYPE == 2
using TileScheduler = cutlass::gemm::StreamKScheduler;
#elif SCHEDULER_TYPE == 1
using TileScheduler = cutlass::gemm::PersistentScheduler;
#else
using TileScheduler = void;
#endif

using gemm_kernel_base = cutlass::gemm::kernel::GemmUniversal<
    cute::Shape<int, int, int, int>,
    gemm_mainloop, gemm_epilogue, TileScheduler>;
struct gemm_kernel : public gemm_kernel_base {};
using gemm_device = cutlass::gemm::device::GemmUniversalAdapter<gemm_kernel>;

// ---- PyTorch binding: gemm_forward(A, B, out?, alpha, beta) ----
// Accepts 2-D or 3-D (batched) inputs. Forward-only.

torch::Tensor gemm_forward(torch::Tensor A, torch::Tensor B,
                            std::optional<torch::Tensor> out = std::nullopt,
                            float alpha = 1.0f, float beta = 0.0f) {
  TORCH_CHECK(A.device().is_xpu(), "A must be an XPU tensor");
  TORCH_CHECK(B.device().is_xpu(), "B must be an XPU tensor");
  TORCH_CHECK(A.dtype() == TORCH_OUTPUT_DTYPE, "A dtype mismatch");
  TORCH_CHECK(B.dtype() == TORCH_OUTPUT_DTYPE, "B dtype mismatch");
  TORCH_CHECK(A.dim() == 2 || A.dim() == 3, "A must be 2-D or 3-D");
  TORCH_CHECK(B.dim() == 2 || B.dim() == 3, "B must be 2-D or 3-D");
  TORCH_CHECK(A.dim() == B.dim(), "A and B must have same number of dimensions");

  const bool batched = (A.dim() == 3);
  if (batched) {
    TORCH_CHECK(A.size(0) == B.size(0), "Batch size mismatch");
  }
  // Dimension check: K must match between A and B
#if LAYOUT_A == 1
  const int64_t A_K = A.size(-2);
#else
  const int64_t A_K = A.size(-1);
#endif
#if LAYOUT_B == 1
  const int64_t B_K = B.size(-1);
#else
  const int64_t B_K = B.size(-2);
#endif
  TORCH_CHECK(A_K == B_K, "Dimension mismatch: A's K=", A_K, " != B's K=", B_K);

  A = A.contiguous();
  B = B.contiguous();

  const int L = batched ? static_cast<int>(A.size(0)) : 1;
#if LAYOUT_A == 1
  const int M = static_cast<int>(A.size(-1));
  const int K = static_cast<int>(A.size(-2));
#else
  const int M = static_cast<int>(A.size(-2));
  const int K = static_cast<int>(A.size(-1));
#endif
#if LAYOUT_B == 1
  const int N = static_cast<int>(B.size(-2));
#else
  const int N = static_cast<int>(B.size(-1));
#endif
  const int64_t batch_stride_A = batched ? int64_t(M) * K : int64_t(0);
  const int64_t batch_stride_B = batched ? int64_t(K) * N : int64_t(0);
  const int64_t batch_stride_D = batched ? int64_t(M) * N : int64_t(0);

  torch::Tensor D;
  if (out.has_value()) {
    D = out.value();
    TORCH_CHECK(D.device() == A.device(), "out must be on same device as A");
    TORCH_CHECK(D.dtype() == TORCH_OUTPUT_DTYPE, "out dtype mismatch");
    if (batched) {
      TORCH_CHECK(D.dim() == 3 && D.size(0) == L && D.size(1) == M && D.size(2) == N,
                  "out shape mismatch");
    } else {
      TORCH_CHECK(D.dim() == 2 && D.size(0) == M && D.size(1) == N,
                  "out shape mismatch");
    }
    D = D.contiguous();
  } else {
    if (batched) {
      D = torch::empty({L, M, N}, A.options());
    } else {
      D = torch::empty({M, N}, A.options());
    }
  }

  using coord_t = cutlass::gemm::GemmCoord::Index;
  const int device_idx = A.device().index();
  cutlass::KernelHardwareInfo hw_info;
  hw_info.sm_count =
      cutlass::KernelHardwareInfo::query_device_multiprocessor_count(device_idx);

  gemm_device::Arguments arguments{
    cutlass::gemm::GemmUniversalMode::kGemm,
    {static_cast<coord_t>(M), static_cast<coord_t>(N),
     static_cast<coord_t>(K), static_cast<coord_t>(L)},
    {
      reinterpret_cast<ELEMENT_INPUT*>(A.data_ptr<TORCH_OUTPUT_CTYPE>()),
#if LAYOUT_A == 1
      {cute::Int<1>{}, int64_t(M), batch_stride_A},
#else
      {int64_t(K), cute::Int<1>{}, batch_stride_A},
#endif
      reinterpret_cast<ELEMENT_INPUT*>(B.data_ptr<TORCH_OUTPUT_CTYPE>()),
#if LAYOUT_B == 1
      {int64_t(K), cute::Int<1>{}, batch_stride_B},
#else
      {cute::Int<1>{}, int64_t(N), batch_stride_B},
#endif
    },
    {
      {alpha, beta},
      (beta != 0.f) ? reinterpret_cast<ELEMENT_OUTPUT*>(D.data_ptr<TORCH_OUTPUT_CTYPE>()) : nullptr,
      {int64_t(N), cute::Int<1>{}, batch_stride_D},
      reinterpret_cast<ELEMENT_OUTPUT*>(D.data_ptr<TORCH_OUTPUT_CTYPE>()),
      {int64_t(N), cute::Int<1>{}, batch_stride_D},
    },
    hw_info
  };

  gemm_device gemm_op;
  size_t ws_bytes = gemm_op.get_workspace_size(arguments);
  auto workspace = torch::empty(
      {static_cast<int64_t>(ws_bytes)},
      torch::TensorOptions().dtype(torch::kByte).device(A.device()));

  auto stream = c10::xpu::getCurrentXPUStream(A.device().index());
  sycl::queue* queue = &stream.queue();

#ifndef CUTLASS_BACKEND_DISABLE_CHECKS
  CUTLASS_CHECK(gemm_op.can_implement(arguments));
#endif
  CUTLASS_CHECK(gemm_op.initialize(
      arguments, static_cast<uint8_t*>(workspace.data_ptr()), queue));
  CUTLASS_CHECK(gemm_op(queue));
  return D;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("gemm_forward", [](torch::Tensor A, torch::Tensor B, py::object out,
                           float alpha, float beta) -> torch::Tensor {
    std::optional<torch::Tensor> opt_out;
    if (!out.is_none()) opt_out = out.cast<torch::Tensor>();
    return gemm_forward(A, B, opt_out, alpha, beta);
  },
        "Forward-only GEMM (no autograd backward). "
        "Accepts 2-D or 3-D (batched) inputs.",
        py::arg("A"), py::arg("B"), py::arg("out") = py::none(),
        py::arg("alpha") = 1.0f, py::arg("beta") = 0.0f);
}
"""


# ---------------------------------------------------------------------------
# JIT compilation
# ---------------------------------------------------------------------------

def _get_cutlass_version(cutlass_root: Path) -> str:
    version_h = cutlass_root / "include" / "cutlass" / "version.h"
    if not version_h.exists():
        return "unknown"
    text = version_h.read_text()
    major = re.search(r"#define\s+CUTLASS_MAJOR\s+(\d+)", text)
    minor = re.search(r"#define\s+CUTLASS_MINOR\s+(\d+)", text)
    patch = re.search(r"#define\s+CUTLASS_PATCH\s+(\d+)", text)
    if major and minor and patch:
        return f"{major.group(1)}.{minor.group(1)}.{patch.group(1)}"
    return "unknown"


def _detect_sycl_target() -> str:
    name = torch.xpu.get_device_name(0).lower()
    if any(x in name for x in ("b580", "b570", "b770", "b50", "b60", "bmg-g21")):
        return "bmg-g21"
    if any(x in name for x in ("b70", "bmg-g31")):
        return "bmg-g31"
    if "bmg" in name or "battlemage" in name:
        return "bmg-g21"
    if any(x in name for x in ("a770", "a750", "a580", "a380", "a310", "dg2")):
        return "acm-g10"
    if "max" in name or "pvc" in name:
        return "pvc"
    if "lunar" in name or "lnl" in name:
        return "lnl-m"
    return "bmg-g21"


def _load_extension(sycl_target: str = "bmg-g21"):
    import torch.utils.cpp_extension as _cpp_ext
    from torch.utils.cpp_extension import load_inline
    import shutil

    _cutlass_path = os.environ.get("CUTLASS_PATH")
    if not _cutlass_path:
        sys.exit("ERROR: CUTLASS_PATH environment variable is not set.")
    cutlass_root = Path(_cutlass_path)

    _cutlass_ver = _get_cutlass_version(cutlass_root)
    _ext_name = (
        f"sycl_tla_gemm_{TILE_M}x{TILE_N}x{TILE_K}_"
        f"grf{GRF_COUNT}_sched{SCHEDULER_TYPE}_epi{EPILOGUE_TYPE}_"
        f"{_cutlass_ver}_{sycl_target}"
    ).replace(".", "_").replace("-", "_")

    extra_include_paths = [
        str(cutlass_root / "include"),
        str(cutlass_root / "tools" / "util" / "include"),
    ]

    _cutlass_defines = [
        "-DCUTLASS_ENABLE_SYCL",
        "-DSYCL_INTEL_TARGET",
        "-DCUTLASS_VERSIONS_GENERATED",
        "-DCUTLASS_BACKEND_DISABLE_CHECKS",
        "-DNDEBUG",
    ]

    _tuning_defines = [
        f"-DTILE_M={TILE_M}",
        f"-DTILE_N={TILE_N}",
        f"-DTILE_K={TILE_K}",
        f"-DGRF_COUNT={GRF_COUNT}",
        f"-DSCHEDULER_TYPE={SCHEDULER_TYPE}",
        f"-DEPILOGUE_TYPE={EPILOGUE_TYPE}",
    ]

    extra_sycl_cflags = [
        "-fno-sycl-instrument-device-code",
        "-fsycl-targets=spir64_gen",
        *_cutlass_defines,
        *_tuning_defines,
        "-O3",
        "-Wno-unused-variable",
        "-Wno-unused-local-typedef",
        "-Wno-unused-but-set-variable",
        "-Wno-uninitialized",
        "-Wno-reorder-ctor",
        "-Wno-logical-op-parentheses",
        "-Wno-unused-function",
        "-Wno-unknown-pragmas",
    ]

    _dlink_extra_flags = [
        "-Xspirv-translator",
        "-spirv-ext=+SPV_INTEL_split_barrier,"
        "+SPV_INTEL_2d_block_io,"
        "+SPV_INTEL_subgroup_matrix_multiply_accumulate",
    ]
    if GRF_COUNT == 256:
        _dlink_extra_flags += [
            "-Xs",
            '"-options -cl-intel-256-GRF-per-thread"',
        ]
    for flag in _dlink_extra_flags:
        if flag not in _cpp_ext._SYCL_DLINK_FLAGS:
            _cpp_ext._SYCL_DLINK_FLAGS.append(flag)

    extra_ldflags = []
    _mklroot = Path(os.environ["MKLROOT"]) if "MKLROOT" in os.environ else None
    if _mklroot and (_mklroot / "lib").exists():
        _mkl_lib = _mklroot / "lib"
        extra_ldflags += [
            f"-Wl,-rpath,{_mkl_lib}", f"-L{_mkl_lib}",
            "-lmkl_intel_ilp64",
            "-lmkl_intel_thread",
            "-lmkl_core",
        ]

    icpx = shutil.which("icpx")
    _compiler_lib = None
    if icpx:
        _icpx_lib = Path(icpx).resolve().parent.parent / "lib"
        if _icpx_lib.exists():
            _compiler_lib = _icpx_lib
    if _compiler_lib is None:
        _cmplr_root = (
            Path(os.environ["CMPLR_ROOT"]) if "CMPLR_ROOT" in os.environ
            else None
        )
        if _cmplr_root and (_cmplr_root / "lib").exists():
            _compiler_lib = _cmplr_root / "lib"
    if _compiler_lib:
        extra_ldflags += [
            f"-Wl,-rpath,{_compiler_lib}", f"-L{_compiler_lib}",
            "-liomp5",
        ]

    os.environ["TORCH_XPU_ARCH_LIST"] = sycl_target

    module = load_inline(
        name=_ext_name,
        cpp_sources=[""],
        sycl_sources=[_KERNEL_SYCL],
        extra_cflags=_cutlass_defines + _tuning_defines,
        extra_sycl_cflags=extra_sycl_cflags,
        extra_include_paths=extra_include_paths,
        extra_ldflags=extra_ldflags,
        with_sycl=True,
        verbose=False,
        no_implicit_headers=True,
    )
    return module


# ---------------------------------------------------------------------------
# Compile the extension at module load time (cached by PyTorch)
# ---------------------------------------------------------------------------
SYCL_TARGET = os.environ.get("SYCL_TLA_TARGET") or _detect_sycl_target()
_module = _load_extension(sycl_target=SYCL_TARGET)


# ---------------------------------------------------------------------------
# Model class — drop-in replacement for the reference
# ---------------------------------------------------------------------------

class Model(nn.Module):
    """
    Model that performs square matrix multiplication (C = A * B)
    using sycl_tla_gemm_template with tuning knobs set via -D flags.
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return _module.gemm_forward(A, B)


N = 2048 * 2


def get_inputs():
    A = torch.rand(N, N)
    B = torch.rand(N, N)
    return [A, B]


def get_init_inputs():
    return []
