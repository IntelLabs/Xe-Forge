// orbit_fused: Xe-Fuse's k2 GEMM+SwiGLU epilogue as loadable torch ops (plan §13.4).
//
// Two ops, composing to the r0/r1 chain with the rmsnorm normalize-pass eliminated:
//
//   add_rms_scale(hidden, residual, eps) -> scale[M]
//       residual += hidden (in place), returns rsqrt(mean(residual^2) + eps) per row.
//       The gamma weight is NOT applied here: it commutes through the GEMM and is
//       folded into the packed B at load time, which is what removes the separate
//       normalized-activation write the unfused path pays for.
//
//   gate_up_swiglu(x, b_packed, scale) -> D[M, N]
//       D = SwiGLU(scale[m] * (x @ b_packed)), Xe-Fuse k2. b_packed is [K, N]
//       row-major with gate/up columns interleaved (even = gate, odd = up) and
//       gamma pre-folded; the SwiGLU result lands in the even columns of D.
//
// Tile dispatch: M <= 16 uses the 16x256x32 instantiation, larger M the 32x256x32
// one — both measured better than the generator's auto pick at their M (plan §13.4).
// The layer patch gates usage at M <= 32, where the fused kernel beats the chain.
//
// Launch-queue note: both ops launch on torch's current in-order XPU stream (the
// cutlass adapter accepts a queue*), with no explicit waits. The first version
// launched on the compat queue with a wait per op — two serializations per layer,
// which converted a kernel-level +3.1% into a measured -2.52% e2e REJECT.

#include "xe-fuse/builder/epilogue_builder.hpp"

#include "cutlass/util/device_memory.h"
#include "cutlass/util/packed_stride.hpp"

#include "sycl_common.hpp"
#include "helper.h"

#include <torch/library.h>
#include <ATen/ATen.h>
#include <c10/xpu/XPUStream.h>

using namespace cute;
namespace b = xe_fuse::builder;
using bf16 = cutlass::bfloat16_t;

namespace orbit_fused {

// ---- add_rms_scale -------------------------------------------------------

at::Tensor add_rms_scale(at::Tensor& hidden, at::Tensor& residual, double epsilon) {
  TORCH_CHECK(hidden.scalar_type() == at::kBFloat16, "bf16 only");
  auto& queue = c10::xpu::getCurrentXPUStream().queue();
  const int64_t k = hidden.size(-1);
  const int64_t m = hidden.numel() / k;
  auto scale = at::empty({m}, hidden.options().dtype(at::kFloat));
  const float eps = static_cast<float>(epsilon);

  using B16 = sycl::ext::oneapi::bfloat16;
  auto* h = reinterpret_cast<B16*>(hidden.data_ptr());
  auto* r = reinterpret_cast<B16*>(residual.data_ptr());
  auto* s = scale.data_ptr<float>();

  constexpr size_t WG = 256;
  queue.submit([&](sycl::handler& cgh) {
    cgh.parallel_for(
        sycl::nd_range<1>(static_cast<size_t>(m) * WG, WG),
        [=](sycl::nd_item<1> it) {
          const int64_t row = it.get_group(0);
          B16* hr = h + row * k;
          B16* rr = r + row * k;
          float local = 0.0f;
          for (int64_t i = it.get_local_id(0); i < k; i += WG) {
            const float x = static_cast<float>(hr[i]) + static_cast<float>(rr[i]);
            rr[i] = static_cast<B16>(x);
            local += x * x;
          }
          const float total =
              sycl::reduce_over_group(it.get_group(), local, sycl::plus<float>());
          if (it.get_local_id(0) == 0) {
            s[row] = sycl::rsqrt(total / k + eps);
          }
        });
  });
  return scale;
}

// ---- gate_up_swiglu ------------------------------------------------------

template <typename TileShape>
struct K2 {
  using EVT = b::SwiGLU<b::ScaleRows<b::Acc, TileShape, float>>;
  using Config = b::MakeGemm<EVT, bf16, bf16, bf16, float, float, TileShape>;
  using Gemm = typename Config::Gemm;

  static void run(sycl::queue& queue, const at::Tensor& x, const at::Tensor& b_packed,
                  const at::Tensor& scale, at::Tensor& out) {
    const int M = static_cast<int>(x.size(0));
    const int K = static_cast<int>(x.size(1));
    const int N = static_cast<int>(b_packed.size(1));
    constexpr int L = 1;

    cutlass::KernelHardwareInfo hw_info;
    hw_info.sm_count =
        cutlass::KernelHardwareInfo::query_device_multiprocessor_count(hw_info.device_id);

    auto stride_A = cutlass::make_cute_packed_stride(
        typename Gemm::GemmKernel::StrideA{}, make_shape(M, K, L));
    auto stride_B = cutlass::make_cute_packed_stride(
        typename Gemm::GemmKernel::StrideB{}, make_shape(N, K, L));
    auto stride_C = cutlass::make_cute_packed_stride(
        typename Config::StrideC{}, make_shape(M, N, L));
    auto stride_D = cutlass::make_cute_packed_stride(
        typename Config::StrideD{}, make_shape(M, N, L));

    typename b::Acc::Arguments accum_args{};
    typename b::ColBroadcast<0, TileShape, float>::Arguments scale_args;
    scale_args.ptr_col = scale.data_ptr<float>();
    scale_args.null_default = 1.0f;
    scale_args.dCol = {cute::Int<1>{}, cute::Int<0>{}, static_cast<int64_t>(M)};
    typename b::MulOp<float, float>::Arguments mul_args{};
    typename b::ScaleRows<b::Acc, TileShape, float>::Arguments rms_args{
        accum_args, scale_args, mul_args};
    typename xe_fuse::XePairwiseCompute<xe_fuse::SwiGLUFn>::Arguments swiglu_args{};
    typename K2::EVT::Arguments evt_args{rms_args, swiglu_args};

    typename Gemm::GemmKernel::EpilogueArguments epilogue_args{
        evt_args, nullptr, stride_C,
        reinterpret_cast<bf16*>(out.data_ptr()), stride_D};
    typename Gemm::GemmKernel::Arguments arguments{
        cutlass::gemm::GemmUniversalMode::kGemm,
        {M, N, K, L},
        {reinterpret_cast<const bf16*>(x.data_ptr()), stride_A,
         reinterpret_cast<const bf16*>(b_packed.data_ptr()), stride_B},
        epilogue_args,
        hw_info};

    Gemm op;
    const size_t ws_bytes = Gemm::get_workspace_size(arguments);
    // Workspace through torch's allocator, so it lives in the same context as the
    // tensors rather than in the compat layer's.
    auto workspace =
        at::empty({static_cast<int64_t>(ws_bytes)}, x.options().dtype(at::kByte));
    TORCH_CHECK(op.can_implement(arguments) == cutlass::Status::kSuccess,
                "k2 cannot run at M=", M, " N=", N, " K=", K);
    CUTLASS_CHECK(op.initialize(arguments, ws_bytes ? workspace.data_ptr() : nullptr));
    // Launch on torch's in-order XPU stream (the adapter takes a queue*), so
    // downstream torch ops serialize correctly with NO explicit wait — the two
    // per-layer waits were what turned a kernel-level win into the measured
    // -2.52% e2e REJECT.
    CUTLASS_CHECK(op.run(&queue));
  }
};

at::Tensor gate_up_swiglu(const at::Tensor& x, const at::Tensor& b_packed,
                          const at::Tensor& scale) {
  TORCH_CHECK(x.dim() == 2 && b_packed.dim() == 2, "x [M,K], b_packed [K,N]");
  TORCH_CHECK(x.size(1) == b_packed.size(0), "K mismatch");
  TORCH_CHECK(x.is_contiguous() && b_packed.is_contiguous(), "contiguous inputs only");
  TORCH_CHECK(x.scalar_type() == at::kBFloat16, "bf16 only");

  auto out = at::empty({x.size(0), b_packed.size(1)}, x.options());
  auto& queue = c10::xpu::getCurrentXPUStream().queue();
  if (x.size(0) <= 16) {
    K2<Shape<_16, _256, _32>>::run(queue, x, b_packed, scale, out);
  } else {
    K2<Shape<_32, _256, _32>>::run(queue, x, b_packed, scale, out);
  }
  return out;
}

}  // namespace orbit_fused

TORCH_LIBRARY(orbit_fused, m) {
  m.def("add_rms_scale(Tensor(a!) hidden, Tensor(b!) residual, float epsilon) -> Tensor");
  m.def("gate_up_swiglu(Tensor x, Tensor b_packed, Tensor scale) -> Tensor");
}

TORCH_LIBRARY_IMPL(orbit_fused, XPU, m) {
  m.impl("add_rms_scale", TORCH_FN(orbit_fused::add_rms_scale));
  m.impl("gate_up_swiglu", TORCH_FN(orbit_fused::gate_up_swiglu));
}
