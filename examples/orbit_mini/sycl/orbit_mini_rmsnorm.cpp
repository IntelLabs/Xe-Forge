// orbit_mini_rmsnorm.cpp — hand-written SYCL RMSNorm, registered as a PyTorch
// dispatcher op (Xe-Orbit plan §11, §13, §15.2).
//
// WHY THIS FILE EXISTS
// --------------------
// §11 is explicit that SYCL is not the exception in Xe-Orbit's kernel taxonomy:
// torch-xpu-ops, IPEX, vLLM-XPU and sgl-kernel-xpu all ship hand-written SYCL
// behind registered dispatcher ops. A reference workload with only Triton
// kernels would leave the entire compiled-language half of the pipeline
// untested. So orbit_mini carries exactly one SYCL kernel, and it covers:
//
//   * build-graph closure (§11.3). For a compiled kernel the closure does NOT
//     come from an AST walk — it comes from the build system.
//     `compile_commands.json` gives the exact compile line for this translation
//     unit: every include path, every define, every flag. CMakeLists.txt next
//     door sets CMAKE_EXPORT_COMPILE_COMMANDS for precisely that reason.
//   * the compiler-option sweep and the icpx harness (§11.4).
//   * the P1 rung of the patch-back ladder (§13) on a SYCL op rather than a
//     Triton kernel. Because this is registered on the XPU dispatch key, an
//     optimized rebuild shadows it as a small out-of-tree extension — no fork
//     of PyTorch, no patched libtorch_xpu, and a revert that is just not
//     loading the library.
//
// WHAT IT IS NOT
// --------------
// It is NOT required to build. orbit_mini must run on a CPU-only machine with
// no oneAPI, no icpx and no XPU runtime; `kernels/sycl_op.py` probes for the op
// and falls back to torch when it is absent, which is always the case on the
// CPU-only CI tier. Nothing in the Python package imports or dlopens this
// unless ORBIT_MINI_SYCL_LIB points at a built library.
//
// BUILD
// -----
//   cmake -S . -B build -DCMAKE_CXX_COMPILER=icpx \
//         -DCMAKE_PREFIX_PATH="$(python -c 'import torch;print(torch.utils.cmake_prefix_path)')"
//   cmake --build build
//   export ORBIT_MINI_SYCL_LIB=$PWD/build/liborbit_mini_sycl.so
//
// NUMERICS
// --------
// Matched to `kernels/sycl_op.py:_rmsnorm_torch_fallback` and to the Triton
// path in `kernels/rmsnorm.py`: accumulate the sum of squares in fp32, scale by
// rsqrt(mean + eps), multiply by the learned per-channel weight. The three
// implementations must agree within the tightened tolerance or §12.12 step 3
// fails, which is the intended signal.

#include <ATen/ATen.h>
#include <torch/library.h>
#include <torch/torch.h>

#include <cstdint>

#ifdef ORBIT_MINI_HAS_SYCL
#include <c10/xpu/XPUStream.h>
#include <sycl/sycl.hpp>
#endif

namespace orbit_mini {

namespace {

constexpr int64_t kSubgroupSize = 32;

// Deliberately kept as a named constant rather than a literal: the extraction
// stage has to notice that this translation unit has compile-time constants of
// its own, exactly as the Triton path has module-level constexpr values.
constexpr int64_t kMaxRowElems = 8192;

void check_inputs(const at::Tensor& input, const at::Tensor& weight) {
  TORCH_CHECK(input.dim() >= 2, "orbit_mini::rmsnorm_xpu expects a >=2-D input");
  TORCH_CHECK(weight.dim() == 1, "orbit_mini::rmsnorm_xpu expects a 1-D weight");
  TORCH_CHECK(input.size(-1) == weight.size(0),
              "orbit_mini::rmsnorm_xpu: last dim of input must match weight");
  TORCH_CHECK(input.size(-1) <= kMaxRowElems,
              "orbit_mini::rmsnorm_xpu: row too wide for the single-pass kernel");
  TORCH_CHECK(input.scalar_type() == at::kFloat || input.scalar_type() == at::kHalf ||
                  input.scalar_type() == at::kBFloat16,
              "orbit_mini::rmsnorm_xpu: unsupported dtype");
}

}  // namespace

#ifdef ORBIT_MINI_HAS_SYCL

// One work-group per row. Each work-group reduces the row's sum of squares in
// fp32, then applies the scale and the learned weight in a second pass over the
// row. Simple on purpose: this is a subject for optimization, not an example of
// it. The whole point of the pipeline is that a candidate rewrite of this
// kernel should beat it.
template <typename scalar_t>
class RmsNormXpuKernel {
 public:
  RmsNormXpuKernel(const scalar_t* input, const scalar_t* weight, scalar_t* output,
                   int64_t n_cols, int64_t stride_row, float eps,
                   sycl::local_accessor<float, 1> scratch)
      : input_(input),
        weight_(weight),
        output_(output),
        n_cols_(n_cols),
        stride_row_(stride_row),
        eps_(eps),
        scratch_(scratch) {}

  void operator()(sycl::nd_item<1> item) const {
    const int64_t row = static_cast<int64_t>(item.get_group(0));
    const int64_t lane = static_cast<int64_t>(item.get_local_id(0));
    const int64_t group_size = static_cast<int64_t>(item.get_local_range(0));

    const scalar_t* row_in = input_ + row * stride_row_;
    scalar_t* row_out = output_ + row * n_cols_;

    float partial = 0.0f;
    for (int64_t i = lane; i < n_cols_; i += group_size) {
      const float v = static_cast<float>(row_in[i]);
      partial += v * v;
    }

    scratch_[lane] = partial;
    item.barrier(sycl::access::fence_space::local_space);

    for (int64_t offset = group_size / 2; offset > 0; offset >>= 1) {
      if (lane < offset) {
        scratch_[lane] += scratch_[lane + offset];
      }
      item.barrier(sycl::access::fence_space::local_space);
    }

    const float mean_sq = scratch_[0] / static_cast<float>(n_cols_);
    const float scale = sycl::rsqrt(mean_sq + eps_);

    for (int64_t i = lane; i < n_cols_; i += group_size) {
      const float v = static_cast<float>(row_in[i]) * scale;
      row_out[i] = static_cast<scalar_t>(v * static_cast<float>(weight_[i]));
    }
  }

 private:
  const scalar_t* input_;
  const scalar_t* weight_;
  scalar_t* output_;
  int64_t n_cols_;
  int64_t stride_row_;
  float eps_;
  sycl::local_accessor<float, 1> scratch_;
};

at::Tensor rmsnorm_xpu(const at::Tensor& input, const at::Tensor& weight, double eps) {
  check_inputs(input, weight);

  // The launch wrapper on the Python side deliberately does not call
  // .contiguous(), so a non-contiguous input reaches here. Handling it is part
  // of the kernel's real behaviour (§12.6 step 5) and the row stride is part of
  // the launch record the bundle has to reproduce (§12.4).
  const at::Tensor x = input.contiguous();
  const at::Tensor w = weight.contiguous();
  at::Tensor out = at::empty_like(x);

  const int64_t n_cols = x.size(-1);
  const int64_t rows = x.numel() / n_cols;

  int64_t group_size = kSubgroupSize;
  while (group_size < n_cols && group_size < 512) {
    group_size <<= 1;
  }

  sycl::queue& queue = c10::xpu::getCurrentXPUStream().queue();

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::kHalf, at::kBFloat16, x.scalar_type(), "orbit_mini::rmsnorm_xpu", [&] {
        const scalar_t* in_ptr = x.const_data_ptr<scalar_t>();
        const scalar_t* w_ptr = w.const_data_ptr<scalar_t>();
        scalar_t* out_ptr = out.mutable_data_ptr<scalar_t>();

        queue.submit([&](sycl::handler& cgh) {
          sycl::local_accessor<float, 1> scratch(
              sycl::range<1>(static_cast<size_t>(group_size)), cgh);
          RmsNormXpuKernel<scalar_t> kernel(in_ptr, w_ptr, out_ptr, n_cols, n_cols,
                                            static_cast<float>(eps), scratch);
          cgh.parallel_for(
              sycl::nd_range<1>(sycl::range<1>(static_cast<size_t>(rows * group_size)),
                                sycl::range<1>(static_cast<size_t>(group_size))),
              kernel);
        });
      });

  return out.view_as(input);
}

#else  // !ORBIT_MINI_HAS_SYCL

// Built without a SYCL compiler: keep the op registerable so the schema, the
// dispatch key and the P1 mechanism can still be exercised, but refuse to run.
// Silently computing something else here would be worse than failing — §13
// verification is a dispatch assertion, and an op that quietly does the wrong
// thing defeats it.
at::Tensor rmsnorm_xpu(const at::Tensor& input, const at::Tensor& weight, double eps) {
  (void)input;
  (void)weight;
  (void)eps;
  TORCH_CHECK(false,
              "orbit_mini::rmsnorm_xpu was built without SYCL support. "
              "Rebuild with icpx, or let kernels/sycl_op.py take the torch fallback.");
}

#endif  // ORBIT_MINI_HAS_SYCL

// Schema. This is the §13 P1 handle: an optimized candidate registers a new
// implementation against this exact name on the XPU key and shadows the one
// below, with the framework left entirely untouched.
TORCH_LIBRARY(orbit_mini, m) {
  m.def("rmsnorm_xpu(Tensor input, Tensor weight, float eps) -> Tensor");
}

TORCH_LIBRARY_IMPL(orbit_mini, XPU, m) {
  m.impl("rmsnorm_xpu", TORCH_FN(rmsnorm_xpu));
}

}  // namespace orbit_mini
