"""Row sums of a scaled matrix, ``y[i] = scale * sum_j x[i, j]``, as a SYCL kernel on the
CPU device, built with ``torch.utils.cpp_extension.load_inline``. One work-item per row."""

import os

import torch
import torch.nn as nn

# No XPU arch list: load_inline builds generic SPIR-V, compiled by the CPU runtime.
os.environ.setdefault("TORCH_XPU_ARCH_LIST", "")
from torch.utils.cpp_extension import load_inline

rows = 8192
cols = 8192
scale = 0.5

_CPP = "at::Tensor scale_rowsum_sycl_forward(const at::Tensor& x, double scale);"

_SYCL = r"""
#include <torch/extension.h>
#include <sycl/sycl.hpp>

namespace scale_rowsum_sycl_v0 {
static sycl::queue& queue() {
  static sycl::queue q{sycl::cpu_selector_v};
  return q;
}
}  // namespace scale_rowsum_sycl_v0

at::Tensor scale_rowsum_sycl_forward(const at::Tensor& x, double scale) {
  auto xc = x.contiguous();
  auto y = torch::empty({xc.size(0)}, xc.options());
  const float* in = xc.data_ptr<float>();
  float* out = y.data_ptr<float>();
  const int64_t m = xc.size(1);
  const float s = static_cast<float>(scale);
  scale_rowsum_sycl_v0::queue()
      .parallel_for<class scale_rowsum_sycl_v0_rows>(
          sycl::range<1>(xc.size(0)),
          [=](sycl::id<1> i) {
            const float* row = in + i[0] * m;
            float acc = 0.f;
            for (int64_t j = 0; j < m; ++j) acc += row[j];
            out[i[0]] = acc * s;
          })
      .wait();
  return y;
}
"""

_mod = load_inline(
    name="scale_rowsum_sycl_v0_ext",
    cpp_sources=_CPP,
    sycl_sources=_SYCL,
    functions=["scale_rowsum_sycl_forward"],
    verbose=False,
)


class Model(nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.scale = float(scale)

    def forward(self, x):
        return _mod.scale_rowsum_sycl_forward(x, self.scale)


def get_inputs():
    return [torch.rand(rows, cols, dtype=torch.float32)]


def get_init_inputs():
    return [scale]
