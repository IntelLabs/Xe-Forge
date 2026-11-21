"""Row sums of a scaled matrix, ``y[i] = scale * sum_j x[i, j]``, as host C++ built with
``torch.utils.cpp_extension.load_inline`` and parallelized over rows with ATen's pool."""

import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

rows = 8192
cols = 8192
scale = 0.5

_CPP = r"""
#include <torch/extension.h>

namespace scale_rowsum_v0 {
at::Tensor forward(const at::Tensor& x, double scale) {
  auto xc = x.contiguous();
  auto y = torch::empty({xc.size(0)}, xc.options());
  const float* in = xc.data_ptr<float>();
  float* out = y.data_ptr<float>();
  const int64_t n = xc.size(0), m = xc.size(1);
  const float s = static_cast<float>(scale);
  at::parallel_for(0, n, 1, [&](int64_t begin, int64_t end) {
    for (int64_t i = begin; i < end; ++i) {
      const float* row = in + i * m;
      float acc = 0.f;
#pragma omp simd reduction(+ : acc)
      for (int64_t j = 0; j < m; ++j) acc += row[j];
      out[i] = acc * s;
    }
  });
  return y;
}
}  // namespace scale_rowsum_v0

at::Tensor scale_rowsum_forward(const at::Tensor& x, double scale) {
  return scale_rowsum_v0::forward(x, scale);
}
"""

_mod = load_inline(
    name="scale_rowsum_v0_ext",
    cpp_sources=_CPP,
    functions=["scale_rowsum_forward"],
    extra_cflags=["-O3", "-fopenmp"],
    extra_ldflags=["-fopenmp"],
    verbose=False,
)


class Model(nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.scale = float(scale)

    def forward(self, x):
        return _mod.scale_rowsum_forward(x, self.scale)


def get_inputs():
    return [torch.rand(rows, cols, dtype=torch.float32)]


def get_init_inputs():
    return [scale]
