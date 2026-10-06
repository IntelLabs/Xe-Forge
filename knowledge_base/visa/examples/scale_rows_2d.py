"""vISA knowledge example (index_2d): 2-D grid: out[r, c] = 2 * x[r, c] with a row stride and a column mask."""

import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def scale_rows_kernel(x_ptr, y_ptr, n_cols, stride, BLOCK_C: tl.constexpr):
    r = tl.program_id(0)
    c = tl.program_id(1) * BLOCK_C + tl.arange(0, BLOCK_C)
    m = c < n_cols
    x = tl.load(x_ptr + r * stride + c, mask=m)
    tl.store(y_ptr + r * stride + c, 2.0 * x, mask=m)


class Model(nn.Module):
    def forward(self, x):
        y = torch.empty_like(x)
        rows, cols = x.shape
        scale_rows_kernel[(rows, triton.cdiv(cols, 128))](x, y, cols, x.stride(0), BLOCK_C=128)
        return y
