"""Lowering ladder L2: row sums of a 2-D tensor; one program per row, one block covers the row."""

import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def row_sum_kernel(x_ptr, out_ptr, n_cols, stride_row, BLOCK_N: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < n_cols
    x = tl.load(x_ptr + row * stride_row + cols, mask=mask, other=0.0)
    tl.store(out_ptr + row, tl.sum(x, axis=0))


class Model(nn.Module):
    def forward(self, x):
        rows, cols = x.shape
        out = torch.empty((rows,), device=x.device, dtype=x.dtype)
        row_sum_kernel[(rows,)](x, out, cols, x.stride(0), BLOCK_N=2048)
        return out
