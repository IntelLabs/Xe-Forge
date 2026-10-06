"""vISA knowledge example (reduction_sum): sum of each row of up to 128 columns. The work-group has two sub-groups and no shared local memory, so sub-group 0 does the whole row (per-lane partial sums, then a cross-lane tree) and sub-group 1 exits."""

import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def subgroup_row_sum_kernel(x_ptr, out_ptr, n_cols, stride, BLOCK_C: tl.constexpr):
    r = tl.program_id(0)
    c = tl.arange(0, BLOCK_C)
    x = tl.load(x_ptr + r * stride + c, mask=c < n_cols, other=0.0)
    tl.store(out_ptr + r, tl.sum(x, axis=0))


class Model(nn.Module):
    def forward(self, x):
        rows, cols = x.shape
        out = torch.empty((rows,), device=x.device, dtype=x.dtype)
        subgroup_row_sum_kernel[(rows,)](x, out, cols, x.stride(0), BLOCK_C=128, num_warps=2)
        return out
