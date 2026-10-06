"""Lowering ladder L1: out[r, c] = x[r, c] + bias[c]; one program per (row, column block)."""

import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def broadcast_add_kernel(x_ptr, bias_ptr, out_ptr, n_cols, stride_row, BLOCK_N: tl.constexpr):
    row = tl.program_id(0)
    col_block = tl.program_id(1)
    cols = col_block * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = cols < n_cols
    x = tl.load(x_ptr + row * stride_row + cols, mask=mask)
    b = tl.load(bias_ptr + cols, mask=mask)
    tl.store(out_ptr + row * stride_row + cols, x + b, mask=mask)


class Model(nn.Module):
    def forward(self, x, bias):
        out = torch.empty_like(x)
        rows, cols = x.shape
        grid = (rows, triton.cdiv(cols, 256))
        broadcast_add_kernel[grid](x, bias, out, cols, x.stride(0), BLOCK_N=256)
        return out
