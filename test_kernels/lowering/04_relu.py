"""Lowering ladder L0: relu_kernel (no mask: n is always a multiple of BLOCK_SIZE)."""

import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def relu_kernel(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    x = tl.load(x_ptr + offsets)
    tl.store(out_ptr + offsets, tl.maximum(x, 0.0))


class Model(nn.Module):
    def forward(self, x):
        out = torch.empty_like(x)
        n = out.numel()
        grid = (n // 1024,)
        relu_kernel[grid](x, out, n, BLOCK_SIZE=1024)
        return out
