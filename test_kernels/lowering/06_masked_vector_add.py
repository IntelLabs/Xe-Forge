"""Lowering ladder L1: vector add where n need not be a multiple of the block."""

import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def masked_add_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    tl.store(out_ptr + offsets, x + y, mask=mask)


class Model(nn.Module):
    def forward(self, x, y):
        out = torch.empty_like(x)
        n = out.numel()
        grid = (triton.cdiv(n, 1024),)
        masked_add_kernel[grid](x, y, out, n, BLOCK_SIZE=1024)
        return out
