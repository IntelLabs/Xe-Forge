"""Lowering ladder L1: in-place axpy, y = a * x + y, with a runtime scalar `a`."""

import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def axpy_kernel(x_ptr, y_ptr, a, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(y_ptr + offsets, a * x + y, mask=mask)


class Model(nn.Module):
    def forward(self, x, y):
        n = x.numel()
        grid = (triton.cdiv(n, 1024),)
        axpy_kernel[grid](x, y, 1.75, n, BLOCK_SIZE=1024)
        return y
