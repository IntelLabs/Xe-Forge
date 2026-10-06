"""vISA knowledge example (relu): fp32 ReLU with a tail mask."""

import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def relu_masked_kernel(x_ptr, y_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    x = tl.load(x_ptr + offs, mask=m)
    tl.store(y_ptr + offs, tl.where(x > 0, x, 0.0), mask=m)


class Model(nn.Module):
    def forward(self, x):
        y = torch.empty_like(x)
        n = x.numel()
        relu_masked_kernel[(triton.cdiv(n, 128),)](x, y, n, BLOCK=128)
        return y
