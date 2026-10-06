"""vISA knowledge example (exp): fp32 natural exponential with a tail mask (vISA exp is base 2)."""

import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def exp_kernel(x_ptr, y_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    x = tl.load(x_ptr + offs, mask=m)
    tl.store(y_ptr + offs, tl.exp(x), mask=m)


class Model(nn.Module):
    def forward(self, x):
        y = torch.empty_like(x)
        n = x.numel()
        exp_kernel[(triton.cdiv(n, 128),)](x, y, n, BLOCK=128)
        return y
