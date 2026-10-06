"""vISA knowledge example (vector_mul): fp32 product scaled by a runtime float scalar, with a tail mask."""

import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def scaled_mul_kernel(a_ptr, b_ptr, c_ptr, alpha, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    a = tl.load(a_ptr + offs, mask=m)
    b = tl.load(b_ptr + offs, mask=m)
    tl.store(c_ptr + offs, a * b * alpha, mask=m)


class Model(nn.Module):
    def forward(self, a, b):
        c = torch.empty_like(a)
        n = a.numel()
        scaled_mul_kernel[(triton.cdiv(n, 128),)](a, b, c, 0.5, n, BLOCK=128)
        return c
