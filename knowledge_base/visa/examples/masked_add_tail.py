"""vISA knowledge example (masked_vector_add): fp32 add with a tail mask, one element per work-item."""

import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def masked_add_kernel(a_ptr, b_ptr, c_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    a = tl.load(a_ptr + offs, mask=m, other=0.0)
    b = tl.load(b_ptr + offs, mask=m, other=0.0)
    tl.store(c_ptr + offs, a + b, mask=m)


class Model(nn.Module):
    def forward(self, a, b):
        c = torch.empty_like(a)
        n = a.numel()
        masked_add_kernel[(triton.cdiv(n, 128),)](a, b, c, n, BLOCK=128)
        return c
