"""vISA knowledge example (vector_add): fp32 add, two elements per work-item (unrolled), no mask."""

import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def add256_kernel(a_ptr, b_ptr, c_ptr, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tl.store(c_ptr + offs, tl.load(a_ptr + offs) + tl.load(b_ptr + offs))


class Model(nn.Module):
    def forward(self, a, b):
        c = torch.empty_like(a)
        add256_kernel[(a.numel() // 256,)](a, b, c, BLOCK=256)
        return c
