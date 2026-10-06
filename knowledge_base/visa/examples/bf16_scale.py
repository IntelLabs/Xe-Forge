"""vISA knowledge example (conversions): bf16 in and out, f32 arithmetic: y = bf16(2 * f32(x)), with a tail mask."""

import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def bf16_scale_kernel(x_ptr, y_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    x = tl.load(x_ptr + offs, mask=m).to(tl.float32)
    tl.store(y_ptr + offs, (x * 2.0 + 1.0).to(tl.bfloat16), mask=m)


class Model(nn.Module):
    def forward(self, x):
        y = torch.empty_like(x)
        n = x.numel()
        bf16_scale_kernel[(triton.cdiv(n, 128),)](x, y, n, BLOCK=128)
        return y
