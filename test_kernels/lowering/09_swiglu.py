"""Lowering ladder L1: SwiGLU, silu(a) * b, bf16 in and out, fp32 arithmetic."""

import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def swiglu_kernel(a_ptr, b_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    a = tl.load(a_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(b_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    silu = a / (1.0 + tl.exp(-a))
    tl.store(out_ptr + offsets, (silu * b).to(tl.bfloat16), mask=mask)


class Model(nn.Module):
    def forward(self, a, b):
        out = torch.empty_like(a)
        n = out.numel()
        grid = (triton.cdiv(n, 1024),)
        swiglu_kernel[grid](a, b, out, n, BLOCK_SIZE=1024)
        return out
