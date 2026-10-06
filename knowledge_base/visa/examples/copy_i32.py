"""vISA knowledge example (copy): int32 copy, one element per work-item, no mask."""

import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def copy_i32_kernel(src_ptr, dst_ptr, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tl.store(dst_ptr + offs, tl.load(src_ptr + offs))


class Model(nn.Module):
    def forward(self, x):
        out = torch.empty_like(x)
        copy_i32_kernel[(x.numel() // 128,)](x, out, BLOCK=128)
        return out
