"""Lowering ladder L1: out = x + y.T over a 2-D tile, with arbitrary strides on every operand."""

import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def strided_add_kernel(
    x_ptr, y_ptr, out_ptr, M, N,
    stride_xm, stride_xn, stride_ym, stride_yn, stride_om, stride_on,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = (rm[:, None] < M) & (rn[None, :] < N)
    x = tl.load(x_ptr + rm[:, None] * stride_xm + rn[None, :] * stride_xn, mask=mask)
    y = tl.load(y_ptr + rm[:, None] * stride_ym + rn[None, :] * stride_yn, mask=mask)
    tl.store(out_ptr + rm[:, None] * stride_om + rn[None, :] * stride_on, x + y, mask=mask)


class Model(nn.Module):
    def forward(self, x, y):
        y_t = y.t()  # non-contiguous view
        M, N = x.shape
        out = torch.empty((M, N), device=x.device, dtype=x.dtype)
        grid = (triton.cdiv(M, 16), triton.cdiv(N, 32))
        strided_add_kernel[grid](
            x, y_t, out, M, N,
            x.stride(0), x.stride(1), y_t.stride(0), y_t.stride(1), out.stride(0), out.stride(1),
            BLOCK_M=16, BLOCK_N=32,
        )
        return out
