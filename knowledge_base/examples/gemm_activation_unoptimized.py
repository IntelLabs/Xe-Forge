# ruff: noqa: E731
# UNOPTIMIZED KERNEL EXAMPLE
# This kernel has multiple performance issues that need to be fixed for Intel XPU
#
# Issues:
# 1. Two separate kernels instead of fused
# 2. Manual pointer arithmetic instead of block pointers
# 3. Small tile sizes (64x64)
# 4. Low warp count (default)
# 5. No grf_mode configuration
# 6. No tile swizzling (GROUP_SIZE_M)
# 7. 2D grid instead of 1D with swizzling

import math

import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _linear_kernel(
    in_ptr,
    wt_ptr,
    bias_ptr,
    out_ptr,
    M,
    N,
    K,
    stride_in0,
    stride_in1,
    stride_w0,
    stride_w1,
    stride_out0,
    stride_out1,
    stride_bias,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    UNOPTIMIZED: Uses manual pointer arithmetic, no block pointers
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Manual offset calculation (SLOW - should use block pointers)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    mask_m = offs_m < M
    mask_n = offs_n < N

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K

        # Manual pointer arithmetic (SLOW)
        in_ptrs = in_ptr + offs_m[:, None] * stride_in0 + offs_k[None, :] * stride_in1
        a = tl.load(in_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)

        wt_ptrs = wt_ptr + offs_n[:, None] * stride_w0 + offs_k[None, :] * stride_w1
        b = tl.load(wt_ptrs, mask=mask_n[:, None] & mask_k[None, :], other=0.0)
        b = b.T  # Transpose

        acc = tl.dot(a.to(tl.float32), b.to(tl.float32), acc)

    # Add bias
    bias = tl.load(bias_ptr + offs_n * stride_bias, mask=mask_n, other=0.0)
    acc = acc + bias[None, :].to(tl.float32)

    # Store with manual pointers
    out_ptrs = out_ptr + offs_m[:, None] * stride_out0 + offs_n[None, :] * stride_out1
    tl.store(out_ptrs, acc.to(tl.float32), mask=mask_m[:, None] & mask_n[None, :])


@triton.jit
def _activation_kernel(
    inp_ptr,
    out_ptr,
    M,
    N,
    stride_row,
    stride_col,
    BLOCK_SIZE: tl.constexpr,
):
    """
    SEPARATE activation kernel (SLOW - should be fused with GEMM)
    swish -> /2 -> clamp -> tanh -> clamp
    """
    pid_col = tl.program_id(0)
    pid_row = tl.program_id(1)

    col_offsets = pid_col * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < N

    row_start = pid_row * stride_row

    ptrs_in = inp_ptr + row_start + col_offsets * stride_col
    ptrs_out = out_ptr + row_start + col_offsets * stride_col

    x = tl.load(ptrs_in, mask=mask, other=0.0)

    # swish
    sig = 1.0 / (1.0 + tl.math.exp(-x))
    swish = x * sig

    # /2, clamp, tanh, clamp
    half = swish * 0.5
    c1 = tl.minimum(tl.maximum(half, -1.0), 1.0)
    exp_p = tl.math.exp(c1)
    exp_n = tl.math.exp(-c1)
    t = (exp_p - exp_n) / (exp_p + exp_n)
    out = tl.minimum(tl.maximum(t, -1.0), 1.0)

    tl.store(ptrs_out, out, mask=mask)


def _forward_unoptimized(x, weight, bias):
    """
    UNOPTIMIZED forward pass - two kernel launches
    """
    M, K = x.shape
    N = weight.shape[0]

    x32 = x.to(torch.float32).contiguous()
    w32 = weight.to(torch.float32).contiguous()
    b32 = bias.to(torch.float32).contiguous()

    # First kernel: GEMM
    linear_out = torch.empty((M, N), device=x.device, dtype=torch.float32)

    BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 64  # Small tiles (SLOW)
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))  # 2D grid (no swizzling)

    _linear_kernel[grid](
        x32,
        w32,
        b32,
        linear_out,
        M,
        N,
        K,
        x32.stride(0),
        x32.stride(1),
        w32.stride(0),
        w32.stride(1),
        linear_out.stride(0),
        linear_out.stride(1),
        b32.stride(0),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
    )

    # Second kernel: Activation (EXTRA kernel launch overhead)
    final_out = torch.empty_like(linear_out)
    BLOCK_SIZE = 256
    grid_act = (triton.cdiv(N, BLOCK_SIZE), M)

    _activation_kernel[grid_act](
        linear_out,
        final_out,
        M,
        N,
        linear_out.stride(0),
        linear_out.stride(1),
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return final_out.to(x.dtype)


class Model(nn.Module):
    """UNOPTIMIZED Model - uses two separate kernels"""

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.bias = None

        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.bias is None:
            bias = torch.zeros(self.out_features, device=x.device, dtype=x.dtype)
        else:
            bias = self.bias
        return _forward_unoptimized(x, self.weight, bias)


# ── test helpers ─────────────────────────────────────────────────────────────
def get_init_inputs():
    return [4096, 4096]  # in_features, out_features


def get_inputs():
    return [torch.rand(1024, 4096, dtype=torch.float32)]
