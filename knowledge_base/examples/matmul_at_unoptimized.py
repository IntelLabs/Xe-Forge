# ruff: noqa: E731
# UNOPTIMIZED KERNEL: matmul_AT (C = A^T @ B)
# A: [K, M], B: [K, N], C: [M, N]
#
# Issues:
# 1. Manual pointer arithmetic (should use block pointers)
# 2. Small tiles (64-128) - XPU prefers 256x256
# 3. Low warps (4-8) - XPU prefers 32
# 4. No grf_mode='256'
# 5. No GROUP_SIZE_M swizzling
# 6. 2D grid (should be 1D with swizzling)
# 7. Transpose inside loop (A_block.T)

import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 16}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=8, num_stages=3),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def _matmul_at_kernel(
    A_ptr,
    B_ptr,
    C_ptr,
    M,
    N,
    K,
    stride_A0,
    stride_A1,
    stride_B0,
    stride_B1,
    stride_C0,
    stride_C1,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    C = A^T @ B
    A: [K, M], B: [K, N], C: [M, N]

    UNOPTIMIZED: Uses manual pointer arithmetic and 2D grid
    """
    # 2D grid (not optimal for cache)
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Manual offset calculation
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)

        # Manual masks
        mask_k = offs_k < K
        mask_m = offs_m < M
        mask_n = offs_n < N

        # Manual pointer arithmetic for A: [K, M] -> load [BLOCK_K, BLOCK_M]
        ptrsA = A_ptr + offs_k[:, None] * stride_A0 + offs_m[None, :] * stride_A1
        A_block = tl.load(ptrsA, mask=mask_k[:, None] & mask_m[None, :], other=0.0)

        # Transpose in loop (suboptimal)
        A_block = A_block.T  # [BLOCK_M, BLOCK_K]

        # Manual pointer arithmetic for B: [K, N] -> load [BLOCK_K, BLOCK_N]
        ptrsB = B_ptr + offs_k[:, None] * stride_B0 + offs_n[None, :] * stride_B1
        B_block = tl.load(ptrsB, mask=mask_k[:, None] & mask_n[None, :], other=0.0)

        acc = tl.dot(A_block, B_block, acc)

    # Manual store
    ptrsC = C_ptr + offs_m[:, None] * stride_C0 + offs_n[None, :] * stride_C1
    mask_C = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(ptrsC, acc, mask=mask_C)


class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        K, M = A.shape
        _, N = B.shape
        C = torch.empty((M, N), device=A.device, dtype=A.dtype)

        # 2D grid
        grid = lambda META: (
            triton.cdiv(M, META["BLOCK_M"]),
            triton.cdiv(N, META["BLOCK_N"]),
        )

        _matmul_at_kernel[grid](
            A,
            B,
            C,
            M,
            N,
            K,
            A.stride(0),
            A.stride(1),
            B.stride(0),
            B.stride(1),
            C.stride(0),
            C.stride(1),
        )
        return C
