# OPTIMIZED KERNEL: matmul_AT (C = A^T @ B) for Intel XPU
# A: [K, M], B: [K, N], C: [M, N]
#
# Optimizations applied:
# 1. Block pointers (tl.make_block_ptr) instead of manual arithmetic
# 2. Large tiles 256x256 for XPU
# 3. 32 warps (XPU optimal)
# 4. grf_mode='256' for large register file
# 5. GROUP_SIZE_M swizzling for L2 cache
# 6. 1D grid with swizzling
# 7. Proper fp16 dot with fp32 accumulation

import torch
import torch.nn as nn
import triton
import triton.language as tl


def get_xpu_configs():
    """Intel XPU optimized autotune configs."""
    return [
        # Primary: Large tiles, 32 warps, grf_mode='256'
        triton.Config(
            {
                "BLOCK_M": 256,
                "BLOCK_N": 256,
                "BLOCK_K": 32,
                "GROUP_SIZE_M": 4,
                "grf_mode": "256",
            },
            num_warps=32,
            num_stages=2,
        ),
        triton.Config(
            {
                "BLOCK_M": 256,
                "BLOCK_N": 128,
                "BLOCK_K": 32,
                "GROUP_SIZE_M": 4,
                "grf_mode": "256",
            },
            num_warps=32,
            num_stages=2,
        ),
        triton.Config(
            {
                "BLOCK_M": 128,
                "BLOCK_N": 256,
                "BLOCK_K": 32,
                "GROUP_SIZE_M": 4,
                "grf_mode": "256",
            },
            num_warps=32,
            num_stages=2,
        ),
        triton.Config(
            {
                "BLOCK_M": 128,
                "BLOCK_N": 128,
                "BLOCK_K": 64,
                "GROUP_SIZE_M": 4,
                "grf_mode": "256",
            },
            num_warps=32,
            num_stages=2,
        ),
    ]


@triton.autotune(
    configs=get_xpu_configs(),
    key=["M", "N", "K"],
)
@triton.jit
def _matmul_at_kernel_optimized(
    # Pointers
    A_ptr,
    B_ptr,
    C_ptr,
    # Dimensions
    M,
    N,
    K,
    # Strides
    stride_ak,  # A is [K, M], stride along K
    stride_am,  # A is [K, M], stride along M
    stride_bk,  # B is [K, N], stride along K
    stride_bn,  # B is [K, N], stride along N
    stride_cm,  # C is [M, N], stride along M
    stride_cn,  # C is [M, N], stride along N
    # Block sizes from autotune (NO default values!)
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    """
    Optimized C = A^T @ B for Intel XPU.

    A: [K, M], B: [K, N], C: [M, N]

    Uses block pointers, tile swizzling, and XPU-optimal configs.
    """
    # === 1D GRID WITH GROUP_SIZE_M SWIZZLING ===
    # Better L2 cache reuse than 2D grid
    pid = tl.program_id(0)

    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n

    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_SIZE_M)

    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # === BLOCK POINTERS ===
    # A is [K, M] - we load [BLOCK_K, BLOCK_M] tiles, then transpose for dot
    A_block_ptr = tl.make_block_ptr(
        base=A_ptr,
        shape=(K, M),
        strides=(stride_ak, stride_am),
        offsets=(0, pid_m * BLOCK_M),
        block_shape=(BLOCK_K, BLOCK_M),
        order=(1, 0),
    )

    # B is [K, N] - we load [BLOCK_K, BLOCK_N] tiles
    B_block_ptr = tl.make_block_ptr(
        base=B_ptr,
        shape=(K, N),
        strides=(stride_bk, stride_bn),
        offsets=(0, pid_n * BLOCK_N),
        block_shape=(BLOCK_K, BLOCK_N),
        order=(1, 0),
    )

    # === ACCUMULATOR ===
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # === MAIN LOOP ===
    for _ in range(0, K, BLOCK_K):
        # Load with boundary checking (block pointers handle this efficiently)
        A_tile = tl.load(A_block_ptr, boundary_check=(0, 1))  # [BLOCK_K, BLOCK_M]
        B_tile = tl.load(B_block_ptr, boundary_check=(0, 1))  # [BLOCK_K, BLOCK_N]

        # Transpose A: [BLOCK_K, BLOCK_M] -> [BLOCK_M, BLOCK_K]
        A_tile_T = A_tile.T

        # Dot product: [BLOCK_M, BLOCK_K] @ [BLOCK_K, BLOCK_N] -> [BLOCK_M, BLOCK_N]
        # Use fp16 for dot, accumulate in fp32
        acc = tl.dot(A_tile_T.to(tl.float16), B_tile.to(tl.float16), acc=acc)

        # Advance block pointers
        A_block_ptr = tl.advance(A_block_ptr, (BLOCK_K, 0))
        B_block_ptr = tl.advance(B_block_ptr, (BLOCK_K, 0))

    # === STORE RESULT ===
    C_block_ptr = tl.make_block_ptr(
        base=C_ptr,
        shape=(M, N),
        strides=(stride_cm, stride_cn),
        offsets=(pid_m * BLOCK_M, pid_n * BLOCK_N),
        block_shape=(BLOCK_M, BLOCK_N),
        order=(1, 0),
    )
    tl.store(C_block_ptr, acc.to(tl.float16), boundary_check=(0, 1))


class Model(nn.Module):
    """
    Optimized matmul_AT: C = A^T @ B
    A: [K, M], B: [K, N], C: [M, N]
    """

    def __init__(self):
        super().__init__()

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        K, M = A.shape
        _, N = B.shape

        # Ensure contiguous
        A = A.contiguous()
        B = B.contiguous()

        C = torch.empty((M, N), device=A.device, dtype=A.dtype)

        # 1D grid for swizzling (total tiles)
        def grid(META):
            return (triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),)

        _matmul_at_kernel_optimized[grid](
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
