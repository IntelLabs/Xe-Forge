"""
GEMM Kernel Template for Intel XPU
===================================

Basic matrix multiplication: C = A @ B

This template demonstrates:
- Tensor descriptors for memory access (preferred on XPU)
- Tile swizzling for cache efficiency
- XPU-specific autotune configurations
- Mixed precision (bf16/fp16 inputs, fp32 accumulator)
- Proper grid setup for swizzling

Usage:
    Adapt this template for your specific GEMM kernel.
    Key sections marked with "TODO" for customization.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def swizzle_tile(tile_id, M, N, K, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, GROUP_SIZE_M):
    """
    Reusable tile swizzling helper for L2 cache efficiency.

    Maps linear tile_id to (pid_m, pid_n) with GROUP_SIZE_M grouping.
    When GROUP_SIZE_M == 0, falls back to linear mapping (no swizzle).
    """
    grid_m = tl.cdiv(M, BLOCK_SIZE_M)
    grid_n = tl.cdiv(N, BLOCK_SIZE_N)

    # Swizzle only if GROUP_SIZE_M > 0
    if GROUP_SIZE_M > 0:
        width = GROUP_SIZE_M * grid_n
        group_id = tile_id // width
        group_size = tl.minimum(GROUP_SIZE_M, grid_m - group_id * GROUP_SIZE_M)
        pid_m = group_id * GROUP_SIZE_M + (tile_id % group_size)
        pid_n = (tile_id % width) // group_size
    else:
        # Linear mapping (no swizzle)
        pid_m = tile_id // grid_n
        pid_n = tile_id % grid_n

    return pid_m, pid_n


@triton.autotune(
    configs=[
        # Large tiles for square GEMMs
        triton.Config(
            {"BLOCK_M": 256, "BLOCK_N": 256, "BLOCK_K": 32, "GROUP_SIZE_M": 4, "grf_mode": "256"},
            num_warps=32,
            num_stages=2,
        ),
        triton.Config(
            {"BLOCK_M": 256, "BLOCK_N": 256, "BLOCK_K": 64, "GROUP_SIZE_M": 4, "grf_mode": "256"},
            num_warps=16,
            num_stages=3,
        ),
        # Medium tiles
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 64, "GROUP_SIZE_M": 4, "grf_mode": "256"},
            num_warps=8,
            num_stages=4,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_SIZE_M": 4, "grf_mode": "256"},
            num_warps=16,
            num_stages=3,
        ),
        # Skinny-M configs (for M < 256)
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_SIZE_M": 2, "grf_mode": "256"},
            num_warps=8,
            num_stages=4,
        ),
        triton.Config(
            {"BLOCK_M": 32, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_SIZE_M": 2, "grf_mode": "256"},
            num_warps=4,
            num_stages=5,
        ),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def gemm_kernel(
    # Pointers
    a_ptr,
    b_ptr,
    c_ptr,
    # Matrix dimensions
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    # Strides
    stride_am: tl.constexpr,
    stride_ak: tl.constexpr,
    stride_bk: tl.constexpr,
    stride_bn: tl.constexpr,
    stride_cm: tl.constexpr,
    stride_cn: tl.constexpr,
    # Meta-parameters (NO defaults!)
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    """
    Optimized GEMM kernel for Intel XPU.

    Computes C[M, N] = A[M, K] @ B[K, N]
    """
    # Get tile indices with swizzling
    pid = tl.program_id(0)
    pid_m, pid_n = swizzle_tile(pid, M, N, K, BLOCK_M, BLOCK_N, BLOCK_K, GROUP_SIZE_M)

    # Create tensor descriptors (preferred on XPU — better codegen than block pointers)
    a_desc = tl.make_tensor_descriptor(
        base=a_ptr,
        shape=[M, K],
        strides=[stride_am, stride_ak],
        block_shape=[BLOCK_M, BLOCK_K],
    )

    b_desc = tl.make_tensor_descriptor(
        base=b_ptr,
        shape=[K, N],
        strides=[stride_bk, stride_bn],
        block_shape=[BLOCK_K, BLOCK_N],
    )

    # Allocate accumulator (fp32 for numerical stability)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Tile offsets
    off_m = pid_m * BLOCK_M
    off_n = pid_n * BLOCK_N

    # K-loop: accumulate partial products
    for off_k in range(0, K, BLOCK_K):
        # Load tiles by coordinate
        a = a_desc.load([off_m, off_k])
        b = b_desc.load([off_k, off_n])

        # Convert to reduced precision for fast dot
        a = a.to(tl.bfloat16)
        b = b.to(tl.bfloat16)

        # Matrix multiply-accumulate
        acc += tl.dot(a, b)

    # Store result
    c_desc = tl.make_tensor_descriptor(
        base=c_ptr,
        shape=[M, N],
        strides=[stride_cm, stride_cn],
        block_shape=[BLOCK_M, BLOCK_N],
    )
    c_desc.store([off_m, off_n], acc)


def matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    PyTorch-compatible wrapper for GEMM kernel.

    Args:
        a: Input tensor [M, K]
        b: Input tensor [K, N]

    Returns:
        Output tensor [M, N]
    """
    # Ensure inputs are on XPU and contiguous
    device = torch.device("xpu")
    a = a.to(device, torch.float16).contiguous()
    b = b.to(device, torch.float16).contiguous()

    # Get dimensions
    assert a.shape[1] == b.shape[0], "Incompatible dimensions for matmul"
    M, K = a.shape
    K, N = b.shape

    # Allocate output
    c = torch.empty((M, N), device=device, dtype=torch.float16)

    # Launch kernel with 1D grid (for swizzling)
    grid = lambda META: (triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),)

    gemm_kernel[grid](
        a,
        b,
        c,
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        c.stride(0),
        c.stride(1),
    )

    return c


# TODO: Adapt this template for your specific use case
# - Add bias in epilogue if needed
# - Fuse activation functions
# - Support batched operations
# - Adjust autotune configs for your workload

if __name__ == "__main__":
    # Simple test
    torch.manual_seed(0)
    a = torch.randn(1024, 512, device="xpu", dtype=torch.float16)
    b = torch.randn(512, 2048, device="xpu", dtype=torch.float16)

    # PyTorch baseline
    c_torch = torch.matmul(a, b)

    # Triton kernel
    c_triton = matmul(a, b)

    # Check correctness
    torch.testing.assert_close(c_triton, c_torch, rtol=1e-2, atol=1e-2)
    print("✓ Correctness test passed!")
