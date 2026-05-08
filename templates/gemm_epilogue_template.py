"""
GEMM with Epilogue Fusion Template for Intel XPU
================================================

Matrix multiplication with fused epilogue operations:
C = activation(A @ B + bias)

This template demonstrates:
- GEMM kernel with fused bias and activation
- Reusable JIT activation helpers
- Module-level compile-time constants
- Proper epilogue fusion patterns

Usage:
    Adapt for your specific GEMM + epilogue pattern.
"""

import math

import torch
import triton
import triton.language as tl

# Module-level compile-time constants
kAlpha = tl.constexpr(math.sqrt(2.0 / math.pi))  # For GeLU
kInvLn2 = tl.constexpr(1.4426950408889634)  # 1/ln(2) for exp2


@triton.jit
def sigmoid_exp2(x):
    """
    exp2-based sigmoid for better XPU performance.
    sigmoid(x) = 1 / (1 + exp(-x)) = 1 / (1 + exp2(-x * 1/ln2))
    """
    return 1.0 / (1.0 + tl.math.exp2((-x) * kInvLn2))


@triton.jit
def tanh(x):
    """
    Sigmoid-based tanh: tanh(x) = 2*sigmoid(2x) - 1
    Avoids separate tl.exp or tl.math.tanh call.
    """
    return 2 * tl.sigmoid(2 * x) - 1


@triton.jit
def gelu(x):
    """
    GeLU activation via tanh approximation.
    Matches PyTorch nn.functional.gelu default.
    """
    return 0.5 * x * (1 + tanh(kAlpha * (x + 0.044715 * x * x * x)))


@triton.jit
def silu(x):
    """
    SiLU (Swish) activation: silu(x) = x * sigmoid(x)
    """
    return x * tl.sigmoid(x)


@triton.jit
def swizzle_tile(tile_id, M, N, K, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, GROUP_SIZE_M):
    """Reusable tile swizzling helper."""
    grid_m = tl.cdiv(M, BLOCK_SIZE_M)
    grid_n = tl.cdiv(N, BLOCK_SIZE_N)

    if GROUP_SIZE_M > 0:
        width = GROUP_SIZE_M * grid_n
        group_id = tile_id // width
        group_size = tl.minimum(GROUP_SIZE_M, grid_m - group_id * GROUP_SIZE_M)
        pid_m = group_id * GROUP_SIZE_M + (tile_id % group_size)
        pid_n = (tile_id % width) // group_size
    else:
        pid_m = tile_id // grid_n
        pid_n = tile_id % grid_n

    return pid_m, pid_n


@triton.autotune(
    configs=[
        # Large tiles - use with caution on heavy epilogues
        triton.Config(
            {"BLOCK_M": 256, "BLOCK_N": 256, "BLOCK_K": 32, "GROUP_SIZE_M": 4, "grf_mode": "256"},
            num_warps=16,
            num_stages=2,  # Note: 16 warps, not 32, for epilogue
        ),
        # Medium tiles - good balance for most epilogues
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 64, "GROUP_SIZE_M": 4, "grf_mode": "256"},
            num_warps=8,
            num_stages=4,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_SIZE_M": 4, "grf_mode": "256"},
            num_warps=8,
            num_stages=3,
        ),
        # Skinny-M configs
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
def gemm_epilogue_kernel(
    # Pointers
    a_ptr,
    b_ptr,
    c_ptr,
    bias_ptr,
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
    # Epilogue parameters
    use_bias: tl.constexpr,
    activation: tl.constexpr,  # 'none', 'relu', 'gelu', 'silu', 'sigmoid'
    # Meta-parameters (NO defaults!)
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    """
    GEMM kernel with fused bias and activation.

    Computes: C = activation(A @ B + bias)
    """
    # Get tile indices
    pid = tl.program_id(0)
    pid_m, pid_n = swizzle_tile(pid, M, N, K, BLOCK_M, BLOCK_N, BLOCK_K, GROUP_SIZE_M)

    # Tensor descriptors (preferred on XPU — better codegen than block pointers)
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

    # Accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Tile offsets
    off_m = pid_m * BLOCK_M
    off_n = pid_n * BLOCK_N

    # K-loop
    for off_k in range(0, K, BLOCK_K):
        a = a_desc.load([off_m, off_k])
        b = b_desc.load([off_k, off_n])

        a = a.to(tl.bfloat16)
        b = b.to(tl.bfloat16)

        acc += tl.dot(a, b)

    # Epilogue: bias + activation
    if use_bias:
        # Load bias (vector broadcast over M dimension)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        mask_n = offs_n < N
        bias = tl.load(bias_ptr + offs_n, mask=mask_n, other=0.0)
        acc = acc + bias[None, :]  # Broadcast

    # Apply activation
    if activation == "relu":
        acc = tl.maximum(acc, 0.0)
    elif activation == "gelu":
        acc = gelu(acc)
    elif activation == "silu":
        acc = silu(acc)
    elif activation == "sigmoid":
        acc = sigmoid_exp2(acc)
    # else: no activation (linear)

    # Store result
    c_desc = tl.make_tensor_descriptor(
        base=c_ptr,
        shape=[M, N],
        strides=[stride_cm, stride_cn],
        block_shape=[BLOCK_M, BLOCK_N],
    )
    c_desc.store([off_m, off_n], acc)


def matmul_epilogue(
    a: torch.Tensor, b: torch.Tensor, bias: torch.Tensor = None, activation: str = "none"
) -> torch.Tensor:
    """
    PyTorch-compatible wrapper for GEMM + epilogue kernel.

    Args:
        a: Input tensor [M, K]
        b: Input tensor [K, N]
        bias: Optional bias vector [N]
        activation: 'none', 'relu', 'gelu', 'silu', 'sigmoid'

    Returns:
        Output tensor [M, N]
    """
    # Move to XPU and ensure contiguous
    device = torch.device("xpu")
    a = a.to(device, torch.float16).contiguous()
    b = b.to(device, torch.float16).contiguous()

    if bias is not None:
        bias = bias.to(device, torch.float16).contiguous()

    # Get dimensions
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    M, K = a.shape
    K, N = b.shape

    if bias is not None:
        assert bias.shape[0] == N, "Bias dimension mismatch"

    # Allocate output
    c = torch.empty((M, N), device=device, dtype=torch.float16)

    # Launch kernel with 1D grid
    grid = lambda META: (triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),)

    gemm_epilogue_kernel[grid](
        a,
        b,
        c,
        bias if bias is not None else a,  # dummy pointer if no bias
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        c.stride(0),
        c.stride(1),
        bias is not None,
        activation,
    )

    return c


# TODO: Customize this template
# - Add more activation functions (tanh, hardtanh, etc.)
# - Support element-wise operations (divide, scale, clamp)
# - Add residual connections
# - Implement batched version

if __name__ == "__main__":
    # Test
    torch.manual_seed(0)
    M, K, N = 1024, 512, 2048

    a = torch.randn(M, K, device="xpu", dtype=torch.float16)
    b = torch.randn(K, N, device="xpu", dtype=torch.float16)
    bias = torch.randn(N, device="xpu", dtype=torch.float16)

    # PyTorch baseline
    c_torch = torch.matmul(a, b) + bias
    c_torch = torch.nn.functional.gelu(c_torch)

    # Triton kernel
    c_triton = matmul_epilogue(a, b, bias, activation="gelu")

    # Check correctness
    torch.testing.assert_close(c_triton, c_torch, rtol=1e-2, atol=1e-2)
    print("✓ Correctness test passed!")
