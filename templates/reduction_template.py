"""
Reduction Kernel Template for Intel XPU
=======================================

Row-wise reduction operations (sum, mean, softmax, layernorm, etc.)

This template demonstrates:
- Multi-row tiling for better utilization
- Hardware work group size query
- Power-of-2 block sizing
- Warp size autotuning
- Proper masking for boundary conditions

Usage:
    Adapt for your specific reduction pattern.
"""

import torch
import triton
import triton.language as tl
from triton.runtime import driver


def get_max_work_group_size():
    """Query Intel XPU max work group size."""
    device = torch.xpu.current_device()
    properties = driver.active.utils.get_device_properties(device)
    return properties["max_work_group_size"]


@triton.autotune(
    configs=[
        # warp_size=32 variants
        triton.Config({"warp_size": 32}, num_warps=32),
        triton.Config({"warp_size": 32}, num_warps=16),
        triton.Config({"warp_size": 32}, num_warps=8),
        triton.Config({"warp_size": 32}, num_warps=4),
        # warp_size=16 variants (allows more warps)
        triton.Config({"warp_size": 16}, num_warps=64),
        triton.Config({"warp_size": 16}, num_warps=32),
        triton.Config({"warp_size": 16}, num_warps=16),
        triton.Config({"warp_size": 16}, num_warps=8),
    ],
    key=["n_cols"],
)
@triton.jit
def row_sum_kernel(
    # Pointers
    x_ptr,
    output_ptr,
    # Dimensions
    n_rows: tl.constexpr,
    n_cols: tl.constexpr,
    # Strides
    stride_xm: tl.constexpr,
    stride_xn: tl.constexpr,
    # Block size
    BLOCK_SIZE_X: tl.constexpr,
    BLOCK_SIZE_Y: tl.constexpr,
):
    """
    Row-wise sum: output[i] = sum(x[i, :])

    Processes BLOCK_SIZE_Y rows per program.
    """
    # Program handles BLOCK_SIZE_Y rows
    pid = tl.program_id(0)
    row_start = pid * BLOCK_SIZE_Y

    # Row offsets
    row_offs = row_start + tl.arange(0, BLOCK_SIZE_Y)
    row_mask = row_offs < n_rows

    # Column offsets (power-of-2 block)
    col_offs = tl.arange(0, BLOCK_SIZE_X)
    col_mask = col_offs < n_cols

    # Pointers for this block
    x_ptrs = x_ptr + row_offs[:, None] * stride_xm + col_offs[None, :] * stride_xn

    # Load and sum
    mask = row_mask[:, None] & col_mask[None, :]
    x = tl.load(x_ptrs, mask=mask, other=0.0)

    # Reduce over columns
    row_sum = tl.sum(x, axis=1)

    # Store result
    output_ptrs = output_ptr + row_offs
    tl.store(output_ptrs, row_sum, mask=row_mask)


@triton.autotune(
    configs=[
        # warp_size variations
        triton.Config({"warp_size": 32}, num_warps=32),
        triton.Config({"warp_size": 32}, num_warps=16),
        triton.Config({"warp_size": 32}, num_warps=8),
        triton.Config({"warp_size": 16}, num_warps=64),
        triton.Config({"warp_size": 16}, num_warps=32),
        triton.Config({"warp_size": 16}, num_warps=16),
    ],
    key=["n_cols"],
)
@triton.jit
def softmax_kernel(
    # Pointers
    x_ptr,
    output_ptr,
    # Dimensions
    n_rows: tl.constexpr,
    n_cols: tl.constexpr,
    # Strides
    stride_xm: tl.constexpr,
    stride_xn: tl.constexpr,
    stride_om: tl.constexpr,
    stride_on: tl.constexpr,
    # Block size
    BLOCK_SIZE_X: tl.constexpr,
    BLOCK_SIZE_Y: tl.constexpr,
):
    """
    Row-wise softmax: output[i, :] = softmax(x[i, :])

    Uses online algorithm with max subtraction for numerical stability.
    """
    # Program handles BLOCK_SIZE_Y rows
    pid = tl.program_id(0)
    row_start = pid * BLOCK_SIZE_Y

    # Row and column offsets
    row_offs = row_start + tl.arange(0, BLOCK_SIZE_Y)
    row_mask = row_offs < n_rows

    col_offs = tl.arange(0, BLOCK_SIZE_X)
    col_mask = col_offs < n_cols

    # Load input
    x_ptrs = x_ptr + row_offs[:, None] * stride_xm + col_offs[None, :] * stride_xn
    mask = row_mask[:, None] & col_mask[None, :]
    x = tl.load(x_ptrs, mask=mask, other=-float("inf"))

    # Numerical stability: subtract max
    row_max = tl.max(x, axis=1, keep_dims=True)
    x_stable = x - row_max

    # Exponentiate
    exp_x = tl.exp(x_stable)

    # Sum for normalization
    row_sum = tl.sum(exp_x, axis=1, keep_dims=True)

    # Normalize
    softmax = exp_x / row_sum

    # Store result
    output_ptrs = output_ptr + row_offs[:, None] * stride_om + col_offs[None, :] * stride_on
    tl.store(output_ptrs, softmax, mask=mask)


def row_sum(x: torch.Tensor) -> torch.Tensor:
    """
    Compute row-wise sum.

    Args:
        x: Input tensor [n_rows, n_cols]

    Returns:
        Output tensor [n_rows]
    """
    device = torch.device("xpu")
    x = x.to(device, torch.float16).contiguous()

    n_rows, n_cols = x.shape

    # Compute block sizes
    BLOCK_SIZE_X = triton.next_power_of_2(n_cols)
    MAX_WORK_GROUP_SIZE = get_max_work_group_size()
    BLOCK_SIZE_Y = MAX_WORK_GROUP_SIZE // BLOCK_SIZE_X
    BLOCK_SIZE_Y = max(BLOCK_SIZE_Y, 1)  # At least 1

    # Allocate output
    output = torch.empty(n_rows, device=device, dtype=torch.float16)

    # Launch kernel
    grid = (triton.cdiv(n_rows, BLOCK_SIZE_Y),)

    row_sum_kernel[grid](
        x,
        output,
        n_rows,
        n_cols,
        x.stride(0),
        x.stride(1),
        BLOCK_SIZE_X=BLOCK_SIZE_X,
        BLOCK_SIZE_Y=BLOCK_SIZE_Y,
    )

    return output


def softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Compute softmax along specified dimension (row-wise only).

    Args:
        x: Input tensor [n_rows, n_cols]
        dim: Dimension to reduce over (must be -1 or 1)

    Returns:
        Output tensor [n_rows, n_cols]
    """
    assert dim in [-1, 1], "This template only supports row-wise softmax"

    device = torch.device("xpu")
    x = x.to(device, torch.float16).contiguous()

    n_rows, n_cols = x.shape

    # Compute block sizes
    BLOCK_SIZE_X = triton.next_power_of_2(n_cols)
    MAX_WORK_GROUP_SIZE = get_max_work_group_size()
    BLOCK_SIZE_Y = MAX_WORK_GROUP_SIZE // BLOCK_SIZE_X
    BLOCK_SIZE_Y = max(BLOCK_SIZE_Y, 1)

    # Allocate output
    output = torch.empty_like(x, device=device, dtype=torch.float16)

    # Launch kernel
    grid = (triton.cdiv(n_rows, BLOCK_SIZE_Y),)

    softmax_kernel[grid](
        x,
        output,
        n_rows,
        n_cols,
        x.stride(0),
        x.stride(1),
        output.stride(0),
        output.stride(1),
        BLOCK_SIZE_X=BLOCK_SIZE_X,
        BLOCK_SIZE_Y=BLOCK_SIZE_Y,
    )

    return output


# TODO: Extend this template for:
# - LayerNorm (mean + variance computation)
# - RMSNorm (root mean square normalization)
# - LogSumExp reduction
# - Max/min reductions
# - Column-wise reductions (transpose first)

if __name__ == "__main__":
    # Test row sum
    torch.manual_seed(0)
    x = torch.randn(4096, 256, device="xpu", dtype=torch.float16)

    # PyTorch baseline
    sum_torch = x.sum(dim=1)

    # Triton kernel
    sum_triton = row_sum(x)

    # Check correctness
    torch.testing.assert_close(sum_triton, sum_torch, rtol=1e-2, atol=1e-2)
    print("✓ Row sum test passed!")

    # Test softmax
    softmax_torch = torch.nn.functional.softmax(x, dim=1)
    softmax_triton = softmax(x, dim=1)

    torch.testing.assert_close(softmax_triton, softmax_torch, rtol=1e-2, atol=1e-2)
    print("✓ Softmax test passed!")
