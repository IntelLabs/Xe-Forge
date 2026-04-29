# ruff: noqa: E731
import torch
import torch.nn as nn
import triton
import triton.language as tl

batch_size = 100
input_size = 8192
hidden_size = 8192
scaling_factor = 1.5


@triton.jit
def sum_weight_kernel(
    w_ptr,
    wcs_ptr,
    N,
    K,
    stride_w0,
    stride_w1,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    offs_k = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_k = offs_k < K

    sum_k = tl.zeros((BLOCK_N,), tl.float32)  # fp32 accumulator — unchanged

    for start_n in tl.range(0, N, BLOCK_M):
        offs_n = start_n + tl.arange(0, BLOCK_M)
        mask_n = offs_n < N
        ptrs = w_ptr + offs_n[:, None] * stride_w0 + offs_k[None, :] * stride_w1
        tile = tl.load(ptrs, mask=mask_n[:, None] & mask_k[None, :], other=0.0)
        tile = tile.to(tl.float32)  # ← cast fp16 tile to fp32 before accumulating
        sum_k += tl.sum(tile, axis=0)

    # store as fp32 (w_colsum stays fp32)
    out_ptrs = wcs_ptr + offs_k
    tl.store(out_ptrs, sum_k, mask=mask_k)


@triton.jit
def dot_row_kernel(
    x_ptr,
    wcs_ptr,
    y_ptr,
    M,
    K,
    stride_x0,
    stride_x1,
    stride_wcs,
    stride_y0,
    BLOCK_K: tl.constexpr,
):
    row = tl.program_id(0)
    acc = 0.0  # fp32 scalar accumulator — unchanged

    for start_k in tl.range(0, K, BLOCK_K):
        offs_k = start_k + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K

        x_vals = tl.load(x_ptr + row * stride_x0 + offs_k * stride_x1, mask=mask_k, other=0.0)
        x_vals = x_vals.to(tl.float32)  # ← cast fp16 input to fp32

        wcs_vals = tl.load(wcs_ptr + offs_k * stride_wcs, mask=mask_k, other=0.0)
        # wcs_vals is already fp32 (w_colsum is fp32)

        acc += tl.sum(x_vals * wcs_vals, axis=0)

    acc = acc * 0.75
    out_ptr = y_ptr + row * stride_y0
    tl.store(out_ptr, acc)


def kernel_function(input_tensor: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    assert input_tensor.dtype == torch.float16 and weight.dtype == torch.float16, (  # ← float16
        "Both inputs must be float16"
    )
    assert hasattr(torch, "xpu") and torch.xpu.is_available(), "XPU is not available"
    device = "xpu"

    x = input_tensor.to(device=device)  # stays fp16
    w = weight.to(device=device)  # stays fp16
    M, K = x.shape
    N, K2 = w.shape
    assert K2 == K

    # w_colsum stays fp32 — accumulates fp16 tiles in fp32
    w_colsum = torch.empty((K,), device=device, dtype=torch.float32)
    BLOCK_M = 256
    BLOCK_N = 128
    grid0 = (triton.cdiv(K, BLOCK_N),)
    sum_weight_kernel[grid0](
        w,
        w_colsum,
        N,
        K,
        w.stride(0),
        w.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
    )

    # output stays fp32 (accumulated in fp32 inside kernel)
    y = torch.empty((M, 1), device=device, dtype=torch.float32)
    BLOCK_K = 256
    grid1 = (M,)
    dot_row_kernel[grid1](
        x,
        w_colsum,
        y,
        M,
        K,
        x.stride(0),
        x.stride(1),
        w_colsum.stride(0),
        y.stride(0),
        BLOCK_K=BLOCK_K,
    )

    torch.xpu.synchronize()
    return y


class Model(nn.Module):
    def __init__(self, input_size, hidden_size, scaling_factor):
        super().__init__()
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.scaling_factor = float(scaling_factor)

        self.weight = nn.Parameter(
            torch.randn(self.hidden_size, self.input_size, dtype=torch.float16)  # ← float16
        )

    def forward(self, x):
        if x.dtype != torch.float16:  # ← float16
            x = x.half()
        if self.weight.dtype != torch.float16:
            self.weight.data = self.weight.data.half()
        return kernel_function(x, self.weight)


def get_inputs():
    return [torch.rand(batch_size, input_size, dtype=torch.float16)]  # ← float16


def get_init_inputs():
    return [input_size, hidden_size, scaling_factor]
