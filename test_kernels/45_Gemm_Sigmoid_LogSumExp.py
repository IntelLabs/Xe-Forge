# ruff: noqa: E731

import torch
import torch.nn as nn
import triton
import triton.language as tl

# -----------------------------------------------------------------------------
# Triton kernels for Intel XPU platform (float16 inputs, fp32 accumulation)
# -----------------------------------------------------------------------------


@triton.jit
def _gemm_sigmoid_kernel(
    A_ptr,
    B_ptr,
    bias_ptr,
    C_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_m = offs_m < M
    mask_n = offs_n < N

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K

        A_ptrs = A_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
        B_ptrs = B_ptr + offs_n[:, None] * stride_bk + offs_k[None, :] * stride_bn

        A_block = tl.load(
            A_ptrs, mask=(mask_m[:, None] & mask_k[None, :]), other=0.0
        ).to(tl.float16)
        B_block = tl.load(
            B_ptrs, mask=(mask_n[:, None] & mask_k[None, :]), other=0.0
        ).to(tl.float16)

        acc = tl.dot(A_block, B_block.T, acc)

    bias_vec = tl.load(bias_ptr + offs_n, mask=mask_n, other=0.0).to(tl.float32)
    acc = acc + bias_vec[None, :]
    acc = 1.0 / (1.0 + tl.exp(-acc))

    C_ptrs = C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    out_mask = mask_m[:, None] & mask_n[None, :]
    tl.store(C_ptrs, acc.to(tl.float16), mask=out_mask)


@triton.jit
def _gemm_bias_kernel(
    A_ptr,
    B_ptr,
    bias_ptr,
    C_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_m = offs_m < M
    mask_n = offs_n < N

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K

        a_ptrs = A_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
        b_ptrs = B_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

        a = tl.load(a_ptrs, mask=(mask_m[:, None] & mask_k[None, :]), other=0.0).to(
            tl.float16
        )
        b = tl.load(b_ptrs, mask=(mask_k[:, None] & mask_n[None, :]), other=0.0).to(
            tl.float16
        )

        acc = tl.dot(a, b, acc)

    bias_vals = tl.load(bias_ptr + offs_n, mask=mask_n, other=0.0).to(tl.float32)
    acc = acc + bias_vals[None, :]

    c_ptrs = C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    mask_c = mask_m[:, None] & mask_n[None, :]
    tl.store(c_ptrs, acc.to(tl.float16), mask=mask_c)


@triton.jit
def _logsumexp_kernel(
    C_ptr,
    Y_ptr,
    M,
    N,
    stride_cm,
    stride_cn,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    row = pid
    offs_n = tl.arange(0, BLOCK_N)

    m_val = -1e20
    for start in range(0, N, BLOCK_N):
        idx = start + offs_n
        mask = idx < N
        ptrs = C_ptr + row * stride_cm + idx * stride_cn
        vals = tl.load(ptrs, mask=mask, other=-1e20).to(tl.float32)
        cur_max = tl.max(vals, axis=0)
        m_val = tl.maximum(m_val, cur_max)

    sum_val = 0.0
    for start in range(0, N, BLOCK_N):
        idx = start + offs_n
        mask = idx < N
        ptrs = C_ptr + row * stride_cm + idx * stride_cn
        vals = tl.load(ptrs, mask=mask, other=0.0).to(tl.float32)
        exp_v = tl.exp(vals - m_val)
        sum_val += tl.sum(exp_v, axis=0)

    res = tl.log(sum_val) + m_val
    out_ptr = Y_ptr + row
    mask_o = row < M
    tl.store(out_ptr, res, mask=mask_o)


def kernel_function(
    input_tensor: torch.Tensor,
    weight1: torch.Tensor,
    bias1: torch.Tensor,
    weight2: torch.Tensor,
    bias2: torch.Tensor,
) -> torch.Tensor:
    """
    input_tensor: [M, K_in]
    weight1: [H, K_in], bias1: [H]
    weight2: [O, H], bias2: [O]
    Returns: [M], the logsumexp output
    """
    assert input_tensor.ndim == 2 and weight1.ndim == 2 and bias1.ndim == 1
    assert weight2.ndim == 2 and bias2.ndim == 1

    M, K_in = input_tensor.shape
    H, K1 = weight1.shape
    O, H2 = weight2.shape

    assert K1 == K_in, "weight1 inner dim mismatch"
    assert H2 == H, "weight2 inner dim mismatch"
    assert bias1.shape[0] == H and bias2.shape[0] == O

    assert hasattr(torch, "xpu") and torch.xpu.is_available(), "Requires XPU"

    dev = torch.device("xpu")
    target_dtype = torch.float16

    A = input_tensor.to(device=dev, dtype=target_dtype)
    W1 = weight1.to(device=dev, dtype=target_dtype)
    B1 = bias1.to(device=dev, dtype=target_dtype)
    W2 = weight2.to(device=dev, dtype=target_dtype)
    B2 = bias2.to(device=dev, dtype=target_dtype)

    inter = torch.empty((M, H), device=dev, dtype=target_dtype)

    BLOCK_M1, BLOCK_N1, BLOCK_K1 = 128, 128, 16
    grid1 = (triton.cdiv(M, BLOCK_M1), triton.cdiv(H, BLOCK_N1))
    _gemm_sigmoid_kernel[grid1](
        A,
        W1,
        B1,
        inter,
        M,
        H,
        K_in,
        A.stride(0),
        A.stride(1),
        W1.stride(0),
        W1.stride(1),
        inter.stride(0),
        inter.stride(1),
        BLOCK_M1,
        BLOCK_N1,
        BLOCK_K1,
    )

    C2 = torch.empty((M, O), device=dev, dtype=target_dtype)
    W2_t = W2.t()

    BLOCK_M2, BLOCK_N2, BLOCK_K2 = 128, 128, 16
    grid2 = (triton.cdiv(M, BLOCK_M2), triton.cdiv(O, BLOCK_N2))
    _gemm_bias_kernel[grid2](
        inter,
        W2_t,
        B2,
        C2,
        M,
        O,
        H,
        inter.stride(0),
        inter.stride(1),
        W2_t.stride(0),
        W2_t.stride(1),
        C2.stride(0),
        C2.stride(1),
        BLOCK_M2,
        BLOCK_N2,
        BLOCK_K2,
    )

    Y = torch.empty((M,), device=dev, dtype=torch.float32)
    BLOCK_REDUCE = 128
    grid3 = (M,)
    _logsumexp_kernel[grid3](C2, Y, M, O, C2.stride(0), C2.stride(1), BLOCK_REDUCE)

    torch.xpu.synchronize()
    return Y.xpu()


# -----------------------------------------------------------------------------
# KernelBench Model (adapted): forward() calls kernel_function()
# -----------------------------------------------------------------------------
def get_inputs():
    BATCH = 4096
    INPUT_SIZE = 2048
    return [torch.rand(BATCH, INPUT_SIZE, dtype=torch.float16)]


def get_init_inputs():
    # (input_size, hidden_size, output_size)
    return [2048, 4096, 1024]


class Model(nn.Module):
    """
    KernelBench wrapper for:
      x -> Linear1 -> Sigmoid -> Linear2 -> LogSumExp(dim=1)

    Init signature matches YAML inits:
      (INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE)

    Note: kernel_function expects weights as:
      weight1: [H, K_in]
      weight2: [O, H]
    We store exactly those shapes.
    """

    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.output_size = int(output_size)

        self.weight1 = nn.Parameter(
            torch.empty(self.hidden_size, self.input_size, dtype=torch.float32)
        )
        self.bias1 = nn.Parameter(torch.empty(self.hidden_size, dtype=torch.float32))
        self.weight2 = nn.Parameter(
            torch.empty(self.output_size, self.hidden_size, dtype=torch.float32)
        )
        self.bias2 = nn.Parameter(torch.empty(self.output_size, dtype=torch.float32))

        nn.init.kaiming_uniform_(self.weight1, a=5**0.5)
        bound1 = 1.0 / (self.input_size**0.5) if self.input_size > 0 else 0.0
        nn.init.uniform_(self.bias1, -bound1, bound1)

        nn.init.kaiming_uniform_(self.weight2, a=5**0.5)
        bound2 = 1.0 / (self.hidden_size**0.5) if self.hidden_size > 0 else 0.0
        nn.init.uniform_(self.bias2, -bound2, bound2)

    def forward(self, x):
        x = x.to(torch.float16)
        return kernel_function(x, self.weight1, self.bias1, self.weight2, self.bias2)