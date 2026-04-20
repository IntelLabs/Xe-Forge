# ruff: noqa: E731
import torch
import torch.nn as nn
import triton
import triton.language as tl

batch_size = 1024
in_features = 8192
out_features = 8192


def get_inputs():
    return [torch.rand(batch_size, in_features, dtype=torch.float16)]  # ← float16


def get_init_inputs():
    return [in_features, out_features]


@triton.jit
def _linear_bias_kernel(
    a_ptr,
    b_ptr,
    bias_ptr,
    c_ptr,
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

        a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
        b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
        a_tile = tl.load(a_ptrs, mask=(mask_m[:, None] & mask_k[None, :]), other=0.0)
        b_tile = tl.load(b_ptrs, mask=(mask_k[:, None] & mask_n[None, :]), other=0.0)

        # ← upcast before dot; use two-arg form so acc stays fp32
        acc += tl.dot(a_tile.to(tl.float32), b_tile.to(tl.float32))

    bias_vals = tl.load(bias_ptr + offs_n, mask=mask_n, other=0.0).to(tl.float32)  # ← upcast
    acc = acc + bias_vals[None, :]

    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc, mask=mask_m[:, None] & mask_n[None, :])


# _swish_div_clamp_tanh_clamp_kernel is UNCHANGED —
# it only ever reads/writes out1/out2 which are fp32
@triton.jit
def _swish_div_clamp_tanh_clamp_kernel(
    x_ptr, y_ptr, M, N, stride_row, stride_col, BLOCK_COL: tl.constexpr
):
    row_idx = tl.program_id(0)
    col_blk = tl.program_id(1)
    offs_c = col_blk * BLOCK_COL + tl.arange(0, BLOCK_COL)
    offs_r = row_idx * stride_row
    ptrs_in = x_ptr + offs_r + offs_c * stride_col
    ptrs_out = y_ptr + offs_r + offs_c * stride_col
    mask = offs_c < N

    x = tl.load(ptrs_in, mask=mask, other=0.0)

    exp_n = tl.exp(-x)
    sig = 1.0 / (1.0 + exp_n)
    y = x * sig
    y = y * 0.5
    y = tl.where(y < -1.0, -1.0, tl.where(y > 1.0, 1.0, y))
    y2 = y * 2.0
    exp_n2 = tl.exp(-y2)
    sig2 = 1.0 / (1.0 + exp_n2)
    t = sig2 * 2.0 - 1.0
    t = tl.where(t < -1.0, -1.0, tl.where(t > 1.0, 1.0, t))
    tl.store(ptrs_out, t, mask=mask)


def kernel_function(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    assert x.ndim == 2 and weight.ndim == 2 and bias.ndim == 1
    M, K = x.shape
    N, K_w = weight.shape
    assert K == K_w
    assert bias.shape[0] == N
    assert (
        x.dtype == torch.float16 and weight.dtype == torch.float16 and bias.dtype == torch.float16
    )  # ← float16

    # CPU fallback — upcast to fp32 for all math, return fp32
    if not (hasattr(torch, "xpu") and torch.xpu.is_available()):
        xf = x.float()
        wf = weight.float()
        bf = bias.float()
        t = xf.matmul(wf.t()).add(bf)
        s = t * (1.0 / (1.0 + torch.exp(-t)))
        s = s * 0.5
        s = torch.clamp(s, min=-1.0, max=1.0)
        s = torch.tanh(s)
        return torch.clamp(s, min=-1.0, max=1.0)

    device = "xpu"

    # out1 is fp32 — _linear_bias_kernel accumulates in fp32
    out1 = torch.empty((M, N), dtype=torch.float32, device=device)
    stride_bk, stride_bn = weight.stride(1), weight.stride(0)
    BLOCK_M, BLOCK_N, BLOCK_K = 128, 128, 64
    _linear_bias_kernel[(triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))](
        x,
        weight,
        bias,
        out1,
        M,
        N,
        K,
        x.stride(0),
        x.stride(1),
        stride_bk,
        stride_bn,
        out1.stride(0),
        out1.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
    )

    # out2 is fp32 — swish kernel only touches fp32 out1
    out2 = torch.empty((M, N), dtype=torch.float32, device=device)
    BLOCK_COL = 256
    _swish_div_clamp_tanh_clamp_kernel[(M, triton.cdiv(N, BLOCK_COL))](
        out1, out2, M, N, out1.stride(0), out1.stride(1), BLOCK_COL=BLOCK_COL
    )

    torch.xpu.synchronize()
    return out2


class Model(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)

        self.weight = nn.Parameter(
            torch.empty(self.out_features, self.in_features, dtype=torch.float16)  # ← float16
        )
        self.bias = nn.Parameter(torch.empty(self.out_features, dtype=torch.float16))  # ← float16

        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        bound = 1.0 / (self.in_features**0.5) if self.in_features > 0 else 0.0
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        if x.dtype != torch.float16:  # ← float16
            x = x.half()
        return kernel_function(x, self.weight, self.bias)
