# ruff: noqa: E731

import torch
import torch.nn as nn
import triton
import triton.language as tl

# ---------------------------------------------------------------------------
# Problem defaults (match YAML bench-gpu)
# ---------------------------------------------------------------------------
batch_size = 1024
in_features = 8192
out_features = 8192


def get_inputs():
    # KernelBench often generates on CPU; Model moves to xpu if available.
    return [torch.rand(batch_size, in_features, dtype=torch.float32)]


def get_init_inputs():
    return [in_features, out_features]


# ---------------------------------------------------------------------------
# Triton kernel: fused Linear + bias (UNCHANGED)
# ---------------------------------------------------------------------------
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
    """
    Computes a BLOCK_M x BLOCK_N tile of out = A @ B + bias.
    A is [M,K], B is treated as [K,N], bias is [N], out is [M,N].
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    m_start = pid_m * BLOCK_M
    n_start = pid_n * BLOCK_N
    offs_m = m_start + tl.arange(0, BLOCK_M)
    offs_n = n_start + tl.arange(0, BLOCK_N)
    mask_m = offs_m < M
    mask_n = offs_n < N
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    # Loop over K dimension
    for k_start in range(0, K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K
        a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
        b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
        a_tile = tl.load(a_ptrs, mask=(mask_m[:, None] & mask_k[None, :]), other=0.0)
        b_tile = tl.load(b_ptrs, mask=(mask_k[:, None] & mask_n[None, :]), other=0.0)
        acc = tl.dot(a_tile, b_tile, acc)
    # add bias
    bias_vals = tl.load(bias_ptr + offs_n, mask=mask_n, other=0.0)
    acc = acc + bias_vals[None, :]
    # store
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    mask_store = mask_m[:, None] & mask_n[None, :]
    tl.store(c_ptrs, acc, mask=mask_store)


# ---------------------------------------------------------------------------
# Triton kernel: Swish -> Div2 -> Clamp -> Tanh -> Clamp (UNCHANGED)
# ---------------------------------------------------------------------------
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
    # Swish: x * sigmoid(x)
    neg_x = -x
    exp_n = tl.exp(neg_x)
    sig = 1.0 / (1.0 + exp_n)
    y = x * sig
    # Divide by 2
    y = y * 0.5
    # Clamp [-1,1]
    y = tl.where(y < -1.0, -1.0, tl.where(y > 1.0, 1.0, y))
    # Tanh: 2*sigmoid(2y)-1
    y2 = y * 2.0
    exp_n2 = tl.exp(-y2)
    sig2 = 1.0 / (1.0 + exp_n2)
    t = sig2 * 2.0 - 1.0
    # Final clamp
    t = tl.where(t < -1.0, -1.0, tl.where(t > 1.0, 1.0, t))
    tl.store(ptrs_out, t, mask=mask)


# ---------------------------------------------------------------------------
# Python wrapper that orchestrates the Triton kernels (UNCHANGED)
# ---------------------------------------------------------------------------
def kernel_function(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """
    Fused forward: out = tanh(clamp(( (x @ weight.T + bias) * sigmoid(...) )/2, -1,1))
    """
    # Basic checks
    assert x.ndim == 2 and weight.ndim == 2 and bias.ndim == 1
    M, K = x.shape
    N, K_w = weight.shape
    assert K == K_w, "Incompatible shapes"
    assert bias.shape[0] == N, "Incompatible bias shape"
    assert (
        x.dtype == torch.float32 and weight.dtype == torch.float32 and bias.dtype == torch.float32
    )
    # CPU fallback if no XPU
    if not (hasattr(torch, "xpu") and torch.xpu.is_available()):
        # linear + bias
        t = x.matmul(weight.t()).add(bias)
        # swish
        s = t * (1.0 / (1.0 + torch.exp(-t)))
        # /2
        s = s * 0.5
        # clamp
        s = torch.clamp(s, min=-1.0, max=1.0)
        # tanh
        s = torch.tanh(s)
        # final clamp
        return torch.clamp(s, min=-1.0, max=1.0)
    # ensure XPU device
    device = "xpu"
    # Linear
    out1 = torch.empty((M, N), dtype=x.dtype, device=device)
    stride_am, stride_ak = x.stride(0), x.stride(1)
    # weight is [N,K] with strides [K,1]; we want B as [K,N]
    stride_bk, stride_bn = weight.stride(1), weight.stride(0)
    stride_cm, stride_cn = out1.stride(0), out1.stride(1)
    BLOCK_M, BLOCK_N, BLOCK_K = 128, 128, 64
    grid1 = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    _linear_bias_kernel[grid1](
        x,
        weight,
        bias,
        out1,
        M,
        N,
        K,
        stride_am,
        stride_ak,
        stride_bk,
        stride_bn,
        stride_cm,
        stride_cn,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
    )
    # Swish->... kernel
    out2 = torch.empty((M, N), dtype=x.dtype, device=device)
    BLOCK_COL = 256
    grid2 = (M, triton.cdiv(N, BLOCK_COL))
    _swish_div_clamp_tanh_clamp_kernel[grid2](
        out1, out2, M, N, out1.stride(0), out1.stride(1), BLOCK_COL=BLOCK_COL
    )
    # synchronize XPU
    torch.xpu.synchronize()
    return out2


# ---------------------------------------------------------------------------
# KernelBench Model (adapted): forward() calls kernel_function()
# ---------------------------------------------------------------------------
class Model(nn.Module):
    """
    KernelBench wrapper for:
      Linear -> Swish -> /2 -> clamp -> tanh -> clamp

    Init signature matches YAML inits:
      (IN_FEAT, OUT_FEAT)

    Note:
    - kernel_function includes a CPU fallback; KernelBench can still run CI on CPU-only.
    """

    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)

        # Store params in shapes kernel_function expects:
        # weight: [N, K], bias: [N]
        self.weight = nn.Parameter(
            torch.empty(self.out_features, self.in_features, dtype=torch.float32)
        )
        self.bias = nn.Parameter(torch.empty(self.out_features, dtype=torch.float32))

        # init similar to nn.Linear
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        bound = 1.0 / (self.in_features**0.5) if self.in_features > 0 else 0.0
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        # KernelBench may pass CPU; keep float32.
        if x.dtype != torch.float32:
            x = x.float()
        return kernel_function(x, self.weight, self.bias)
